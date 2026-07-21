"""Camera clip capture and processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


"""
Split into two halves with very different testability, same reasoning as
voc.py: PROCESSING a recorded clip into a feature vector uses
opencv-python (cv2), which is genuinely cross-platform and installs fine
on a normal dev machine -- so that half is real, tested code (real .mp4
fixture files, not a synthetic stand-in for a camera). CAPTURING a clip
needs picamera2, which is Raspberry-Pi-only, so those tests are marked
`@pytest.mark.hardware` and only run for real, on the Pi.

Confirmed real hardware (2026-07-16): Raspberry Pi Camera Module v1
(OV5647 sensor), connected over CSI (the ribbon-cable camera port, not
USB), via `picamera2`.
"""


@dataclass
class ClipFeatures:
    """The result of processing one recorded clip."""

    """
    Its per-frame-averaged [red, green, blue] feature vector (matching
    jaxsr_calibration.logging_.schema.CAMERA_RAW_SCHEMA's fixed column
    order) plus how many frames actually contributed to that average.
    """

    rgb: list[float]

    frame_count: int


def process_clip(video_path: Path) -> ClipFeatures:
    """Open a recorded video clip and compute its MEAN [red, green, blue]
    feature vector."""

    """
    Averaged across every frame -- the "mean across frames" decision made
    earlier for how a clip's many frames collapse into the one feature
    vector CAMERA_RAW_SCHEMA stores per capture.
    """

    """
    `cv2.VideoCapture` opens a video file for reading, frame by frame --
    conceptually similar to opening a text file and reading it line by
    line, except each "line" here is a full image (a numpy array).
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"process_clip: could not open video file at {video_path}")

    """
    Accumulate a running sum (not a running mean) across frames, and
    divide once at the end -- simpler and no less accurate than updating a
    mean incrementally, and we don't know the frame count in advance
    without a separate (sometimes unreliable) metadata read.
    """
    channel_sums = np.zeros(3, dtype=np.float64)
    frame_count = 0

    try:
        while True:
            """
            `capture.read()` returns (success, frame). `frame` is a
            (height, width, 3) array of pixel values; `success` is False
            once there are no more frames (or the file couldn't be read),
            which is how this loop knows when to stop -- no need to know
            the total frame count ahead of time.
            """
            got_frame, frame = capture.read()
            if not got_frame:
                break

            """
            OpenCV reads color channels in BGR order (blue, green, red) --
            a long-standing OpenCV quirk, not a mistake -- so frame[:, :,
            0] is blue, not red. `.mean(axis=(0, 1))` averages over every
            pixel's row and column, leaving one value per color channel: a
            single [mean_blue, mean_green, mean_red] triple summarizing
            this one frame.
            """
            channel_sums += frame.mean(axis=(0, 1))
            frame_count += 1
    finally:
        """
        Always release the capture handle, even if something above raised
        -- same reasoning as SMBus.close() in jaxsr_calibration's scan_i2c.
        """
        capture.release()

    if frame_count == 0:
        raise ValueError(f"process_clip: {video_path} contains no readable frames")

    mean_bgr = channel_sums / frame_count

    """
    Reorder BGR -> [red, green, blue] here, once, at the boundary where
    OpenCV's convention meets this project's fixed convention (matching
    CAMERA_RAW_SCHEMA and
    jaxsr_calibration.camera.calibration.greenness_index) -- so nothing
    downstream of this function ever has to think about BGR vs RGB again.
    """
    mean_rgb = [float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])]

    return ClipFeatures(rgb=mean_rgb, frame_count=frame_count)


class CameraCapture(Protocol):
    """Anything that can record a video clip of a given length to a file."""

    def record_clip(self, duration_s: float, frame_rate_fps: float, out_path: Path) -> Path:
        """Record for `duration_s` seconds at `frame_rate_fps`, writing the
        result to `out_path`, and return that same path."""
        ...


@dataclass
class Picamera2CameraCapture:
    """Real `CameraCapture` backed by the Raspberry Pi camera module, via
    `picamera2` (this package's `[hardware]` extra)."""

    """
    picamera2 is the current standard library for the Pi camera (it
    replaced the older, now-deprecated `picamera`).

    Confirmed on real hardware 2026-07-21: `H264Encoder` always writes a
    raw H.264 elementary stream regardless of the output path's extension
    -- `run_camera_tick` (service.py) names clips `.mp4`, so `process_clip`
    (via `cv2.VideoCapture`) reliably failed to open them (`moov atom not
    found` -- the file genuinely isn't a valid MP4 container, no matter
    the OpenCV build). Fixed by muxing directly into a real MP4 via
    picamera2's `FfmpegOutput`, which shells out to the real `ffmpeg`
    binary (a required system dependency now, not just an occasional
    manual-remux tool -- see docs/hardware_setup.md).
    """

    resolution_wh: tuple[int, int] = (640, 480)

    def record_clip(self, duration_s: float, frame_rate_fps: float, out_path: Path) -> Path:
        try:
            from picamera2 import Picamera2
            from picamera2.encoders import H264Encoder
            from picamera2.outputs import FfmpegOutput
        except ImportError as exc:
            raise ImportError(
                "Picamera2CameraCapture requires the 'hardware' extra "
                "(picamera2). Install with `pip install algaesense-edge[hardware]` "
                "on a Raspberry Pi with a camera module attached."
            ) from exc

        import time

        camera = Picamera2()

        """
        `create_video_configuration` builds the settings dict picamera2
        needs before it can start capturing -- resolution comes from
        self.resolution_wh (set to match CameraConfig.resolution_wh,
        jaxsr_calibration.camera.config, when this is constructed).
        `controls={"FrameRate": ...}` is picamera2's documented way to
        request a specific capture frame rate (matching
        CameraConfig.frame_rate_fps) rather than leaving it at whatever the
        sensor's default is.
        """
        video_config = camera.create_video_configuration(
            main={"size": self.resolution_wh}, controls={"FrameRate": frame_rate_fps}
        )
        camera.configure(video_config)
        encoder = H264Encoder()

        """
        `FfmpegOutput` (rather than a plain path string) pipes the
        encoder's raw H.264 output through the real `ffmpeg` binary,
        which muxes it into whatever container format `out_path`'s
        extension actually calls for -- so a `.mp4` path now genuinely
        contains a playable MP4, not a raw stream wearing an `.mp4` name.
        """
        output = FfmpegOutput(str(out_path))

        """
        `camera.close()` is required, not optional -- confirmed on real
        hardware that skipping it breaks every capture after the first
        one in the same process (test or real service run alike):
        `Picamera2.__init__` registers every instance with a global
        camera-manager singleton, which keeps a live reference and holds
        the device in `Configured` state until explicitly closed. Without
        this, a NEW `Picamera2()` on the next call fails with
        `RuntimeError: Failed to acquire camera: Device or resource busy`
        -- Python's own garbage collection never reaches it, since the
        manager singleton, not just this local variable, is holding the
        reference. This is exactly the failure mode a real long-running
        `algaesense-edge start` process would hit on its second hourly
        camera capture, not just a test-isolation artifact.
        """
        try:
            camera.start_recording(encoder, output)
            time.sleep(duration_s)
        finally:
            camera.stop_recording()
            camera.close()

        return out_path


def create_hardware_camera_capture() -> CameraCapture:
    """Construct the real picamera2-backed camera capture (see
    `Picamera2CameraCapture`)."""
    return Picamera2CameraCapture()
