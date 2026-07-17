"""Unit tests for algaesense_edge.acquisition.camera.

`process_clip` is genuinely hardware-independent (it just reads a video
file with cv2), so it's tested here against a real, directly-written
test video file -- not a stand-in for a camera, just test input data for
pure processing logic, the same way jaxsr-calibration's tests use
synthetic numeric fixtures for pure math functions. Only actually
recording from the physical camera needs `@pytest.mark.hardware`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from algaesense_edge.acquisition.camera import (
    ClipFeatures,
    Picamera2CameraCapture,
    create_hardware_camera_capture,
    process_clip,
)


def _write_solid_color_clip(
    out_path: Path, color_bgr: tuple[int, int, int], duration_s: float, frame_rate_fps: float, resolution_wh: tuple[int, int] = (64, 64)
) -> None:
    """Write a real, readable .mp4 test fixture: every frame the same
    solid color, so `process_clip`'s recovered color/frame-count can be
    checked against a known value."""

    n_frames = max(1, round(duration_s * frame_rate_fps))
    width, height = resolution_wh

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, frame_rate_fps, (width, height))
    frame = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def test_capture_then_process_round_trip_recovers_the_written_color(tmp_path: Path) -> None:
    """
    color_bgr=(120, 150, 100) -- OpenCV's B,G,R order -- means the RGB
    feature vector this should recover is approximately [red=100,
    green=150, blue=120].
    """
    out_path = tmp_path / "clip.mp4"
    _write_solid_color_clip(out_path, color_bgr=(120, 150, 100), duration_s=1.0, frame_rate_fps=10.0)

    features = process_clip(out_path)

    assert isinstance(features, ClipFeatures)
    assert features.frame_count == 10
    """
    abs=5.0 tolerance: mp4v is a lossy codec, so the recovered color
    won't be bit-exact -- confirmed by hand while prototyping this
    module (a written [120,150,100] BGR round-tripped to roughly
    [118,149,99]).
    """
    assert features.rgb[0] == pytest.approx(100.0, abs=5.0)  # red
    assert features.rgb[1] == pytest.approx(150.0, abs=5.0)  # green
    assert features.rgb[2] == pytest.approx(120.0, abs=5.0)  # blue


def test_process_clip_frame_count_matches_duration_times_frame_rate(tmp_path: Path) -> None:
    out_path = tmp_path / "clip.mp4"
    _write_solid_color_clip(out_path, color_bgr=(80, 120, 90), duration_s=2.0, frame_rate_fps=15.0)

    features = process_clip(out_path)

    assert features.frame_count == 30  # 2.0s * 15fps


def test_process_clip_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not open"):
        process_clip(tmp_path / "does_not_exist.mp4")


def test_create_hardware_camera_capture_returns_a_real_picamera2_capture() -> None:
    capture = create_hardware_camera_capture()

    assert isinstance(capture, Picamera2CameraCapture)


def test_picamera2_capture_fails_clearly_without_hardware_extra_installed(tmp_path: Path) -> None:
    """
    picamera2 isn't installed on this dev machine (it's part of the
    Pi-only 'hardware' extra, and needs the real OV5647 camera module
    attached even when installed) -- the first actual recording attempt
    should fail with a clear, actionable ImportError.
    """
    capture = Picamera2CameraCapture()
    with pytest.raises(ImportError, match="hardware"):
        capture.record_clip(duration_s=1.0, frame_rate_fps=10.0, out_path=tmp_path / "clip.h264")


@pytest.mark.hardware
def test_picamera2_records_a_real_clip_from_hardware(tmp_path: Path) -> None:
    """Run only on the Pi, with the OV5647 camera module attached via CSI."""
    capture = create_hardware_camera_capture()
    out_path = tmp_path / "clip.h264"

    result_path = capture.record_clip(duration_s=1.0, frame_rate_fps=10.0, out_path=out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
