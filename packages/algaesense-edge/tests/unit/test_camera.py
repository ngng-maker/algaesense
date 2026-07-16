"""Unit tests for algaesense_edge.acquisition.camera.

test_capture_then_process_round_trip is the one worth calling out: it uses
REAL cv2 to write a synthetic clip to a temp file and REAL cv2 to read it
back, on this dev machine, with no camera hardware or mocking of the
video I/O itself -- only the "point a physical camera at something" step is
mocked, not the actual file format handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from algaesense_edge.acquisition.camera import (
    ClipFeatures,
    MockCameraCapture,
    Picamera2CameraCapture,
    create_hardware_camera_capture,
    process_clip,
)


def test_mock_camera_capture_writes_a_real_readable_video_file(tmp_path: Path) -> None:
    capture = MockCameraCapture(color_bgr=(120, 150, 100), resolution_wh=(64, 64))
    out_path = tmp_path / "clip.mp4"

    result_path = capture.record_clip(duration_s=1.0, frame_rate_fps=10.0, out_path=out_path)

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_capture_then_process_round_trip_recovers_the_written_color(tmp_path: Path) -> None:
    # color_bgr=(120, 150, 100) -- OpenCV's B,G,R order -- means the RGB
    # feature vector this should recover is approximately [red=100, green=150, blue=120].
    capture = MockCameraCapture(color_bgr=(120, 150, 100), resolution_wh=(64, 64))
    out_path = tmp_path / "clip.mp4"
    capture.record_clip(duration_s=1.0, frame_rate_fps=10.0, out_path=out_path)

    features = process_clip(out_path)

    assert isinstance(features, ClipFeatures)
    assert features.frame_count == 10
    # abs=5.0 tolerance: mp4v is a lossy codec, so the recovered color won't
    # be bit-exact -- confirmed by hand while prototyping this module (a
    # written [120,150,100] BGR round-tripped to roughly [118,149,99]).
    assert features.rgb[0] == pytest.approx(100.0, abs=5.0)  # red
    assert features.rgb[1] == pytest.approx(150.0, abs=5.0)  # green
    assert features.rgb[2] == pytest.approx(120.0, abs=5.0)  # blue


def test_process_clip_frame_count_matches_duration_times_frame_rate(tmp_path: Path) -> None:
    capture = MockCameraCapture()
    out_path = tmp_path / "clip.mp4"
    capture.record_clip(duration_s=2.0, frame_rate_fps=15.0, out_path=out_path)

    features = process_clip(out_path)

    assert features.frame_count == 30  # 2.0s * 15fps


def test_process_clip_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not open"):
        process_clip(tmp_path / "does_not_exist.mp4")


def test_create_hardware_camera_capture_returns_a_real_picamera2_capture() -> None:
    capture = create_hardware_camera_capture()

    assert isinstance(capture, Picamera2CameraCapture)


def test_picamera2_capture_fails_clearly_without_hardware_extra_installed(tmp_path: Path) -> None:
    # picamera2 isn't installed on this dev machine (it's part of the
    # Pi-only 'hardware' extra, and needs a real Pi camera module attached
    # even when installed) -- the first actual recording attempt should fail
    # with a clear, actionable ImportError.
    capture = Picamera2CameraCapture()
    with pytest.raises(ImportError, match="hardware"):
        capture.record_clip(duration_s=1.0, frame_rate_fps=10.0, out_path=tmp_path / "clip.h264")
