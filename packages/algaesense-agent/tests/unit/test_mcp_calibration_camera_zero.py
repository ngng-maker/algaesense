"""Unit tests for the guided camera zero-point calibration wizard."""

from __future__ import annotations

from pathlib import Path

from algaesense_agent.mcp_calibration.camera_zero import (
    finish_camera_zero_session,
    record_camera_zero_step,
    start_camera_zero_session,
)


def test_wizard_requires_min_captures_then_computes_real_baseline(tmp_path: Path) -> None:
    session, next_step = start_camera_zero_session(
        tmp_path, experiment_id="exp_01", camera_id="CAM01", min_captures=3
    )
    assert "Step 1/3" in next_step

    """
    Tiny jitter around a fixed [red, green, blue] baseline -- low enough
    relative std to land as PASS, confirming the wizard's feature-vector
    collection wires correctly into the real compute_blank_baseline.
    """
    readings = [[80.0, 120.0, 90.0], [81.0, 119.0, 91.0], [79.0, 121.0, 89.0]]

    for rgb in readings[:-1]:
        session, next_step = record_camera_zero_step(tmp_path, session.session_id, rgb)
        assert "capture a clip" in next_step

    session, next_step = record_camera_zero_step(tmp_path, session.session_id, readings[-1])
    assert "finish_camera_zero_session" in next_step

    out_dir = tmp_path / "calibrations"
    model = finish_camera_zero_session(tmp_path, session.session_id, out_dir=out_dir)

    assert model.camera_id == "CAM01"
    assert model.n_captures == 3
    assert model.status == "PASS"
    assert (out_dir / f"{model.calibration_run_id}.yaml").exists()
