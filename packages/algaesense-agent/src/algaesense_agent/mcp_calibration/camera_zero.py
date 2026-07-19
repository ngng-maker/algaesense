"""Guided camera zero-point (blank) biomass calibration: walks the
operator through capturing clean-medium clips, then computes the real
blank baseline.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from jaxsr_calibration.camera.calibration import BiomassCameraModel, compute_blank_baseline, persist_biomass_calibration

from algaesense_agent.mcp_calibration.sessions import (
    CalibrationSession,
    append_step,
    create_session,
    load_session,
    mark_finished,
)


def _next_instruction(session: CalibrationSession) -> str:
    done = len(session.steps)
    min_captures = session.context["min_captures"]

    if done >= min_captures:
        return (
            f"{done} captures recorded (minimum {min_captures} met). Call "
            "finish_camera_zero_session when ready, or keep recording more for a "
            "more stable baseline."
        )

    return (
        f"Step {done + 1}/{min_captures}: capture a clip against clean, cell-free "
        "medium, then record its [red, green, blue] feature vector via "
        "record_camera_zero_step."
    )


def start_camera_zero_session(
    sessions_dir: Path,
    experiment_id: str,
    camera_id: str,
    min_captures: int = 10,
) -> tuple[CalibrationSession, str]:
    context = {"camera_id": camera_id, "min_captures": min_captures}
    session = create_session(sessions_dir, kind="camera_zero", experiment_id=experiment_id, context=context)
    return session, _next_instruction(session)


def record_camera_zero_step(
    sessions_dir: Path, session_id: str, rgb: list[float]
) -> tuple[CalibrationSession, str]:
    step = {"rgb": list(rgb)}
    session = append_step(sessions_dir, session_id, step)
    return session, _next_instruction(session)


def finish_camera_zero_session(
    sessions_dir: Path,
    session_id: str,
    out_dir: Path,
    max_relative_std: float = 0.10,
    now: dt.datetime | None = None,
) -> BiomassCameraModel:
    """Compute the real blank baseline from whatever captures have been
    recorded, and persist it via the existing real persistence function."""

    session = load_session(sessions_dir, session_id)
    camera_id = session.context["camera_id"]
    min_captures = session.context["min_captures"]

    now = now or dt.datetime.now(dt.timezone.utc)
    calibration_run_id = f"camera_zero_{camera_id}_{now.strftime('%Y-%m-%dT%H-%M-%S')}"

    feature_vectors = [step["rgb"] for step in session.steps]
    model = compute_blank_baseline(
        feature_vectors,
        camera_id=camera_id,
        calibration_run_id=calibration_run_id,
        min_captures=min_captures,
        max_relative_std=max_relative_std,
    )

    persist_biomass_calibration(model, out_dir=out_dir)
    mark_finished(sessions_dir, session_id)

    return model
