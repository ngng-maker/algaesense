"""Guided reference-jar cross-sensor drift check: walks the operator
through dwelling each sensor at the shared reference jar, then computes
real fleet ratios.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from jaxsr_calibration.calibration.reference_jar import compute_fleet_ratios

from algaesense_agent.mcp_calibration.sessions import (
    CalibrationSession,
    append_step,
    create_session,
    load_session,
    mark_finished,
)


def _next_instruction(session: CalibrationSession) -> str:
    sensors = session.context["sensors"]
    dwell_min = session.context["dwell_min"]

    """
    Multiple readings per sensor are allowed (compute_fleet_ratios
    averages over every row sharing a sensor_id) -- "remaining" only
    tracks which sensors have at least one reading yet, so the operator
    can record extra dwell readings for a sensor without that changing
    which sensor gets suggested next.
    """
    recorded = {step["sensor_id"] for step in session.steps}
    remaining = [s for s in sensors if s not in recorded]

    if not remaining:
        return "All sensors have at least one reading. Call finish_reference_jar_session when ready."

    return (
        f"Step: disconnect sensor {remaining[0]!r}, connect it to the reference jar, "
        f"dwell {dwell_min} minutes, then record its reading via record_reference_jar_reading."
    )


def start_reference_jar_session(
    sessions_dir: Path,
    experiment_id: str,
    sensors: list[str],
    dwell_min: int = 10,
) -> tuple[CalibrationSession, str]:
    context = {"sensors": sensors, "dwell_min": dwell_min}
    session = create_session(sessions_dir, kind="reference_jar", experiment_id=experiment_id, context=context)
    return session, _next_instruction(session)


def record_reference_jar_reading(
    sessions_dir: Path, session_id: str, sensor_id: str, pid_voltage_mv: float
) -> tuple[CalibrationSession, str]:
    step = {"sensor_id": sensor_id, "pid_voltage_mv": pid_voltage_mv}
    session = append_step(sessions_dir, session_id, step)
    return session, _next_instruction(session)


def finish_reference_jar_session(sessions_dir: Path, session_id: str) -> dict[str, float]:
    """Compute real fleet ratios from whatever readings have been recorded."""

    session = load_session(sessions_dir, session_id)
    df = pl.DataFrame(session.steps)

    ratios = compute_fleet_ratios(df)
    mark_finished(sessions_dir, session_id)

    return ratios
