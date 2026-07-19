"""Unit tests for the guided reference-jar cross-sensor calibration wizard."""

from __future__ import annotations

from pathlib import Path

from algaesense_agent.mcp_calibration.reference_jar import (
    finish_reference_jar_session,
    record_reference_jar_reading,
    start_reference_jar_session,
)


def test_wizard_cycles_through_each_sensor_and_computes_real_ratios(tmp_path: Path) -> None:
    session, next_step = start_reference_jar_session(
        tmp_path, experiment_id="exp_01", sensors=["PID01", "PID02", "PID03"], dwell_min=5
    )
    assert "PID01" in next_step

    session, next_step = record_reference_jar_reading(tmp_path, session.session_id, "PID01", 100.0)
    assert "PID02" in next_step

    session, next_step = record_reference_jar_reading(tmp_path, session.session_id, "PID02", 100.0)
    assert "PID03" in next_step

    session, next_step = record_reference_jar_reading(tmp_path, session.session_id, "PID03", 120.0)
    assert "finish_reference_jar_session" in next_step

    """
    Fleet median is 100.0 (PID01/PID02), so PID03's ratio (120/100) should
    come out to 1.2 -- confirms this wizard's DataFrame construction wires
    correctly into the real compute_fleet_ratios.
    """
    ratios = finish_reference_jar_session(tmp_path, session.session_id)

    assert ratios == {"PID01": 1.0, "PID02": 1.0, "PID03": 1.2}


def test_multiple_readings_for_the_same_sensor_are_averaged(tmp_path: Path) -> None:
    session, _ = start_reference_jar_session(tmp_path, experiment_id="exp_01", sensors=["PID01"])

    record_reference_jar_reading(tmp_path, session.session_id, "PID01", 90.0)
    record_reference_jar_reading(tmp_path, session.session_id, "PID01", 110.0)

    ratios = finish_reference_jar_session(tmp_path, session.session_id)

    # Average of 90 and 110 is 100, which is also the (only) fleet median.
    assert ratios == {"PID01": 1.0}
