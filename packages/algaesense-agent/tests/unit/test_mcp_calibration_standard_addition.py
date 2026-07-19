"""Unit tests for the guided standard-addition calibration wizard.

Uses a clean, exact linear relationship (voltage = 0.5 + 2.0 * ppm) so a
successful fit with r_squared ~= 1.0 is a real signal that the session ->
DataFrame -> fit_sensitivity_per_sensor wiring works correctly, not that
the model got lucky on noisy data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from algaesense_agent.mcp_calibration.sessions import load_session
from algaesense_agent.mcp_calibration.standard_addition import (
    finish_standard_addition_session,
    record_standard_addition_step,
    start_standard_addition_session,
)

_B0 = 0.5
_B1 = 2.0


def _voltage_for(ppm: float) -> float:
    return _B0 + _B1 * ppm


def test_start_session_with_builtin_gas_resolves_it_and_returns_step_one(tmp_path: Path) -> None:
    session, next_step = start_standard_addition_session(
        tmp_path, experiment_id="exp_01", sensor_id="PID01", calibration_gas_name="isoprene"
    )

    assert session.context["gas_name"] == "isoprene"
    assert session.context["sensor_id"] == "PID01"
    assert "Step 1/4" in next_step
    assert "BASELINE" in next_step


def test_start_session_with_custom_gas_requires_compound_name_and_mw(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="compound_name and mw"):
        start_standard_addition_session(
            tmp_path, experiment_id="exp_01", sensor_id="PID01", calibration_gas_name="custom"
        )


def test_recording_all_steps_then_finishing_fits_the_known_relationship(tmp_path: Path) -> None:
    session, _ = start_standard_addition_session(
        tmp_path, experiment_id="exp_01", sensor_id="PID01", calibration_gas_name="isoprene"
    )

    for expected_ppm in session.context["spike_ppm_list"]:
        session, next_step = record_standard_addition_step(
            tmp_path,
            session.session_id,
            pid_voltage_mv=_voltage_for(expected_ppm),
            sample_t_c=25.0,
            sample_rh_pct=50.0,
            lamp_hours=10.0,
        )

    assert "finish_standard_addition_session" in next_step

    out_dir = tmp_path / "calibrations"
    model = finish_standard_addition_session(
        tmp_path, session.session_id, calibration_run_id="cal_run_01", out_dir=out_dir
    )

    assert model.b0_mv == pytest.approx(_B0, abs=1e-6)
    assert model.b1_mv_per_ppm_asgas == pytest.approx(_B1, abs=1e-6)
    assert model.r_squared > 0.999
    assert model.status == "PASS"

    # persist_calibration's real output files landed where expected.
    assert (out_dir / "cal_run_01.parquet").exists()
    assert (out_dir / "cal_run_01.yaml").exists()

    finished_session = load_session(tmp_path, session.session_id)
    assert finished_session.status == "finished"


def test_finishing_with_only_one_distinct_level_raises_the_real_fit_error(tmp_path: Path) -> None:
    session, _ = start_standard_addition_session(
        tmp_path,
        experiment_id="exp_01",
        sensor_id="PID01",
        calibration_gas_name="isoprene",
        spike_ppm_list=[0.0],
    )
    session, _ = record_standard_addition_step(
        tmp_path, session.session_id, pid_voltage_mv=0.5, sample_t_c=25.0, sample_rh_pct=50.0, lamp_hours=10.0
    )

    with pytest.raises(ValueError, match="at least 2 distinct"):
        finish_standard_addition_session(
            tmp_path, session.session_id, calibration_run_id="cal_run_02", out_dir=tmp_path / "calibrations"
        )
