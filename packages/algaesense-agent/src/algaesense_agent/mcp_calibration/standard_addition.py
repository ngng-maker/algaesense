"""Guided standard-addition gas calibration: walks the operator through
each spike level, then fits the real sensitivity model.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from jaxsr_calibration.calibration.apply import persist_calibration
from jaxsr_calibration.calibration.models import CalibrationGas, SensitivityModel
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor

from algaesense_agent.mcp_calibration.sessions import (
    CalibrationSession,
    append_step,
    create_session,
    load_session,
    mark_finished,
)


"""
The wizard's own job is only sequencing and data collection -- every
number this eventually produces comes from the real, already-tested
`fit_sensitivity_per_sensor`, not a reimplementation. `spike_ppm_list`
defaults to a baseline (0 ppm) plus three spike levels, giving
`fit_sensitivity_per_sensor` more than the 2 distinct levels it requires
without forcing every session to use exactly this shape -- a caller can
pass a different list, and `finish_standard_addition_session` accepts
whatever's actually been recorded rather than demanding every entry in
the original list.
"""

_DEFAULT_SPIKE_PPM_LIST = [0.0, 1.0, 2.0, 5.0]


def _resolve_gas(
    calibration_gas_name: str,
    compound_name: str | None,
    mw: float | None,
    response_factor: float | None,
) -> CalibrationGas:
    if calibration_gas_name == "custom":
        if compound_name is None or mw is None:
            raise ValueError("calibration_gas_name='custom' requires compound_name and mw")
        return CalibrationGas.custom(name=compound_name, mw=mw, response_factor=response_factor)
    return CalibrationGas.builtin(calibration_gas_name)


def _next_instruction(session: CalibrationSession) -> str:
    spike_list = session.context["spike_ppm_list"]
    done = len(session.steps)

    if done >= len(spike_list):
        return "All planned spike levels recorded. Call finish_standard_addition_session when ready."

    next_ppm = spike_list[done]
    sensor_id = session.context["sensor_id"]

    if next_ppm == 0.0:
        return (
            f"Step {done + 1}/{len(spike_list)}: record a BASELINE reading (0 ppm, "
            f"no injection) for sensor {sensor_id!r}, then call record_standard_addition_step."
        )

    return (
        f"Step {done + 1}/{len(spike_list)}: inject the chamber with {next_ppm} ppm of "
        f"{session.context['gas_name']}, dwell, then record the reading for sensor "
        f"{sensor_id!r} via record_standard_addition_step."
    )


def start_standard_addition_session(
    sessions_dir: Path,
    experiment_id: str,
    sensor_id: str,
    calibration_gas_name: str,
    compound_name: str | None = None,
    mw: float | None = None,
    response_factor: float | None = None,
    spike_ppm_list: list[float] | None = None,
) -> tuple[CalibrationSession, str]:
    """Resolve the calibration gas and start a new guided session."""

    gas = _resolve_gas(calibration_gas_name, compound_name, mw, response_factor)
    spike_ppm_list = spike_ppm_list or list(_DEFAULT_SPIKE_PPM_LIST)

    context = {
        "sensor_id": sensor_id,
        "gas_name": gas.name,
        "gas_mw": gas.mw,
        "gas_response_factor": gas.response_factor,
        "gas_response_factor_stderr": gas.response_factor_stderr,
        "gas_source": gas.source,
        "gas_is_builtin": gas.is_builtin,
        "spike_ppm_list": spike_ppm_list,
    }
    session = create_session(
        sessions_dir, kind="standard_addition", experiment_id=experiment_id, context=context
    )

    return session, _next_instruction(session)


def record_standard_addition_step(
    sessions_dir: Path,
    session_id: str,
    pid_voltage_mv: float,
    sample_t_c: float,
    sample_rh_pct: float,
    lamp_hours: float,
) -> tuple[CalibrationSession, str]:
    """Record one spike level's reading, in the order `spike_ppm_list` names."""

    session = load_session(sessions_dir, session_id)
    spike_list = session.context["spike_ppm_list"]
    step_index = len(session.steps)

    if step_index >= len(spike_list):
        raise ValueError(
            "All planned spike levels are already recorded; call finish_standard_addition_session "
            "instead, or start a new session for more levels."
        )

    step = {
        "spike_ppm_asgas": spike_list[step_index],
        "pid_voltage_mv": pid_voltage_mv,
        "sample_t_c": sample_t_c,
        "sample_rh_pct": sample_rh_pct,
        "lamp_hours": lamp_hours,
    }
    session = append_step(sessions_dir, session_id, step)

    return session, _next_instruction(session)


def finish_standard_addition_session(
    sessions_dir: Path,
    session_id: str,
    calibration_run_id: str,
    out_dir: Path,
    method: str = "ols",
) -> SensitivityModel:
    """Fit the real sensitivity model from whatever's been recorded, and
    persist it via the existing real calibration-persistence function."""

    session = load_session(sessions_dir, session_id)
    ctx = session.context

    rows = [
        {
            "sensor_id": ctx["sensor_id"],
            "spike_ppm_asgas": step["spike_ppm_asgas"],
            "pid_voltage_mv": step["pid_voltage_mv"],
            "sample_t_c": step["sample_t_c"],
            "sample_rh_pct": step["sample_rh_pct"],
            "lamp_hours": step["lamp_hours"],
            "calibration_compound": ctx["gas_name"],
            "mw_g_mol": ctx["gas_mw"],
            "response_factor": ctx["gas_response_factor"],
        }
        for step in session.steps
    ]
    df = pl.DataFrame(rows)

    results = fit_sensitivity_per_sensor(df, method=method)
    model = results[ctx["sensor_id"]]

    persist_calibration(
        {ctx["sensor_id"]: model},
        calibration_run_id=calibration_run_id,
        experiment_id=session.experiment_id,
        out_dir=out_dir,
    )
    mark_finished(sessions_dir, session_id)

    return model
