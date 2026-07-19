"""MCP server exposing guided calibration wizards for all three
calibration types as step-by-step tools Hermes can walk an operator
through conversationally.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from algaesense_agent.mcp_actuators.edge_client import EdgeClient
from algaesense_agent.mcp_calibration import camera_zero, reference_jar, standard_addition
from algaesense_agent.mcp_calibration.sessions import CalibrationSession


"""
No LLM-specific logic lives here -- the actual "guide the operator
step by step" behavior comes from Hermes's own conversational turn-taking
plus the system prompt (see profile/system_prompt.md), reading the
`next_step` text every tool below returns and relaying it to the human.
These tools only track state and do the real math/persistence; none of
them are side-effecting in the `mcp_actuators` sense (no hardware is
touched), so there's no propose/apply split needed here.
"""

mcp = FastMCP("algaesense-calibration")


def _sessions_dir() -> Path:
    return Path(os.environ.get("ALGAESENSE_CALIBRATION_SESSIONS_DIR", "data/calibration_sessions"))


def _data_dir() -> Path:
    return Path(os.environ.get("ALGAESENSE_DATA_DIR", "data"))


def _session_result(session: CalibrationSession, next_step: str) -> dict:
    return {"session": asdict(session), "next_step": next_step}


# --- Standard-addition gas calibration -------------------------------------


@mcp.tool()
def start_standard_addition_session(
    experiment_id: str,
    sensor_id: str,
    calibration_gas_name: str,
    compound_name: str | None = None,
    mw: float | None = None,
    response_factor: float | None = None,
    spike_ppm_list: list[float] | None = None,
) -> dict:
    """Start a guided standard-addition gas calibration for one sensor.
    `calibration_gas_name` is a built-in compound name (e.g. "isoprene"),
    or "custom" (then compound_name/mw are required)."""
    session, next_step = standard_addition.start_standard_addition_session(
        _sessions_dir(), experiment_id, sensor_id, calibration_gas_name, compound_name, mw, response_factor, spike_ppm_list
    )
    return _session_result(session, next_step)


@mcp.tool()
def record_standard_addition_step(
    session_id: str, pid_voltage_mv: float, sample_t_c: float, sample_rh_pct: float, lamp_hours: float
) -> dict:
    """Record one spike level's reading for an in-progress standard-addition session."""
    session, next_step = standard_addition.record_standard_addition_step(
        _sessions_dir(), session_id, pid_voltage_mv, sample_t_c, sample_rh_pct, lamp_hours
    )
    return _session_result(session, next_step)


@mcp.tool()
def finish_standard_addition_session(session_id: str, calibration_run_id: str, method: str = "ols") -> dict:
    """Fit and persist the real sensitivity model from whatever's been recorded so far."""
    model = standard_addition.finish_standard_addition_session(
        _sessions_dir(),
        session_id,
        calibration_run_id,
        out_dir=_data_dir() / "derived" / "calibrations" / "standard_addition",
        method=method,
    )
    return asdict(model)


# --- Reference-jar cross-sensor check ---------------------------------------


@mcp.tool()
def start_reference_jar_session(experiment_id: str, sensors: list[str], dwell_min: int = 10) -> dict:
    """Start a guided reference-jar rotation across the given sensors."""
    session, next_step = reference_jar.start_reference_jar_session(_sessions_dir(), experiment_id, sensors, dwell_min)
    return _session_result(session, next_step)


@mcp.tool()
def record_reference_jar_reading(session_id: str, sensor_id: str, pid_voltage_mv: float) -> dict:
    """Record one sensor's reference-jar dwell reading."""
    session, next_step = reference_jar.record_reference_jar_reading(_sessions_dir(), session_id, sensor_id, pid_voltage_mv)
    return _session_result(session, next_step)


@mcp.tool()
def finish_reference_jar_session(session_id: str) -> dict:
    """Compute real fleet ratios from whatever readings have been recorded so far."""
    return reference_jar.finish_reference_jar_session(_sessions_dir(), session_id)


# --- Camera zero-point (blank) biomass calibration --------------------------


@mcp.tool()
def start_camera_zero_session(experiment_id: str, camera_id: str, min_captures: int = 10) -> dict:
    """Start a guided camera zero-point calibration against clean, cell-free medium."""
    session, next_step = camera_zero.start_camera_zero_session(_sessions_dir(), experiment_id, camera_id, min_captures)
    return _session_result(session, next_step)


@mcp.tool()
def record_camera_zero_step(session_id: str, rgb: list[float]) -> dict:
    """Record one clean-medium clip's [red, green, blue] feature vector."""
    session, next_step = camera_zero.record_camera_zero_step(_sessions_dir(), session_id, rgb)
    return _session_result(session, next_step)


def _build_edge_client() -> EdgeClient:
    """Construct the EdgeClient `record_camera_zero_step_from_edge` talks
    through."""

    """
    A separate, overridable function (rather than inlining
    `EdgeClient(...)` directly in the tool below) so a test can
    monkeypatch this one function to return a client backed by an
    in-process fake edge service, exercising the real MCP tool call
    instead of only the plain function underneath it -- same pattern as
    mcp_actuators/server.py's `_build_edge_client`.
    """
    edge_base_url = os.environ.get("ALGAESENSE_EDGE_BASE_URL", "http://localhost:8000")
    return EdgeClient(edge_base_url)


@mcp.tool()
async def record_camera_zero_step_from_edge(session_id: str) -> dict:
    """Record the most recent camera reading already buffered by
    algaesense-edge, instead of the operator supplying an rgb vector by
    hand -- convenient since captures happen on the edge service's own
    hourly schedule rather than on demand."""
    edge = _build_edge_client()
    try:
        readings = await edge.recent_camera_readings(limit=1)
    finally:
        await edge.close()

    if not readings:
        raise ValueError("No camera readings available yet from algaesense-edge.")

    rgb = readings[-1]["image_feature_vector"]
    session, next_step = camera_zero.record_camera_zero_step(_sessions_dir(), session_id, rgb)
    return _session_result(session, next_step)


@mcp.tool()
def finish_camera_zero_session(session_id: str, max_relative_std: float = 0.10) -> dict:
    """Compute and persist the real blank baseline from whatever's been recorded so far."""
    model = camera_zero.finish_camera_zero_session(
        _sessions_dir(),
        session_id,
        out_dir=_data_dir() / "derived" / "calibrations" / "camera_zero",
        max_relative_std=max_relative_std,
    )
    return asdict(model)


def main() -> None:
    """Entry point for the `algaesense-mcp-calibration` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
