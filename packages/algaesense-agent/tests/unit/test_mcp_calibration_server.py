"""Unit tests for the mcp_calibration FastMCP server's tool wrappers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from tests.fixtures.real_edge_app import build_real_edge_app, edge_transport


def _payload(raw_result):
    """See test_labwiki_server.py's `_tool_payload` for why both shapes
    need handling."""
    if isinstance(raw_result, tuple):
        return raw_result[1]["result"]
    return json.loads(raw_result[0].text)


async def test_standard_addition_tools_round_trip_through_the_server(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALGAESENSE_CALIBRATION_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path / "data"))

    from algaesense_agent.mcp_calibration import server as server_module

    importlib.reload(server_module)

    start_result = await server_module.mcp.call_tool(
        "start_standard_addition_session",
        {"experiment_id": "exp_01", "sensor_id": "PID01", "calibration_gas_name": "isoprene"},
    )
    start_payload = _payload(start_result)
    session_id = start_payload["session"]["session_id"]
    spike_ppm_list = start_payload["session"]["context"]["spike_ppm_list"]

    step_payload = None
    for ppm in spike_ppm_list:
        step_result = await server_module.mcp.call_tool(
            "record_standard_addition_step",
            {
                "session_id": session_id,
                "pid_voltage_mv": 0.5 + 2.0 * ppm,
                "sample_t_c": 25.0,
                "sample_rh_pct": 50.0,
                "lamp_hours": 10.0,
            },
        )
        step_payload = _payload(step_result)
    assert "finish_standard_addition_session" in step_payload["next_step"]

    finish_result = await server_module.mcp.call_tool(
        "finish_standard_addition_session", {"session_id": session_id, "calibration_run_id": "cal_run_server_01"}
    )
    finished_payload = _payload(finish_result)

    assert finished_payload["b1_mv_per_ppm_asgas"] == pytest.approx(2.0, abs=1e-6)
    assert (tmp_path / "data" / "derived" / "calibrations" / "standard_addition" / "cal_run_server_01.parquet").exists()


async def test_record_camera_zero_step_from_edge_pulls_the_real_edge_services_latest_reading(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ALGAESENSE_CALIBRATION_SESSIONS_DIR", str(tmp_path / "sessions"))

    from algaesense_agent.mcp_calibration import server as server_module

    importlib.reload(server_module)

    app, state = build_real_edge_app()
    state.record_camera_reading({"image_feature_vector": [80.0, 120.0, 90.0]})

    def _fake_edge_client():
        return server_module.EdgeClient("http://fake-edge", transport=edge_transport(app))

    monkeypatch.setattr(server_module, "_build_edge_client", _fake_edge_client)

    start_result = await server_module.mcp.call_tool(
        "start_camera_zero_session", {"experiment_id": "exp_01", "camera_id": "CAM01", "min_captures": 1}
    )
    session_id = _payload(start_result)["session"]["session_id"]

    step_result = await server_module.mcp.call_tool(
        "record_camera_zero_step_from_edge", {"session_id": session_id}
    )
    step_payload = _payload(step_result)

    assert step_payload["session"]["steps"] == [{"rgb": [80.0, 120.0, 90.0]}]
