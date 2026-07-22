"""Unit tests for the mcp_actuators FastMCP server's tool wrappers.

Most of this file covers the tools that need no network access
(propose_led_change, the not-implemented stubs) -- apply_led_change's
server-level wiring against a real edge service is covered by the Phase 2
end-to-end test instead, since it needs a full fake-edge harness.
get_recent_voc_readings/get_recent_camera_readings are the exception: a
lighter, local version of that same real-edge-app + _build_edge_client
monkeypatch pattern is used below, since they're read-only (no hardware,
no propose/apply gate needed) and don't warrant the full Phase 2 harness.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from algaesense_agent.mcp_actuators import server as actuators_server
from algaesense_agent.mcp_actuators.edge_client import EdgeClient
from algaesense_agent.mcp_actuators.server import mcp
from tests.fixtures.real_edge_app import build_real_edge_app, edge_transport


async def test_propose_led_change_tool_returns_a_structured_proposal() -> None:
    result = await mcp.call_tool("propose_led_change", {"reactor_id": "R01", "par_umol_m2_s": 250.0})

    payload = json.loads(result[0].text)

    assert payload["reactor_id"] == "R01"
    assert payload["requested_value"] == 250.0


async def test_propose_led_profile_change_tool_returns_a_structured_proposal() -> None:
    profile = {"shape": "constant", "par_umol_m2_s": 100.0}

    result = await mcp.call_tool("propose_led_profile_change", {"reactor_id": "R01", "profile": profile})

    payload = json.loads(result[0].text)
    assert payload["reactor_id"] == "R01"
    assert payload["profile"] == profile


async def test_propose_temperature_change_tool_reports_not_implemented() -> None:
    """
    FastMCP re-raises a tool's exception as a `ToolError` rather than
    returning it as a normal successful result -- confirmed by running
    this test before assuming otherwise. Checking for the explanatory
    text in the raised error is what matters here.
    """
    with pytest.raises(ToolError, match="no temperature-control hardware"):
        await mcp.call_tool("propose_temperature_change", {"reactor_id": "R01", "temperature_c": 30.0})


def _wire_real_edge_app(monkeypatch: pytest.MonkeyPatch):
    """Point this server's _build_edge_client at a real, in-process edge
    app (same pattern as test_phase2_end_to_end.py's wired_servers
    fixture, just local to this file since these two tools need no
    hardware and no propose/apply gate)."""
    app, state = build_real_edge_app()

    def _build_real_edge_client() -> EdgeClient:
        return EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    monkeypatch.setattr(actuators_server, "_build_edge_client", _build_real_edge_client)
    return state


def _payload(raw_result):
    """A list-returning tool's `call_tool` result isn't wrapped in a
    single text content block the way a dict-returning tool's is (see
    the other tests in this file) -- FastMCP hands a list result back
    structured already. See test_phase2_end_to_end.py's `_payload` for
    the same dual-shape handling."""
    if isinstance(raw_result, tuple):
        return raw_result[1]["result"]
    if isinstance(raw_result, list) and raw_result and hasattr(raw_result[0], "text"):
        return json.loads(raw_result[0].text)
    return raw_result


async def test_get_recent_voc_readings_tool_returns_real_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _wire_real_edge_app(monkeypatch)
    state.record_voc_reading({"pid_voltage_mv": 1.2})
    state.record_voc_reading({"pid_voltage_mv": 1.5})

    result = await mcp.call_tool("get_recent_voc_readings", {"limit": 5})

    assert _payload(result) == [{"pid_voltage_mv": 1.2}, {"pid_voltage_mv": 1.5}]


async def test_get_recent_camera_readings_tool_returns_real_readings(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _wire_real_edge_app(monkeypatch)
    state.record_camera_reading({"image_feature_vector": [1.0, 2.0, 3.0]})

    result = await mcp.call_tool("get_recent_camera_readings", {"limit": 5})

    assert _payload(result) == [{"image_feature_vector": [1.0, 2.0, 3.0]}]


async def test_propose_start_new_experiment_run_tool_returns_a_structured_proposal() -> None:
    result = await mcp.call_tool("propose_start_new_experiment_run", {"reactor_id": "R01"})

    payload = json.loads(result[0].text)
    assert payload["reactor_id"] == "R01"
    assert "restart" in payload["note"].lower()


async def test_apply_start_new_experiment_run_tool_reads_pi_env_vars_and_reports_dashboard_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real SSH mechanics are exhaustively covered (against a genuine
    local SSH server) in test_experiment_run.py -- this test only checks
    the server layer's own job: reading the ALGAESENSE_PI_*/
    ALGAESENSE_DASHBOARD_URL env vars and passing them through to
    apply_new_experiment_run correctly."""
    monkeypatch.setenv("ALGAESENSE_PI_HOST", "pi.example")
    monkeypatch.setenv("ALGAESENSE_PI_USERNAME", "someuser")
    monkeypatch.setenv("ALGAESENSE_PI_PASSWORD", "somepassword")
    monkeypatch.setenv("ALGAESENSE_DASHBOARD_URL", "http://example-dashboard:8501")

    captured_kwargs = {}

    async def _fake_apply_new_experiment_run(reactor_id, **kwargs):
        captured_kwargs["reactor_id"] = reactor_id
        captured_kwargs.update(kwargs)
        return {"reactor_id": reactor_id, "status": "restarted", "dashboard_url": kwargs.get("dashboard_url")}

    monkeypatch.setattr(actuators_server, "apply_new_experiment_run", _fake_apply_new_experiment_run)

    result = await mcp.call_tool("apply_start_new_experiment_run", {"reactor_id": "R01"})

    payload = json.loads(result[0].text)
    assert payload == {"reactor_id": "R01", "status": "restarted", "dashboard_url": "http://example-dashboard:8501"}
    assert captured_kwargs["host"] == "pi.example"
    assert captured_kwargs["username"] == "someuser"
    assert captured_kwargs["password"] == "somepassword"
    assert captured_kwargs["dashboard_url"] == "http://example-dashboard:8501"
    assert captured_kwargs["led_profile"] is None
    assert captured_kwargs["edge"] is None


async def test_apply_start_new_experiment_run_tool_builds_an_edge_client_when_led_profile_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only when led_profile is actually given should this tool bother
    building an EdgeClient at all -- confirmed here by wiring
    _build_edge_client at a real (if not network-exercised further)
    in-process edge app, same pattern as _wire_real_edge_app."""
    monkeypatch.setenv("ALGAESENSE_PI_HOST", "pi.example")
    monkeypatch.setenv("ALGAESENSE_PI_USERNAME", "someuser")
    monkeypatch.setenv("ALGAESENSE_PI_PASSWORD", "somepassword")
    _wire_real_edge_app(monkeypatch)

    captured_kwargs = {}

    async def _fake_apply_new_experiment_run(reactor_id, **kwargs):
        captured_kwargs["reactor_id"] = reactor_id
        captured_kwargs.update(kwargs)
        return {"reactor_id": reactor_id, "status": "restarted"}

    monkeypatch.setattr(actuators_server, "apply_new_experiment_run", _fake_apply_new_experiment_run)

    profile = {"shape": "constant", "par_umol_m2_s": 100.0}
    await mcp.call_tool("apply_start_new_experiment_run", {"reactor_id": "R01", "led_profile": profile})

    assert captured_kwargs["led_profile"] == profile
    assert isinstance(captured_kwargs["edge"], EdgeClient)
