"""Unit tests for algaesense_agent.mcp_actuators: the propose/apply
human-in-the-loop split, and the EdgeClient's handling of the edge
service's own safety responses.
"""

from __future__ import annotations

import httpx
import pytest

from algaesense_agent.mcp_actuators.actuators import (
    ActuatorNotImplementedError,
    apply_led_setpoint,
    propose_led_setpoint,
    propose_stirring_setpoint,
    propose_temperature_setpoint,
)
from algaesense_agent.mcp_actuators.edge_client import (
    EdgeClient,
    SetpointRejectedError,
    UnknownReactorError,
)


def test_propose_led_setpoint_has_no_side_effect_and_describes_the_change() -> None:
    proposal = propose_led_setpoint("R01", 250.0)

    assert proposal.reactor_id == "R01"
    assert proposal.requested_value == 250.0
    assert proposal.kind == "led_par"
    assert "not yet applied" in proposal.note.lower()


def test_propose_temperature_and_stirring_raise_not_implemented() -> None:
    with pytest.raises(ActuatorNotImplementedError):
        propose_temperature_setpoint("R01", 30.0)

    with pytest.raises(ActuatorNotImplementedError):
        propose_stirring_setpoint("R01", 200.0)


def _mock_edge_client(handler) -> EdgeClient:
    """Build an EdgeClient backed by `httpx.MockTransport` instead of a
    real running algaesense-edge instance."""
    return EdgeClient(base_url="http://fake-edge", transport=httpx.MockTransport(handler))


async def test_apply_led_setpoint_returns_the_edge_services_applied_value() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/actuators/led/R01"
        return httpx.Response(200, json={"reactor_id": "R01", "applied_par_umol_m2_s": 250.0})

    edge = _mock_edge_client(handler)

    result = await apply_led_setpoint(edge, "R01", 250.0)

    assert result == {"reactor_id": "R01", "applied_par_umol_m2_s": 250.0}
    await edge.close()


async def test_apply_led_setpoint_raises_for_unknown_reactor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No LED actuator configured for reactor 'R99'"})

    edge = _mock_edge_client(handler)

    with pytest.raises(UnknownReactorError):
        await apply_led_setpoint(edge, "R99", 250.0)

    await edge.close()


async def test_apply_led_setpoint_raises_when_edge_rejects_an_unsafe_setpoint() -> None:
    """The edge service's own bounds-check (UnsafeSetpointError, one
    network hop away) is what actually protects the hardware -- this
    confirms that rejection surfaces back to the caller as a clear
    exception rather than being swallowed or mis-parsed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Requested PAR 9999.0 exceeds reactor 'R01's configured maximum"})

    edge = _mock_edge_client(handler)

    with pytest.raises(SetpointRejectedError, match="exceeds reactor"):
        await apply_led_setpoint(edge, "R01", 9999.0)

    await edge.close()


async def test_recent_voc_readings_passes_through_the_edge_services_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sensors/voc/recent"
        assert request.url.params["limit"] == "5"
        return httpx.Response(200, json=[{"pid_voltage_mv": 1.2}])

    edge = _mock_edge_client(handler)

    readings = await edge.recent_voc_readings(limit=5)

    assert readings == [{"pid_voltage_mv": 1.2}]
    await edge.close()
