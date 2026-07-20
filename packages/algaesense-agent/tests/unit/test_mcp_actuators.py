"""Unit tests for algaesense_agent.mcp_actuators: the propose/apply
human-in-the-loop split, and the EdgeClient's handling of the edge
service's own safety responses.

Every EdgeClient test below runs against a REAL algaesense-edge FastAPI
app (`httpx.ASGITransport`, via tests/fixtures/real_edge_app.py) -- not a
hand-written fake HTTP handler standing in for it. Most of them don't
need `@pytest.mark.hardware` at all: rejection (404 unknown reactor, 422
unsafe setpoint) happens inside real routing/validation code before
anything would ever touch a physical LED. Only the one test that needs a
setpoint to actually succeed is hardware-marked, since applying it for
real means the actuator calling through to real GPIO.
"""

from __future__ import annotations

import pytest

from algaesense_agent.mcp_actuators.actuators import (
    ActuatorNotImplementedError,
    apply_led_profile,
    apply_led_setpoint,
    propose_led_profile,
    propose_led_setpoint,
    propose_stirring_setpoint,
    propose_temperature_setpoint,
    stop_led_profile,
)
from algaesense_agent.mcp_actuators.edge_client import (
    EdgeClient,
    EdgeRequestShapeError,
    ProfileRejectedError,
    SetpointRejectedError,
    UnknownReactorError,
)
from tests.fixtures.real_edge_app import build_real_edge_app, edge_transport


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


async def test_apply_led_setpoint_raises_for_unknown_reactor() -> None:
    app, _ = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(UnknownReactorError):
        await apply_led_setpoint(edge, "R99", 250.0)

    await edge.close()


async def test_apply_led_setpoint_raises_when_edge_rejects_an_unsafe_setpoint() -> None:
    """The edge service's own bounds-check (UnsafeSetpointError) is what
    actually protects the hardware -- this confirms that rejection
    surfaces back to the caller as a clear exception rather than being
    swallowed or mis-parsed. Real routing, real validation, real
    LEDActuator -- no hardware needed, since the request is rejected
    before ever reaching it."""

    app, _ = build_real_edge_app(reactor_id="R01", max_par=500.0)
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(SetpointRejectedError, match="exceeds reactor"):
        await apply_led_setpoint(edge, "R01", 9999.0)

    await edge.close()


async def test_apply_led_setpoint_raises_request_shape_error_for_a_malformed_request_body() -> None:
    """Regression test for a real bug: a 422 whose `detail` is a LIST
    (FastAPI's own request-body validation, e.g. `par_umol_m2_s` failing
    to parse as a float) used to be conflated with SetpointRejectedError
    (the edge service's own domain-level safety rejection, always a plain
    string `detail`) -- misreporting a bug in the request this client
    built as if it were the edge service's safety validation correctly
    doing its job."""

    app, _ = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(EdgeRequestShapeError):
        await apply_led_setpoint(edge, "R01", "not-a-number")  # fails pydantic's float parsing

    await edge.close()


async def test_recent_voc_readings_passes_through_the_edge_services_response() -> None:
    app, state = build_real_edge_app()
    state.record_voc_reading({"pid_voltage_mv": 1.2})
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    readings = await edge.recent_voc_readings(limit=5)

    assert readings == [{"pid_voltage_mv": 1.2}]
    await edge.close()


async def test_recent_camera_readings_passes_through_the_edge_services_response() -> None:
    app, state = build_real_edge_app()
    state.record_camera_reading({"image_feature_vector": [1.0, 2.0, 3.0]})
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    readings = await edge.recent_camera_readings(limit=5)

    assert readings == [{"image_feature_vector": [1.0, 2.0, 3.0]}]
    await edge.close()


def test_propose_led_profile_has_no_side_effect_and_describes_the_change() -> None:
    profile = {"shape": "constant", "par_umol_m2_s": 100.0}

    proposal = propose_led_profile("R01", profile)

    assert proposal.reactor_id == "R01"
    assert proposal.profile == profile
    assert "not yet started" in proposal.note.lower()


"""
Starting/stopping a profile only reaches AppState's in-memory
validate-and-record logic on the edge service (see
algaesense_edge.api.app's start_led_profile/stop_led_profile endpoints)
-- it never calls `hardware.set_duty_cycle()`, so none of the tests below
need @pytest.mark.hardware, unlike apply_led_setpoint's success case
above.
"""


async def test_apply_led_profile_raises_for_unknown_reactor() -> None:
    app, _ = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(UnknownReactorError):
        await apply_led_profile(edge, "R99", {"shape": "constant", "par_umol_m2_s": 100.0})

    await edge.close()


async def test_apply_led_profile_raises_when_edge_rejects_an_unknown_shape() -> None:
    app, _ = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(ProfileRejectedError, match="Unknown control profile shape"):
        await apply_led_profile(edge, "R01", {"shape": "spiral"})

    await edge.close()


async def test_apply_led_profile_raises_when_edge_rejects_missing_required_keys() -> None:
    app, _ = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(ProfileRejectedError, match="missing required keys"):
        await apply_led_profile(edge, "R01", {"shape": "ramp"})

    await edge.close()


async def test_apply_led_profile_raises_request_shape_error_for_a_malformed_request_body() -> None:
    """Same shape-conflation bug as apply_led_setpoint's, for the profile
    endpoint: `profile` failing to parse as a dict at all (FastAPI's own
    request-body validation) is a bug in the request this client built,
    not validate_control_profile rejecting a well-formed-but-invalid
    profile (that's ProfileRejectedError's job, string `detail`)."""

    app, _ = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    with pytest.raises(EdgeRequestShapeError):
        await apply_led_profile(edge, "R01", "not-a-dict")  # fails pydantic's dict-type check

    await edge.close()


async def test_apply_led_profile_starts_a_well_formed_profile() -> None:
    app, state = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))
    profile = {"shape": "constant", "par_umol_m2_s": 100.0}

    result = await apply_led_profile(edge, "R01", profile)

    assert result == {"reactor_id": "R01", "profile": profile}
    assert ("R01", "led") in state.active_control_profiles
    await edge.close()


async def test_stop_led_profile_reports_whether_one_was_running() -> None:
    app, state = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))
    await apply_led_profile(edge, "R01", {"shape": "constant", "par_umol_m2_s": 100.0})

    first = await stop_led_profile(edge, "R01")
    second = await stop_led_profile(edge, "R01")

    assert first == {"reactor_id": "R01", "was_running": True}
    assert second == {"reactor_id": "R01", "was_running": False}
    await edge.close()


@pytest.mark.hardware
async def test_apply_led_setpoint_returns_the_edge_services_applied_value() -> None:
    """Run only on the Pi -- this is the one case in this file where the
    setpoint is actually accepted and reaches real GPIO."""
    app, _ = build_real_edge_app(reactor_id="R01", max_par=500.0, par_per_full_duty=1000.0)
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    result = await apply_led_setpoint(edge, "R01", 250.0)

    assert result == {"reactor_id": "R01", "applied_par_umol_m2_s": 250.0}
    await edge.close()
