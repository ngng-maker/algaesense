"""Propose/apply actuator-change logic: the actual human-in-the-loop
safety boundary for this package.
"""

from __future__ import annotations

from dataclasses import dataclass

from algaesense_agent.mcp_actuators.edge_client import EdgeClient


"""
Split into two functions on purpose: `propose_led_setpoint` has NO side
effect (it only describes what would happen), and `apply_led_setpoint` is
the only function in this whole package that actually reaches hardware.
The Hermes profile's system prompt (see profile/) is what instructs the
agent to always call propose, show the human the result in Slack, and
wait for explicit confirmation before ever calling apply -- but even if
the agent (or a bug, or a prompt injection) skipped straight to calling
apply, the request still has to pass algaesense_edge's own independent
`UnsafeSetpointError` bounds-check before anything physically changes.
This is defense in depth, not the only layer.
"""


class ActuatorNotImplementedError(NotImplementedError):
    """Raised for actuator kinds that have no real hardware yet."""

    """
    Mirrors algaesense_edge.actuators.actuators.TemperatureActuator/
    StirringActuator, which are themselves unimplemented Protocol stubs --
    this is that same "not built yet" state, surfaced at the MCP-tool
    layer with the same honesty rather than pretending the capability
    exists.
    """


@dataclass
class ActuatorProposal:
    """A described, not-yet-applied actuator change."""

    reactor_id: str
    kind: str
    requested_value: float
    unit: str
    note: str


def propose_led_setpoint(reactor_id: str, par_umol_m2_s: float) -> ActuatorProposal:
    """Describe an LED setpoint change without applying it."""

    """
    Deliberately makes no network call -- there is nothing to "preview"
    from the edge service (it has no GET endpoint for a reactor's current
    LED setpoint), so this just formats the request clearly for whoever
    is about to approve or reject it in Slack. The real safety check
    still happens inside `apply_led_setpoint`, on the edge service.
    """
    return ActuatorProposal(
        reactor_id=reactor_id,
        kind="led_par",
        requested_value=par_umol_m2_s,
        unit="umol_m2_s",
        note=(
            f"Proposing to set reactor {reactor_id!r}'s LED to {par_umol_m2_s} umol/m^2/s. "
            "Not yet applied -- requires explicit confirmation before calling apply_led_setpoint."
        ),
    )


async def apply_led_setpoint(edge: EdgeClient, reactor_id: str, par_umol_m2_s: float) -> dict:
    """Actually command a reactor's LED to a new setpoint over the network."""
    return await edge.set_led(reactor_id, par_umol_m2_s)


@dataclass
class LedProfileProposal:
    """A described, not-yet-started time-varying control profile (see
    algaesense_edge.actuators.control_profiles for the supported
    shapes)."""

    reactor_id: str
    profile: dict
    note: str


def propose_led_profile(reactor_id: str, profile: dict) -> LedProfileProposal:
    """Describe a control profile to start on a reactor's LED, without
    starting it."""

    """
    Same "no network call" reasoning as propose_led_setpoint -- the real
    shape/parameter validation happens on the edge service, inside
    apply_led_profile, not duplicated here. This just formats the profile
    clearly for the human to review and confirm.
    """
    return LedProfileProposal(
        reactor_id=reactor_id,
        profile=profile,
        note=(
            f"Proposing to start a {profile.get('shape')!r} control profile on reactor "
            f"{reactor_id!r}'s LED: {profile}. Not yet started -- requires explicit "
            "confirmation before calling apply_led_profile."
        ),
    )


async def apply_led_profile(edge: EdgeClient, reactor_id: str, profile: dict) -> dict:
    """Actually start a control profile on a reactor's LED over the
    network. Every value the running profile computes is still
    re-validated per-tick by the edge service's own LEDActuator bounds
    check -- this call only starts it, it does not bypass that."""
    return await edge.start_led_profile(reactor_id, profile)


async def stop_led_profile(edge: EdgeClient, reactor_id: str) -> dict:
    """Stop whatever control profile is currently running on a reactor's
    LED, if any. Has an immediate side effect (the LED stops following
    the profile), but stopping something already-approved needs no
    separate propose step -- there is nothing new to review."""
    return await edge.stop_led_profile(reactor_id)


def propose_temperature_setpoint(reactor_id: str, temperature_c: float) -> ActuatorProposal:
    """Not implemented -- no temperature-control hardware exists yet."""
    raise ActuatorNotImplementedError(
        "propose_temperature_setpoint: no temperature-control hardware exists yet "
        "(see algaesense_edge.actuators.actuators.TemperatureActuator)."
    )


def propose_stirring_setpoint(reactor_id: str, speed_rpm: float) -> ActuatorProposal:
    """Not implemented -- no stirring-control hardware exists yet."""
    raise ActuatorNotImplementedError(
        "propose_stirring_setpoint: no stirring-control hardware exists yet "
        "(see algaesense_edge.actuators.actuators.StirringActuator)."
    )
