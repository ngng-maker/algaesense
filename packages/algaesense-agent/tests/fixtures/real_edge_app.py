"""A real, in-process algaesense-edge FastAPI app for tests that need to
exercise the actual actuator/API code -- not a hand-written fake HTTP
handler standing in for it.
"""

from __future__ import annotations

import httpx
from algaesense_edge.actuators.actuators import LEDActuator, NeoPixelLEDHardware
from algaesense_edge.api.app import create_app
from algaesense_edge.api.state import AppState
from jaxsr_calibration.calibration.config import ReactorConfig


"""
`NeoPixelLEDHardware` is the real LED hardware class -- constructing one
never touches GPIO (that only happens lazily, on the first actual
set_duty_cycle call, which never runs in these tests since they route
through the reactor's own safety validation the same way any real caller
would). `httpx.ASGITransport` runs `create_app`'s real FastAPI app
in-process, so requests through it exercise real routing, real pydantic
validation, and the real `LEDActuator` bounds-check -- the only thing
"fake" here is that there's no real network socket or physical strip
involved, same as `fastapi.testclient.TestClient` used throughout this
project's other API tests.
"""


def build_real_edge_app(reactor_id: str = "R01", max_par: float = 500.0, par_per_full_duty: float = 1000.0):
    """Return `(app, state)` for a real algaesense-edge instance with one
    LED actuator configured."""
    state = AppState()
    reactor_config = ReactorConfig(id=reactor_id, model="pioreactor_20mL", max_par_umol_m2_s=max_par)
    state.led_actuators[reactor_id] = LEDActuator(
        hardware=NeoPixelLEDHardware(gpio_pin=18, num_pixels=30),
        reactor_config=reactor_config,
        par_per_full_duty_umol_m2_s=par_per_full_duty,
    )
    return create_app(state), state


def edge_transport(app) -> httpx.ASGITransport:
    return httpx.ASGITransport(app=app)
