"""The FastAPI service the brain machine talks to instead of needing SSH
access to the Pi: read recent sensor data, command the LED.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from algaesense_edge.actuators.actuators import UnsafeSetpointError
from algaesense_edge.api.state import AppState


"""
`create_app(state)` is a factory (not a single module-level `app` object)
so tests can build a fresh app around a known `AppState` with mock
hardware -- no real server, network, or hardware needed to exercise every
endpoint.
"""


class LEDSetpointRequest(BaseModel):
    par_umol_m2_s: float


class LEDSetpointResponse(BaseModel):
    reactor_id: str
    applied_par_umol_m2_s: float


def create_app(state: AppState) -> FastAPI:
    app = FastAPI(title="algaesense-edge")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/sensors/voc/recent")
    def get_recent_voc(limit: int | None = Query(default=None, gt=0)) -> list[dict]:
        return state.recent_voc_readings(limit=limit)

    @app.get("/sensors/camera/recent")
    def get_recent_camera(limit: int | None = Query(default=None, gt=0)) -> list[dict]:
        return state.recent_camera_readings(limit=limit)

    @app.post("/actuators/led/{reactor_id}", response_model=LEDSetpointResponse)
    def set_led(reactor_id: str, request: LEDSetpointRequest) -> LEDSetpointResponse:
        actuator = state.led_actuators.get(reactor_id)
        if actuator is None:
            raise HTTPException(
                status_code=404,
                detail=f"No LED actuator configured for reactor {reactor_id!r}",
            )

        try:
            """
            This is the actual safety re-validation the whole network
            boundary exists for -- LEDActuator.set_par() re-checks the
            request against the reactor's configured bounds itself,
            regardless of what the caller (an agent, a script, a person)
            believed was safe.
            """
            applied = actuator.set_par(request.par_umol_m2_s)
        except UnsafeSetpointError as exc:
            """
            422 Unprocessable Entity: the request was well-formed JSON,
            but its VALUE was rejected -- distinct from 400 (malformed
            request body) or 404 (no such reactor).
            """
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return LEDSetpointResponse(reactor_id=reactor_id, applied_par_umol_m2_s=applied)

    return app
