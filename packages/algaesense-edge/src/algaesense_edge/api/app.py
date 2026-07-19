"""The FastAPI service the brain machine talks to instead of needing SSH
access to the Pi: read recent sensor data, command the LED.
"""

from __future__ import annotations

import datetime as dt

import yaml
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


class LEDProfileRequest(BaseModel):
    """`profile` is the plain dict described in
    algaesense_edge.actuators.control_profiles -- a `shape` discriminator
    plus that shape's typed parameters, e.g.
    `{"shape": "ramp", "start_par_umol_m2_s": 0.0, "end_par_umol_m2_s": 300.0, "duration_s": 3600.0}`."""

    profile: dict


class LEDProfileResponse(BaseModel):
    reactor_id: str
    profile: dict


class LEDProfileStopResponse(BaseModel):
    reactor_id: str
    was_running: bool


def _log_started_profile(
    state: AppState, reactor_id: str, actuator_kind: str, profile: dict, started_at: dt.datetime
) -> None:
    """Best-effort record of "this profile was started, at this time" under
    the experiment's raw data directory -- separate from `AppState`'s
    in-memory tracking, which is what actually drives
    `AcquisitionService.tick_control_profiles` and doesn't survive a
    restart.
    """

    """
    Silently skipped (not an error) when `state` wasn't wired up with a
    real experiment/data directory -- true for every test's plain
    `AppState()`, and a deliberate choice not to force every caller to
    supply logging plumbing just to exercise the in-memory start/stop
    behavior.
    """
    if state.experiment_id is None or state.raw_data_dir is None:
        return

    profile_dir = state.raw_data_dir / "experiments" / state.experiment_id / "control_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    """
    Colons aren't valid in Windows filenames, hence the replace -- same
    reasoning as service.py's camera clip naming.
    """
    file_name = f"{reactor_id}_{started_at.isoformat().replace(':', '-')}.yaml"
    record = {
        "reactor_id": reactor_id,
        "actuator_kind": actuator_kind,
        "started_at": started_at.isoformat(),
        "profile": profile,
    }
    (profile_dir / file_name).write_text(yaml.safe_dump(record, sort_keys=False))


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

        """
        Cached generically (see AppState.last_applied_setpoint) so other
        code -- e.g. AcquisitionService.run_voc_tick recording what PAR a
        VOC sample was taken under -- can know what this actuator is
        actually doing without re-reading real hardware.
        """
        state.last_applied_setpoint[(reactor_id, "led")] = applied

        return LEDSetpointResponse(reactor_id=reactor_id, applied_par_umol_m2_s=applied)

    @app.post("/actuators/led/{reactor_id}/profile", response_model=LEDProfileResponse)
    def start_led_profile(reactor_id: str, request: LEDProfileRequest) -> LEDProfileResponse:
        if reactor_id not in state.led_actuators:
            raise HTTPException(
                status_code=404,
                detail=f"No LED actuator configured for reactor {reactor_id!r}",
            )

        now = dt.datetime.now(dt.timezone.utc)
        try:
            """
            `AppState.start_control_profile` validates the profile's shape --
            an unknown shape or a shape missing required keys is rejected
            here, before it's ever handed to
            `AcquisitionService.tick_control_profiles`. "led" is hardcoded
            here since this route is LED-specific by design -- a future
            heater/cooler route would pass its own kind the same way.
            """
            state.start_control_profile(reactor_id, "led", request.profile, now=now)
        except ValueError as exc:
            """
            Covers both `UnknownProfileShapeError` (an unrecognized
            `shape`) and plain `ValueError` (a recognized shape missing
            required keys) -- the former is a subclass of the latter, and
            both are the caller's fault (a malformed request), not a
            server error.
            """
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        _log_started_profile(state, reactor_id, "led", request.profile, now)

        return LEDProfileResponse(reactor_id=reactor_id, profile=request.profile)

    @app.delete("/actuators/led/{reactor_id}/profile", response_model=LEDProfileStopResponse)
    def stop_led_profile(reactor_id: str) -> LEDProfileStopResponse:
        was_running = state.stop_control_profile(reactor_id, "led")
        return LEDProfileStopResponse(reactor_id=reactor_id, was_running=was_running)

    return app
