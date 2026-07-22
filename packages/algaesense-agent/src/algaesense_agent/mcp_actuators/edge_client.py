"""Thin HTTP proxy to one algaesense-edge instance's network API."""

from __future__ import annotations

import httpx


"""
This is the only place in `algaesense-agent` that talks to a Raspberry
Pi -- and even here, only over the network API algaesense-edge already
exposes (see algaesense_edge.api.app), never anything hardware-specific.
`httpx.AsyncClient` accepts an injectable `transport=`, which is what lets
tests run this against a real, in-process algaesense-edge FastAPI app via
`httpx.ASGITransport` (see tests/fixtures/real_edge_app.py) instead of a
separately running edge service -- exercising the real routing and
validation code, not a hand-written stand-in for it.
"""


class UnknownReactorError(LookupError):
    """Raised when the edge service has no actuator configured for the
    requested reactor_id."""


class SetpointRejectedError(ValueError):
    """Raised when the edge service's own safety validation rejected a
    requested setpoint."""

    """
    This mirrors algaesense_edge.actuators.actuators.UnsafeSetpointError
    one network hop away -- the edge service re-validates every setpoint
    independent of whatever this client sends, and this exception is just
    that rejection surfacing back to the caller here, not a duplicate
    safety check performed on this side.
    """


class EdgeRequestShapeError(RuntimeError):
    """Raised when the edge service's 422 response came from FastAPI's own
    request-body validation (a missing/mistyped field in what this client
    sent), not from this project's own domain-level rejection
    (SetpointRejectedError/ProfileRejectedError)."""

    """
    FastAPI's own pydantic validation errors shape `detail` as a LIST of
    per-field error dicts (`[{"loc": [...], "msg": ..., "type": ...}, ...]`),
    while this project's own `HTTPException(422, detail="...")` calls
    (the real safety/shape rejections SetpointRejectedError/
    ProfileRejectedError exist to surface) always pass a plain string.
    Treating both shapes as the same kind of rejection would misreport a
    bug in the request this client built as if it were the edge service's
    safety validation correctly doing its job.
    """


class ProfileRejectedError(ValueError):
    """Raised when the edge service rejected a control profile as
    malformed (unknown shape, or missing/invalid parameters for its
    shape)."""

    """
    A distinct exception from SetpointRejectedError on purpose -- this is
    algaesense_edge.actuators.control_profiles.validate_control_profile
    rejecting the profile's SHAPE before it's ever started, not
    LEDActuator rejecting a specific PAR value the profile computed. The
    latter can still happen later, per-tick, on the edge service (see
    AcquisitionService.tick_led_profiles) -- there is no way for this
    client to observe that after the fact, since it happens on the edge
    service's own schedule, not in response to a request this client made.
    """


def _raise_for_domain_rejection(response: httpx.Response, domain_error_cls: type[Exception], reactor_id: str) -> None:
    """Raise `domain_error_cls` for a real domain-level 422 rejection
    (string `detail`), or `EdgeRequestShapeError` for FastAPI's own
    request-shape validation 422 (list `detail`) -- see
    EdgeRequestShapeError's docstring for why these can't be treated the
    same."""
    detail = response.json().get("detail", response.text)
    if isinstance(detail, str):
        raise domain_error_cls(detail)
    raise EdgeRequestShapeError(
        f"algaesense-edge rejected the request shape for reactor {reactor_id!r}: {detail!r}"
    )


class EdgeClient:
    """Talks to one algaesense-edge instance's network API."""

    def __init__(self, base_url: str, timeout: float = 10.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    async def health(self) -> dict:
        """Check whether the edge service is up and responding at all --
        used to wait out a restart before sending anything else, e.g. an
        LED setpoint right after starting a new experiment run."""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def recent_voc_readings(self, limit: int | None = None) -> list[dict]:
        """Fetch the most recent VOC readings the edge service has buffered."""
        response = await self._client.get(
            "/sensors/voc/recent", params={"limit": limit} if limit is not None else None
        )
        response.raise_for_status()
        return response.json()

    async def recent_camera_readings(self, limit: int | None = None) -> list[dict]:
        """Fetch the most recent camera readings the edge service has buffered."""
        response = await self._client.get(
            "/sensors/camera/recent", params={"limit": limit} if limit is not None else None
        )
        response.raise_for_status()
        return response.json()

    async def set_led(self, reactor_id: str, par_umol_m2_s: float) -> dict:
        """Command a reactor's LED to a new PAR setpoint; the edge service
        validates it before applying anything."""

        response = await self._client.post(
            f"/actuators/led/{reactor_id}", json={"par_umol_m2_s": par_umol_m2_s}
        )

        if response.status_code == 404:
            raise UnknownReactorError(
                f"algaesense-edge has no LED actuator configured for reactor {reactor_id!r}"
            )

        if response.status_code == 422:
            _raise_for_domain_rejection(response, SetpointRejectedError, reactor_id)

        response.raise_for_status()
        return response.json()

    async def start_led_profile(self, reactor_id: str, profile: dict) -> dict:
        """Start a time-varying control profile on a reactor's LED; the
        edge service validates the profile's shape before ever recording
        it as active."""

        response = await self._client.post(f"/actuators/led/{reactor_id}/profile", json={"profile": profile})

        if response.status_code == 404:
            raise UnknownReactorError(
                f"algaesense-edge has no LED actuator configured for reactor {reactor_id!r}"
            )

        if response.status_code == 422:
            _raise_for_domain_rejection(response, ProfileRejectedError, reactor_id)

        response.raise_for_status()
        return response.json()

    async def stop_led_profile(self, reactor_id: str) -> dict:
        """Stop whatever control profile is currently running on a
        reactor's LED, if any."""
        response = await self._client.delete(f"/actuators/led/{reactor_id}/profile")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
