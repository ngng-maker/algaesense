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


class EdgeClient:
    """Talks to one algaesense-edge instance's network API."""

    def __init__(self, base_url: str, timeout: float = 10.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

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
            detail = response.json().get("detail", response.text)
            raise SetpointRejectedError(detail)

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
            detail = response.json().get("detail", response.text)
            raise ProfileRejectedError(detail)

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
