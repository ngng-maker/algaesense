"""In-memory state the API serves from: a rolling buffer of recent sensor
readings (pushed in by the acquisition loop) plus the actuators the API is
allowed to command.
"""

from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from algaesense_edge.actuators.actuators import ControlProfileActuator, LEDActuator
from algaesense_edge.actuators.control_profiles import validate_control_profile


"""
How many recent readings to keep in memory per sensor stream -- at ~1 Hz,
300 covers the last 5 minutes, which is plenty for "what's this sensor
reading right now" without unbounded memory growth over a long-running
experiment. Not spec-mandated, a reasonable operational default.
"""
_DEFAULT_BUFFER_SIZE = 300


@dataclass
class ActiveControlProfile:
    """One reactor's currently-running control profile for one actuator
    kind (see algaesense_edge.actuators.control_profiles): the profile
    data itself, plus when it started -- everything
    `AcquisitionService.tick_control_profiles` needs to compute "what
    should this actuator be doing right now"."""

    profile: dict
    started_at: dt.datetime


@dataclass
class AppState:
    """Owns the rolling reading buffers and the actuators this Pi instance
    controls."""

    """
    One instance is created per running service and shared by every
    request handler (via FastAPI's dependency injection, see app.py) --
    NOT recreated per-request, since the whole point is remembering
    readings across requests.
    """

    """
    The manual single-setpoint path (`app.py`'s `/actuators/led/{reactor_id}`
    route, and `LEDActuator.set_par()` directly) only ever uses this dict --
    it's LED-specific on purpose, unlike `control_actuators` below, since
    there's no present need to generalize the manual-setpoint surface.
    """
    led_actuators: dict[str, LEDActuator] = field(default_factory=dict)

    """
    A generic registry any actuator kind can register into, keyed by
    `(reactor_id, actuator_kind)` -- e.g. `("R01", "led")` today,
    `("R01", "heater")` once that hardware exists. This is what
    `AcquisitionService.tick_control_profiles` iterates and drives, so a
    future actuator kind only needs to (a) implement
    `ControlProfileActuator` and (b) register itself here -- no engine
    code changes needed. `cli.py` dual-registers the LED actuator into
    both this dict and `led_actuators` above (same object, two purposes).
    """
    control_actuators: dict[tuple[str, str], ControlProfileActuator] = field(default_factory=dict)

    """
    Empty for a reactor+kind with no profile running -- a single static
    setpoint call (the existing propose/apply path, LED-specific today)
    never touches this at all, only `start_control_profile`/
    `stop_control_profile` do. Keyed the same way as `control_actuators`.
    """
    active_control_profiles: dict[tuple[str, str], ActiveControlProfile] = field(default_factory=dict)

    """
    "Last successfully applied setpoint" per `(reactor_id, actuator_kind)` --
    a generic cache any actuator kind can use to let other code (e.g. VOC
    row logging) know what an actuator was actually doing, without
    re-reading real hardware on every use. Updated wherever
    `ControlProfileActuator.apply_setpoint`/a manual setpoint actually
    succeeds; not touched otherwise.
    """
    last_applied_setpoint: dict[tuple[str, str], float] = field(default_factory=dict)

    """
    Both optional and `None` by default -- most tests construct a bare
    `AppState()` with no interest in profile-start logging at all. Only
    set (by cli.py's `start` command, from the same `--experiment`/
    `--raw-data-dir` options AcquisitionService already takes) when
    running for real, so `app.py`'s profile-start endpoint knows where to
    log a record of what was started.
    """
    experiment_id: str | None = None

    raw_data_dir: Path | None = None

    buffer_size: int = _DEFAULT_BUFFER_SIZE

    """
    `deque(maxlen=N)` is a list-like container that automatically drops
    its oldest item once it's full and a new one is added -- exactly the
    "keep only the most recent N readings" behavior this buffer needs,
    without us having to manually trim it on every insert.
    """
    _voc_readings: deque = field(init=False)

    _camera_readings: deque = field(init=False)

    def __post_init__(self) -> None:
        """
        `__post_init__` runs automatically right after a dataclass's
        generated `__init__` finishes -- used here because
        `deque(maxlen=...)` needs `self.buffer_size` (itself a constructor
        argument) to already be set, which a plain
        `field(default_factory=...)` can't reference.
        """
        self._voc_readings = deque(maxlen=self.buffer_size)
        self._camera_readings = deque(maxlen=self.buffer_size)

    def record_voc_reading(self, row: dict) -> None:
        self._voc_readings.append(row)

    def record_camera_reading(self, row: dict) -> None:
        self._camera_readings.append(row)

    def recent_voc_readings(self, limit: int | None = None) -> list[dict]:
        """`limit=None` returns every buffered reading, `limit=0` returns
        none, and a positive `limit` returns the last N -- `readings[-limit:]`
        alone gets `limit=0` wrong (`readings[-0:]` is `readings[0:]`, the
        WHOLE list, not an empty one), so 0 is special-cased."""
        readings = list(self._voc_readings)
        if limit is None:
            return readings
        return readings[-limit:] if limit > 0 else []

    def recent_camera_readings(self, limit: int | None = None) -> list[dict]:
        """See recent_voc_readings' docstring -- same `limit=0` slicing fix."""
        readings = list(self._camera_readings)
        if limit is None:
            return readings
        return readings[-limit:] if limit > 0 else []

    def start_control_profile(self, reactor_id: str, actuator_kind: str, profile: dict, now: dt.datetime) -> None:
        """Validate and record a new control profile for one reactor's
        actuator of the given kind (e.g. "led"), replacing whatever
        profile (if any) was already running there."""

        """
        Validating here (not just trusting the caller) means a malformed
        profile is rejected immediately, with a clear error, rather than
        being silently accepted and only failing later on its first tick.
        """
        validate_control_profile(profile)
        self.active_control_profiles[(reactor_id, actuator_kind)] = ActiveControlProfile(
            profile=profile, started_at=now
        )

    def stop_control_profile(self, reactor_id: str, actuator_kind: str) -> bool:
        """Stop whatever profile is running for one reactor's actuator of
        the given kind, if any. Returns whether a profile was actually
        running."""
        return self.active_control_profiles.pop((reactor_id, actuator_kind), None) is not None
