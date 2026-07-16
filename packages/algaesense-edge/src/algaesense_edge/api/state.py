"""In-memory state the API serves from: a rolling buffer of recent sensor
readings (pushed in by the acquisition loop) plus the actuators the API is
allowed to command.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from algaesense_edge.actuators.actuators import LEDActuator


"""
How many recent readings to keep in memory per sensor stream -- at ~1 Hz,
300 covers the last 5 minutes, which is plenty for "what's this sensor
reading right now" without unbounded memory growth over a long-running
experiment. Not spec-mandated, a reasonable operational default.
"""
_DEFAULT_BUFFER_SIZE = 300


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

    led_actuators: dict[str, LEDActuator] = field(default_factory=dict)

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
        readings = list(self._voc_readings)
        return readings[-limit:] if limit is not None else readings

    def recent_camera_readings(self, limit: int | None = None) -> list[dict]:
        readings = list(self._camera_readings)
        return readings[-limit:] if limit is not None else readings
