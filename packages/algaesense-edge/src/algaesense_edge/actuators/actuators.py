"""Actuator control: turns a requested setpoint into real hardware output, with safety checks that always happen here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from jaxsr_calibration.calibration.config import ReactorConfig


"""
Combines what used to be three files (errors.py, base.py, led.py) -- LED
control is the only actuator actually built so far; the safety-validation
logic never trusts whatever asked for a change, independent of the caller
(a script, a person, or later an AI agent).

Split the same way as the acquisition readers: `LEDHardware` is the thin,
swappable "how do I actually set a PWM pin" contract (mock for tests, real
`gpiozero`-backed implementation for a Pi); `LEDActuator` is the safety and
unit-conversion logic, which never touches hardware directly and is fully
testable regardless of which `LEDHardware` it's given.
`TemperatureActuator`/`StirringActuator` at the bottom are not implemented --
no such hardware exists in this project yet -- kept here as the template to
follow (same hardware-Protocol-plus-safety-wrapper shape as LED) once it does,
rather than in their own near-empty file.
"""


class UnsafeSetpointError(ValueError):
    """Raised when a requested actuator setpoint is unsafe."""

    """
    Specifically: negative, or exceeding the reactor's configured safety
    maximum. Deliberately a REJECTION, not a silent clamp -- a caller asking
    for far more light than intended (e.g. a typo, or a bad value from an
    upstream agent) should find out immediately, not have their request
    quietly coerced into something smaller that still might not be what they
    meant.
    """


class LEDHardware(Protocol):
    """Anything that can set/read a PWM duty cycle, as a fraction 0.0-1.0."""

    def set_duty_cycle(self, fraction: float) -> None: ...
    def read_duty_cycle(self) -> float: ...


@dataclass
class MockLEDHardware:
    """A fake LED for tests and dev machines with no real hardware wired up."""

    """
    Just remembers whatever duty cycle was last set, in memory.
    """

    duty_cycle: float = 0.0

    def set_duty_cycle(self, fraction: float) -> None:
        self.duty_cycle = fraction

    def read_duty_cycle(self) -> float:
        return self.duty_cycle


@dataclass
class GpiozeroLEDHardware:
    """The real LED, controlled via GPIO PWM."""

    """
    Backed by `gpiozero`'s `PWMLED`, this package's chosen library for
    GPIO/PWM control on a Pi (simpler, more modern API than the older
    `RPi.GPIO`). Can't be run or verified without real Pi GPIO hardware.
    """

    gpio_pin: int

    """
    Deliberately unannotated -- see acquisition/voc.py's
    Ads1115VOCSensorReader for why (keeps this out of the
    dataclass-generated __init__/__repr__).
    """
    _pwm_led = None

    def _connect(self):
        """Open the actual GPIO connection, the first time it's needed."""

        try:
            from gpiozero import PWMLED
        except ImportError as exc:
            raise ImportError(
                "GpiozeroLEDHardware requires the 'hardware' extra (gpiozero). "
                "Install with `pip install algaesense-edge[hardware]` on a "
                "Raspberry Pi."
            ) from exc
        self._pwm_led = PWMLED(self.gpio_pin)

    def set_duty_cycle(self, fraction: float) -> None:
        if self._pwm_led is None:
            self._connect()

        """
        gpiozero's PWMLED.value IS the duty cycle, as a 0.0-1.0 fraction --
        setting it directly drives the PWM signal, no separate "apply" call
        needed.
        """
        self._pwm_led.value = fraction

    def read_duty_cycle(self) -> float:
        if self._pwm_led is None:
            self._connect()
        return float(self._pwm_led.value)


def create_hardware_led(gpio_pin: int) -> LEDHardware:
    """Get a real, GPIO-backed LED hardware handle."""
    return GpiozeroLEDHardware(gpio_pin=gpio_pin)


@dataclass
class LEDActuator:
    """Turns a requested light level into safe, real LED output."""

    """
    The safety-and-units layer in front of an `LEDHardware`.

    `reactor_config` supplies the safety ceiling (`max_par_umol_m2_s`) --
    reusing jaxsr_calibration.calibration.config.ReactorConfig's existing
    field rather than inventing a second, separate bound that could drift
    out of sync with it.

    `par_per_full_duty_umol_m2_s` is a per-installation CALIBRATION CONSTANT
    ("how much PAR this specific LED/vial setup produces at 100% duty
    cycle") -- required explicitly, with no made-up default, because the
    hardware protocol's own installation steps (measuring illuminance at the
    vial surface with a lux meter, per the experimentalist protocol) are
    exactly how a real value for this gets measured; a fabricated default
    number here would be worse than requiring the real measurement.
    """

    hardware: LEDHardware
    reactor_config: ReactorConfig
    par_per_full_duty_umol_m2_s: float

    def set_par(self, par_umol_m2_s: float) -> float:
        """Request a light level (PAR, umol/m^2/s). Returns what was actually applied."""

        """
        Raises UnsafeSetpointError (not a silent clamp) for a negative
        request or one exceeding this reactor's configured maximum.
        """

        if math.isnan(par_umol_m2_s) or par_umol_m2_s < 0:
            raise UnsafeSetpointError(f"Requested PAR {par_umol_m2_s} is invalid (must be >= 0).")

        if par_umol_m2_s > self.reactor_config.max_par_umol_m2_s:
            raise UnsafeSetpointError(
                f"Requested PAR {par_umol_m2_s} exceeds reactor "
                f"{self.reactor_config.id!r}'s configured maximum of "
                f"{self.reactor_config.max_par_umol_m2_s} umol/m^2/s."
            )

        duty_fraction = par_umol_m2_s / self.par_per_full_duty_umol_m2_s

        """
        A safe request in PAR terms should always translate to a physically
        valid 0.0-1.0 duty cycle given a correct calibration constant, but
        this second clamp is defense-in-depth against a miscalibrated
        `par_per_full_duty_umol_m2_s` producing an out-of-range fraction --
        the PWM hardware itself only accepts 0-1.
        """
        duty_fraction = min(max(duty_fraction, 0.0), 1.0)

        self.hardware.set_duty_cycle(duty_fraction)
        return par_umol_m2_s

    def read_par(self) -> float:
        """Read back the currently-applied light level."""

        """
        Computed from the hardware's current duty cycle, not a separately
        tracked value, so this can never drift out of sync with what the
        hardware is actually doing.
        """
        return self.hardware.read_duty_cycle() * self.par_per_full_duty_umol_m2_s

    def turn_off(self) -> None:
        """Turn the LED fully off."""
        self.hardware.set_duty_cycle(0.0)


class TemperatureActuator(Protocol):
    """Not implemented -- no temperature-control hardware in this project yet."""

    """
    When it exists, follow LEDActuator's pattern above: a thin hardware
    Protocol (mock + real implementations) plus a safety-validating wrapper
    class that clamps/rejects requests against a configured min/max
    temperature (jaxsr_calibration.calibration.config.ReactorConfig already
    has min_reactor_temp_c/max_reactor_temp_c fields ready for exactly this).
    """

    def set_temperature_c(self, temperature_c: float) -> float: ...


class StirringActuator(Protocol):
    """Not implemented -- no stirring-control hardware in this project yet."""

    """
    Same pattern to follow as TemperatureActuator once real hardware exists.
    """

    def set_speed_rpm(self, speed_rpm: float) -> float: ...
