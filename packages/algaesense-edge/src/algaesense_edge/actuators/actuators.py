"""Actuator control: turns a requested setpoint into real hardware output, with safety checks that always happen here."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from jaxsr_calibration.calibration.config import ReactorConfig


"""
LED control is the only actuator actually built so far; the
safety-validation logic never trusts whatever asked for a change,
independent of the caller (a script, a person, or later an AI agent).

Confirmed real hardware (2026-07-16): a WS2811 addressable RGB LED strip
(ALITOVE), driven from GPIO18 (BCM numbering) through a 74AHCT125 logic
level shifter (3.3V -> 5V) and a 470ohm series resistor into the strip's
data line, powered from a separate 12V supply with a common ground back
to the Pi. This is NOT simple PWM dimming (a single-brightness gpiozero
PWMLED, this project's earlier assumption) -- WS2811 is a timed serial
protocol (each bit is a precisely-timed HIGH pulse width, ~0.4us for a 0
bit and ~0.85us for a 1 bit, at 800kHz), driven via the Pi's DMA+PWM
peripheral through `adafruit-circuitpython-neopixel` (built on
`rpi_ws281x`), not a bare PWM duty cycle.

Split the same way as the acquisition readers: `LEDHardware` is the thin,
swappable "how do I actually drive the strip" contract; `LEDActuator` is
the safety and unit-conversion logic, which never touches hardware
directly. `LEDActuator` itself needed no changes for this hardware
switch -- it only calls `set_duty_cycle`/`read_duty_cycle`, so swapping
what's behind that Protocol was enough.

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
    """Anything that can set/read an overall light output level, as a
    fraction 0.0-1.0."""

    def set_duty_cycle(self, fraction: float) -> None: ...
    def read_duty_cycle(self) -> float: ...


@dataclass
class NeoPixelLEDHardware:
    """The real WS2811 LED strip, driven via `adafruit-circuitpython-neopixel`."""

    """
    Every pixel is held at `color` (full white, (255, 255, 255), by
    default -- a reasonable general grow-light color absent a specific
    spectrum requirement) and the strip's overall `.brightness` is what
    `set_duty_cycle` actually adjusts -- NeoPixel's `.brightness` scales
    every pixel's set color down uniformly, which is exactly the
    "single dimmable light source" interface `LEDActuator` expects,
    without needing to re-set each pixel's raw color on every setpoint
    change.

    `num_pixels` has no default -- it must match the physical strip
    actually wired up, and guessing a number here would silently produce
    wrong PAR-per-pixel behavior rather than a loud, obvious error.
    `pixel_order="BRG"` matches this project's specific ALITOVE strip,
    confirmed by testing -- NOT the library's own default (GRB), which is
    a different strip's channel order and would swap colors silently if
    assumed here instead of set explicitly. "BRG" isn't one of the
    library's own predefined name constants (`neopixel.RGB`/`GRB`/`BGR`/
    etc.) -- `pixel_order` accepts any such string directly, so this is
    passed straight through rather than looked up as a module attribute.

    Honesty note, same as the acquisition readers: built against
    `adafruit-circuitpython-neopixel`'s standard, documented usage and
    this project's own confirmed wiring, but can't be run or verified
    without the real strip and GPIO access -- untested on this dev
    machine by nature.
    """

    gpio_pin: int
    num_pixels: int
    pixel_order: str = "BRG"
    color: tuple[int, int, int] = (255, 255, 255)

    """
    Deliberately unannotated -- see acquisition/voc.py's
    Ads1115VOCSensorReader for why (keeps this out of the
    dataclass-generated __init__/__repr__).
    """
    _pixels = None

    def _connect(self):
        """Open the actual GPIO/DMA connection, the first time it's needed."""

        try:
            import board
            import neopixel
        except ImportError as exc:
            raise ImportError(
                "NeoPixelLEDHardware requires the 'hardware' extra "
                "(adafruit-circuitpython-neopixel). Install with "
                "`pip install algaesense-edge[hardware]` on a Raspberry Pi."
            ) from exc

        """
        `getattr(board, f"D{self.gpio_pin}")` maps a plain BCM pin number
        (18) onto the `board` module's named pin object (`board.D18`) --
        the same "map an int onto the library's own naming" pattern
        already used for the ADS1115 channel, applied here to a pin
        instead of a channel.
        """
        pin = getattr(board, f"D{self.gpio_pin}")

        """
        `pixel_order` takes a plain string directly (e.g. "BRG") -- the
        library's `neopixel.RGB`/`GRB`/`BGR`/etc. names are just
        convenience aliases for those same strings, and "BRG" specifically
        isn't one of the library's predefined names, so this passes
        `self.pixel_order` straight through rather than looking it up as
        a module attribute.
        """
        self._pixels = neopixel.NeoPixel(
            pin,
            self.num_pixels,
            brightness=0.0,
            auto_write=True,
            pixel_order=self.pixel_order,
        )
        self._pixels.fill(self.color)

    def set_duty_cycle(self, fraction: float) -> None:
        if self._pixels is None:
            self._connect()

        """
        Setting `.brightness` (rather than re-filling every pixel with a
        scaled-down color) lets the strip's own hardware/driver-level
        brightness scaling do the dimming -- `auto_write=True` means this
        takes effect immediately, no separate "show"/"apply" call needed.
        """
        self._pixels.brightness = fraction

    def read_duty_cycle(self) -> float:
        if self._pixels is None:
            self._connect()
        return float(self._pixels.brightness)


def create_hardware_led(
    gpio_pin: int, num_pixels: int, pixel_order: str = "BRG", color: tuple[int, int, int] = (255, 255, 255)
) -> LEDHardware:
    """Get a real, GPIO-backed LED strip hardware handle (see
    `NeoPixelLEDHardware`)."""
    return NeoPixelLEDHardware(gpio_pin=gpio_pin, num_pixels=num_pixels, pixel_order=pixel_order, color=color)


class ControlProfileActuator(Protocol):
    """Anything a time-varying control profile (see
    algaesense_edge.actuators.control_profiles) can drive -- one apply
    call per tick, plus a way to turn it off if a computed value gets
    rejected."""

    """
    Deliberately just these two methods, not a merge of every actuator
    kind's own richer interface (`LEDActuator.set_par`/`read_par` stay as
    they are) -- this Protocol exists purely so
    `AcquisitionService.tick_control_profiles` can drive ANY actuator kind
    generically, without knowing whether it's actually an LED, a future
    heater, or a future stirrer underneath. `LEDActuator.apply_setpoint`
    below is a thin delegate to `set_par`, added FOR this Protocol; a
    future `TemperatureActuator`/`StirringActuator` should add their own
    equivalent thin delegate the same way, not rename their existing
    domain-specific method.
    """

    def apply_setpoint(self, value: float) -> float: ...
    def turn_off(self) -> None: ...


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
    cycle") -- required explicitly, with no made-up default, because a
    fabricated default number here would be worse than requiring a real
    measurement.

    NOT the same thing as the experimentalist protocol's lux-meter step
    (Resources/spirulina_voc_experimentalist_protocol.md's LED installation
    section) -- that's a one-time, coarse photoinhibition SAFETY CEILING
    check ("keep illuminance under ~15,000 lux"), not a PAR calibration, and
    a plain lux meter can't give you PAR directly: lux is weighted to human
    eye sensitivity (peaks green ~555nm) while PAR is a flat photon count
    over 400-700nm, so the conversion between them depends on the specific
    LED's spectrum, not a universal constant. Until a real quantum/PAR
    meter is available, derive this value approximately: measure lux at the
    vial surface at 100% duty cycle (same physical setup as the protocol's
    ceiling check, just read at full brightness), then multiply by an
    LED-appropriate lux-to-PPFD conversion factor (see docs/hardware_setup.md
    for the derivation and its caveats) -- an approximation with real,
    acknowledged uncertainty, not a calibrated measurement.
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
        the hardware itself only accepts 0-1.
        """
        duty_fraction = min(max(duty_fraction, 0.0), 1.0)

        self.hardware.set_duty_cycle(duty_fraction)
        return par_umol_m2_s

    def apply_setpoint(self, value: float) -> float:
        """`ControlProfileActuator` protocol entry point -- delegates to
        `set_par`, so the generic control-profile tick loop can drive an
        LED without knowing it's an LED specifically."""
        return self.set_par(value)

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
    Protocol plus a safety-validating wrapper class that clamps/rejects
    requests against a configured min/max temperature
    (jaxsr_calibration.calibration.config.ReactorConfig already has
    min_reactor_temp_c/max_reactor_temp_c fields ready for exactly this).
    For control-profile support (ramps/cycles, not just static setpoints),
    also add a thin `apply_setpoint`/`turn_off` pair satisfying
    `ControlProfileActuator` above, the same way `LEDActuator.apply_setpoint`
    delegates to `set_par` -- then register it in `AppState.control_actuators`
    keyed by `(reactor_id, "temperature")` and no other engine code needs
    to change.
    """

    def set_temperature_c(self, temperature_c: float) -> float: ...


class StirringActuator(Protocol):
    """Not implemented -- no stirring-control hardware in this project yet."""

    """
    Same pattern to follow as TemperatureActuator once real hardware exists.
    """

    def set_speed_rpm(self, speed_rpm: float) -> float: ...
