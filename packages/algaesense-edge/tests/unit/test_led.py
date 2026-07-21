"""Unit tests for algaesense_edge.actuators.actuators (LED-related classes).

LEDActuator's safety-validation logic (reject negative/NaN/over-max
requests) never touches hardware at all -- it raises before ever calling
`hardware.set_duty_cycle()` -- so those tests run against a real
(unconnected) `NeoPixelLEDHardware` instance with no GPIO or neopixel
library needed. Only the tests that actually call
`set_duty_cycle`/`read_duty_cycle` for real need `@pytest.mark.hardware`.
"""

from __future__ import annotations

import math

import pytest
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.actuators.actuators import (
    LEDActuator,
    NeoPixelLEDHardware,
    UnsafeSetpointError,
    create_hardware_led,
)
from tests.conftest import hardware_extra_importable

"""
Placeholder pixel count for constructing NeoPixelLEDHardware in tests
that never actually reach hardware -- adjust to the real strip's length
before running the @pytest.mark.hardware tests below for real on the Pi.
"""
_TEST_NUM_PIXELS = 40


def _reactor(max_par: float = 500.0) -> ReactorConfig:
    return ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=max_par)


def _hardware() -> NeoPixelLEDHardware:
    return NeoPixelLEDHardware(gpio_pin=18, num_pixels=_TEST_NUM_PIXELS)


class _InertLEDHardware:
    """A bare `LEDHardware` Protocol-conforming recorder: no GPIO, no
    simulated safety logic, just "remember the last duty cycle set."

    This is NOT a stand-in for NeoPixelLEDHardware's own behavior (which
    this project's no-mock-hardware convention rules out) -- it has no
    behavior to stand in for. `LEDActuator.set_par`'s exact-maximum
    boundary comparison (`>` vs `>=` against `reactor_config.max_par_umol_m2_s`)
    is pure arithmetic that happens to sit inside a class whose OTHER
    methods touch real hardware; this fake exists only so that one pure
    comparison can run on every test invocation, not just on the Pi.
    """

    def __init__(self) -> None:
        self.duty_cycle = 0.0

    def set_duty_cycle(self, fraction: float) -> None:
        self.duty_cycle = fraction

    def read_duty_cycle(self) -> float:
        return self.duty_cycle


def test_set_par_rejects_request_above_reactor_max() -> None:
    actuator = LEDActuator(
        hardware=_hardware(), reactor_config=_reactor(max_par=500.0), par_per_full_duty_umol_m2_s=1000.0
    )

    with pytest.raises(UnsafeSetpointError, match="exceeds reactor"):
        actuator.set_par(600.0)


def test_set_par_rejects_negative_request() -> None:
    actuator = LEDActuator(hardware=_hardware(), reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    with pytest.raises(UnsafeSetpointError, match="invalid"):
        actuator.set_par(-10.0)


def test_set_par_rejects_nan_request() -> None:
    actuator = LEDActuator(hardware=_hardware(), reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    with pytest.raises(UnsafeSetpointError, match="invalid"):
        actuator.set_par(math.nan)


def test_create_hardware_led_returns_a_real_neopixel_hardware_handle() -> None:
    hardware = create_hardware_led(gpio_pin=18, num_pixels=60, pixel_order="BRG")

    assert isinstance(hardware, NeoPixelLEDHardware)
    assert hardware.gpio_pin == 18
    assert hardware.num_pixels == 60
    assert hardware.pixel_order == "BRG"


def test_neopixel_hardware_fails_clearly_without_hardware_extra_installed() -> None:
    """
    adafruit-circuitpython-neopixel (and the Blinka `board` module it
    needs) isn't installed on this dev machine -- the first actual
    hardware call should fail with a clear, actionable ImportError.
    """
    if hardware_extra_importable("board", "neopixel"):
        pytest.skip("board/neopixel are installed in this environment (e.g. on the Pi) -- "
                     "this test only verifies the ImportError path when they're absent.")
    hardware = _hardware()
    with pytest.raises(ImportError, match="hardware"):
        hardware.set_duty_cycle(0.5)


@pytest.mark.hardware
def test_set_par_within_bounds_applies_the_correct_duty_cycle() -> None:
    """Run only on the Pi, with the WS2811 strip wired per this module's
    confirmed pinout (GPIO18, BRG order)."""
    hardware = _hardware()
    # par_per_full_duty=1000 means "1000 PAR at 100% duty" -- so requesting
    # 250 PAR should set a 25% duty cycle.
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    applied = actuator.set_par(250.0)

    assert applied == 250.0
    assert hardware.read_duty_cycle() == pytest.approx(0.25)


@pytest.mark.hardware
def test_read_par_reflects_current_hardware_duty_cycle() -> None:
    hardware = _hardware()
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    actuator.set_par(500.0)

    assert actuator.read_par() == pytest.approx(500.0)


def test_set_par_at_exactly_the_maximum_is_allowed() -> None:
    """The exact-maximum boundary comparison itself is pure arithmetic
    (`set_par` never reaches `hardware.set_duty_cycle()` until AFTER this
    comparison passes) -- runs against `_InertLEDHardware`, a bare
    Protocol-conforming recorder with no simulated hardware behavior, so
    this doesn't need @pytest.mark.hardware or the Pi to verify."""
    hardware = _InertLEDHardware()
    actuator = LEDActuator(
        hardware=hardware, reactor_config=_reactor(max_par=500.0), par_per_full_duty_umol_m2_s=1000.0
    )

    applied = actuator.set_par(500.0)  # exactly at the boundary, not over it

    assert applied == 500.0
    assert hardware.duty_cycle == pytest.approx(0.5)


@pytest.mark.hardware
def test_set_par_at_exactly_the_maximum_is_allowed_on_real_hardware() -> None:
    """Same boundary check as the unit test above, run for real on the Pi
    against the actual NeoPixel strip."""
    hardware = _hardware()
    actuator = LEDActuator(
        hardware=hardware, reactor_config=_reactor(max_par=500.0), par_per_full_duty_umol_m2_s=1000.0
    )

    applied = actuator.set_par(500.0)  # exactly at the boundary, not over it

    assert applied == 500.0


@pytest.mark.hardware
def test_turn_off_sets_duty_cycle_to_zero() -> None:
    hardware = _hardware()
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    actuator.set_par(400.0)
    actuator.turn_off()

    assert hardware.read_duty_cycle() == 0.0
