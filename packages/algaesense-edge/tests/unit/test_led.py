"""Unit tests for algaesense_edge.actuators.actuators (LED-related classes)."""

from __future__ import annotations

import math

import pytest
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.actuators.actuators import (
    GpiozeroLEDHardware,
    LEDActuator,
    MockLEDHardware,
    UnsafeSetpointError,
    create_hardware_led,
)


def _reactor(max_par: float = 500.0) -> ReactorConfig:
    return ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=max_par)


def test_set_par_within_bounds_applies_the_correct_duty_cycle() -> None:
    hardware = MockLEDHardware()
    # par_per_full_duty=1000 means "1000 PAR at 100% duty" -- so requesting
    # 250 PAR should set a 25% duty cycle.
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    applied = actuator.set_par(250.0)

    assert applied == 250.0
    assert hardware.duty_cycle == pytest.approx(0.25)


def test_read_par_reflects_current_hardware_duty_cycle() -> None:
    hardware = MockLEDHardware(duty_cycle=0.5)
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    assert actuator.read_par() == pytest.approx(500.0)


def test_set_par_rejects_request_above_reactor_max() -> None:
    hardware = MockLEDHardware()
    actuator = LEDActuator(
        hardware=hardware, reactor_config=_reactor(max_par=500.0), par_per_full_duty_umol_m2_s=1000.0
    )

    with pytest.raises(UnsafeSetpointError, match="exceeds reactor"):
        actuator.set_par(600.0)

    # The rejected request must not have reached the hardware at all --
    # the duty cycle should still be at its untouched default.
    assert hardware.duty_cycle == 0.0


def test_set_par_rejects_negative_request() -> None:
    hardware = MockLEDHardware()
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    with pytest.raises(UnsafeSetpointError, match="invalid"):
        actuator.set_par(-10.0)


def test_set_par_rejects_nan_request() -> None:
    hardware = MockLEDHardware()
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    with pytest.raises(UnsafeSetpointError, match="invalid"):
        actuator.set_par(math.nan)


def test_set_par_at_exactly_the_maximum_is_allowed() -> None:
    hardware = MockLEDHardware()
    actuator = LEDActuator(
        hardware=hardware, reactor_config=_reactor(max_par=500.0), par_per_full_duty_umol_m2_s=1000.0
    )

    applied = actuator.set_par(500.0)  # exactly at the boundary, not over it

    assert applied == 500.0


def test_turn_off_sets_duty_cycle_to_zero() -> None:
    hardware = MockLEDHardware(duty_cycle=0.8)
    actuator = LEDActuator(hardware=hardware, reactor_config=_reactor(), par_per_full_duty_umol_m2_s=1000.0)

    actuator.turn_off()

    assert hardware.duty_cycle == 0.0


def test_create_hardware_led_returns_a_real_gpiozero_hardware_handle() -> None:
    hardware = create_hardware_led(gpio_pin=17)

    assert isinstance(hardware, GpiozeroLEDHardware)
    assert hardware.gpio_pin == 17


def test_gpiozero_hardware_fails_clearly_without_hardware_extra_installed() -> None:
    # gpiozero isn't installed on this dev machine (it's part of the Pi-only
    # 'hardware' extra) -- the first actual hardware call should fail with a
    # clear, actionable ImportError.
    hardware = GpiozeroLEDHardware(gpio_pin=17)
    with pytest.raises(ImportError, match="hardware"):
        hardware.set_duty_cycle(0.5)
