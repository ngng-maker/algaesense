"""Unit tests for AcquisitionService.tick_control_profiles: evaluating and
applying each (reactor, actuator_kind)'s currently-running control profile,
plus run_voc_tick's reading of the generic last-applied-setpoint cache.

Every path through tick_control_profiles that actually reaches the LED (a
successfully-applied setpoint, or a rejected one that gets turned off)
calls real `NeoPixelLEDHardware.set_duty_cycle`, which needs the real
GPIO/neopixel stack -- so, same as actuators/test_led.py, those cases are
`@pytest.mark.hardware`. The two tick_control_profiles paths that never
reach hardware at all (no active profile, no registered actuator) are
plain unit tests. run_voc_tick's PAR-cache lookup itself is hardware-free,
but run_voc_tick as a whole always touches the real VOC reader too, so
those tests stay hardware-marked as well.
"""

from __future__ import annotations

import datetime as dt

import pytest
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.acquisition.camera import create_hardware_camera_capture
from algaesense_edge.acquisition.voc import create_hardware_voc_reader
from algaesense_edge.actuators.actuators import LEDActuator, NeoPixelLEDHardware
from algaesense_edge.api.state import AppState
from algaesense_edge.service import AcquisitionService

_START = dt.datetime(2026, 7, 25, 8, 0, 0, tzinfo=dt.timezone.utc)


def _service(tmp_path, state: AppState) -> AcquisitionService:
    return AcquisitionService(
        experiment_id="exp_control_profile_test",
        reactor_id="R01",
        sensor_id="PID01",
        camera_id="CAM01",
        voc_reader=create_hardware_voc_reader(),
        trh_reader=None,
        camera_capture=create_hardware_camera_capture(),
        camera_clip_dir=tmp_path / "clips",
        raw_data_dir=tmp_path / "raw",
        state=state,
    )


def _led_actuator(max_par: float = 500.0) -> LEDActuator:
    return LEDActuator(
        hardware=NeoPixelLEDHardware(gpio_pin=18, num_pixels=30),
        reactor_config=ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=max_par),
        par_per_full_duty_umol_m2_s=1000.0,
    )


def test_tick_with_no_active_profiles_does_nothing(tmp_path) -> None:
    state = AppState()
    service = _service(tmp_path, state)

    results = service.tick_control_profiles(_START)

    assert results == {}


def test_tick_skips_a_reactor_with_no_registered_control_actuator(tmp_path) -> None:
    """A profile started for a (reactor, kind) this Pi instance has no
    registered control_actuators entry for (a configuration mismatch) is
    silently skipped rather than raising -- it has no actuator to apply
    anything to or turn off."""
    state = AppState()
    state.start_control_profile("R01", "led", {"shape": "constant", "par_umol_m2_s": 100.0}, now=_START)
    service = _service(tmp_path, state)

    results = service.tick_control_profiles(_START)

    assert results == {}


@pytest.mark.hardware
def test_tick_applies_an_in_range_profile_value(tmp_path) -> None:
    state = AppState()
    actuator = _led_actuator()
    state.led_actuators["R01"] = actuator
    state.control_actuators[("R01", "led")] = actuator
    state.start_control_profile("R01", "led", {"shape": "constant", "par_umol_m2_s": 200.0}, now=_START)
    service = _service(tmp_path, state)

    results = service.tick_control_profiles(_START + dt.timedelta(seconds=5))

    assert results == {("R01", "led"): "applied"}
    assert ("R01", "led") in state.active_control_profiles  # still running -- constant never finishes
    assert state.last_applied_setpoint[("R01", "led")] == 200.0


@pytest.mark.hardware
def test_tick_rejects_an_out_of_range_profile_value_and_stops_it(tmp_path) -> None:
    state = AppState()
    actuator = _led_actuator(max_par=500.0)
    state.led_actuators["R01"] = actuator
    state.control_actuators[("R01", "led")] = actuator
    # A ramp that overshoots the reactor's configured max well before it finishes.
    state.start_control_profile(
        "R01",
        "led",
        {"shape": "ramp", "start_par_umol_m2_s": 0.0, "end_par_umol_m2_s": 1000.0, "duration_s": 100.0},
        now=_START,
    )
    service = _service(tmp_path, state)

    results = service.tick_control_profiles(_START + dt.timedelta(seconds=90))  # 900 PAR requested, above max=500

    assert results == {("R01", "led"): "rejected"}
    assert ("R01", "led") not in state.active_control_profiles  # stopped, not left running
    assert actuator.read_par() == 0.0  # turned off
    assert state.last_applied_setpoint[("R01", "led")] == 0.0


"""
The PAR-cache lookup itself never touches hardware, but `run_voc_tick` as
a whole always does (it also calls the real VOC reader's
`read_voltage_mv()`, same as every other run_voc_tick test in this
codebase, e.g. test_phase1b_end_to_end.py) -- so these are still
hardware-marked, pre-seeding the cache directly rather than going through
an actuator.
"""


@pytest.mark.hardware
def test_run_voc_tick_records_the_cached_par_for_this_reactor(tmp_path) -> None:
    state = AppState()
    state.last_applied_setpoint[("R01", "led")] = 250.0
    service = _service(tmp_path, state)

    row = service.run_voc_tick(_START)

    assert row["reactor_par_umol_m2_s"] == 250.0
    service.close()


@pytest.mark.hardware
def test_run_voc_tick_records_none_when_nothing_cached_yet(tmp_path) -> None:
    state = AppState()
    service = _service(tmp_path, state)

    row = service.run_voc_tick(_START)

    assert row["reactor_par_umol_m2_s"] is None
    service.close()
