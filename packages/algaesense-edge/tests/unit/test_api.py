"""Unit tests for algaesense_edge.api.app.

`TestClient` drives real FastAPI request/response handling (routing,
pydantic validation, status codes) in-process -- no real network socket or
running server needed, same idea as click's CliRunner used throughout
jaxsr-calibration's CLI tests.

The rejection-path tests (out-of-range, negative, unknown reactor,
malformed body) never reach `hardware.set_duty_cycle()` at all -- routing
and validation reject the request first -- so they run against a real
(unconnected) `NeoPixelLEDHardware` with no GPIO needed. Only the
successful-apply test actually reaches hardware, so it's
`@pytest.mark.hardware`.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.actuators.actuators import LEDActuator, NeoPixelLEDHardware
from algaesense_edge.api.app import create_app
from algaesense_edge.api.state import AppState

_TEST_NUM_PIXELS = 40


def _state_with_led(max_par: float = 500.0, par_per_full_duty: float = 1000.0) -> AppState:
    state = AppState()
    reactor = ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=max_par)
    state.led_actuators["R01"] = LEDActuator(
        hardware=NeoPixelLEDHardware(gpio_pin=18, num_pixels=_TEST_NUM_PIXELS),
        reactor_config=reactor,
        par_per_full_duty_umol_m2_s=par_per_full_duty,
    )
    return state


def test_health_endpoint() -> None:
    client = TestClient(create_app(AppState()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recent_voc_readings_empty_by_default() -> None:
    client = TestClient(create_app(AppState()))

    response = client.get("/sensors/voc/recent")

    assert response.status_code == 200
    assert response.json() == []


def test_recent_voc_readings_returns_recorded_rows() -> None:
    state = AppState()
    state.record_voc_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc), "pid_voltage_mv": 1.5})
    state.record_voc_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, 1, tzinfo=dt.timezone.utc), "pid_voltage_mv": 1.6})
    client = TestClient(create_app(state))

    response = client.get("/sensors/voc/recent")

    body = response.json()
    assert len(body) == 2
    assert body[-1]["pid_voltage_mv"] == 1.6


def test_recent_voc_readings_respects_limit() -> None:
    state = AppState()
    for i in range(10):
        state.record_voc_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, i, tzinfo=dt.timezone.utc), "pid_voltage_mv": float(i)})
    client = TestClient(create_app(state))

    response = client.get("/sensors/voc/recent", params={"limit": 3})

    body = response.json()
    assert len(body) == 3
    assert [row["pid_voltage_mv"] for row in body] == [7.0, 8.0, 9.0]  # the 3 MOST RECENT


def test_recent_voc_readings_limit_zero_returns_none_not_everything() -> None:
    """Regression test for a real bug: `readings[-limit:]` alone gets
    limit=0 wrong -- `readings[-0:]` is `readings[0:]`, the WHOLE list, not
    an empty one. Called directly on AppState (not through the HTTP layer)
    since this is pure slicing logic with no routing/serialization
    involved."""
    state = AppState()
    for i in range(5):
        state.record_voc_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, i, tzinfo=dt.timezone.utc), "pid_voltage_mv": float(i)})
        state.record_camera_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, i, tzinfo=dt.timezone.utc), "image_feature_vector": [float(i)]})

    assert state.recent_voc_readings(limit=0) == []
    assert state.recent_camera_readings(limit=0) == []
    # Sanity check the non-zero cases still behave as before this fix.
    assert len(state.recent_voc_readings(limit=None)) == 5
    assert len(state.recent_voc_readings(limit=2)) == 2


def test_recent_camera_readings_returns_recorded_rows() -> None:
    state = AppState()
    state.record_camera_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc), "image_feature_vector": [1.0, 2.0, 3.0]})
    client = TestClient(create_app(state))

    response = client.get("/sensors/camera/recent")

    assert response.json()[0]["image_feature_vector"] == [1.0, 2.0, 3.0]


@pytest.mark.hardware
def test_set_led_within_bounds_succeeds() -> None:
    """Run only on the Pi, with the WS2811 strip wired up -- this is the
    one test in this file that actually reaches hardware."""
    state = _state_with_led(max_par=500.0, par_per_full_duty=1000.0)
    client = TestClient(create_app(state))

    response = client.post("/actuators/led/R01", json={"par_umol_m2_s": 250.0})

    assert response.status_code == 200
    assert response.json() == {"reactor_id": "R01", "applied_par_umol_m2_s": 250.0}
    assert state.last_applied_setpoint[("R01", "led")] == 250.0


def test_set_led_out_of_range_is_rejected_with_422() -> None:
    client = TestClient(create_app(_state_with_led(max_par=500.0)))

    response = client.post("/actuators/led/R01", json={"par_umol_m2_s": 999.0})

    assert response.status_code == 422
    assert "exceeds reactor" in response.json()["detail"]


def test_set_led_negative_value_is_rejected_with_422() -> None:
    client = TestClient(create_app(_state_with_led()))

    response = client.post("/actuators/led/R01", json={"par_umol_m2_s": -5.0})

    assert response.status_code == 422


def test_set_led_unknown_reactor_returns_404() -> None:
    client = TestClient(create_app(_state_with_led()))

    response = client.post("/actuators/led/R99", json={"par_umol_m2_s": 100.0})

    assert response.status_code == 404


def test_set_led_malformed_body_returns_422_from_pydantic_validation() -> None:
    client = TestClient(create_app(_state_with_led()))

    """
    Missing the required par_umol_m2_s field entirely -- FastAPI/pydantic
    itself should reject this before our endpoint code even runs.
    """
    response = client.post("/actuators/led/R01", json={})

    assert response.status_code == 422


"""
Starting/stopping a profile only validates and records it in AppState's
in-memory dict -- it never calls `hardware.set_duty_cycle()` (that only
happens later, per-tick, in AcquisitionService.tick_control_profiles) --
so every test below runs with no hardware needed at all, unlike the
immediate-apply `/actuators/led/{reactor_id}` endpoint above.
"""


def test_start_led_profile_succeeds_for_a_known_reactor() -> None:
    state = _state_with_led()
    client = TestClient(create_app(state))

    response = client.post("/actuators/led/R01/profile", json={"profile": {"shape": "constant", "par_umol_m2_s": 100.0}})

    assert response.status_code == 200
    assert response.json() == {"reactor_id": "R01", "profile": {"shape": "constant", "par_umol_m2_s": 100.0}}
    assert ("R01", "led") in state.active_control_profiles


def test_start_led_profile_unknown_reactor_returns_404() -> None:
    client = TestClient(create_app(_state_with_led()))

    response = client.post("/actuators/led/R99/profile", json={"profile": {"shape": "constant", "par_umol_m2_s": 100.0}})

    assert response.status_code == 404


def test_start_led_profile_rejects_unknown_shape_with_422() -> None:
    client = TestClient(create_app(_state_with_led()))

    response = client.post("/actuators/led/R01/profile", json={"profile": {"shape": "spiral"}})

    assert response.status_code == 422
    assert "Unknown control profile shape" in response.json()["detail"]


def test_start_led_profile_rejects_missing_required_keys_with_422() -> None:
    client = TestClient(create_app(_state_with_led()))

    response = client.post("/actuators/led/R01/profile", json={"profile": {"shape": "ramp"}})

    assert response.status_code == 422
    assert "missing required keys" in response.json()["detail"]


def test_start_led_profile_logs_a_yaml_record_when_experiment_wiring_is_present(tmp_path) -> None:
    state = _state_with_led()
    state.experiment_id = "exp_control_profile_test"
    state.raw_data_dir = tmp_path / "raw"
    client = TestClient(create_app(state))

    client.post("/actuators/led/R01/profile", json={"profile": {"shape": "constant", "par_umol_m2_s": 100.0}})

    profile_dir = tmp_path / "raw" / "experiments" / "exp_control_profile_test" / "control_profiles"
    logged_files = list(profile_dir.glob("R01_*.yaml"))
    assert len(logged_files) == 1
    assert "constant" in logged_files[0].read_text()


def test_start_led_profile_does_not_log_without_experiment_wiring() -> None:
    """A bare AppState() (no experiment_id/raw_data_dir -- true for every
    other test in this file) must not try to write anywhere."""
    state = _state_with_led()
    client = TestClient(create_app(state))

    response = client.post("/actuators/led/R01/profile", json={"profile": {"shape": "constant", "par_umol_m2_s": 100.0}})

    assert response.status_code == 200


def test_stop_led_profile_reports_whether_one_was_running() -> None:
    state = _state_with_led()
    client = TestClient(create_app(state))
    client.post("/actuators/led/R01/profile", json={"profile": {"shape": "constant", "par_umol_m2_s": 100.0}})

    first = client.delete("/actuators/led/R01/profile")
    second = client.delete("/actuators/led/R01/profile")

    assert first.json() == {"reactor_id": "R01", "was_running": True}
    assert second.json() == {"reactor_id": "R01", "was_running": False}
    assert ("R01", "led") not in state.active_control_profiles
