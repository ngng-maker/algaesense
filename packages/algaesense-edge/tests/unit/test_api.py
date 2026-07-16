"""Unit tests for algaesense_edge.api.app.

`TestClient` drives real FastAPI request/response handling (routing,
pydantic validation, status codes) in-process -- no real network socket or
running server needed, same idea as click's CliRunner used throughout
jaxsr-calibration's CLI tests.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.actuators.actuators import LEDActuator, MockLEDHardware
from algaesense_edge.api.app import create_app
from algaesense_edge.api.state import AppState


def _state_with_led(max_par: float = 500.0, par_per_full_duty: float = 1000.0) -> AppState:
    state = AppState()
    reactor = ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=max_par)
    state.led_actuators["R01"] = LEDActuator(
        hardware=MockLEDHardware(),
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


def test_recent_camera_readings_returns_recorded_rows() -> None:
    state = AppState()
    state.record_camera_reading({"timestamp": dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc), "image_feature_vector": [1.0, 2.0, 3.0]})
    client = TestClient(create_app(state))

    response = client.get("/sensors/camera/recent")

    assert response.json()[0]["image_feature_vector"] == [1.0, 2.0, 3.0]


def test_set_led_within_bounds_succeeds() -> None:
    client = TestClient(create_app(_state_with_led(max_par=500.0, par_per_full_duty=1000.0)))

    response = client.post("/actuators/led/R01", json={"par_umol_m2_s": 250.0})

    assert response.status_code == 200
    assert response.json() == {"reactor_id": "R01", "applied_par_umol_m2_s": 250.0}


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

    # Missing the required par_umol_m2_s field entirely -- FastAPI/pydantic
    # itself should reject this before our endpoint code even runs.
    response = client.post("/actuators/led/R01", json={})

    assert response.status_code == 422
