"""Unit tests for jaxsr_calibration.diagnostics.ambient.run_ambient_baseline."""

from __future__ import annotations

import polars as pl
import pytest

from jaxsr_calibration.diagnostics.ambient import run_ambient_baseline
from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from tests.fixtures.synthetic_readings import make_ambient_readings


def test_run_ambient_baseline_raises_without_readings() -> None:
    with pytest.raises(LiveAcquisitionNotAvailableError):
        run_ambient_baseline(duration_h=12)


def test_run_ambient_baseline_fits_one_model_per_sensor() -> None:
    readings = make_ambient_readings(
        {
            "PID01": {"alpha": 10.0, "beta_rh": 0.2, "gamma_t": 0.5, "delta_rh_t": 0.0},
            "PID02": {"alpha": 8.0, "beta_rh": 0.1, "gamma_t": 0.3, "delta_rh_t": 0.005},
        },
        seed=10,
    )

    result = run_ambient_baseline(duration_h=12, readings=readings)

    assert set(result.covariate_models.keys()) == {"PID01", "PID02"}
    assert set(result.r_squared_per_sensor.keys()) == {"PID01", "PID02"}
    for sensor_id, model in result.covariate_models.items():
        assert model.sensor_id == sensor_id
        # Clean synthetic data with small noise should fit very well.
        assert result.r_squared_per_sensor[sensor_id] > 0.9


def test_run_ambient_baseline_skips_sensor_with_insufficient_rh_range() -> None:
    # Two separately-generated tables, one per sensor, each with its own RH
    # range -- simpler and clearer than trying to selectively narrow one
    # sensor's column out of an already-combined table.
    healthy = make_ambient_readings(
        {"PID01": {"alpha": 10.0, "beta_rh": 0.2, "gamma_t": 0.5}},
        rh_range=(20.0, 80.0),
        seed=11,
    )
    # Only a 3% RH spread -- below fit_covariate_model's default 20% minimum,
    # so this sensor should be silently skipped rather than blowing up the
    # whole call.
    narrow = make_ambient_readings(
        {"PID02": {"alpha": 8.0, "beta_rh": 0.1, "gamma_t": 0.3}},
        rh_range=(49.0, 52.0),
        seed=12,
    )
    combined = pl.concat([healthy, narrow])

    result = run_ambient_baseline(duration_h=12, readings=combined)

    assert "PID01" in result.covariate_models
    assert "PID02" not in result.covariate_models
