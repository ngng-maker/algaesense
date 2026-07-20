"""Unit tests for jaxsr_calibration.calibration.reference_jar."""

from __future__ import annotations

import polars as pl
import pytest

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.calibration.reference_jar import (
    compute_fleet_ratios,
    run_reference_jar_rotation,
)


def test_run_reference_jar_rotation_always_needs_live_acquisition() -> None:
    with pytest.raises(LiveAcquisitionNotAvailableError):
        run_reference_jar_rotation(sensors="all")


def test_compute_fleet_ratios_sensor_at_median_gets_ratio_one() -> None:
    readings = pl.DataFrame(
        {
            "sensor_id": ["PID01", "PID02", "PID03"],
            "pid_voltage_mv": [8.0, 10.0, 12.0],
        }
    )

    ratios = compute_fleet_ratios(readings)

    # Median of [8, 10, 12] is 10 -- PID02 (the median sensor) gets ratio 1.0,
    # PID01 is 20% low (0.8), PID03 is 20% high (1.2).
    assert ratios["PID02"] == pytest.approx(1.0)
    assert ratios["PID01"] == pytest.approx(0.8)
    assert ratios["PID03"] == pytest.approx(1.2)


def test_compute_fleet_ratios_averages_multiple_readings_per_sensor() -> None:
    readings = pl.DataFrame(
        {
            "sensor_id": ["PID01", "PID01", "PID02"],
            "pid_voltage_mv": [9.0, 11.0, 10.0],  # PID01 mean = 10.0
        }
    )

    ratios = compute_fleet_ratios(readings)

    assert ratios["PID01"] == pytest.approx(1.0)


def test_compute_fleet_ratios_rejects_empty_readings() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_fleet_ratios(pl.DataFrame({"sensor_id": [], "pid_voltage_mv": []}))


def test_compute_fleet_ratios_rejects_zero_fleet_median() -> None:
    readings = pl.DataFrame({"sensor_id": ["PID01", "PID02"], "pid_voltage_mv": [0.0, 0.0]})

    with pytest.raises(ValueError, match="median reading is exactly 0"):
        compute_fleet_ratios(readings)


def test_compute_fleet_ratios_rejects_readings_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="compute_fleet_ratios"):
        compute_fleet_ratios(pl.DataFrame({"sensor_id": ["PID01"]}))
