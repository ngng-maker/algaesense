"""Unit tests for jaxsr_calibration.diagnostics.fleet_zero.run_fleet_zero."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.diagnostics.fleet_zero import run_fleet_zero
from jaxsr_calibration.processing.config import DiagnosticThresholds
from tests.fixtures.synthetic_readings import make_fleet_readings


def test_run_fleet_zero_raises_without_readings() -> None:
    # No live-acquisition backend exists yet (see errors.py's module
    # docstring) -- calling without `readings=` must fail clearly rather than
    # hang or silently return an empty result.
    with pytest.raises(LiveAcquisitionNotAvailableError):
        run_fleet_zero(duration_min=60)


def test_run_fleet_zero_all_healthy_sensors_are_green() -> None:
    readings = make_fleet_readings(
        {
            "PID01": {"mean_mv": 0.5, "std_mv": 0.1, "slope_mv_per_min": 0.0},
            "PID02": {"mean_mv": -0.3, "std_mv": 0.2, "slope_mv_per_min": 0.001},
        },
        seed=1,
    )

    result = run_fleet_zero(duration_min=5, readings=readings)

    assert result.summary_status == "GREEN"
    assert result.per_sensor["PID01"]["status"] == "PASS"
    assert result.per_sensor["PID02"]["status"] == "PASS"


def test_run_fleet_zero_one_bad_sensor_makes_summary_red() -> None:
    thresholds = DiagnosticThresholds()  # default max_mean_mv=5.0
    readings = make_fleet_readings(
        {
            "PID01": {"mean_mv": 0.5, "std_mv": 0.1, "slope_mv_per_min": 0.0},
            # Mean far beyond even fail_multiplier x the 5.0 mV limit --
            # should land as FAIL, dragging the whole fleet to RED.
            "PID02": {"mean_mv": 50.0, "std_mv": 0.2, "slope_mv_per_min": 0.0},
        },
        seed=2,
    )

    result = run_fleet_zero(duration_min=5, thresholds=thresholds, readings=readings)

    assert result.per_sensor["PID01"]["status"] == "PASS"
    assert result.per_sensor["PID02"]["status"] == "FAIL"
    assert result.summary_status == "RED"


def test_run_fleet_zero_borderline_sensor_is_suspect_not_fail() -> None:
    thresholds = DiagnosticThresholds()  # max_mean_mv=5.0, fail_multiplier=2.0
    readings = make_fleet_readings(
        # 7.5 mV is 1.5x the 5.0 mV limit -- over threshold, but under the
        # 2.0x fail_multiplier, so should classify as SUSPECT rather than FAIL.
        {"PID01": {"mean_mv": 7.5, "std_mv": 0.05, "slope_mv_per_min": 0.0}},
        seed=3,
    )

    result = run_fleet_zero(duration_min=5, thresholds=thresholds, readings=readings)

    assert result.per_sensor["PID01"]["status"] == "SUSPECT"
    assert result.summary_status == "YELLOW"


def test_run_fleet_zero_detects_drift_even_with_low_noise() -> None:
    # Deliberately low std_mv (would PASS on its own) but a steep slope --
    # this proves drift is checked independently of mean/std, not just
    # folded into the noise statistic.
    readings = make_fleet_readings(
        {"PID01": {"mean_mv": 0.0, "std_mv": 0.01, "slope_mv_per_min": 1.0}},
        seed=4,
    )

    result = run_fleet_zero(duration_min=5, readings=readings)

    assert result.per_sensor["PID01"]["status"] == "FAIL"
    assert result.per_sensor["PID01"]["slope_mv_per_min"] == pytest.approx(1.0, rel=0.2)


def test_run_fleet_zero_writes_parquet_when_output_dir_given(tmp_path: Path) -> None:
    readings = make_fleet_readings(
        {"PID01": {"mean_mv": 0.5, "std_mv": 0.1, "slope_mv_per_min": 0.0}}, seed=5
    )
    out_dir = tmp_path / "diagnostics" / "fleet_zero"

    run_fleet_zero(duration_min=5, readings=readings, output_dir=out_dir)

    written = list(out_dir.glob("*.parquet"))
    assert len(written) == 1
    table = pl.read_parquet(written[0])
    assert table["sensor_id"].to_list() == ["PID01"]
