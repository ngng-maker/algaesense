"""Unit tests for jaxsr_calibration.calibration.standard_addition."""

from __future__ import annotations

import pytest

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.calibration.models import CalibrationGas
from jaxsr_calibration.calibration.standard_addition import (
    fit_sensitivity_per_sensor,
    run_standard_addition,
)
from tests.fixtures.synthetic_readings import make_standard_addition_readings


def test_run_standard_addition_always_needs_live_acquisition() -> None:
    gas = CalibrationGas.builtin("isoprene")
    with pytest.raises(LiveAcquisitionNotAvailableError):
        run_standard_addition(
            experiment_id="exp_test",
            calibration_gas=gas,
            spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        )


def test_fit_sensitivity_per_sensor_recovers_known_line_with_known_rf() -> None:
    df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 2.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.1}},
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        calibration_compound="isoprene",
        mw_g_mol=68.12,
        response_factor=0.63,
        seed=30,
    )

    models = fit_sensitivity_per_sensor(df)

    assert set(models.keys()) == {"PID01"}
    model = models["PID01"]
    assert model.sensor_id == "PID01"
    assert model.b0_mv == pytest.approx(2.0, abs=0.3)
    assert model.b1_mv_per_ppm_asgas == pytest.approx(4.0, abs=0.1)
    assert model.r_squared > 0.99
    assert model.status == "PASS"
    assert model.fit_method == "ols"
    assert model.calibration_gas.name == "isoprene"
    assert model.calibration_gas.has_rf is True
    # b1_iso_equiv = b1_asgas * RF = 4.0 * 0.63 = 2.52
    assert model.b1_mv_per_ppm_iso_equiv == pytest.approx(4.0 * 0.63, abs=0.05)


def test_fit_sensitivity_per_sensor_handles_unknown_response_factor() -> None:
    df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 3.0, "noise_std": 0.1}},
        spike_ppm_list=[0.0, 2.0, 10.0],
        calibration_compound="my_unlisted_voc",
        mw_g_mol=100.0,
        response_factor=None,
        seed=31,
    )

    models = fit_sensitivity_per_sensor(df)

    model = models["PID01"]
    assert model.calibration_gas.has_rf is False
    assert model.b1_mv_per_ppm_iso_equiv is None


def test_fit_sensitivity_per_sensor_fits_multiple_sensors_independently() -> None:
    df = make_standard_addition_readings(
        {
            "PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.1},
            "PID02": {"b0_mv": -0.5, "b1_mv_per_ppm": 6.0, "noise_std": 0.1},
        },
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        seed=32,
    )

    models = fit_sensitivity_per_sensor(df)

    assert set(models.keys()) == {"PID01", "PID02"}
    assert models["PID01"].b1_mv_per_ppm_asgas == pytest.approx(4.0, abs=0.1)
    assert models["PID02"].b1_mv_per_ppm_asgas == pytest.approx(6.0, abs=0.1)


def test_fit_sensitivity_per_sensor_flags_poor_fit_as_fail() -> None:
    df = make_standard_addition_readings(
        # Huge noise relative to the signal -- the fitted line should barely
        # explain any variance, landing well under the SUSPECT bar.
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 0.5, "noise_std": 20.0}},
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        seed=33,
    )

    models = fit_sensitivity_per_sensor(df)

    assert models["PID01"].status == "FAIL"


def test_fit_sensitivity_per_sensor_requires_at_least_two_spike_levels() -> None:
    df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0}},
        spike_ppm_list=[5.0],  # only one level, no baseline -- slope undefined
        seed=34,
    )

    with pytest.raises(ValueError, match="at least 2 distinct"):
        fit_sensitivity_per_sensor(df)


def test_fit_sensitivity_per_sensor_rejects_mixed_compounds_for_one_sensor() -> None:
    isoprene_rows = make_standard_addition_readings(
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0}},
        spike_ppm_list=[0.0, 5.0],
        calibration_compound="isoprene",
        seed=35,
    )
    acetone_rows = make_standard_addition_readings(
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0}},
        spike_ppm_list=[0.0, 5.0],
        calibration_compound="acetone",
        seed=36,
    )
    import polars as pl

    combined = pl.concat([isoprene_rows, acetone_rows])

    with pytest.raises(ValueError, match="exactly one calibration_compound"):
        fit_sensitivity_per_sensor(combined)


def test_fit_sensitivity_per_sensor_rejects_unimplemented_methods() -> None:
    df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0}}, spike_ppm_list=[0.0, 5.0], seed=37
    )
    for method in ("robust", "polynomial_deg2"):
        with pytest.raises(NotImplementedError):
            fit_sensitivity_per_sensor(df, method=method)
