"""Unit tests for jaxsr_calibration.processing.covariate.apply_covariate_correction."""

from __future__ import annotations

import polars as pl
import pytest

from jaxsr_calibration.processing.covariate import apply_covariate_correction, fit_covariate_model
from tests.fixtures.synthetic_readings import make_ambient_readings


def test_apply_covariate_correction_removes_ambient_signal_in_clean_air() -> None:
    # Fit a CovariateModel on pure ambient (zero-VOC) data, then apply that
    # SAME model back to the data it was fit on -- the corrected signal
    # should collapse to ~0 everywhere, since by construction the model's
    # prediction should match the data almost exactly (small noise aside).
    readings = make_ambient_readings(
        {"PID01": {"alpha": 10.0, "beta_rh": 0.2, "gamma_t": 0.5, "delta_rh_t": 0.0, "noise_std": 0.05}},
        seed=60,
    )
    mask = pl.Series([True] * readings.height)
    model = fit_covariate_model(readings, mask)

    corrected = apply_covariate_correction(readings, {"PID01": model})

    assert "pid_voltage_mv_covariate_corrected" in corrected.columns
    corrected_values = corrected["pid_voltage_mv_covariate_corrected"].to_numpy()
    # Not exactly 0 (there's real measurement noise in the synthetic data),
    # but should be small relative to the raw voltage's own scale (which
    # ranges roughly 10-30 mV given alpha=10, beta_rh=0.2*80=16).
    assert abs(corrected_values.mean()) < 0.5
    assert corrected_values.std() < 0.5


def test_apply_covariate_correction_passes_through_sensor_with_no_model() -> None:
    readings = make_ambient_readings({"PID02": {"alpha": 5.0}}, seed=61)

    corrected = apply_covariate_correction(readings, {})  # no models at all

    # With no fitted model for PID02, the corrected column should just equal
    # the raw voltage, not be dropped or set to null.
    raw = corrected["pid_voltage_mv"].to_numpy()
    corrected_values = corrected["pid_voltage_mv_covariate_corrected"].to_numpy()
    assert corrected_values == pytest.approx(raw)


def test_apply_covariate_correction_applies_a_robust_model_not_just_ols() -> None:
    """Regression test for a real bug the covariate work uncovered: this
    predicate used to check `model.method != "ols"` exactly, which meant a
    fitted "robust" model would silently never actually get applied here."""
    readings = make_ambient_readings(
        {"PID01": {"alpha": 10.0, "beta_rh": 0.2, "gamma_t": 0.5, "delta_rh_t": 0.0, "noise_std": 0.05}},
        seed=63,
    )
    mask = pl.Series([True] * readings.height)
    model = fit_covariate_model(readings, mask, method="robust")

    corrected = apply_covariate_correction(readings, {"PID01": model})

    # A correctly-applied correction should leave residuals near zero, not
    # equal to the raw (uncorrected) voltage.
    corrected_values = corrected["pid_voltage_mv_covariate_corrected"].to_numpy()
    assert abs(corrected_values.mean()) < 0.5
    assert corrected_values.std() < 0.5


def test_apply_covariate_correction_handles_multiple_sensors_independently() -> None:
    readings = make_ambient_readings(
        {
            "PID01": {"alpha": 10.0, "beta_rh": 0.2, "gamma_t": 0.5, "noise_std": 0.05},
            "PID02": {"alpha": 5.0, "beta_rh": 0.05, "gamma_t": 0.2, "noise_std": 0.05},
        },
        seed=62,
    )
    mask = pl.Series([True] * readings.height)
    models = {}
    for (sensor_id,), sensor_df in readings.partition_by("sensor_id", as_dict=True).items():
        sensor_mask = pl.Series([True] * sensor_df.height)
        models[sensor_id] = fit_covariate_model(sensor_df, sensor_mask)

    corrected = apply_covariate_correction(readings, models)

    assert corrected.filter(pl.col("sensor_id") == "PID01").height > 0
    assert corrected.filter(pl.col("sensor_id") == "PID02").height > 0
    # Both sensors' corrected signals should be small, each relative to its
    # own fitted model, even though their true coefficients differ.
    for sensor_id in ("PID01", "PID02"):
        values = corrected.filter(pl.col("sensor_id") == sensor_id)[
            "pid_voltage_mv_covariate_corrected"
        ].to_numpy()
        assert abs(values.mean()) < 0.5
