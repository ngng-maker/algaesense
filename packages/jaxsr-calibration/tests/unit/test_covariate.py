"""Unit tests for jaxsr_calibration.processing.covariate.fit_covariate_model."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.processing.covariate import fit_covariate_model
from jaxsr_calibration.processing.errors import TrainingWindowInsufficientError


def _synthetic_covariate_df(
    n: int = 200,
    alpha: float = 10.0,
    beta_rh: float = 0.2,
    gamma_t: float = 0.5,
    delta_rh_t: float = 0.01,
    rh_min: float = 20.0,
    rh_max: float = 80.0,
    noise_std: float = 0.05,
    sensor_id: str = "PID01",
    seed: int = 0,
) -> pl.DataFrame:
    # `np.random.default_rng(seed)` is numpy's modern random-number-generator
    # API (replacing the older, implicitly-global `np.random.seed(...)` +
    # `np.random.normal(...)` style) -- using a fixed seed makes the "random"
    # data fully reproducible between test runs, which matters for a test
    # that checks fitted coefficients land close to known true values.
    rng = np.random.default_rng(seed)
    rh = rng.uniform(rh_min, rh_max, size=n)
    temp = rng.uniform(28.0, 34.0, size=n)
    noise = rng.normal(0.0, noise_std, size=n)
    voltage = alpha + beta_rh * rh + gamma_t * temp + delta_rh_t * (rh * temp) + noise

    base_time = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
    timestamps = [base_time + dt.timedelta(seconds=i) for i in range(n)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": [sensor_id] * n,
            "sample_rh_pct": rh,
            "sample_t_c": temp,
            "pid_voltage_mv": voltage,
        }
    )


def test_fit_covariate_model_recovers_known_coefficients() -> None:
    true_alpha, true_beta_rh, true_gamma_t, true_delta = 10.0, 0.2, 0.5, 0.01
    df = _synthetic_covariate_df(
        alpha=true_alpha, beta_rh=true_beta_rh, gamma_t=true_gamma_t, delta_rh_t=true_delta
    )
    # `pl.Series([True] * n)` : every row counts as "training data" here --
    # ambient-baseline fitting (Milestone 2's use case) doesn't need to carve
    # out a sub-window the way per-experiment covariate fitting will.
    mask = pl.Series([True] * df.height)

    model = fit_covariate_model(df, mask, method="ols", min_rh_range_pct=20.0)

    assert model is not None
    assert model.sensor_id == "PID01"
    # `pytest.approx(x, abs=...)` : floating-point-safe equality -- with 200
    # noisy samples the fit won't recover the *exact* true coefficient, only
    # something close to it, so we assert "close" rather than "equal".
    assert model.alpha == pytest.approx(true_alpha, abs=0.5)
    assert model.beta_rh == pytest.approx(true_beta_rh, abs=0.05)
    assert model.gamma_t == pytest.approx(true_gamma_t, abs=0.05)
    assert model.delta_rh_t == pytest.approx(true_delta, abs=0.01)
    assert model.r_squared > 0.95
    assert model.covariance.shape == (4, 4)
    assert model.method == "ols"
    assert model.symbolic_regressor is None


def test_fit_covariate_model_returns_none_for_empty_training_window() -> None:
    df = _synthetic_covariate_df(n=50)
    # An all-False mask selects zero rows -- simulating "this sensor has no
    # data in the requested training window".
    mask = pl.Series([False] * df.height)

    assert fit_covariate_model(df, mask) is None


def test_fit_covariate_model_raises_on_narrow_rh_range() -> None:
    # rh_min/rh_max only 5% apart -- far below the default 20% requirement.
    df = _synthetic_covariate_df(rh_min=40.0, rh_max=45.0)
    mask = pl.Series([True] * df.height)

    with pytest.raises(TrainingWindowInsufficientError):
        fit_covariate_model(df, mask, min_rh_range_pct=20.0)


def test_fit_covariate_model_raises_on_too_few_rows() -> None:
    # Wide RH range but only 3 rows -- fails the minimum-row-count guard even
    # though the RH-range check alone would have passed.
    df = _synthetic_covariate_df(n=3, rh_min=10.0, rh_max=90.0)
    mask = pl.Series([True] * df.height)

    with pytest.raises(TrainingWindowInsufficientError):
        fit_covariate_model(df, mask)


def test_fit_covariate_model_rejects_multi_sensor_data() -> None:
    df1 = _synthetic_covariate_df(sensor_id="PID01")
    df2 = _synthetic_covariate_df(sensor_id="PID02", seed=1)
    # `pl.concat` stacks two DataFrames with the same columns row-wise into one.
    combined = pl.concat([df1, df2])
    mask = pl.Series([True] * combined.height)

    with pytest.raises(ValueError, match="exactly one sensor_id"):
        fit_covariate_model(combined, mask)


def test_fit_covariate_model_robust_and_symbolic_not_implemented_yet() -> None:
    df = _synthetic_covariate_df()
    mask = pl.Series([True] * df.height)

    for method in ("robust", "symbolic"):
        with pytest.raises(NotImplementedError):
            fit_covariate_model(df, mask, method=method)
