"""Tests for the two calibration fit methods added for real-sensor
behaviour: weighted least squares (noise that grows with concentration)
and a quadratic response curve (a sensor whose output bends).

Both are checked against a KNOWN generating truth, so the tests assert
recovery of that truth rather than merely that a number came back.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.calibration.apply import (
    apply_calibration,
    load_calibration,
    persist_calibration,
)
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor


LEVELS = [0.0, 100.0, 250.0, 500.0, 800.0]
N_PER_LEVEL = 12
TRUE_B0 = 20.0
TRUE_B1 = 0.60
TRUE_B2 = -0.00012
"""A negative quadratic term: response compresses at high concentration,
the usual PID saturation direction."""


def _recording(
    *,
    b2: float = 0.0,
    noise_per_ppm: float = 0.0,
    base_noise: float = 0.3,
    seed: int = 0,
) -> pl.DataFrame:
    """A standard-addition bench recording against a known gas, with
    optional curvature and optional concentration-dependent noise."""
    rng = np.random.default_rng(seed)
    spike = np.repeat(np.array(LEVELS), N_PER_LEVEL)
    sigma = base_noise + noise_per_ppm * spike
    voltage = TRUE_B0 + TRUE_B1 * spike + b2 * spike**2 + rng.normal(0.0, sigma)
    n = spike.size

    return pl.DataFrame(
        {
            "timestamp": [dt.datetime(2026, 7, 27, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=i) for i in range(n)],
            "sensor_id": ["PID01"] * n,
            "spike_ppm_asgas": spike,
            "pid_voltage_mv": voltage,
            "sample_t_c": np.full(n, 30.0),
            "sample_rh_pct": np.full(n, 55.0),
            "lamp_hours": np.full(n, 12.0),
            "calibration_compound": ["isoprene"] * n,
            "mw_g_mol": np.full(n, 68.12),
            "response_factor": np.full(n, 0.63),
            "response_factor_stderr": [None] * n,
            "calibration_source": ["test"] * n,
            "calibration_is_builtin": [True] * n,
        }
    )


def test_wls_recovers_the_true_slope_from_heteroscedastic_data() -> None:
    """Noise growing 20x across the range should still yield the true slope."""
    df = _recording(noise_per_ppm=0.006, seed=1)

    model = fit_sensitivity_per_sensor(df, method="wls")["PID01"]

    assert model.b1_mv_per_ppm_asgas == pytest.approx(TRUE_B1, rel=0.02)
    assert model.fit_method == "wls"
    assert model.b2_mv_per_ppm2_asgas is None


def test_wls_reports_a_tighter_slope_uncertainty_than_ols_on_the_same_data() -> None:
    """The point of weighting: precise low-concentration points should
    count for more than noisy high-concentration ones, which is what
    buys back precision on the slope."""
    df = _recording(noise_per_ppm=0.006, seed=2)

    ols = fit_sensitivity_per_sensor(df, method="ols")["PID01"]
    wls = fit_sensitivity_per_sensor(df, method="wls")["PID01"]

    assert wls.b1_stderr < ols.b1_stderr


def test_polynomial_deg2_recovers_a_curved_response_a_line_cannot() -> None:
    """A saturating sensor: the quadratic fit should capture the curvature
    and beat a straight line on the same data."""
    df = _recording(b2=TRUE_B2, seed=3)

    linear = fit_sensitivity_per_sensor(df, method="ols")["PID01"]
    quad = fit_sensitivity_per_sensor(df, method="polynomial_deg2")["PID01"]

    assert quad.b2_mv_per_ppm2_asgas == pytest.approx(TRUE_B2, rel=0.1)
    assert quad.r_squared > linear.r_squared


def test_a_quadratic_calibration_survives_persistence_and_inverts_correctly(tmp_path: Path) -> None:
    """The failure this guards against is silent: if b2 were dropped on
    save or load, apply_calibration would fall back to linear inversion
    and return biased concentrations with no error raised anywhere."""
    df = _recording(b2=TRUE_B2, seed=4)
    models = fit_sensitivity_per_sensor(df, method="polynomial_deg2")
    persist_calibration(models, "run_quad", "exp_quad", tmp_path)

    reloaded = load_calibration("run_quad", tmp_path)["PID01"]
    assert reloaded.b2_mv_per_ppm2_asgas == pytest.approx(TRUE_B2, rel=0.1)

    """
    Round-trip a known concentration: build the voltage the true curve
    would produce at 400 ppm, then check the calibration inverts back to
    400 rather than to whatever a straight line would have said.
    """
    true_ppm = 400.0
    voltage = TRUE_B0 + TRUE_B1 * true_ppm + TRUE_B2 * true_ppm**2

    recovered, _, _ = apply_calibration(
        pl.Series([voltage]),
        "PID01",
        pl.Series([30.0]),
        pl.Series([55.0]),
        "run_quad",
        data_dir=tmp_path,
    )
    assert recovered[0] == pytest.approx(true_ppm, rel=0.02)


def test_linear_calibrations_still_invert_linearly(tmp_path: Path) -> None:
    """Regression guard: adding the quadratic branch must not disturb the
    ordinary linear path."""
    df = _recording(seed=5)
    models = fit_sensitivity_per_sensor(df, method="ols")
    persist_calibration(models, "run_lin", "exp_lin", tmp_path)

    true_ppm = 300.0
    voltage = TRUE_B0 + TRUE_B1 * true_ppm

    recovered, _, _ = apply_calibration(
        pl.Series([voltage]),
        "PID01",
        pl.Series([30.0]),
        pl.Series([55.0]),
        "run_lin",
        data_dir=tmp_path,
    )
    assert recovered[0] == pytest.approx(true_ppm, rel=0.02)
    assert load_calibration("run_lin", tmp_path)["PID01"].b2_mv_per_ppm2_asgas is None
