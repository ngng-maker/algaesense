"""Tests for normalising ambient drift out of a signal that will then be
calibrated.

The distinction being guarded is easy to lose: `apply_covariate_correction`
drives clean air to zero, which is right for inspecting a sensor on its
own and wrong as an input to `apply_calibration`, because the calibration
subtracts the same baseline a second time.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.calibration.apply import (
    apply_calibration,
    persist_calibration,
)
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from jaxsr_calibration.processing.covariate import (
    apply_covariate_correction,
    fit_covariate_model,
    normalise_to_reference_conditions,
)


TRUE_B0 = 55.0
TRUE_B1 = 45.0
RH_COEFF = 0.9
T_COEFF = 2.2
REFERENCE_RH = 50.0
REFERENCE_T = 25.0

N = 400


def _ambient(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hours = np.linspace(0.0, 4.0, N)
    rh = 50.0 + 20.0 * np.sin(2 * np.pi * hours / 2.0) + rng.normal(0.0, 0.3, size=N)
    temp = 25.0 + 4.0 * np.sin(2 * np.pi * hours / 2.0 + 0.5) + rng.normal(0.0, 0.1, size=N)
    return rh, temp


def _frame(voltage: np.ndarray, rh: np.ndarray, temp: np.ndarray) -> pl.DataFrame:
    base = dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc)
    return pl.DataFrame(
        {
            "timestamp": [base + dt.timedelta(seconds=10 * i) for i in range(voltage.size)],
            "sensor_id": ["PID01"] * voltage.size,
            "pid_voltage_mv": voltage,
            "sample_rh_pct": rh,
            "sample_t_c": temp,
        }
    )


def _fitted_model():
    """A model learned from a clean-air baseline, as the real protocol
    fits it."""
    rh, temp = _ambient(seed=1)
    clean_air_mv = TRUE_B0 + RH_COEFF * (rh - 50.0) + T_COEFF * (temp - 25.0)
    frame = _frame(clean_air_mv, rh, temp)
    model = fit_covariate_model(frame, pl.Series(np.ones(N, dtype=bool)))
    assert model is not None
    return {"PID01": model}


def test_ambient_drift_is_removed_but_the_baseline_is_kept() -> None:
    """Clean air should come back at the sensor's own offset -- not at
    zero, and not swinging with the room."""
    models = _fitted_model()
    rh, temp = _ambient(seed=2)
    clean_air_mv = TRUE_B0 + RH_COEFF * (rh - 50.0) + T_COEFF * (temp - 25.0)

    out = normalise_to_reference_conditions(
        _frame(clean_air_mv, rh, temp), models, REFERENCE_RH, REFERENCE_T
    )
    normalised = out["pid_voltage_mv_ambient_normalised"].to_numpy()

    assert np.mean(normalised) == pytest.approx(TRUE_B0, abs=1.0)

    """The swing is what had to go: raw varies by tens of mV with the room."""
    assert np.std(normalised) < 0.1 * np.std(clean_air_mv)


def test_the_diagnostic_correction_drives_the_same_signal_to_zero() -> None:
    """Both functions are correct; they answer different questions. This
    one exists so the difference is asserted rather than assumed."""
    models = _fitted_model()
    rh, temp = _ambient(seed=2)
    clean_air_mv = TRUE_B0 + RH_COEFF * (rh - 50.0) + T_COEFF * (temp - 25.0)

    out = apply_covariate_correction(_frame(clean_air_mv, rh, temp), models)

    assert np.mean(out["pid_voltage_mv_covariate_corrected"].to_numpy()) == pytest.approx(0.0, abs=1.0)


def test_composing_with_a_calibration_recovers_the_true_concentration(tmp_path) -> None:
    """The failure this guards is silent and biased, not loud: feeding the
    zero-centred correction into a calibration subtracts the baseline
    twice and reads low by roughly b0/b1 -- here 55/45, about 1.2 ppm."""
    models = _fitted_model()

    bench_ppm = np.repeat([0.0, 1.0, 3.0, 5.0], 5)
    bench = pl.DataFrame(
        {
            "timestamp": [
                dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=i)
                for i in range(bench_ppm.size)
            ],
            "sensor_id": ["PID01"] * bench_ppm.size,
            "spike_ppm_asgas": bench_ppm,
            "pid_voltage_mv": TRUE_B0 + TRUE_B1 * bench_ppm,
            "sample_t_c": np.full(bench_ppm.size, REFERENCE_T),
            "sample_rh_pct": np.full(bench_ppm.size, REFERENCE_RH),
            "lamp_hours": np.zeros(bench_ppm.size),
            "calibration_compound": ["isobutylene"] * bench_ppm.size,
            "mw_g_mol": np.full(bench_ppm.size, 56.11),
            "response_factor": np.ones(bench_ppm.size),
            "response_factor_stderr": [None] * bench_ppm.size,
            "calibration_source": ["test"] * bench_ppm.size,
            "calibration_is_builtin": [False] * bench_ppm.size,
        }
    )
    persist_calibration(fit_sensitivity_per_sensor(bench, method="ols"), "run", "exp", tmp_path)

    rh, temp = _ambient(seed=3)
    true_ppm = 2.5
    observed = TRUE_B0 + TRUE_B1 * true_ppm + RH_COEFF * (rh - 50.0) + T_COEFF * (temp - 25.0)
    frame = _frame(observed, rh, temp)

    normalised = normalise_to_reference_conditions(frame, models, REFERENCE_RH, REFERENCE_T)
    recovered, _, _ = apply_calibration(
        normalised["pid_voltage_mv_ambient_normalised"],
        "PID01",
        normalised["sample_t_c"],
        normalised["sample_rh_pct"],
        "run",
        data_dir=tmp_path,
    )
    assert float(np.mean(recovered.to_numpy())) == pytest.approx(true_ppm, abs=0.05)

    zero_centred = apply_covariate_correction(frame, models)
    biased, _, _ = apply_calibration(
        zero_centred["pid_voltage_mv_covariate_corrected"],
        "PID01",
        zero_centred["sample_t_c"],
        zero_centred["sample_rh_pct"],
        "run",
        data_dir=tmp_path,
    )
    assert float(np.mean(biased.to_numpy())) == pytest.approx(true_ppm - TRUE_B0 / TRUE_B1, abs=0.1)


def test_a_sensor_with_no_model_passes_through_untouched() -> None:
    """Losing the correction for one sensor must not discard its data."""
    rh, temp = _ambient(seed=4)
    voltage = TRUE_B0 + RH_COEFF * (rh - 50.0)

    out = normalise_to_reference_conditions(_frame(voltage, rh, temp), {}, REFERENCE_RH, REFERENCE_T)

    assert out["pid_voltage_mv_ambient_normalised"].to_list() == pytest.approx(voltage.tolist())
