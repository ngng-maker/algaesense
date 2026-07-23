"""Test 3 -- ground truth #1: given ONE specific, time-varying PAR(t)
schedule within a single experiment, does discover_led_response_dynamics
recover the true dynamic law governing how VOC unfolds over time?

Distinct from Test 1/2 (ground truth #2: how VOC varies ACROSS many
different static settings). Here there is exactly one experiment, one
real control-profile schedule (a sinusoid, driven by the actual
`evaluate_control_profile` function this project's edge service uses to
drive the LED), and the question is purely about the WITHIN-experiment
time dynamics: dVOC/dt = (1/tau) * (true_voc_ppm(par(t), temp) - VOC(t))
-- see ground_truth.py's dynamic-ground-truth section for why this is
tied to the same static surface as its steady-state target.

Runs both the REAL top-level `discover_led_response_dynamics` (for a
genuine end-to-end proof the real tool works, exactly as production
code exposes it -- which applies calibration to raw voltage only, no
ambient-baseline correction) AND a parallel internal-pieces variant that
adds ambient-baseline covariate correction before calibration, so the
two can be compared the same way Test 1 compared raw vs. corrected --
this doubles as a real test of whether discover_led_response_dynamics
would benefit from a correction step it doesn't currently apply.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jaxsr
import numpy as np
import polars as pl

from ground_truth import (
    AmbientCovariateTruth,
    DYNAMIC_RELAXATION_TAU_S,
    NoiseConfig,
    PAR_BOUNDS,
    SensorCalibrationTruth,
    generate_ambient_blank_recording,
    generate_calibration_recording,
    generate_dynamic_experiment_recording,
    simulate_true_dynamic_trajectory,
    true_voc_ppm,
)

from algaesense_agent.mcp_pipeline.pipeline import discover_led_response_dynamics
from algaesense_agent.raw_readers import load_raw_voc_readings
from jaxsr_calibration.calibration.apply import apply_calibration, persist_calibration
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from jaxsr_calibration.processing.covariate import apply_covariate_correction, fit_covariate_model
from jaxsr_calibration.processing.features import load_timeseries_for_jaxsr


EXPERIMENT_ID = "exp_dynamics_bench"
REACTOR_ID = "R01"
SENSOR_ID = "PID01"
CALIBRATION_RUN_ID = "benchmark_dynamics_cal_01"
TRUE_CALIBRATION = SensorCalibrationTruth(b0_mv=20.0, b1_mv_per_ppm=0.60)
TEMP = 30.0
DURATION_S = 1800  # 3 full periods of the profile below
PROFILE = {"shape": "sinusoid", "mean_par_umol_m2_s": 220.0, "amplitude_par_umol_m2_s": 180.0, "period_s": 600.0}


def true_derivative(voc: np.ndarray, par: np.ndarray) -> np.ndarray:
    """The exact right-hand side of the ground-truth ODE, evaluated at
    an arbitrary (voc, par) state -- not just along the one trajectory
    actually simulated, so this can score a model against a dense
    held-out grid the same way Test 1 does."""
    return (1.0 / DYNAMIC_RELAXATION_TAU_S) * (true_voc_ppm(par, TEMP) - voc)


@dataclass
class DynamicsRecoveryResult:
    label: str
    selected_features: list[str]
    equation: str
    rmse_vs_true_derivative: float
    r2_vs_true_derivative: float


@dataclass
class DynamicsRecoveryRun:
    results: dict[str, DynamicsRecoveryResult]
    t: np.ndarray
    par_values: np.ndarray
    true_voc_values: np.ndarray


def _score_model(model, state_names: list[str]) -> tuple[float, float]:
    """Dense held-out (voc, par) grid, mirroring Test 1's dense-grid
    approach -- 'true_derivative' is a real function of state, so it can
    be evaluated anywhere, not just at points the trajectory visited."""
    voc_grid, par_grid = np.meshgrid(np.linspace(0.0, 900.0, 40), np.linspace(*PAR_BOUNDS, 40))
    par_idx = state_names.index("reactor_par_umol_m2_s")
    voc_idx = state_names.index("ppm_asgas")
    X_test = np.zeros((voc_grid.size, 2))
    X_test[:, voc_idx] = voc_grid.ravel()
    X_test[:, par_idx] = par_grid.ravel()

    true_rate = true_derivative(voc_grid.ravel(), par_grid.ravel())
    predicted_rate = np.asarray(model.predict(X_test))

    rmse = float(np.sqrt(np.mean((predicted_rate - true_rate) ** 2)))
    ss_res = np.sum((true_rate - predicted_rate) ** 2)
    ss_tot = np.sum((true_rate - np.mean(true_rate)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)
    return rmse, r2


def _discover_dynamics_from_readings(readings: pl.DataFrame, max_terms: int = 5):
    """Mirrors discover_led_response_dynamics's own internal steps
    exactly (load_timeseries_for_jaxsr -> jaxsr.discover_dynamics), just
    called directly on an in-memory DataFrame so the live model objects
    are available for scoring -- the public function deliberately strips
    those (see its DynamicsDiscoveryResult docstring), same reasoning
    Test 2 already used for its own internal scoring fit."""
    X, t, state_names = load_timeseries_for_jaxsr(readings, state_columns=["ppm_asgas", "reactor_par_umol_m2_s"])
    result = jaxsr.discover_dynamics(X, t, state_names=state_names, max_terms=max_terms)
    return result, state_names


def run_dynamics_recovery_test(seed: int = 0, verbose: bool = True) -> DynamicsRecoveryRun:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        calibration_dir = data_dir / "derived" / "calibrations" / "standard_addition"
        calibration_dir.mkdir(parents=True)

        cal_df = generate_calibration_recording(
            [SENSOR_ID],
            {SENSOR_ID: TRUE_CALIBRATION},
            spike_ppm_list=[0.0, 50.0, 100.0, 200.0, 400.0, 800.0],
            n_per_level=8,
            seed=seed,
        )
        sensitivity_models = fit_sensitivity_per_sensor(cal_df)
        persist_calibration(sensitivity_models, CALIBRATION_RUN_ID, "cal_experiment", calibration_dir)

        ambient_truth = AmbientCovariateTruth()
        ambient_df = generate_ambient_blank_recording([SENSOR_ID], ambient_truth, seed=seed + 1)
        covariate_model = fit_covariate_model(ambient_df, pl.Series([True] * ambient_df.height), method="ols")

        def par_fn(elapsed_s: float) -> float:
            from algaesense_edge.actuators.control_profiles import evaluate_control_profile

            return evaluate_control_profile(PROFILE, elapsed_s)

        t, par_values, true_voc_values = simulate_true_dynamic_trajectory(par_fn, TEMP, DURATION_S)

        noise = NoiseConfig(ambient=ambient_truth)
        readings = generate_dynamic_experiment_recording(
            EXPERIMENT_ID, REACTOR_ID, SENSOR_ID, par_values, TEMP, true_voc_values,
            TRUE_CALIBRATION, noise, seed=seed + 2,
        )

        """
        Step 1 -- prove the REAL, public discover_led_response_dynamics
        works end to end, exactly as production code exposes it: write
        the raw Parquet layout it actually reads from and call it
        directly, no internal shortcuts.
        """
        raw_dir = data_dir / "raw" / "experiments" / EXPERIMENT_ID / f"sensor_id={SENSOR_ID}"
        raw_dir.mkdir(parents=True)
        readings.write_parquet(raw_dir / "hour=0.parquet")

        real_result = discover_led_response_dynamics(
            EXPERIMENT_ID, REACTOR_ID, SENSOR_ID, CALIBRATION_RUN_ID, data_dir=data_dir, max_terms=5
        )
        if verbose:
            print(f"  [real discover_led_response_dynamics] equation: {real_result.equations.get('ppm_asgas')}")
            print(f"  [real discover_led_response_dynamics] selected_features: {real_result.selected_features.get('ppm_asgas')}")

        """
        Step 2 -- score BOTH raw and ambient-corrected variants against
        the true derivative law on a dense held-out grid, using the same
        underlying pieces the real function calls (just with the live
        model kept around, unlike the public function's JSON-safe
        return type).
        """
        real_readings = load_raw_voc_readings(data_dir, EXPERIMENT_ID).filter(
            (pl.col("reactor_id") == REACTOR_ID) & (pl.col("sensor_id") == SENSOR_ID)
        )

        raw_ppm, _, _ = apply_calibration(
            real_readings["pid_voltage_mv"], SENSOR_ID, real_readings["sample_t_c"],
            real_readings["sample_rh_pct"], CALIBRATION_RUN_ID, data_dir=calibration_dir,
        )
        raw_readings = real_readings.with_columns(raw_ppm.alias("ppm_asgas"))
        raw_result, raw_state_names = _discover_dynamics_from_readings(raw_readings)

        corrected_df = apply_covariate_correction(real_readings, {SENSOR_ID: covariate_model})
        corrected_ppm, _, _ = apply_calibration(
            corrected_df["pid_voltage_mv_covariate_corrected"], SENSOR_ID, corrected_df["sample_t_c"],
            corrected_df["sample_rh_pct"], CALIBRATION_RUN_ID, data_dir=calibration_dir,
        )
        corrected_readings = real_readings.with_columns(corrected_ppm.alias("ppm_asgas"))
        corrected_result, corrected_state_names = _discover_dynamics_from_readings(corrected_readings)

        results = {}
        for label, result, state_names in [
            ("raw (matches real discover_led_response_dynamics)", raw_result, raw_state_names),
            ("ambient-corrected (not yet done by the real tool)", corrected_result, corrected_state_names),
        ]:
            model = result.models["ppm_asgas"]
            rmse, r2 = _score_model(model, state_names)
            results[label] = DynamicsRecoveryResult(
                label=label,
                selected_features=list(model.selected_features_),
                equation=result.equations.get("ppm_asgas", ""),
                rmse_vs_true_derivative=rmse,
                r2_vs_true_derivative=r2,
            )
            if verbose:
                print(f"\n{label}:")
                print(f"  equation: {result.equations.get('ppm_asgas')}")
                print(f"  selected_features: {model.selected_features_}")
                print(f"  RMSE vs true derivative (ppm/s): {rmse:.4f}")
                print(f"  R^2 vs true derivative (dense grid): {r2:.4f}")

        return DynamicsRecoveryRun(results=results, t=t, par_values=par_values, true_voc_values=true_voc_values)


if __name__ == "__main__":
    run_dynamics_recovery_test()
