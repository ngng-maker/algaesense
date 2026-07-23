"""Test 1 -- does the pre-calibration package's correction pipeline
actually let us recover the true VOC(PAR, temp) function, compared to
using raw, uncorrected sensor voltage?

Runs synthetic, deliberately contaminated raw sensor data (fleet-zero-
style per-sensor bias baked into a real standard-addition calibration,
an ambient RH/T covariate nuisance, common-mode fleet noise, and AR(1)
autocorrelated noise) through the REAL jaxsr_calibration functions:
fit_sensitivity_per_sensor / persist_calibration / apply_calibration,
fit_covariate_model / apply_covariate_correction, and
subtract_common_mode -- then checks how close the recovered VOC values
(and a curve_fit of the known functional form against them) land to
ground truth, for both a "raw" and a "corrected" processing path.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from ground_truth import (
    AmbientCovariateTruth,
    NoiseConfig,
    SensorCalibrationTruth,
    generate_ambient_blank_recording,
    generate_calibration_recording,
    generate_common_mode_check_recording,
    generate_experiment_recording,
    true_voc_ppm,
)

from jaxsr_calibration.calibration.apply import apply_calibration, persist_calibration
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from jaxsr_calibration.diagnostics.fleet_zero import run_fleet_zero
from jaxsr_calibration.processing.common_mode import subtract_common_mode
from jaxsr_calibration.processing.covariate import apply_covariate_correction, fit_covariate_model


SENSOR_IDS = ["PID01", "PID02", "PID03"]
REACTOR_IDS = ["R01", "R02", "R03"]
SENSOR_FOR_REACTOR = dict(zip(REACTOR_IDS, SENSOR_IDS))

TRUE_CALIBRATION = {
    "PID01": SensorCalibrationTruth(b0_mv=20.0, b1_mv_per_ppm=0.60),
    "PID02": SensorCalibrationTruth(b0_mv=17.0, b1_mv_per_ppm=0.55),
    "PID03": SensorCalibrationTruth(b0_mv=23.0, b1_mv_per_ppm=0.65),
}

CALIBRATION_RUN_ID = "benchmark_cal_01"


def _true_voc_curve_form(xy, vmax, k_m, temp_slope, gamma, photo_k, baseline):
    """The exact functional shape of true_voc_ppm, with unknown
    coefficients -- what curve_fit tries to recover from noisy/corrected
    observations, so we can quote a real percent-error-per-parameter
    number rather than only a generic RMSE. photo_threshold_par (380) and
    temp_ref (28.0) are treated as known constants, not fitted -- not
    smoothly identifiable via gradient-based curve_fit the way a
    coefficient is."""
    par, temp = xy
    light_term = vmax * par / (k_m + par)
    temp_term = temp_slope * (temp - 28.0)
    interaction = gamma * par * (temp - 28.0)
    photoinhibition = -photo_k * np.maximum(par - 380.0, 0.0) ** 2
    return baseline + light_term + temp_term + interaction + photoinhibition


@dataclass
class RecoveryResult:
    label: str
    recovered_params: dict[str, float]
    param_pct_error: dict[str, float]
    rmse_vs_true_ppm: float
    r2_on_dense_grid: float
    par_values: list[float] | None = None
    temp_values: list[float] | None = None
    measured_ppm: list[float] | None = None
    true_ppm: list[float] | None = None


def _fit_and_score(par_values: np.ndarray, temp_values: np.ndarray, ppm_values: np.ndarray) -> RecoveryResult:
    true_params = {
        "vmax": 800.0, "k_m": 150.0, "temp_slope": 3.0, "gamma": 0.05, "photo_k": 0.0104, "baseline": 30.0,
    }
    p0 = [600.0, 200.0, 1.0, 0.02, 0.005, 10.0]
    bounds = ([1.0, 1.0, -20.0, -5.0, 0.0, -200.0], [5000.0, 5000.0, 20.0, 5.0, 1.0, 200.0])
    popt, _ = curve_fit(
        _true_voc_curve_form, (par_values, temp_values), ppm_values, p0=p0, bounds=bounds, maxfev=20000
    )
    recovered = dict(zip(["vmax", "k_m", "temp_slope", "gamma", "photo_k", "baseline"], popt))
    pct_error = {
        name: 100.0 * abs(recovered[name] - true_params[name]) / abs(true_params[name])
        for name in true_params
    }

    rmse = float(np.sqrt(np.mean((ppm_values - true_voc_ppm(par_values, temp_values)) ** 2)))

    par_grid, temp_grid = np.meshgrid(np.linspace(0, 500, 40), np.linspace(20, 40, 40))
    true_grid = true_voc_ppm(par_grid.ravel(), temp_grid.ravel())
    predicted_grid = _true_voc_curve_form((par_grid.ravel(), temp_grid.ravel()), *popt)
    ss_res = np.sum((true_grid - predicted_grid) ** 2)
    ss_tot = np.sum((true_grid - np.mean(true_grid)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot)

    return RecoveryResult(
        label="",
        recovered_params=recovered,
        param_pct_error=pct_error,
        rmse_vs_true_ppm=rmse,
        r2_on_dense_grid=r2,
        par_values=[float(v) for v in par_values],
        temp_values=[float(v) for v in temp_values],
        measured_ppm=[float(v) for v in ppm_values],
        true_ppm=[float(v) for v in true_voc_ppm(par_values, temp_values)],
    )


def run_calibration_recovery_test(seed: int = 0, verbose: bool = True) -> dict[str, RecoveryResult]:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        calibration_dir = data_dir / "derived" / "calibrations" / "standard_addition"
        calibration_dir.mkdir(parents=True)

        """
        Step 1 -- a clean, controlled multi-level ('N-point') standard-
        addition calibration, exactly the way it's really run: isolated
        from the reactor room's ambient/common-mode contamination.
        """
        cal_df = generate_calibration_recording(
            SENSOR_IDS,
            TRUE_CALIBRATION,
            spike_ppm_list=[0.0, 50.0, 100.0, 200.0, 400.0, 800.0],
            n_per_level=8,
            seed=seed,
        )
        sensitivity_models = fit_sensitivity_per_sensor(cal_df)
        persist_calibration(sensitivity_models, CALIBRATION_RUN_ID, "cal_experiment", calibration_dir)

        if verbose:
            for sensor_id, model in sensitivity_models.items():
                true = TRUE_CALIBRATION[sensor_id]
                print(
                    f"  calibration[{sensor_id}]: recovered b0={model.b0_mv:.2f} (true {true.b0_mv}), "
                    f"b1={model.b1_mv_per_ppm_asgas:.4f} (true {true.b1_mv_per_ppm}), r2={model.r_squared:.4f}"
                )

        """
        Step 2 -- a zero-VOC ambient/blank recording spanning a real RH/T
        swing, to fit each sensor's nuisance ambient-covariate model
        (the thing run_ambient_baseline/fit_covariate_model exist for).
        """
        ambient_truth = AmbientCovariateTruth()
        ambient_df = generate_ambient_blank_recording(SENSOR_IDS, ambient_truth, seed=seed + 1)
        covariate_models = {}
        for sensor_id, sensor_df in ambient_df.partition_by("sensor_id", as_dict=True).items():
            sensor_id = sensor_id[0] if isinstance(sensor_id, tuple) else sensor_id
            mask = pl.Series([True] * sensor_df.height)
            model = fit_covariate_model(sensor_df, mask, method="ols")
            if model is not None:
                covariate_models[sensor_id] = model
        if verbose:
            for sensor_id, model in covariate_models.items():
                print(
                    f"  ambient[{sensor_id}]: recovered beta_rh={model.beta_rh:.4f} "
                    f"(true {ambient_truth.beta_rh}), gamma_t={model.gamma_t:.4f} (true {ambient_truth.gamma_t}), "
                    f"r2={model.r_squared:.4f}"
                )

        """
        Step 3 -- a synchronized fleet-wide zero check with a shared
        common-mode artifact on top. First, run_fleet_zero recovers each
        sensor's own fixed bias (averaging over full common-mode periods
        cancels the shared sine wave out of the mean, the same way a real
        multi-hour fleet-zero check would). Then, on the bias-corrected
        residual -- where every sensor's remaining true value genuinely
        IS the same (~0) -- subtract_common_mode is used for what it's
        actually valid for: confirming the shared artifact is removed,
        not blended with real per-reactor signal differences (see
        NoiseConfig's docstring on why it's never applied to the main
        experiment data below).
        """
        fleet_zero_bias = {sid: t.b0_mv for sid, t in TRUE_CALIBRATION.items()}
        common_mode_df = generate_common_mode_check_recording(
            SENSOR_IDS, REACTOR_IDS, fleet_zero_bias, seed=seed + 2
        )
        fleet_zero_result = run_fleet_zero(duration_min=2, readings=common_mode_df)
        residual_frames = []
        for sensor_id, sensor_df in common_mode_df.partition_by("sensor_id", as_dict=True).items():
            sensor_id = sensor_id[0] if isinstance(sensor_id, tuple) else sensor_id
            recovered_bias = fleet_zero_result.per_sensor[sensor_id]["mean_mv"]
            residual_frames.append(
                sensor_df.with_columns((pl.col("pid_voltage_mv") - recovered_bias).alias("pid_voltage_mv"))
            )
            if verbose:
                print(
                    f"  fleet-zero[{sensor_id}]: recovered bias={recovered_bias:.2f} "
                    f"(true {fleet_zero_bias[sensor_id]})"
                )
        residual_df = pl.concat(residual_frames)
        common_mode_removed = subtract_common_mode(residual_df, method="median")
        if verbose:
            before_std = float(residual_df["pid_voltage_mv"].std())
            after_std = float(common_mode_removed["pid_voltage_mv_common_mode_subtracted"].std())
            print(f"  common-mode residual std before removal: {before_std:.2f} mV, after: {after_std:.2f} mV")

        """
        Step 4 -- experiments spanning the (PAR, temp) domain, each
        contaminated with the ambient covariate nuisance and AR(1) noise
        (see NoiseConfig's docstring for why common-mode isn't layered
        in here). Every experiment is processed TWO ways: 'raw' (average
        the uncorrected voltage straight into apply_calibration) and
        'corrected' (apply_covariate_correction first).
        """
        rng = np.random.default_rng(seed + 3)
        par_grid_points = np.linspace(20.0, 480.0, 4)
        temp_grid_points = np.linspace(21.0, 39.0, 3)
        conditions = [(par, temp) for par in par_grid_points for temp in temp_grid_points]

        noise = NoiseConfig(ambient=ambient_truth)

        raw_par, raw_temp, raw_ppm = [], [], []
        corrected_par, corrected_temp, corrected_ppm = [], [], []

        for i, (par, temp) in enumerate(conditions):
            reactor_conditions = {REACTOR_IDS[i % 3]: (par, temp)}
            experiment_df = generate_experiment_recording(
                experiment_id=f"exp_{i:02d}",
                reactor_conditions=reactor_conditions,
                sensor_for_reactor=SENSOR_FOR_REACTOR,
                calibration_truth=TRUE_CALIBRATION,
                noise=noise,
                duration_s=300,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            sensor_id = SENSOR_FOR_REACTOR[REACTOR_IDS[i % 3]]

            raw_ppm_series, _, _ = apply_calibration(
                experiment_df["pid_voltage_mv"],
                sensor_id,
                experiment_df["sample_t_c"],
                experiment_df["sample_rh_pct"],
                CALIBRATION_RUN_ID,
                data_dir=calibration_dir,
            )
            raw_par.append(par)
            raw_temp.append(temp)
            raw_ppm.append(float(raw_ppm_series.mean()))

            corrected_df = apply_covariate_correction(experiment_df, covariate_models)
            corrected_ppm_series, _, _ = apply_calibration(
                corrected_df["pid_voltage_mv_covariate_corrected"],
                sensor_id,
                corrected_df["sample_t_c"],
                corrected_df["sample_rh_pct"],
                CALIBRATION_RUN_ID,
                data_dir=calibration_dir,
            )
            corrected_par.append(par)
            corrected_temp.append(temp)
            corrected_ppm.append(float(corrected_ppm_series.mean()))

        raw_result = _fit_and_score(np.array(raw_par), np.array(raw_temp), np.array(raw_ppm))
        raw_result.label = "raw (uncorrected)"
        corrected_result = _fit_and_score(
            np.array(corrected_par), np.array(corrected_temp), np.array(corrected_ppm)
        )
        corrected_result.label = "corrected (ambient-baseline applied)"

        return {"raw": raw_result, "corrected": corrected_result}


if __name__ == "__main__":
    results = run_calibration_recovery_test()
    for key, result in results.items():
        print(f"\n{result.label}:")
        print(f"  RMSE vs true VOC (ppm): {result.rmse_vs_true_ppm:.2f}")
        print(f"  R^2 of recovered curve vs true surface: {result.r2_on_dense_grid:.4f}")
        for name, err in result.param_pct_error.items():
            print(f"  {name} recovered={result.recovered_params[name]:.4f}, pct_error={err:.1f}%")
