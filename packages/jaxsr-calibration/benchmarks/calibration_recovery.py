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

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

"""
Makes this file's own sibling import below (ground_truth) resolve regardless
of how this script is invoked -- see the identical note in run_all.py.
"""

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    generate_cross_sensor_consistency_recording,
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

CROSS_SENSOR_PAR = 250.0
CROSS_SENSOR_TEMP = 30.0
CROSS_SENSOR_DURATION_S = 600
CROSS_SENSOR_REACTOR_ID = "R_cross_sensor"
"""A mid-domain (PAR, temp) point all 3 sensors observe SIMULTANEOUSLY --
see generate_cross_sensor_consistency_recording's docstring for why this
checks a genuinely different question from the rest of Test 1 (does
correction bring 3 sensors observing the SAME true value into agreement
with each other, not just recover one sensor's own value)."""


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


@dataclass
class CrossSensorConsistencyResult:
    """Three sensors observing the EXACT SAME (par, temp) condition
    simultaneously -- checks whether the correction pipeline brings
    them into agreement WITH EACH OTHER (cross_sensor_spread) and with
    the TRUE value (rmse_vs_true), not just whether it recovers one
    sensor's own reading correctly."""

    par: float
    temp: float
    true_ppm: float
    t: list[float]
    sensor_ids: list[str]
    raw_ppm: dict[str, list[float]]
    corrected_ppm: dict[str, list[float]]
    raw_cross_sensor_spread: float
    corrected_cross_sensor_spread: float
    raw_rmse_vs_true: dict[str, float]
    corrected_rmse_vs_true: dict[str, float]


def run_cross_sensor_consistency_test(
    calibration_dir: Path, covariate_models: dict, seed: int, verbose: bool = True,
) -> CrossSensorConsistencyResult:
    """Reuses the SAME already-fitted calibration/covariate models the
    rest of Test 1 fits (passed in, not refit) -- this sub-test isn't
    about whether calibration/covariate fitting works (already checked
    elsewhere), it's specifically about whether applying that already-
    correct correction to 3 sensors observing the SAME true value makes
    them converge with each other and with the truth.

    Deliberately does NOT inject the sluggish/zigzag/curvy drift
    artifacts (see ground_truth.py's generate_cross_sensor_consistency_recording
    docstring for why) -- those represent a genuinely different noise
    class this pipeline was never built to correct for, so including
    them here would misleadingly look like a pipeline failure rather
    than an honest scope limitation. That limitation is called out as a
    plain-text note in REPORT.md instead of demonstrated live."""
    noise = NoiseConfig(ambient=AmbientCovariateTruth())
    readings = generate_cross_sensor_consistency_recording(
        experiment_id="exp_cross_sensor",
        reactor_id=CROSS_SENSOR_REACTOR_ID,
        sensor_ids=SENSOR_IDS,
        calibration_truth=TRUE_CALIBRATION,
        noise=noise,
        par=CROSS_SENSOR_PAR,
        temp=CROSS_SENSOR_TEMP,
        duration_s=CROSS_SENSOR_DURATION_S,
        seed=seed,
    )

    true_ppm = float(true_voc_ppm(CROSS_SENSOR_PAR, CROSS_SENSOR_TEMP))
    t_values: list[float] | None = None
    raw_ppm: dict[str, list[float]] = {}
    corrected_ppm: dict[str, list[float]] = {}
    raw_rmse: dict[str, float] = {}
    corrected_rmse: dict[str, float] = {}

    for sensor_id in SENSOR_IDS:
        sensor_df = readings.filter(pl.col("sensor_id") == sensor_id)
        if t_values is None:
            t0 = sensor_df["timestamp"][0]
            t_values = [(ts - t0).total_seconds() for ts in sensor_df["timestamp"]]

        raw_series, _, _ = apply_calibration(
            sensor_df["pid_voltage_mv"], sensor_id, sensor_df["sample_t_c"], sensor_df["sample_rh_pct"],
            CALIBRATION_RUN_ID, data_dir=calibration_dir,
        )
        raw_ppm[sensor_id] = [float(v) for v in raw_series]
        raw_rmse[sensor_id] = float(np.sqrt(np.mean((raw_series.to_numpy() - true_ppm) ** 2)))

        corrected_df = apply_covariate_correction(sensor_df, covariate_models)
        corrected_series, _, _ = apply_calibration(
            corrected_df["pid_voltage_mv_covariate_corrected"], sensor_id, corrected_df["sample_t_c"],
            corrected_df["sample_rh_pct"], CALIBRATION_RUN_ID, data_dir=calibration_dir,
        )
        corrected_ppm[sensor_id] = [float(v) for v in corrected_series]
        corrected_rmse[sensor_id] = float(np.sqrt(np.mean((corrected_series.to_numpy() - true_ppm) ** 2)))

    """
    Cross-sensor spread: at each timestamp, the standard deviation ACROSS
    the 3 sensors' readings (not vs. the true value) -- averaged over the
    whole window. This is "do the sensors agree with EACH OTHER,"
    distinct from raw_rmse/corrected_rmse ("does each sensor agree with
    the truth").
    """
    raw_matrix = np.array([raw_ppm[sid] for sid in SENSOR_IDS])
    corrected_matrix = np.array([corrected_ppm[sid] for sid in SENSOR_IDS])
    raw_spread = float(np.mean(np.std(raw_matrix, axis=0)))
    corrected_spread = float(np.mean(np.std(corrected_matrix, axis=0)))

    if verbose:
        print(f"  cross-sensor spread (ppm, std across sensors, averaged over time): raw={raw_spread:.2f}, corrected={corrected_spread:.2f}")
        for sensor_id in SENSOR_IDS:
            print(
                f"  cross-sensor[{sensor_id}]: raw RMSE vs true={raw_rmse[sensor_id]:.2f}, "
                f"corrected RMSE vs true={corrected_rmse[sensor_id]:.2f}"
            )

    return CrossSensorConsistencyResult(
        par=CROSS_SENSOR_PAR,
        temp=CROSS_SENSOR_TEMP,
        true_ppm=true_ppm,
        t=t_values or [],
        sensor_ids=list(SENSOR_IDS),
        raw_ppm=raw_ppm,
        corrected_ppm=corrected_ppm,
        raw_cross_sensor_spread=raw_spread,
        corrected_cross_sensor_spread=corrected_spread,
        raw_rmse_vs_true=raw_rmse,
        corrected_rmse_vs_true=corrected_rmse,
    )


def run_calibration_recovery_test(
    seed: int = 0, verbose: bool = True,
) -> tuple[dict[str, RecoveryResult], CrossSensorConsistencyResult]:
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
        Step 3.5 -- cross-sensor consistency: all 3 sensors observe the
        SAME (PAR, temp) condition simultaneously, reusing the SAME
        calibration/covariate models just fitted above. Checks a
        genuinely different question from the rest of Test 1: does
        correction bring several sensors observing the SAME true value
        into agreement with each other, not just recover one sensor's
        own value.
        """
        cross_sensor_result = run_cross_sensor_consistency_test(
            calibration_dir, covariate_models, seed=seed + 4, verbose=verbose,
        )

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

        return {"raw": raw_result, "corrected": corrected_result}, cross_sensor_result


if __name__ == "__main__":
    results, cross_sensor = run_calibration_recovery_test()
    for key, result in results.items():
        print(f"\n{result.label}:")
        print(f"  RMSE vs true VOC (ppm): {result.rmse_vs_true_ppm:.2f}")
        print(f"  R^2 of recovered curve vs true surface: {result.r2_on_dense_grid:.4f}")
        for name, err in result.param_pct_error.items():
            print(f"  {name} recovered={result.recovered_params[name]:.4f}, pct_error={err:.1f}%")

    print(f"\nCross-sensor consistency (PAR={cross_sensor.par}, temp={cross_sensor.temp}, true={cross_sensor.true_ppm:.1f} ppm):")
    print(f"  cross-sensor spread: raw={cross_sensor.raw_cross_sensor_spread:.2f} ppm, corrected={cross_sensor.corrected_cross_sensor_spread:.2f} ppm")
    for sensor_id in cross_sensor.sensor_ids:
        print(
            f"  {sensor_id}: raw RMSE vs true={cross_sensor.raw_rmse_vs_true[sensor_id]:.2f}, "
            f"corrected RMSE vs true={cross_sensor.corrected_rmse_vs_true[sensor_id]:.2f}"
        )
