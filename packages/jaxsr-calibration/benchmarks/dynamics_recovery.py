"""Test 3 -- ground truth #1: given ONE static PAR level held for a
whole, long-running experiment, does discover_led_response_dynamics
recover the true relaxation dynamics governing how VOC settles toward
its steady state?

Distinct from Test 1/2 (ground truth #2: how VOC varies ACROSS many
different static settings sampled as one-off points). Here, several
SEPARATE experiments are each run at their OWN fixed static PAR level
(see PAR_LEVELS below), for a long duration (>= 1 week), and the
question is purely about the WITHIN-experiment time dynamics:
dVOC/dt = (1/tau) * (true_voc_ppm(par, temp) - VOC(t)) -- see
ground_truth.py's dynamic-ground-truth section for why this is tied to
the same static surface as its steady-state target.

**Redesigned 2026-07-23, at the user's explicit request**: the earlier
version drove PAR with a single continuously-varying sinusoid within
one experiment. This version uses several fixed-PAR step experiments
instead, deliberately NOT pooled together before fitting -- each
experiment is analyzed entirely on its own (the user's own reasoning:
"there may be interesting dynamics that happen for a given static
PAR"), rather than combined into one joint PAR-dependence fit. One real
consequence of not pooling: within any SINGLE experiment, PAR is
literally constant, so `discover_led_response_dynamics` (which always
declares reactor_par_umol_m2_s as a state variable) has no within-
experiment PAR variation to learn a genuine PAR-coefficient from --
correctly, the discovered equation should end up dominated by a plain
self-decay term in ppm_asgas, with PAR terms carrying little-to-no real
information (see the report's own honest accounting of this).
Temperature stays fixed across all PAR levels for now.

Runs the REAL, public `discover_led_response_dynamics` for BOTH the raw
path (no `ambient_baseline_run_id`) and the ambient-corrected path
(passing a real, persisted `ambient_baseline_run_id`) -- this parameter
was added directly to the production tool after this benchmark first
confirmed the correction helps (see CLAUDE.md's dev log). A parallel
internal-pieces variant (same load_timeseries_for_jaxsr ->
jaxsr.discover_dynamics calls the public function makes internally) is
still used for SCORING and for the predicted-trajectory overlay plot,
since the public function's DynamicsDiscoveryResult deliberately strips
the live model objects both of those need; its equation is cross-
checked against the real function's own output to confirm the two stay
in sync (a soft note, not a hard assert -- see the comment further down).
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

"""
Makes this file's own sibling import below (ground_truth) resolve regardless
of how this script is invoked -- see the identical note in run_all.py.
"""

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from jaxsr_calibration.processing.covariate import (
    apply_covariate_correction,
    fit_covariate_model,
    persist_covariate_models,
)
from jaxsr_calibration.processing.features import load_timeseries_for_jaxsr


REACTOR_ID = "R01"
SENSOR_ID = "PID01"
CALIBRATION_RUN_ID = "benchmark_dynamics_cal_01"
AMBIENT_BASELINE_RUN_ID = "benchmark_dynamics_ambient_01"
TRUE_CALIBRATION = SensorCalibrationTruth(b0_mv=20.0, b1_mv_per_ppm=0.60)
TEMP = 30.0

PAR_LEVELS = [50.0, 175.0, 300.0, 450.0]
"""Four fixed PAR levels spanning the domain -- 300 and 450 straddle
PHOTO_THRESHOLD_PAR=380, so photoinhibition's effect on the steady-state
target shows up in at least one level. Each becomes its own completely
independent, long-running static-PAR experiment."""

DURATION_S = 7 * 24 * 3600  # >= 1 week, per the user's explicit request
DT_S = 10.0
"""10s sampling -- 12 samples per relaxation time constant (tau=120s),
comfortably enough to resolve the relaxation curve without generating/
fitting an unwieldy ~604,800-row-per-experiment dataset at full 1s
resolution (a week at 1s would be 60x more data for no real gain in
identifying a tau=120s decay)."""
FIT_DECIMATION = 6
"""Only every FIT_DECIMATION-th generated row is written to Parquet and
fed into discover_led_response_dynamics's fit -- see the comment at its
call site for why full-week-at-10s resolution isn't needed there."""
PREDICT_DT_S = 300.0
"""Step size for integrating the DISCOVERED equation forward for the
predicted-trajectory overlay plot -- see _simulate_predicted_trajectory's
docstring for why this is decoupled from DT_S."""


def true_derivative(voc: np.ndarray, par: np.ndarray) -> np.ndarray:
    """The exact right-hand side of the ground-truth ODE, evaluated at
    an arbitrary (voc, par) state -- not just along the one trajectory
    actually simulated, so this can score a model against a held-out
    grid the same way Test 1 does."""
    return (1.0 / DYNAMIC_RELAXATION_TAU_S) * (true_voc_ppm(par, TEMP) - voc)


@dataclass
class DynamicsRecoveryResult:
    label: str
    par_level: float
    selected_features: list[str]
    equation: str
    rmse_vs_true_derivative: float
    r2_vs_true_derivative: float
    diverged: bool


@dataclass
class ParLevelTrajectories:
    """One static-PAR experiment's true trajectory plus each mode's
    (raw/corrected) discovered-equation-predicted trajectory, integrated
    forward from the same initial condition -- what the overlay plot
    actually shows."""

    par_level: float
    t: np.ndarray
    true_voc_values: np.ndarray
    t_predicted: np.ndarray
    predicted_voc_values: dict[str, np.ndarray]


@dataclass
class DynamicsRecoveryRun:
    per_level_results: dict[float, dict[str, DynamicsRecoveryResult]]
    trajectories: dict[float, ParLevelTrajectories]


def _score_model_at_fixed_par(model, state_names: list[str], par_level: float) -> tuple[float, float]:
    """A 1D held-out VOC grid at THIS experiment's own fixed PAR --
    unlike the earlier sinusoid design, this single experiment never saw
    any other PAR value, so scoring it against a full 2D (VOC, PAR) grid
    would be checking something the fit was never in a position to
    learn. Mirrors Test 1's dense-grid RMSE/R^2 approach, just 1D."""
    voc_grid = np.linspace(0.0, 900.0, 200)
    par_idx = state_names.index("reactor_par_umol_m2_s")
    voc_idx = state_names.index("ppm_asgas")
    X_test = np.zeros((voc_grid.size, 2))
    X_test[:, voc_idx] = voc_grid
    X_test[:, par_idx] = par_level

    true_rate = true_derivative(voc_grid, np.full_like(voc_grid, par_level))
    predicted_rate = np.asarray(model.predict(X_test))

    rmse = float(np.sqrt(np.mean((predicted_rate - true_rate) ** 2)))
    ss_res = float(np.sum((true_rate - predicted_rate) ** 2))
    ss_tot = float(np.sum((true_rate - np.mean(true_rate)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, r2


DIVERGENCE_BOUND_PPM = 1.0e4
"""If a discovered equation's forward integration ever exceeds this
(roughly 10x any physically plausible true VOC value in this
benchmark), treat it as diverged rather than let it run away to
astronomical values -- a real, honest possibility for a SINDy-style
discovered equation: a term that fits the local derivative reasonably
well can still have the WRONG sign on a higher-order coefficient, which
is invisible in a single-step RMSE but compounds into runaway growth
over thousands of integration steps. See the report's own discussion of
this."""


def _simulate_predicted_trajectory(
    model, state_names: list[str], par_level: float, duration_s: float, voc0: float = 0.0,
    predict_dt_s: float = PREDICT_DT_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler integration of the DISCOVERED equation (not the
    true one) at this experiment's own fixed PAR, from the same initial
    condition as the true trajectory -- what gets overlaid against the
    true VOC(t) in the comparison plot. Uses its OWN, much coarser time
    step than the true trajectory's DT_S -- each step calls the fitted
    model's own `.predict()`, and jaxsr/JAX's per-call dispatch overhead
    made looping this at the full week-at-10s resolution (60,480 calls)
    the actual runtime bottleneck of this benchmark (confirmed directly:
    decimating the FIT input alone barely changed wall time; this loop
    was the real cost). A 5-minute predicted-trajectory step is still
    visually indistinguishable from the true curve at the scale a
    week-long plot renders at.

    If the integration diverges (a real, observed failure mode -- a
    positive-signed higher-order term makes the discovered ODE
    numerically unstable even when it locally fits dVOC/dt reasonably
    well), the remainder is filled with NaN rather than astronomical
    values, so matplotlib simply stops drawing that line at the point
    of divergence instead of collapsing the whole chart's y-axis scale.
    """
    par_idx = state_names.index("reactor_par_umol_m2_s")
    voc_idx = state_names.index("ppm_asgas")
    n = int(duration_s / predict_dt_s)
    t_predicted = np.arange(n) * predict_dt_s
    voc = np.empty(n)
    voc[0] = voc0
    x_row = np.zeros((1, 2))
    x_row[0, par_idx] = par_level
    diverged_at = None
    for i in range(1, n):
        x_row[0, voc_idx] = voc[i - 1]
        rate = float(np.asarray(model.predict(x_row))[0])
        voc[i] = voc[i - 1] + predict_dt_s * rate
        if not np.isfinite(voc[i]) or abs(voc[i]) > DIVERGENCE_BOUND_PPM:
            diverged_at = i
            break
    if diverged_at is not None:
        voc[diverged_at:] = np.nan
    return t_predicted, voc


def _discover_dynamics_from_readings(readings: pl.DataFrame, max_terms: int = 5, strategy: str = "exhaustive"):
    """Mirrors discover_led_response_dynamics's own internal steps
    exactly (load_timeseries_for_jaxsr -> jaxsr.discover_dynamics), just
    called directly on an in-memory DataFrame so the live model objects
    are available for scoring/simulation -- the public function
    deliberately strips those (see its DynamicsDiscoveryResult
    docstring), same reasoning Test 2 already used for its own internal
    scoring fit. `strategy` defaults to "exhaustive" to match the
    production function's own default (see its docstring for why) --
    keeping this in sync matters since run_dynamics_recovery_test
    cross-checks this internal-pieces result's equation against the
    real function's own output."""
    X, t, state_names = load_timeseries_for_jaxsr(readings, state_columns=["ppm_asgas", "reactor_par_umol_m2_s"])
    result = jaxsr.discover_dynamics(X, t, state_names=state_names, max_terms=max_terms, strategy=strategy)
    return result, state_names


def _run_one_static_par_experiment(
    par_level: float,
    seed: int,
    data_dir: Path,
    calibration_dir: Path,
    covariate_model,
    verbose: bool,
) -> tuple[dict[str, DynamicsRecoveryResult], ParLevelTrajectories]:
    experiment_id = f"exp_dynamics_par_{par_level:.0f}"

    t, par_values, true_voc_values = simulate_true_dynamic_trajectory(
        lambda _t, p=par_level: p, TEMP, DURATION_S, dt_s=DT_S,
    )

    ambient_truth = AmbientCovariateTruth()
    noise = NoiseConfig(ambient=ambient_truth)
    readings = generate_dynamic_experiment_recording(
        experiment_id, REACTOR_ID, SENSOR_ID, par_values, TEMP, true_voc_values,
        TRUE_CALIBRATION, noise, dt_s=DT_S, seed=seed + int(par_level),
    )

    """
    Fitting is done on a DECIMATED subset of the full week's readings
    (every FIT_DECIMATION-th row -- still ~2 samples per relaxation time
    constant at DT_S=10s*FIT_DECIMATION=60s) -- a full week at 10s
    resolution is ~60,480 rows, and jaxsr's exhaustive strategy refits
    several candidate term-subsets, so fit cost scales with row count for
    no real gain here (the whole point of >=1 week is realism/steady-
    state settling, not that discover_dynamics needs 60k samples to
    resolve a tau=120s decay). The TRUE and PREDICTED trajectories
    plotted later still use the full DT_S=10s resolution `t` array,
    independent of this decimation.
    """
    raw_dir = data_dir / "raw" / "experiments" / experiment_id / f"sensor_id={SENSOR_ID}"
    raw_dir.mkdir(parents=True)
    readings[::FIT_DECIMATION].write_parquet(raw_dir / "hour=0.parquet")

    real_raw_result = discover_led_response_dynamics(
        experiment_id, REACTOR_ID, SENSOR_ID, CALIBRATION_RUN_ID, data_dir=data_dir, max_terms=5
    )
    real_corrected_result = discover_led_response_dynamics(
        experiment_id, REACTOR_ID, SENSOR_ID, CALIBRATION_RUN_ID, data_dir=data_dir, max_terms=5,
        ambient_baseline_run_id=AMBIENT_BASELINE_RUN_ID,
    )
    if verbose:
        print(f"  [PAR={par_level:.0f}, raw] equation: {real_raw_result.equations.get('ppm_asgas')}")
        print(f"  [PAR={par_level:.0f}, corrected] equation: {real_corrected_result.equations.get('ppm_asgas')}")

    real_readings = load_raw_voc_readings(data_dir, experiment_id).filter(
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

    """
    NOT a hard assert -- jaxsr.SymbolicRegressor.fit() is independently
    non-deterministic even given identical input data (see Test 2's own
    finding), so two separate fit() calls on the same X/t can
    legitimately land on different equations. Both code paths (real
    function, internal-pieces scoring/simulation variant) are still the
    exact same load_timeseries_for_jaxsr -> jaxsr.discover_dynamics
    sequence -- only which of jaxsr's own non-deterministic fit outcomes
    gets drawn can differ.
    """
    if verbose and corrected_result.equations.get("ppm_asgas") != real_corrected_result.equations.get("ppm_asgas"):
        print(
            f"  [note, PAR={par_level:.0f}] internal-pieces scoring variant landed on a different "
            "equation than the real function's own call, on identical data -- expected "
            "occasionally, per jaxsr's documented fit() non-determinism, not a bug in either path."
        )

    results: dict[str, DynamicsRecoveryResult] = {}
    predicted_voc_values: dict[str, np.ndarray] = {}
    t_predicted = None
    for label, result, state_names in [
        ("raw", raw_result, raw_state_names),
        ("corrected", corrected_result, corrected_state_names),
    ]:
        model = result.models["ppm_asgas"]
        rmse, r2 = _score_model_at_fixed_par(model, state_names, par_level)
        t_predicted, predicted_voc_values[label] = _simulate_predicted_trajectory(
            model, state_names, par_level, DURATION_S, voc0=0.0,
        )
        results[label] = DynamicsRecoveryResult(
            label=label,
            par_level=par_level,
            selected_features=list(model.selected_features_),
            equation=result.equations.get("ppm_asgas", ""),
            rmse_vs_true_derivative=rmse,
            r2_vs_true_derivative=r2,
            diverged=bool(np.any(np.isnan(predicted_voc_values[label]))),
        )
        if verbose:
            print(f"  [PAR={par_level:.0f}, {label}] RMSE vs true derivative (ppm/s): {rmse:.4f}, R^2: {r2:.4f}")

    trajectories = ParLevelTrajectories(
        par_level=par_level, t=t, true_voc_values=true_voc_values,
        t_predicted=t_predicted, predicted_voc_values=predicted_voc_values,
    )
    return results, trajectories


def run_dynamics_recovery_test(seed: int = 0, verbose: bool = True) -> DynamicsRecoveryRun:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        calibration_dir = data_dir / "derived" / "calibrations" / "standard_addition"
        calibration_dir.mkdir(parents=True)

        """
        Calibration and ambient-baseline correction are built ONCE per
        seed and reused across every PAR-level experiment -- realistic
        (one sensor calibration serves a whole campaign of experiments),
        and keeps this shared setup from itself being a source of
        cross-level variation.
        """
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
        persist_covariate_models(
            {SENSOR_ID: covariate_model},
            AMBIENT_BASELINE_RUN_ID,
            data_dir / "derived" / "diagnostics" / "ambient_baseline",
        )

        per_level_results: dict[float, dict[str, DynamicsRecoveryResult]] = {}
        trajectories: dict[float, ParLevelTrajectories] = {}
        for par_level in PAR_LEVELS:
            if verbose:
                print(f" PAR level {par_level:.0f} umol/m^2/s:")
            results, traj = _run_one_static_par_experiment(
                par_level, seed, data_dir, calibration_dir, covariate_model, verbose,
            )
            per_level_results[par_level] = results
            trajectories[par_level] = traj

        return DynamicsRecoveryRun(per_level_results=per_level_results, trajectories=trajectories)


if __name__ == "__main__":
    run_dynamics_recovery_test()
