"""Part 1 -- can the pre-calibration package strip the noise off three
very different-LOOKING raw sensor traces and recover ONE true VOC signal?

Three sensors watch the same reactor through the same week-long run at a
fixed (PAR, temp). Their raw traces look nothing like each other -- one
flat, one haphazardly rising, one oscillating on a daily cycle -- purely
because each sits in its own ambient micro-environment (see
ground_truth.ambient_micro_environment). The true VOC they are all
observing is identical by construction, so every difference between their
raw traces is contamination, and what survives correction is the measure.

Runs the REAL pipeline: fit_sensitivity_per_sensor / persist_calibration /
apply_calibration and fit_covariate_model / apply_covariate_correction.

Two cases are run:

  "correctable"   -- the shape differences are driven by each sensor's own
                     MEASURED RH/T, which is exactly the contamination
                     class the covariate model characterizes. These should
                     converge.

  "uncorrectable" -- the same three shapes injected straight onto the
                     voltage with no covariate signature at all. Nothing
                     in this pipeline models arbitrary sensor drift, so
                     these should NOT converge. It is run as an explicit
                     negative control: the point is to show where the
                     pipeline's competence ends rather than assert it.

Deliberately does NOT apply subtract_common_mode. That estimator subtracts
the across-sensor median, which is only meaningful when the shared true
value is zero (a synchronized blank/zero check). Here every sensor is
watching the same NON-zero VOC curve, so subtracting the median would
remove the signal itself, not the noise.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from ground_truth import (
    AmbientCovariateTruth,
    NoiseConfig,
    SensorCalibrationTruth,
    generate_ambient_blank_recording,
    generate_calibration_recording,
    generate_week_long_sensor_recording,
    true_tau_hours,
    true_voc_ppm,
)

from jaxsr_calibration.calibration.apply import apply_calibration, persist_calibration
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from jaxsr_calibration.processing.covariate import apply_covariate_correction, fit_covariate_model


SENSOR_IDS = ["PID01", "PID02", "PID03"]

SENSOR_ENVIRONMENTS = {
    "PID01": "stable",
    "PID02": "warming",
    "PID03": "diurnal",
}
"""Which micro-environment each sensor sits in -- this is what makes the
three raw traces look qualitatively different despite an identical truth."""

TRUE_CALIBRATION = {
    "PID01": SensorCalibrationTruth(b0_mv=20.0, b1_mv_per_ppm=0.60),
    "PID02": SensorCalibrationTruth(b0_mv=17.0, b1_mv_per_ppm=0.55),
    "PID03": SensorCalibrationTruth(b0_mv=23.0, b1_mv_per_ppm=0.65),
}

FIXED_PAR = 250.0
FIXED_TEMP = 30.0
"""One representative mid-domain operating point -- bright enough to be
well clear of the dark baseline, below the photoinhibition threshold, and
warm enough that tau is a comfortable ~15h."""

DURATION_S = 7 * 24 * 3600
DT_S = 300.0
CALIBRATION_RUN_ID = "bench_sensor_consistency"
SPIKE_LEVELS = [0.0, 100.0, 250.0, 500.0, 800.0]
"""Standard-addition levels spanning the plateau range this run reaches --
a calibration only anchored near 0 would be extrapolating everywhere."""


@dataclass
class SensorTrace:
    """One sensor's before/after view of the same underlying truth."""

    sensor_id: str
    environment: str
    raw_ppm: list[float]
    corrected_ppm: list[float]
    raw_rmse_vs_true: float
    corrected_rmse_vs_true: float
    raw_median_abs_err: float = 0.0
    corrected_median_abs_err: float = 0.0
    """
    Median absolute error is carried alongside RMSE because once sparse
    spikes are present the two answer different questions: RMSE is
    dominated by the handful of contaminated samples, while the median
    describes the bulk of the trace an operator actually reads. Reporting
    only one would misrepresent the result in whichever direction that one
    happened to favour.
    """


@dataclass
class ConsistencyResult:
    """One case (correctable or uncorrectable) across all three sensors."""

    case: str
    elapsed_hours: list[float]
    true_ppm: list[float]
    traces: dict[str, SensorTrace] = field(default_factory=dict)
    raw_spread_ppm: float = 0.0
    corrected_spread_ppm: float = 0.0

    @property
    def spread_reduction_pct(self) -> float:
        if self.raw_spread_ppm == 0.0:
            return 0.0
        return 100.0 * (1.0 - self.corrected_spread_ppm / self.raw_spread_ppm)


def _fit_pipeline_models(seed: int, calibration_dir: Path):
    """Fit the two real models the correction path needs: each sensor's
    standard-addition sensitivity line, and each sensor's ambient
    covariate model. Both come from their own dedicated recordings, the
    way a real calibration/diagnostic run would."""

    cal_df = generate_calibration_recording(SENSOR_IDS, TRUE_CALIBRATION, SPIKE_LEVELS, seed=seed)
    sensitivity_models = fit_sensitivity_per_sensor(cal_df)
    persist_calibration(sensitivity_models, CALIBRATION_RUN_ID, "cal_experiment", calibration_dir)

    """
    The blank recording has to span the RH/T ranges the sensors actually
    experience during the run -- the warming sensor climbs to ~33C and the
    diurnal one swings +/-4C, both inside generate_ambient_blank_recording's
    default (22-34C, 30-80% RH). A blank narrower than the run would leave
    the covariate model extrapolating.
    """
    ambient_df = generate_ambient_blank_recording(SENSOR_IDS, AmbientCovariateTruth(), seed=seed + 1)
    covariate_models = {}
    for sensor_id, sensor_df in ambient_df.partition_by("sensor_id", as_dict=True).items():
        sensor_id = sensor_id[0] if isinstance(sensor_id, tuple) else sensor_id
        mask = pl.Series([True] * sensor_df.height)
        model = fit_covariate_model(sensor_df, mask, method="ols")
        if model is not None:
            covariate_models[sensor_id] = model

    return sensitivity_models, covariate_models


def run_consistency_case(
    case: str,
    seed: int = 0,
    verbose: bool = False,
) -> ConsistencyResult:
    """Run one week-long 3-sensor recording through the real pipeline.

    `case` is "correctable" (shape differences driven by measured RH/T)
    or "uncorrectable" (same shapes as unexplained drift).
    """
    inject_drift = case == "uncorrectable"

    with tempfile.TemporaryDirectory() as tmp:
        calibration_dir = Path(tmp) / "derived" / "calibrations" / "standard_addition"
        _, covariate_models = _fit_pipeline_models(seed, calibration_dir)

        recording = generate_week_long_sensor_recording(
            experiment_id="exp_sensor_consistency",
            reactor_id="R01",
            sensor_ids=SENSOR_IDS,
            sensor_environments=SENSOR_ENVIRONMENTS,
            calibration_truth=TRUE_CALIBRATION,
            noise=NoiseConfig(),
            par=FIXED_PAR,
            temp=FIXED_TEMP,
            duration_s=DURATION_S,
            dt_s=DT_S,
            inject_uncorrectable_drift=inject_drift,
            seed=seed + 2,
        )

        by_sensor = recording.partition_by("sensor_id", as_dict=True)
        traces: dict[str, SensorTrace] = {}
        raw_matrix = []
        corrected_matrix = []
        true_ppm = None
        elapsed_hours = None

        for sensor_id in SENSOR_IDS:
            sensor_df = by_sensor[(sensor_id,)] if (sensor_id,) in by_sensor else by_sensor[sensor_id]
            true_ppm = sensor_df["true_voc_ppm"].to_numpy()
            elapsed_hours = np.arange(sensor_df.height) * DT_S / 3600.0

            """
            "Before" deliberately still goes through apply_calibration --
            otherwise it would be millivolts and could not share a y-axis
            with the corrected ppm trace. This is the honest comparison:
            what the sensor's own nominal calibration alone tells you,
            versus what it tells you after the ambient correction too.
            """
            raw_series, _, _ = apply_calibration(
                sensor_df["pid_voltage_mv"],
                sensor_id,
                sensor_df["sample_t_c"],
                sensor_df["sample_rh_pct"],
                CALIBRATION_RUN_ID,
                data_dir=calibration_dir,
            )
            raw_arr = raw_series.to_numpy()

            corrected_df = apply_covariate_correction(sensor_df, covariate_models)
            corrected_series, _, _ = apply_calibration(
                corrected_df["pid_voltage_mv_covariate_corrected"],
                sensor_id,
                corrected_df["sample_t_c"],
                corrected_df["sample_rh_pct"],
                CALIBRATION_RUN_ID,
                data_dir=calibration_dir,
            )
            corrected_arr = corrected_series.to_numpy()

            raw_matrix.append(raw_arr)
            corrected_matrix.append(corrected_arr)

            traces[sensor_id] = SensorTrace(
                sensor_id=sensor_id,
                environment=SENSOR_ENVIRONMENTS[sensor_id],
                raw_ppm=[float(v) for v in raw_arr],
                corrected_ppm=[float(v) for v in corrected_arr],
                raw_rmse_vs_true=float(np.sqrt(np.mean((raw_arr - true_ppm) ** 2))),
                corrected_rmse_vs_true=float(np.sqrt(np.mean((corrected_arr - true_ppm) ** 2))),
                raw_median_abs_err=float(np.median(np.abs(raw_arr - true_ppm))),
                corrected_median_abs_err=float(np.median(np.abs(corrected_arr - true_ppm))),
            )

        """
        Cross-sensor spread: the std ACROSS the three sensors at each
        instant, averaged over the run. This is the "do they agree with
        each other" number, deliberately separate from RMSE-vs-true ("are
        they each right") -- a pipeline could in principle make three
        sensors agree on a value that is wrong, and those two failures
        should not be collapsed into one statistic.
        """
        raw_spread = float(np.mean(np.std(np.vstack(raw_matrix), axis=0)))
        corrected_spread = float(np.mean(np.std(np.vstack(corrected_matrix), axis=0)))

    result = ConsistencyResult(
        case=case,
        elapsed_hours=[float(h) for h in elapsed_hours],
        true_ppm=[float(v) for v in true_ppm],
        traces=traces,
        raw_spread_ppm=raw_spread,
        corrected_spread_ppm=corrected_spread,
    )

    if verbose:
        print(f"  [{case}] cross-sensor spread: raw={raw_spread:.2f} ppm -> corrected={corrected_spread:.2f} ppm")
        for sensor_id, trace in result.traces.items():
            print(
                f"    {sensor_id} ({trace.environment}): "
                f"RMSE {trace.raw_rmse_vs_true:6.2f} -> {trace.corrected_rmse_vs_true:6.2f} ppm | "
                f"median|err| {trace.raw_median_abs_err:6.2f} -> {trace.corrected_median_abs_err:5.2f} ppm"
            )

    return result


def run_sensor_consistency_test(seed: int = 0, verbose: bool = False) -> dict[str, ConsistencyResult]:
    """Both cases, returned keyed by case name."""
    if verbose:
        print(
            f"  true signal: plateau={float(true_voc_ppm(FIXED_PAR, FIXED_TEMP)):.1f} ppm, "
            f"tau={float(true_tau_hours(FIXED_PAR, FIXED_TEMP)):.2f} h "
            f"at PAR={FIXED_PAR:.0f}, temp={FIXED_TEMP:.0f}C"
        )
    return {case: run_consistency_case(case, seed=seed, verbose=verbose) for case in ("correctable", "uncorrectable")}


if __name__ == "__main__":
    run_sensor_consistency_test(verbose=True)
