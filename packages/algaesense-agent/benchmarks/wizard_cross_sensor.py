"""Does calibrating each sensor through the real wizard make three
disagreeing sensors agree -- with each other, and with the truth?

Three synthetic sensors watch the SAME reactor, so at every instant there
is exactly one correct answer. Each carries its own imperfections: a
different zero offset, a different sensitivity, and for one of them a
genuine bend near the top of its range. On top of that sits the
contamination that varies over time rather than between units -- humidity
and temperature dragging the baseline, autocorrelated drift, and
occasional per-sensor electrical glitches.

The question is deliberately split in two, because a calibration curve can
only answer half of it. Inverting a per-sensor response corrects the ways
units differ *about the same gas* -- offset, gain, curvature. It cannot
touch anything that moves over time, which is what the ambient-baseline
correction is for. Reporting one combined number would credit calibration
with work it does not do, so each stage is measured separately:

    raw            -- one nominal datasheet conversion for all three,
                      i.e. what you get by not calibrating at all
    calibrated     -- each sensor through its own wizard-produced curve
    + ambient      -- that, plus the fitted RH/T covariate correction

Runs the ACTUAL wizard -- its session rules, its minimum-levels guard, its
fit and its save path -- not the fitting function underneath it. The
existing jaxsr-calibration benchmark already covers the pipeline; what was
never covered is the thing an operator actually drives.

Self-contained on purpose: it needs one true VOC trace at fixed
conditions, not the PAR-by-temperature surface the other benchmark is
built around, so importing that machinery would drag in far more than it
uses.

Run:  python packages/algaesense-agent/benchmarks/wizard_cross_sensor.py
"""

from __future__ import annotations

import datetime as dt
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from algaesense_agent.dashboard.calibration_session import (
    DEFAULT_LEVELS_PPM,
    MIN_READINGS_PER_LEVEL,
    CalibrationSession,
    LevelRecording,
    assess_quality,
    fit_calibration,
    save_calibration,
)
from jaxsr_calibration.calibration.apply import apply_calibration
from jaxsr_calibration.processing.despike import flag_glitches_across_sensors
from jaxsr_calibration.processing.covariate import (
    fit_covariate_model,
    normalise_to_reference_conditions,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

DURATION_S = 6 * 3600
DT_S = 10.0

BASELINE_DURATION_S = 2 * 3600
"""A separate clean-air run before the experiment, which is what the
ambient-baseline check actually is -- the sensor on zero gas while the
room cycles, so the model learns RH/T coupling and nothing else."""

NOMINAL_B0_MV = 25.0
NOMINAL_B1_MV_PER_PPM = 60.0
"""The datasheet response an operator would fall back on with no
calibration of their own -- the baseline this whole exercise is measured
against."""


@dataclass(frozen=True)
class SensorTruth:
    """What a given unit really does, which no one gets to see directly."""

    sensor_id: str
    b0_mv: float
    b1_mv_per_ppm: float
    b2_mv_per_ppm2: float
    rh_coeff_mv_per_pct: float
    t_coeff_mv_per_c: float
    noise_mv: float


SENSORS = [
    SensorTruth("PID01", b0_mv=20.0, b1_mv_per_ppm=62.0, b2_mv_per_ppm2=0.0,
                rh_coeff_mv_per_pct=0.55, t_coeff_mv_per_c=1.10, noise_mv=0.9),
    SensorTruth("PID02", b0_mv=55.0, b1_mv_per_ppm=45.0, b2_mv_per_ppm2=0.0,
                rh_coeff_mv_per_pct=0.95, t_coeff_mv_per_c=2.30, noise_mv=1.4),
    SensorTruth("PID03", b0_mv=8.0, b1_mv_per_ppm=78.0, b2_mv_per_ppm2=-2.4,
                rh_coeff_mv_per_pct=0.30, t_coeff_mv_per_c=0.70, noise_mv=1.1),
]
"""Three units that disagree in every way a PID can: a 47 mV spread in
zero offset, sensitivities running -25% to +30% either side of the
datasheet nominal, and one that compresses near full scale. PID03's bend
is what makes the wizard's fourth level earn its keep.

Deliberately NOT centred on the nominal response. An earlier draft had
them straddling it so evenly that the uncalibrated conversion came out
accidentally close to correct, leaving calibration almost nothing to fix
-- which flattered the baseline rather than testing anything."""

AR1_RHO = 0.92
GLITCH_RATE_PER_HOUR = 4.0
GLITCH_AMPLITUDE_MV = 60.0


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


def true_voc_trace(t_s: np.ndarray) -> np.ndarray:
    """One true VOC concentration over time, shared by all three sensors.

    A culture relaxing toward its plateau, with two genuine transient
    emissions riding on top. The events matter: they are what a naive
    filter would strip out, and they must survive into the corrected
    signal.
    """
    plateau, tau_s = 3.2, 1.5 * 3600
    voc = plateau * (1.0 - np.exp(-t_s / tau_s))

    for onset_s, height, width_s in ((2.2 * 3600, 1.1, 420.0), (4.6 * 3600, 0.7, 300.0)):
        voc += height * np.exp(-(((t_s - onset_s) / width_s) ** 2))

    return voc


def ambient_conditions(t_s: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Room humidity and temperature over the run.

    Given a wide enough swing on purpose: the covariate fit refuses to run
    below a minimum RH range, on the grounds that a coefficient fitted
    across a narrow band is not worth trusting.
    """
    hours = t_s / 3600.0

    """
    A two-hour cycle, roughly what a room's HVAC does, rather than one slow
    swing across the whole run. It matters for more than realism: the
    covariate fit refuses to run unless its training window spans at least
    20% RH, on the sound grounds that a coefficient fitted across a narrow
    band is not worth trusting -- and a training window taken before the
    first emission event only sees a full cycle if the cycle is short
    enough to fit inside it.
    """
    rh = 52.0 + 18.0 * np.sin(2 * np.pi * hours / 2.0) + rng.normal(0.0, 0.6, size=t_s.size)
    temp = 24.0 + 3.5 * np.sin(2 * np.pi * hours / 2.0 + 0.9) + rng.normal(0.0, 0.15, size=t_s.size)
    return rh, temp


def _ar1_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Drift that remembers where it was, rather than independent jitter --
    the shape real sensor noise actually has."""
    innovations = rng.normal(0.0, sigma * np.sqrt(1 - AR1_RHO**2), size=n)
    out = np.empty(n)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        out[i] = AR1_RHO * out[i - 1] + innovations[i]
    return out


def _glitches(t_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-sensor electrical spikes -- one instrument's own fault, seen by
    no other sensor, unlike the shared transient events in the true
    trace."""
    out = np.zeros(t_s.size)
    n_glitches = rng.poisson(GLITCH_RATE_PER_HOUR * DURATION_S / 3600.0)
    for index in rng.integers(0, t_s.size, size=n_glitches):
        out[index] += rng.choice([-1.0, 1.0]) * GLITCH_AMPLITUDE_MV * rng.uniform(0.5, 1.5)
    return out


def observe(sensor: SensorTruth, true_ppm: np.ndarray, rh: np.ndarray, temp: np.ndarray,
            t_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """What this unit actually outputs, in millivolts."""
    response = (
        sensor.b0_mv
        + sensor.b1_mv_per_ppm * true_ppm
        + sensor.b2_mv_per_ppm2 * true_ppm**2
    )
    ambient = sensor.rh_coeff_mv_per_pct * (rh - 50.0) + sensor.t_coeff_mv_per_c * (temp - 25.0)
    return response + ambient + _ar1_noise(t_s.size, sensor.noise_mv, rng) + _glitches(t_s, rng)


# --------------------------------------------------------------------------
# The wizard, driven exactly as an operator would
# --------------------------------------------------------------------------


def calibrate_through_wizard(sensor: SensorTruth, data_dir: Path, rng: np.random.Generator) -> str:
    """Bench-calibrate one sensor by building the wizard's own session and
    running its own fit and save.

    The bench readings are what this unit would really produce at each
    certified level, including its own bend and a little scatter -- so the
    wizard has to recover the response rather than being handed it.
    """
    session = CalibrationSession(
        reactor_id="R1",
        sensor_id=sensor.sensor_id,
        calibration_run_id=f"bench_{sensor.sensor_id}",
        experiment_id="exp_bench",
        calibration_compound="isobutylene",
        mw_g_mol=56.11,
        response_factor=1.0,
        levels=[LevelRecording(ppm=ppm) for ppm in DEFAULT_LEVELS_PPM],
    )

    for level in session.levels:
        settled_mv = (
            sensor.b0_mv
            + sensor.b1_mv_per_ppm * level.ppm
            + sensor.b2_mv_per_ppm2 * level.ppm**2
        )
        level.readings_mv = list(
            settled_mv + rng.normal(0.0, 0.25, size=MIN_READINGS_PER_LEVEL + 2)
        )

    quality = assess_quality(session)

    """
    The wizard's own recommendation is followed rather than overridden --
    that recommendation firing correctly for PID03 and staying quiet for
    the other two is part of what this benchmark is checking.
    """
    method = "polynomial_deg2" if quality.suggests_quadratic else "wls"
    model = fit_calibration(session, method=method)
    save_calibration(session, model, data_dir)

    print(
        f"  {sensor.sensor_id}: fitted {method:16s} "
        f"b0={model.b0_mv:7.2f} (true {sensor.b0_mv:6.2f})  "
        f"b1={model.b1_mv_per_ppm_asgas:6.2f} (true {sensor.b1_mv_per_ppm:5.2f})  "
        f"curvature share {100 * quality.curvature_share:5.1f}%"
    )
    return session.calibration_run_id


# --------------------------------------------------------------------------
# The three stages
# --------------------------------------------------------------------------


@dataclass
class StageResult:
    name: str
    ppm: dict[str, np.ndarray]
    spread_ppm: float
    """Mean over time of the gap between the highest and lowest sensor --
    do the three agree with each other."""

    rmse_ppm: dict[str, float]
    """Per sensor, against the truth -- do they agree with reality. Kept
    separate because a pipeline could in principle deliver one without the
    other."""


def _summarise(name: str, ppm: dict[str, np.ndarray], true_ppm: np.ndarray) -> StageResult:
    stacked = np.vstack([ppm[s.sensor_id] for s in SENSORS])
    spread = float(np.mean(stacked.max(axis=0) - stacked.min(axis=0)))
    rmse = {
        sid: float(np.sqrt(np.mean((values - true_ppm) ** 2))) for sid, values in ppm.items()
    }
    return StageResult(name=name, ppm=ppm, spread_ppm=spread, rmse_ppm=rmse)


def run(seed: int = 0, verbose: bool = True) -> dict[str, StageResult]:
    rng = np.random.default_rng(seed)
    t_s = np.arange(0.0, DURATION_S, DT_S)
    true_ppm = true_voc_trace(t_s)
    rh, temp = ambient_conditions(t_s, rng)

    timestamps = [
        dt.datetime(2026, 7, 29, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=float(s))
        for s in t_s
    ]

    observed_mv = {s.sensor_id: observe(s, true_ppm, rh, temp, t_s, rng) for s in SENSORS}

    if verbose:
        print("Calibrating each sensor through the real wizard:")

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        run_ids = {s.sensor_id: calibrate_through_wizard(s, data_dir, rng) for s in SENSORS}
        calibration_dir = data_dir / "derived" / "calibrations" / "standard_addition"

        raw_ppm = {
            sid: (mv - NOMINAL_B0_MV) / NOMINAL_B1_MV_PER_PPM for sid, mv in observed_mv.items()
        }

        calibrated_ppm = {}
        for sensor in SENSORS:
            ppm, _, _ = apply_calibration(
                pl.Series(observed_mv[sensor.sensor_id]),
                sensor.sensor_id,
                pl.Series(temp),
                pl.Series(rh),
                run_ids[sensor.sensor_id],
                data_dir=calibration_dir,
            )
            calibrated_ppm[sensor.sensor_id] = np.asarray(ppm, dtype=float)

        """
        The covariate model is fitted on a separate CLEAN-AIR baseline
        recording, not on the experiment itself.

        This is the protocol's own ambient-baseline check, and doing it any
        other way does not work: fitted on experiment data, where VOC is
        rising at the same time as the room cycles, humidity and
        temperature become proxies for the VOC trend, and subtracting the
        fitted baseline strips out real signal. Measured directly here
        before the workflow was corrected -- RMSE went from 0.17 to 1.65
        ppm, an order of magnitude worse than not correcting at all.
        """
        baseline_t_s = np.arange(0.0, BASELINE_DURATION_S, DT_S)
        baseline_rh, baseline_temp = ambient_conditions(baseline_t_s, rng)
        baseline_true = np.zeros_like(baseline_t_s)
        baseline_timestamps = [
            dt.datetime(2026, 7, 28, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=float(s))
            for s in baseline_t_s
        ]

        models = {}
        for sensor in SENSORS:
            baseline_mv = observe(
                sensor, baseline_true, baseline_rh, baseline_temp, baseline_t_s, rng
            )
            baseline_frame = pl.DataFrame(
                {
                    "timestamp": baseline_timestamps,
                    "sensor_id": [sensor.sensor_id] * baseline_t_s.size,
                    "pid_voltage_mv": baseline_mv,
                    "sample_rh_pct": baseline_rh,
                    "sample_t_c": baseline_temp,
                }
            )
            model = fit_covariate_model(
                baseline_frame, pl.Series(np.ones(baseline_t_s.size, dtype=bool))
            )
            if model is not None:
                models[sensor.sensor_id] = model

        experiment_frame = pl.DataFrame(
            {
                "timestamp": timestamps * len(SENSORS),
                "sensor_id": [s.sensor_id for s in SENSORS for _ in t_s],
                "pid_voltage_mv": np.concatenate([observed_mv[s.sensor_id] for s in SENSORS]),
                "sample_rh_pct": np.tile(rh, len(SENSORS)),
                "sample_t_c": np.tile(temp, len(SENSORS)),
            }
        )

        """
        Normalised to the conditions the calibration was recorded at, not
        driven to zero. `apply_covariate_correction` does the latter, which
        is right for inspecting a sensor on its own but wrong here: it
        removes the baseline offset that `apply_calibration` is about to
        remove again.
        """
        corrected_frame = normalise_to_reference_conditions(
            experiment_frame, models, reference_rh_pct=50.0, reference_t_c=25.0
        )
        corrected_column = "pid_voltage_mv_ambient_normalised"

        def _calibrate(source: pl.DataFrame, column: str) -> dict[str, np.ndarray]:
            out = {}
            for sensor in SENSORS:
                sensor_frame = source.filter(pl.col("sensor_id") == sensor.sensor_id)
                ppm, _, _ = apply_calibration(
                    sensor_frame[column],
                    sensor.sensor_id,
                    sensor_frame["sample_t_c"],
                    sensor_frame["sample_rh_pct"],
                    run_ids[sensor.sensor_id],
                    data_dir=calibration_dir,
                )
                out[sensor.sensor_id] = np.asarray(ppm, dtype=float)
            return out

        ambient_ppm = _calibrate(corrected_frame, corrected_column)

        """
        Despiking last, and only possible at all because there are three
        sensors: an excursion every sensor registers at the same instant is
        in the gas and must survive, while one no other sensor corroborates
        is that instrument's own electrical fault. With a single sensor the
        two are indistinguishable and the function refuses to guess.
        """
        despiked_frame = flag_glitches_across_sensors(
            corrected_frame, value_column=corrected_column
        )
        despiked_ppm = _calibrate(despiked_frame, f"{corrected_column}_despiked")

    stages = {
        "raw": _summarise("Uncalibrated (nominal response)", raw_ppm, true_ppm),
        "calibrated": _summarise("Wizard calibration", calibrated_ppm, true_ppm),
        "ambient": _summarise("+ ambient correction", ambient_ppm, true_ppm),
        "despiked": _summarise("+ despiking", despiked_ppm, true_ppm),
    }

    if verbose:
        print("\nCross-sensor spread (mean gap between highest and lowest sensor):")
        for stage in stages.values():
            print(f"  {stage.name:45s} {stage.spread_ppm:7.3f} ppm")
        print("\nAccuracy against the true VOC (RMSE per sensor):")
        for stage in stages.values():
            per_sensor = "  ".join(f"{sid} {v:6.3f}" for sid, v in stage.rmse_ppm.items())
            print(f"  {stage.name:45s} {per_sensor}")

    _plot(t_s, true_ppm, stages, seed)
    return stages


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------


SENSOR_COLORS = {"PID01": "#1f77b4", "PID02": "#2ca02c", "PID03": "#d62728"}
STAGE_ORDER = ["raw", "calibrated", "ambient", "despiked"]


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def _plot(t_s: np.ndarray, true_ppm: np.ndarray, stages: dict[str, StageResult], seed: int) -> None:
    _apply_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hours = t_s / 3600.0

    """
    All three panels share one y-range, chosen from the widest stage. The
    whole claim is that the traces converge, and that is only readable if
    the axis does not quietly rescale to flatter each panel in turn.
    """
    lo = min(float(np.min(v)) for stage in stages.values() for v in stage.ppm.values())
    hi = max(float(np.max(v)) for stage in stages.values() for v in stage.ppm.values())
    pad = 0.05 * (hi - lo)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5), sharey=True)
    for ax, key in zip(axes, STAGE_ORDER):
        stage = stages[key]
        for sensor in SENSORS:
            ax.plot(
                hours,
                stage.ppm[sensor.sensor_id],
                color=SENSOR_COLORS[sensor.sensor_id],
                linewidth=1.1,
                alpha=0.85,
                label=sensor.sensor_id,
            )
        ax.plot(hours, true_ppm, color="black", linewidth=2.0, linestyle="--", label="True VOC")
        ax.set_title(f"{stage.name}\nspread {stage.spread_ppm:.2f} ppm")
        ax.set_xlabel("Hours since start")
        ax.set_ylim(lo - pad, hi + pad)

    axes[0].set_ylabel("VOC (ppm)")
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Three sensors watching one reactor: what each correction stage actually fixes",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "wizard_cross_sensor_traces.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax_spread, ax_rmse) = plt.subplots(1, 2, figsize=(13, 5))

    positions = np.arange(len(STAGE_ORDER))
    ax_spread.bar(
        positions,
        [stages[k].spread_ppm for k in STAGE_ORDER],
        color=["#8c8c8c", "#1f77b4", "#ff7f0e", "#9467bd"],
        width=0.6,
    )
    for x, key in zip(positions, STAGE_ORDER):
        ax_spread.text(
            x, stages[key].spread_ppm, f" {stages[key].spread_ppm:.2f}",
            ha="center", va="bottom", fontweight="bold",
        )
    ax_spread.set_xticks(positions)
    ax_spread.set_xticklabels(["Uncalibrated", "Calibrated", "+ ambient", "+ despiked"])
    ax_spread.set_ylabel("Mean sensor-to-sensor gap (ppm)")
    ax_spread.set_title("Do the sensors agree with each other?")

    width = 0.25
    for offset, sensor in zip((-width, 0.0, width), SENSORS):
        ax_rmse.bar(
            positions + offset,
            [stages[k].rmse_ppm[sensor.sensor_id] for k in STAGE_ORDER],
            width=width,
            color=SENSOR_COLORS[sensor.sensor_id],
            label=sensor.sensor_id,
        )
    ax_rmse.set_xticks(positions)
    ax_rmse.set_xticklabels(["Uncalibrated", "Calibrated", "+ ambient", "+ despiked"])
    ax_rmse.set_ylabel("RMSE against true VOC (ppm)")
    ax_rmse.set_title("Do they agree with the truth?")
    ax_rmse.legend()

    fig.suptitle(
        "Agreement and accuracy are measured separately, because a pipeline can deliver one without the other",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "wizard_cross_sensor_summary.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots written to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
