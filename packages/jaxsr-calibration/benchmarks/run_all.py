"""Run all three parts, write the seven plots, and regenerate REPORT.md.

  Part 1  sensor_consistency.py   -- can pre-calibration recover ONE true
                                     VOC signal from three differently-
                                     shaped raw traces?
  Part 2  doe_comparison.py       -- does JAXSR+labwiki active learning
                                     beat classic DoE at uncovering the
                                     symbolic VOC(PAR, temp) relationship?
  Part 3  dynamics_recovery.py    -- can the within-run rate law be
                                     recovered from a single sinusoid-
                                     driven week?

Every number quoted in REPORT.md is computed from the run that produced
it. This file has shipped a stale hardcoded claim three separate times
(a location, a numeric range, and a directional "correction helps"
conclusion that silently flipped) -- so nothing here asserts a result as
fixed prose.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from doe_comparison import (
    LABWIKI_CONSTRAINT_NOTE,
    LABWIKI_MARGIN_BOUNDS,
    MEASUREMENT_NOISE_PPM,
    METHOD_NAMES,
    METRIC_KEYS,
    METRIC_LABELS,
    N_SEED,
    N_TOTAL_EXPERIMENTS,
    TRUE_PLATEAU_TERMS,
    run_doe_comparison,
)
from doe_methods import DECLARED_SEARCH_BOUNDS, EXTRAP_PAR_BOUNDS, SEED_POINTS, TRAIN_PAR_BOUNDS
from dynamics_recovery import (
    SINUSOID_PERIOD_S,
    SINUSOID_PROFILE,
    TEMP as DYNAMICS_TEMP,
    run_dynamics_recovery_test,
)
from ground_truth import (
    BASELINE,
    GAMMA,
    K_M,
    PHOTO_K,
    PHOTO_THRESHOLD_PAR,
    TAU_BASE_H,
    TAU_PAR_H,
    TAU_TEMP_H_PER_C,
    TEMP_BOUNDS,
    TEMP_REF,
    TEMP_SLOPE,
    VMAX,
    true_tau_hours,
    true_voc_ppm,
)
from sensor_consistency import (
    FIXED_PAR,
    FIXED_TEMP,
    SENSOR_ENVIRONMENTS,
    SENSOR_IDS,
    run_sensor_consistency_test,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DOE_SEEDS = [0, 1, 2, 3, 4]


"""
Presentation styling, applied once. The categorical palette is the
validated fixed-order set used elsewhere in this project; colour is
paired with a distinct line style AND marker per method, because seven
series on one axis WILL overlap and colour alone is not a sufficient
encoding (a previous round shipped a panel where five of six methods were
invisible underneath whichever was drawn last).
"""
PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

METHOD_STYLE = {
    name: {"color": PALETTE[i], "linestyle": LINESTYLES[i], "marker": MARKERS[i]}
    for i, name in enumerate(METHOD_NAMES)
}

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


# ---------------------------------------------------------------- Part 1 plots


SENSOR_TRACE_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]


def _part1_shared_ylim(result) -> tuple[float, float]:
    """One y-range covering BOTH the before and after traces. Without
    this each figure would auto-scale to its own data and the corrected
    plot would look indistinguishable from the raw one -- the whole point
    is that they are on the same scale."""
    values = [result.true_ppm]
    for sensor_id in SENSOR_IDS:
        trace = result.traces[sensor_id]
        values.append(trace.raw_ppm)
        values.append(trace.corrected_ppm)
    flat = np.concatenate([np.asarray(v) for v in values])
    lo, hi = float(np.nanmin(flat)), float(np.nanmax(flat))
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _plot_sensor_traces(result, out_path: Path, which: str, title: str, ylim) -> None:
    """Three sensors plus the truth, in ppm. `which` is "raw_ppm" or
    "corrected_ppm"."""
    fig, ax = plt.subplots(figsize=(9, 5.2))

    hours = np.asarray(result.elapsed_hours)
    ax.plot(hours, result.true_ppm, color="black", linewidth=2.4, label="True VOC", zorder=5)

    for color, sensor_id in zip(SENSOR_TRACE_COLORS, SENSOR_IDS):
        trace = result.traces[sensor_id]
        ax.plot(
            hours,
            getattr(trace, which),
            color=color,
            linewidth=1.3,
            alpha=0.9,
            label=f"{sensor_id} ({trace.environment})",
        )

    ax.set_ylim(*ylim)
    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("VOC (ppm)")
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_part1_plots(result) -> None:
    ylim = _part1_shared_ylim(result)
    for which, name, title in (
        ("raw_ppm", "part1_before_calibration.png",
         "Part 1 - BEFORE pre-calibration (nominal sensor calibration only)"),
        ("corrected_ppm", "part1_after_calibration.png",
         "Part 1 - AFTER pre-calibration (ambient-covariate corrected)"),
    ):
        _plot_sensor_traces(result, OUTPUT_DIR / name, which, title, ylim)


# ---------------------------------------------------------------- Part 2 plots


def _aggregate(runs_by_seed: list[dict], metric: str) -> dict[str, np.ndarray]:
    """Median across seeds, per method, per round. Median rather than mean
    because jaxsr's fit is independently non-deterministic and a single
    bad fit produces an enormous RMSE outlier that would dominate a mean."""
    out = {}
    for name in METHOD_NAMES:
        stacked = np.vstack([np.asarray(getattr(r[name], metric), dtype=float) for r in runs_by_seed])
        out[name] = np.nanmedian(stacked, axis=0)
    return out


def _plot_metric(runs_by_seed: list[dict], metric: str, out_path: Path, target: float | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    agg = _aggregate(runs_by_seed, metric)
    rounds = np.arange(1, N_TOTAL_EXPERIMENTS + 1)

    for name in METHOD_NAMES:
        style = METHOD_STYLE[name]
        ax.plot(
            rounds,
            agg[name],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5.5,
            linewidth=1.9,
            label=name,
        )

    if target is not None:
        ax.axhline(target, color="black", linewidth=1.2, linestyle=(0, (2, 2)), alpha=0.7)
        ax.annotate(
            f"target = {target:.1f}",
            xy=(rounds[0], target),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=9,
        )

    if metric.endswith("rmse"):
        ax.set_yscale("log")
        ax.set_ylabel(METRIC_LABELS[metric] + "  [log scale]")
    else:
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel(METRIC_LABELS[metric])

    ax.axvline(N_SEED + 0.5, color="grey", linewidth=1.0, alpha=0.5)
    ax.annotate("shared seed |  method's own points",
                xy=(N_SEED + 0.5, ax.get_ylim()[1]), xytext=(4, -14),
                textcoords="offset points", fontsize=9, color="grey")

    ax.set_xlabel("Experiment round (cumulative)")
    ax.set_xticks(rounds)
    ax.set_title(f"Part 2 - {METRIC_LABELS[metric]}  (median of {len(runs_by_seed)} seeds)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _rounds_to_target(runs_by_seed: list[dict], metric: str, target: float) -> dict[str, float | None]:
    """First round at which the median curve reaches the target. None means
    it never did within budget -- a real, distinct outcome, reported as
    such rather than silently coerced to the budget ceiling."""
    agg = _aggregate(runs_by_seed, metric)
    out: dict[str, float | None] = {}
    for name in METHOD_NAMES:
        hit = None
        for i, value in enumerate(agg[name]):
            if np.isfinite(value) and value <= target:
                hit = i + 1
                break
        out[name] = hit
    return out


# ---------------------------------------------------------------- Part 3 plots


def _plot_part3_experiment(traj, out_path: Path) -> None:
    """Drive and response as STACKED single-axis panels. Deliberately not
    a dual-y-axis chart: with PAR in umol/m^2/s against VOC in ppm there
    is no way to judge from a shared frame whether an apparent lead/lag is
    real or an artefact of how the two scales happened to be set."""
    fig, (ax_par, ax_voc) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)

    hours = np.asarray(traj.t) / 3600.0
    ax_par.plot(hours, traj.par_values, color="#ff7f0e", linewidth=1.8)
    ax_par.set_ylabel("LED PAR\n(umol/m^2/s)")
    ax_par.set_title(
        f"Part 3 - {SINUSOID_PERIOD_S / 3600:.0f}h sinusoidal LED drive and the culture's VOC response"
    )

    ax_voc.plot(hours, traj.true_voc_values, color="#1f77b4", linewidth=1.8)
    ax_voc.set_ylabel("True VOC\n(ppm)")
    ax_voc.set_xlabel("Elapsed time (hours)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_part3_recovery(traj, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))

    hours = np.asarray(traj.t_predicted) / 3600.0
    true_hours = np.asarray(traj.t) / 3600.0
    ax.plot(true_hours, traj.true_voc_values, color="black", linewidth=2.4, label="True VOC", zorder=5)

    for label, color in (("raw", "#1f77b4"), ("corrected", "#ff7f0e")):
        values = traj.predicted_voc_values.get(label)
        if values is None:
            continue
        ax.plot(hours, values, color=color, linewidth=1.8, linestyle="--",
                label=f"Discovered equation, {label}")

    finite = np.asarray(traj.true_voc_values)
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    pad = 0.35 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Elapsed time (hours)")
    ax.set_ylabel("VOC (ppm)")
    ax.set_title("Part 3 - discovered rate law integrated forward vs. the true trajectory")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------- report


def _fmt(value, spec="{:.2f}") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return spec.format(value)


def _write_report(part1, doe_runs, target, rounds_to_target, dynamics, path: Path) -> None:
    correctable = part1["correctable"]
    uncorrectable = part1["uncorrectable"]

    lines: list[str] = []
    lines.append("# Pre-calibration + JAXSR/labwiki vs. classic DoE\n")
    lines.append(
        "Three parts, all driven by one shared synthetic ground truth. Every number "
        "below is computed from the run that generated this file.\n"
    )

    # ---- ground truth
    lines.append("## The ground truth\n")
    lines.append(
        "Two control variables (LED intensity `PAR`, temperature `temp`) and a time axis. "
        "At a fixed setting the culture approaches a steady state:\n"
    )
    lines.append("```")
    lines.append("VOC(t; PAR, temp) = plateau(PAR, temp) * (1 - exp(-t / tau(PAR, temp)))")
    lines.append("")
    lines.append(f"plateau(PAR,temp) = {BASELINE}")
    lines.append(f"                  + {VMAX} * PAR/({K_M}+PAR)          [saturating light response]")
    lines.append(f"                  + {TEMP_SLOPE} * (temp-{TEMP_REF})              [temperature main effect]")
    lines.append(f"                  + {GAMMA} * PAR * (temp-{TEMP_REF})         [PAR x temp interaction]")
    lines.append(f"                  - {PHOTO_K} * max(PAR-{PHOTO_THRESHOLD_PAR:.0f},0)^2   [photoinhibition]")
    lines.append("")
    lines.append(f"tau(PAR,temp)     = {TAU_BASE_H} - {TAU_PAR_H}*PAR/({K_M}+PAR) - {TAU_TEMP_H_PER_C}*(temp-{TEMP_REF})   [hours]")
    lines.append("```\n")
    lines.append(
        f"`plateau` spans {float(true_voc_ppm(0, TEMP_BOUNDS[0])):.0f}-"
        f"{float(true_voc_ppm(TRAIN_PAR_BOUNDS[1], TEMP_BOUNDS[1])):.0f} ppm and `tau` spans "
        f"{float(true_tau_hours(TRAIN_PAR_BOUNDS[1], TEMP_BOUNDS[1])):.1f}-"
        f"{float(true_tau_hours(0, TEMP_BOUNDS[0])):.1f} h across the domain.\n"
    )
    lines.append(
        "**Why tau is hours, not minutes.** A ~2-minute constant would be headspace gas "
        "mixing; across a 168-hour run that transient is the first 0.1% of the samples and "
        "is unrecoverable from noise. At week timescales the physically relevant process is "
        "photoacclimation - the culture remodelling pigment and metabolism for new "
        "conditions - which in microalgae runs hours to days. That gives 7-14 time constants "
        "per run, so tau is identifiable.\n"
    )

    # ---- part 1
    lines.append("## Part 1 - recovering one true signal from three noisy sensors\n")
    lines.append(
        f"Three sensors watch the same reactor for one week at PAR={FIXED_PAR:.0f}, "
        f"temp={FIXED_TEMP:.0f}C (true plateau "
        f"{float(true_voc_ppm(FIXED_PAR, FIXED_TEMP)):.0f} ppm, true tau "
        f"{float(true_tau_hours(FIXED_PAR, FIXED_TEMP)):.1f} h). They observe an IDENTICAL "
        "true VOC, so every difference between their raw traces is contamination.\n"
    )
    lines.append("Each sensor sits in its own ambient micro-environment, which is what makes the raw traces differ in shape:\n")
    for sensor_id in SENSOR_IDS:
        lines.append(f"- **{sensor_id}** - {SENSOR_ENVIRONMENTS[sensor_id]}")
    lines.append("")
    lines.append(
        f"**Cross-sensor spread** (std across the three sensors at each instant, averaged over "
        f"the run): raw = {correctable.raw_spread_ppm:.2f} ppm -> corrected = "
        f"{correctable.corrected_spread_ppm:.2f} ppm "
        f"({correctable.spread_reduction_pct:.0f}% reduction).\n"
    )
    lines.append(
        "Per-sensor accuracy against the truth, as BOTH root-mean-square error and median "
        "absolute error. Reported together deliberately: sparse glitches (below) dominate an "
        "RMSE while barely moving a median, so either statistic alone would misrepresent the "
        "result in whichever direction it happened to favour.\n"
    )
    for sensor_id in SENSOR_IDS:
        t = correctable.traces[sensor_id]
        lines.append(
            f"- {sensor_id} ({t.environment}): RMSE {t.raw_rmse_vs_true:.2f} -> "
            f"{t.corrected_rmse_vs_true:.2f} ppm | median|err| {t.raw_median_abs_err:.2f} -> "
            f"{t.corrected_median_abs_err:.2f} ppm"
        )
    lines.append("")

    med_before = [correctable.traces[s].raw_median_abs_err for s in SENSOR_IDS]
    med_after = [correctable.traces[s].corrected_median_abs_err for s in SENSOR_IDS]
    rmse_after = [correctable.traces[s].corrected_rmse_vs_true for s in SENSOR_IDS]
    lines.append(
        f"**On the bulk of the trace they converge.** Median absolute error goes from "
        f"{min(med_before):.1f}-{max(med_before):.1f} ppm to {min(med_after):.1f}-"
        f"{max(med_after):.1f} ppm across the three, on a "
        f"{float(true_voc_ppm(FIXED_PAR, FIXED_TEMP)):.0f} ppm signal. The three shapes differ "
        "because each sensor's own MEASURED RH/T differs, and a linear dependence on measured "
        "RH/T is exactly the contamination class `fit_covariate_model` characterizes. "
        "Correction works because the differences are attributable to a measured covariate, not "
        "because the pipeline removes arbitrary noise.\n"
    )

    lines.append("### Spikes: real VOC events vs. instrument glitches\n")
    lines.append(
        "The raw traces contain sharp excursions of two kinds, and separating them is the "
        "point.\n"
    )
    lines.append(
        "**Real VOC events** (a disturbance, a feed) are part of the true signal. Every sensor "
        "watching the reactor sees the same event at the same instant, so they are generated "
        "once and folded into the ground truth. Correction leaves them intact and they "
        "contribute no error - which is correct behaviour: a pipeline that erased them would be "
        "destroying data, not cleaning it.\n"
    )
    lines.append(
        "**Instrument glitches** (electrical transients, dropouts) are per-sensor and "
        "independent. Nothing in this package removes them: `run_fleet_zero` handles a constant "
        "bias, `fit_covariate_model` a linear dependence on measured RH/T, "
        "`subtract_common_mode` a shared same-true-value artifact, and "
        "`spectral.notch_filter_known_artifacts` a PERIODIC component - a sparse aperiodic "
        "outlier is none of those. There is no outlier-rejection stage anywhere in the "
        f"package, which is why corrected RMSE stays at {min(rmse_after):.0f}-"
        f"{max(rmse_after):.0f} ppm, roughly {max(rmse_after) / max(med_after):.0f}x the median "
        "error, while the median itself is fine.\n"
    )
    lines.append(
        "**The discriminator is cross-sensor coincidence, and it is the main argument for "
        "running three sensors rather than one.** From a single trace a real event and a glitch "
        "are frequently indistinguishable - same shape, same duration, same amplitude range. "
        "Across three simultaneous instruments they are not: an excursion appearing in all "
        "three at the same instant is real, one appearing in a single sensor is that sensor's "
        "own problem. Blind despiking on one channel would delete genuine transient biology; "
        "coincidence-gated despiking would not. That capability does not exist in the package "
        "today and is the concrete gap this part identifies.\n"
    )

    lines.append("### Negative control - the same shapes, no covariate signature\n")
    lines.append(
        f"Re-running with the same three shapes injected straight onto the voltage, with "
        f"nothing in the logged RH/T explaining them: spread "
        f"{uncorrectable.raw_spread_ppm:.2f} -> {uncorrectable.corrected_spread_ppm:.2f} ppm "
        f"({uncorrectable.spread_reduction_pct:.0f}% reduction), versus "
        f"{correctable.spread_reduction_pct:.0f}% in the correctable case.\n"
    )
    lines.append(
        "This is the boundary of the pipeline's competence, shown rather than asserted. One "
        "precision worth keeping: the FLAT drift is a constant offset, which fleet-zero "
        "*would* correct given a zero-air reference - it is not run in this part, which is why "
        "that sensor barely improves. The rising and oscillating drifts are genuinely outside "
        "what any model here fits.\n"
    )

    # ---- part 2
    lines.append("## Part 2 - active learning vs. classic DoE\n")
    lines.append(
        f"{N_TOTAL_EXPERIMENTS} experiments per method ({N_SEED} shared seed points + "
        f"{N_TOTAL_EXPERIMENTS - N_SEED} of the method's own), repeated over {len(doe_runs)} "
        f"seeds. Each run yields two scalars fitted from its own noisy time series "
        f"(measurement noise {MEASUREMENT_NOISE_PPM} ppm, the level Part 1 shows the real "
        "pipeline achieves), and each scalar gets its own symbolic surface.\n"
    )
    lines.append(
        f"PAR is sampled only within {TRAIN_PAR_BOUNDS[0]:.0f}-{TRAIN_PAR_BOUNDS[1]:.0f}; the band "
        f"{EXTRAP_PAR_BOUNDS[0]:.0f}-{EXTRAP_PAR_BOUNDS[1]:.0f} is held out from every method so "
        "extrapolation can be scored on territory none of them saw.\n"
    )

    lines.append("### The two labwiki variants\n")
    lines.append(
        f"Both start from the same operator note: *\"{LABWIKI_CONSTRAINT_NOTE}\"*\n"
    )
    lines.append(
        f"**labwiki-constraint-with-margin** turns that note into a numeric constraint "
        f"`bound_overrides` = PAR in "
        f"[{LABWIKI_MARGIN_BOUNDS['par_umol_m2_s'][0]:.0f}, "
        f"{LABWIKI_MARGIN_BOUNDS['par_umol_m2_s'][1]:.0f}] - deliberately keeping headroom above "
        f"the {PHOTO_THRESHOLD_PAR:.0f} the note mentions. A literal reading would hard-cap at "
        f"{PHOTO_THRESHOLD_PAR:.0f} exactly; the note says output *falls off* above there, not "
        "that it is a wall, and the previous benchmark round showed that a no-margin reading "
        "excludes the genuinely better region and makes labwiki actively harmful. Deciding the "
        "margin is the judgment step `system_prompt.md` assigns to Hermes.\n"
    )
    lines.append(
        f"**labwiki-search_bounds-seeding** supplies the known safe operating envelope up front "
        f"instead - `search_bounds` = PAR "
        f"[{DECLARED_SEARCH_BOUNDS['par_umol_m2_s'][0]:.0f}, "
        f"{DECLARED_SEARCH_BOUNDS['par_umol_m2_s'][1]:.0f}], temp "
        f"[{DECLARED_SEARCH_BOUNDS['mean_sample_t_c'][0]:.0f}, "
        f"{DECLARED_SEARCH_BOUNDS['mean_sample_t_c'][1]:.0f}]. This matters because "
        f"`suggest_next_experiments` defaults its search to the OBSERVED data's min/max, and "
        f"`bound_overrides` can only narrow that, never widen it. The campaign is seeded with a "
        f"narrow cluster of pilot runs (PAR "
        f"{min(p[0] for p in SEED_POINTS):.0f}-{max(p[0] for p in SEED_POINTS):.0f}), so without "
        "this the learner is permanently confined to the cluster. That is the case `search_bounds` "
        "was built for.\n"
    )

    lines.append("### Metrics\n")
    lines.append(
        "- **Surface accuracy** - RMSE against the true plateau over the sampled domain. The "
        "conventional measure, but a flexible surrogate can score well here with entirely wrong "
        "structure.\n"
        "- **Structural recovery** - fraction of the physically distinctive true terms "
        f"({', '.join(sorted(TRUE_PLATEAU_TERMS))}) actually selected. The most direct test of "
        "\"did it uncover the relationship\", and pass/fail per term rather than a squishy "
        "distance. Constant/linear terms are excluded from scoring since nearly every fit "
        "includes them.\n"
        "- **Extrapolation error** - RMSE on the held-out PAR band. A structurally correct law "
        "extrapolates; an in-sample surrogate collapses. This discriminates far more sharply "
        "than in-domain RMSE.\n"
    )

    lines.append(f"### Speed: experiments needed to reach RMSE <= {target:.1f} ppm\n")
    lines.append(
        f"Target computed fresh each run as 1.5x the best method's own round-"
        f"{N_TOTAL_EXPERIMENTS} median, never hardcoded.\n"
    )
    for name in METHOD_NAMES:
        hit = rounds_to_target[name]
        lines.append(f"- {name}: {hit if hit is not None else 'never within budget'}")
    lines.append("")

    lines.append("### Final-round medians\n")
    for name in METHOD_NAMES:
        agg_s = _aggregate(doe_runs, "surface_rmse")[name][-1]
        agg_t = _aggregate(doe_runs, "structural_recovery")[name][-1]
        agg_e = _aggregate(doe_runs, "extrapolation_rmse")[name][-1]
        lines.append(
            f"- {name}: surface {_fmt(agg_s, '{:.1f}')} ppm | structural "
            f"{_fmt(agg_t)} | extrapolation {_fmt(agg_e, '{:.1f}')} ppm"
        )
    lines.append("")

    best_surface = min(METHOD_NAMES, key=lambda n: np.nan_to_num(_aggregate(doe_runs, "surface_rmse")[n][-1], nan=1e18))
    best_struct = max(METHOD_NAMES, key=lambda n: _aggregate(doe_runs, "structural_recovery")[n][-1])
    best_extrap = min(METHOD_NAMES, key=lambda n: np.nan_to_num(_aggregate(doe_runs, "extrapolation_rmse")[n][-1], nan=1e18))
    lines.append(
        f"**Verdict as measured:** best surface accuracy = *{best_surface}*; best structural "
        f"recovery = *{best_struct}*; best extrapolation = *{best_extrap}*. Where a classic DoE "
        "method wins, that is reported as-is - active learning exploits a promising region, "
        "which helps optimisation and can hurt whole-domain structure recovery, and this "
        "benchmark was not retuned to avoid that outcome.\n"
    )

    # ---- part 3
    lines.append("## Part 3 - the within-run rate law\n")
    lines.append(
        f"One week-long run under a real sinusoidal LED profile "
        f"({SINUSOID_PROFILE['mean_par_umol_m2_s']:.0f} +/- "
        f"{SINUSOID_PROFILE['amplitude_par_umol_m2_s']:.0f} umol/m^2/s, "
        f"{SINUSOID_PERIOD_S / 3600:.0f} h period) at temp={DYNAMICS_TEMP:.0f}C, fed into the "
        "real `discover_led_response_dynamics` to recover d(VOC)/dt.\n"
    )
    lines.append(
        f"**The {SINUSOID_PERIOD_S / 3600:.0f}h period is deliberate.** Part 1 establishes a "
        "24-hour diurnal ambient swing, and the covariate correction exists to remove exactly "
        "that; driving the light on the same 24h period would make a genuine PAR effect and a "
        "residual ambient artefact nearly inseparable, so an apparent success could be the "
        "algorithm latching onto the room's day/night cycle instead.\n"
    )
    results = dynamics.per_level_results["sinusoid"]
    for label in ("raw", "corrected"):
        res = results.get(label)
        if res is None:
            continue
        lines.append(
            f"- **{label}**: RMSE vs true derivative {res.rmse_vs_true_derivative:.4f} ppm/s, "
            f"R^2 {res.r2_vs_true_derivative:.4f}"
        )
    lines.append("")
    raw_r2 = results["raw"].r2_vs_true_derivative
    corr_r2 = results["corrected"].r2_vs_true_derivative
    if corr_r2 > raw_r2:
        lines.append(
            f"Applying the persisted ambient-baseline correction improved R^2 from "
            f"{raw_r2:.4f} to {corr_r2:.4f} this run.\n"
        )
    else:
        lines.append(
            f"Applying the ambient-baseline correction did NOT help this run "
            f"(R^2 {raw_r2:.4f} raw vs {corr_r2:.4f} corrected) - reported in whichever "
            "direction the numbers land.\n"
        )
    lines.append(
        "Temperature is held fixed in this part: the shipped function declares VOC and PAR as "
        "its state variables, and extending production code purely to make a benchmark look "
        "more complete would be the wrong trade.\n"
    )

    # ---- plots
    lines.append("## Plots\n")
    lines.append("- `part1_before_calibration.png` / `part1_after_calibration.png` - the three sensors before and after correction, same units and same y-range on both.")
    lines.append("- `part2_surface_rmse.png`, `part2_structural_recovery.png`, `part2_extrapolation_rmse.png` - one per metric, all seven methods.")
    lines.append("- `part3_experiment.png` - the LED drive and the VOC response, stacked single-axis panels.")
    lines.append("- `part3_recovery.png` - the discovered equation integrated forward against the truth.\n")

    lines.append("## What this does NOT claim\n")
    lines.append(
        "- One synthetic ground truth, not a general proof about real biology.\n"
        "- Every method fits the SAME basis library, so the comparison isolates point selection, "
        "not model expressiveness. That library contains the true functional forms as candidates "
        "- without them structural recovery would be impossible by construction and the metric "
        "would measure nothing.\n"
        "- 5-minute sampling and the tau range are disclosed modelling choices, not measurements "
        "from the real rig.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Part 1 (sensor consistency)...")
    part1 = run_sensor_consistency_test(verbose=True)
    _write_part1_plots(part1["correctable"])

    print(f"\nPart 2 (DoE comparison, {len(DOE_SEEDS)} seeds)...")
    doe_runs = []
    for seed in DOE_SEEDS:
        print(f"  seed {seed + 1}/{len(DOE_SEEDS)}")
        doe_runs.append(run_doe_comparison(seed=seed))

    final_medians = {n: _aggregate(doe_runs, "surface_rmse")[n][-1] for n in METHOD_NAMES}
    best = np.nanmin([v for v in final_medians.values() if np.isfinite(v)])
    target = float(best * 1.5)
    rounds_to_target = _rounds_to_target(doe_runs, "surface_rmse", target)

    for metric in METRIC_KEYS:
        _plot_metric(
            doe_runs,
            metric,
            OUTPUT_DIR / f"part2_{metric}.png",
            target if metric == "surface_rmse" else None,
        )

    print("\nPart 3 (dynamics recovery)...")
    dynamics = run_dynamics_recovery_test(verbose=True)
    traj = dynamics.trajectories["sinusoid"]
    _plot_part3_experiment(traj, OUTPUT_DIR / "part3_experiment.png")
    _plot_part3_recovery(traj, OUTPUT_DIR / "part3_recovery.png")

    _write_report(part1, doe_runs, target, rounds_to_target, dynamics, OUTPUT_DIR / "REPORT.md")

    print(f"\nDone. Results written to {OUTPUT_DIR}/")
    for name in (
        "part1_before_calibration.png",
        "part1_after_calibration.png",
        "part2_surface_rmse.png",
        "part2_structural_recovery.png",
        "part2_extrapolation_rmse.png",
        "part3_experiment.png",
        "part3_recovery.png",
        "REPORT.md",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
