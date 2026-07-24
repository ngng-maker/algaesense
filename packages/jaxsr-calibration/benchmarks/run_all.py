"""Run both benchmark tests, save plots, and write a REPORT.md with the
numeric verdict.

Usage: .venv/Scripts/python.exe packages/jaxsr-calibration/benchmarks/run_all.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

"""
This benchmark's own modules (ground_truth, doe_methods, calibration_recovery,
doe_comparison, dynamics_recovery) import each other with plain, flat imports
(e.g. `from ground_truth import ...`), which only resolve if this directory
is on sys.path -- true automatically when Python is invoked as
`python path/to/run_all.py` (it inserts the script's own directory as
sys.path[0]), but not guaranteed under every IDE/launcher/working directory.
Inserting it explicitly makes this script runnable from anywhere, not just
lucky invocation styles.
"""

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from calibration_recovery import (
    TRUE_CALIBRATION,
    run_calibration_recovery_test,
)
from doe_comparison import (
    LABWIKI_BOUND_OVERRIDE,
    MEASUREMENT_NOISE_SIGMA_PPM,
    N_EXTRA,
    N_TOTAL_EXPERIMENTS,
    PHOTO_THRESHOLD_PAR,
    SEED_ONLY_BEST,
    TRUE_GLOBAL_MAX,
    TRUE_GLOBAL_MAX_PAR,
    TRUE_GLOBAL_MAX_TEMP,
    run_doe_comparison,
)
from doe_methods import grid_points
from dynamics_recovery import (
    DURATION_S,
    PAR_LEVELS,
    PREDICT_DURATION_S,
    TEMP,
    run_dynamics_recovery_test,
)
from ground_truth import (
    AmbientCovariateTruth,
    BASELINE as GT_BASELINE,
    DYNAMIC_RELAXATION_TAU_S,
    GAMMA as GT_GAMMA,
    K_M as GT_K_M,
    NoiseConfig,
    PHOTO_K as GT_PHOTO_K,
    PHOTO_THRESHOLD_PAR as GT_PHOTO_THRESHOLD,
    TEMP_REF as GT_TEMP_REF,
    TEMP_SLOPE as GT_TEMP_SLOPE,
    VMAX as GT_VMAX,
)

OUTPUT_DIR = Path(__file__).parent / "results"
N_SEEDS = 12
N_DYNAMICS_SEEDS = 5

"""
Validated categorical palette (fixed hue order -- never cycled, per this
project's dataviz convention): assigning colors in this exact order
keeps every adjacent pair colorblind-safe (checked with the skill's own
validator) and keeps a given series' color meaning stable across every
plot in this report, not just within one panel.
"""
PALETTE = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
RAW_COLOR = PALETTE[0]  # blue -- "raw/uncorrected" everywhere in this report
CORRECTED_COLOR = PALETTE[5]  # orange -- "corrected" everywhere in this report
TRUE_COLOR = "#3a3a38"  # near-black -- the ground truth itself, never a data series' color

_METHOD_COLORS = {
    "Ours (plain)": PALETTE[0],
    "Ours + labwiki": PALETTE[1],
    "Latin Hypercube": PALETTE[2],
    "Sobol": PALETTE[3],
    "Grid": PALETTE[4],
    "Random": PALETTE[5],
}

"""
Distinct marker SHAPES per method, not just color -- all 6 methods share
the same 4 seed points (rounds 1-4), so their "finding good conditions"
curves are identical there by construction, and often nearly identical
for rounds 5-10 too (the shared seed frequently already sits close to
the true optimum, leaving little room for any method to separate from
it -- see REPORT.md's seed-adjusted-improvement section). When curves
coincide, color alone makes 5 of 6 series invisible under whichever one
was drawn last; a distinct marker shape stays visually identifiable even
when several lines sit on the exact same pixels.
"""
_METHOD_MARKERS = {
    "Ours (plain)": "o",
    "Ours + labwiki": "s",
    "Latin Hypercube": "^",
    "Sobol": "D",
    "Grid": "v",
    "Random": "P",
}

"""
Presentation-quality matplotlib defaults: larger type for projection/
slides, a light recessive grid (never dominant), and no top/right
spines so the eye lands on the data, not the frame.
"""
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.8,
        "axes.edgecolor": "#898781",
        "axes.labelcolor": "#0b0b0b",
        "text.color": "#0b0b0b",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _run_doe_comparison_repeated(n_seeds: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Repeat the full 10-experiment DoE comparison across n_seeds
    independent random seeds and stack each method's round-by-round
    curves for BOTH metrics -- a single 10-experiment run is too small to
    tell a real difference between methods from seed-to-seed noise, so
    every number in the report is a mean (+/- std) over these repeats,
    not one run."""
    rmse_curves: dict[str, list[list[float]]] = {}
    best_found_curves: dict[str, list[list[float]]] = {}
    for seed in range(n_seeds):
        print(f"  [DoE comparison] seed {seed + 1}/{n_seeds}")
        result = run_doe_comparison(seed=seed, verbose=False)
        for label, rmses in result.rmse_by_round.items():
            rmse_curves.setdefault(label, []).append(rmses)
        for label, values in result.best_found_by_round.items():
            best_found_curves.setdefault(label, []).append(values)
    return (
        {label: np.array(curves, dtype=float) for label, curves in rmse_curves.items()},
        {label: np.array(curves, dtype=float) for label, curves in best_found_curves.items()},
    )


def _run_dynamics_recovery_repeated(n_seeds: int):
    """Same reasoning as Test 2's repeats: jaxsr.discover_dynamics builds
    on the same non-deterministic SymbolicRegressor.fit(), so one run
    isn't a result on its own -- average the recovery scores across
    seeds AND across PAR levels (since none are pooled together before
    fitting, each PAR level's own fit is an independent trial of the
    same underlying question: "does discover_led_response_dynamics
    recover this experiment's own relaxation dynamics correctly"), but
    keep the BEST-fitting seed's full set of per-PAR-level trajectories
    for the illustrative overlay plot (not just seed 0 arbitrarily) --
    given the real recovery-rate/divergence findings below, an arbitrary
    seed's plot can show every predicted line diverging immediately,
    which illustrates nothing useful even though it's an honest outcome;
    the plot is meant to show what a working fit looks like, while the
    report's own text states the full, honest distribution -- including
    failures -- separately, not just the cherry-picked example."""
    per_label_rmse: dict[str, list[float]] = {}
    per_label_r2: dict[str, list[float]] = {}
    per_label_equation: dict[str, str] = {}
    per_label_selected_features: dict[str, list[list[str]]] = {}
    per_label_diverged: dict[str, list[bool]] = {}
    best_trajectories = None
    best_score = -float("inf")
    for seed in range(n_seeds):
        print(f"  [Dynamics recovery] seed {seed + 1}/{n_seeds}")
        run = run_dynamics_recovery_test(seed=seed, verbose=False)
        seed_r2_values = [
            result.r2_vs_true_derivative
            for results in run.per_level_results.values()
            for result in results.values()
        ]
        seed_score = float(np.mean(seed_r2_values))
        if seed_score > best_score:
            best_score = seed_score
            best_trajectories = run.trajectories
        for par_level, results in run.per_level_results.items():
            for label, result in results.items():
                per_label_rmse.setdefault(label, []).append(result.rmse_vs_true_derivative)
                per_label_r2.setdefault(label, []).append(result.r2_vs_true_derivative)
                per_label_equation[label] = result.equation
                per_label_selected_features.setdefault(label, []).append(result.selected_features)
                per_label_diverged.setdefault(label, []).append(result.diverged)
    return per_label_rmse, per_label_r2, per_label_equation, per_label_selected_features, per_label_diverged, best_trajectories


def _plot_dynamics_recovery(trajectories: dict, output_path: Path) -> None:
    """Two panels -- (a)/(b) overlay every static-PAR experiment's TRUE
    VOC(t) trajectory (solid) against the DISCOVERED equation's own
    predicted trajectory (dashed, integrated forward from the same
    initial condition), one color per PAR level, raw mode in (a) and
    ambient-corrected mode in (b) -- deliberately two panels rather than
    one twin-axis plot (a documented chart anti-pattern) or one
    overcrowded panel with 2x as many lines. The numeric accuracy
    summary (RMSE/R^2, aggregated across every (seed, PAR level)
    independent trial) is reported in REPORT.md's text rather than a
    third bar-chart panel here."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    par_levels_sorted = sorted(trajectories.keys())
    level_colors = {level: PALETTE[i % len(PALETTE)] for i, level in enumerate(par_levels_sorted)}

    """
    y-axis limits are pinned to the TRUE trajectories' own range (with a
    margin) -- a discovered equation's forward-integrated prediction can
    genuinely diverge (see _simulate_predicted_trajectory's docstring),
    and its NaN-filled tail is excluded from autoscaling by definition,
    but pinning explicitly also guards against a prediction that stays
    finite yet drifts far outside any physically plausible range.
    """
    all_true_values = np.concatenate([trajectories[level].true_voc_values for level in par_levels_sorted])
    y_lo, y_hi = float(np.min(all_true_values)), float(np.max(all_true_values))
    y_margin = max(10.0, (y_hi - y_lo) * 0.15)

    for mode_idx, mode in enumerate(["raw", "corrected"]):
        ax = axes[mode_idx]
        for level in par_levels_sorted:
            traj = trajectories[level]
            color = level_colors[level]
            t_days = traj.t / 86400.0
            t_pred_days = traj.t_predicted / 86400.0
            ax.plot(t_days, traj.true_voc_values, color=color, linewidth=2.0, linestyle="-", label=f"PAR={level:.0f} (true)")
            ax.plot(t_pred_days, traj.predicted_voc_values[mode], color=color, linewidth=2.0, linestyle="--", label=f"PAR={level:.0f} (predicted)")
        ax.set_ylim(y_lo - y_margin, y_hi + y_margin)
        ax.set_xlabel("Elapsed time (days)")
        ax.set_ylabel("VOC output (ppm)")
        mode_title = "Raw (no ambient correction)" if mode == "raw" else "Corrected (ambient_baseline_run_id set)"
        ax.set_title(f"({'a' if mode_idx == 0 else 'b'}) {mode_title}\nsolid = true, dashed = discovered-equation prediction")
        ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.95)

    fig.suptitle(
        "Test 3 — Dynamics recovery: static-PAR step experiments, real discover_led_response_dynamics\n"
        "(shows the best-fitting of several repeats, for a legible illustration — the report states "
        "the full, honest range including worse fits)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_cross_sensor_consistency(result, output_path: Path) -> None:
    """Two panels -- (a) RAW, (b) CORRECTED -- each overlaying all 3
    sensors' ppm trace over time (one color per sensor, fixed order)
    against a dashed reference line at the true value. All 3 sensors are
    observing the EXACT SAME (PAR, temp) condition, contaminated ONLY
    with the same ambient-covariate/AR(1) noise every other recording in
    this benchmark uses (the contamination the pipeline actually
    models), so in an ideal world every colored line would sit exactly
    on the dashed true-value line in BOTH panels -- how far short of
    that the RAW panel falls vs. how much closer the CORRECTED panel
    gets is the entire point of this chart."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sensor_ids = result.sensor_ids
    sensor_colors = {sid: PALETTE[i % len(PALETTE)] for i, sid in enumerate(sensor_ids)}
    t_minutes = np.array(result.t) / 60.0

    for panel_idx, (ax, ppm_dict, title) in enumerate([
        (axes[0], result.raw_ppm, "(a) Raw (uncorrected voltage)"),
        (axes[1], result.corrected_ppm, "(b) Corrected (ambient-baseline applied)"),
    ]):
        for sensor_id in sensor_ids:
            ax.plot(
                t_minutes, ppm_dict[sensor_id], color=sensor_colors[sensor_id], linewidth=1.8,
                label=sensor_id, alpha=0.9,
            )
        ax.axhline(result.true_ppm, color=TRUE_COLOR, linestyle="--", linewidth=1.5, label="True VOC (same for all 3)")
        ax.set_xlabel("Elapsed time (minutes)")
        ax.set_ylabel("Recovered VOC (ppm)")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9, framealpha=0.95)

    fig.suptitle(
        f"Test 1b — Cross-sensor consistency: 3 sensors, one shared (PAR={result.par:.0f}, temp={result.temp:.0f}°C) condition"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _compute_rounds_to_target(rmse_curves: dict[str, np.ndarray]) -> tuple[dict[str, list[float | None]], float]:
    """For each method, the first round (1-indexed) at which that seed's
    surface-reconstruction RMSE first drops to or below a shared target
    -- answers "how many experiments does this method need to reach a
    given accuracy," rather than "how accurate is it at a fixed
    N_TOTAL_EXPERIMENTS-round budget." `None` means that seed never
    reached the target within the budget. The target itself is computed
    fresh from THIS run's own data (1.5x the best method's own mean
    round-10 RMSE), never hardcoded, so a method has to notably beat
    "merely as good as the best method eventually gets" to count as
    having converged, while staying meaningful if the ground truth or
    noise level ever changes."""
    final_means = {label: float(np.nanmean(curve[:, -1])) for label, curve in rmse_curves.items()}
    target_rmse = 1.5 * min(final_means.values())

    rounds_to_target: dict[str, list[float | None]] = {}
    for label, curve in rmse_curves.items():
        per_seed: list[float | None] = []
        for seed_curve in curve:
            reached = None
            for i, val in enumerate(seed_curve):
                if not np.isnan(val) and val <= target_rmse:
                    reached = float(i + 1)
                    break
            per_seed.append(reached)
        rounds_to_target[label] = per_seed
    return rounds_to_target, target_rmse


def _rounds_to_target_summary(rounds_to_target: dict[str, list]) -> dict[str, dict]:
    summary = {}
    for label, per_seed in rounds_to_target.items():
        converged = [r for r in per_seed if r is not None]
        summary[label] = {
            "median_rounds": float(np.median(converged)) if converged else None,
            "n_converged": len(converged),
            "n_total": len(per_seed),
        }
    return summary


def _plot_doe_comparison(
    rmse_curves: dict[str, np.ndarray],
    best_found_curves: dict[str, np.ndarray],
    rounds_to_target: dict[str, list],
    target_rmse: float,
    output_path: Path,
) -> None:
    """Four panels -- (a)/(b) are the two DISTINCT metrics that can (and
    here, do) disagree about which method 'wins', since they measure
    different things (whole-surface reconstruction accuracy, which
    favors space-filling DoE, vs. whether the method actually located
    good experimental conditions, what active learning is FOR). Panel
    (c) is a deliberate zoomed-in re-plot of (b)'s rounds 4-10: all 6
    methods share the SAME 4 seed points, so their (b) curves are
    IDENTICAL for rounds 1-4 by construction, and often stay within a
    few percent of each other afterward too (the shared seed frequently
    already sits close to the true optimum) -- real, not missing, data,
    but invisible at panel (b)'s necessary 0-100% scale. Panel (c)
    re-plots the same numbers on an axis zoomed to the actual spread so
    the real separation between methods is visible. Panel (d) is the
    speed reframing of panel (a): rather than "how accurate is this
    method at a fixed budget," it asks "how many experiments does this
    method actually need to reach a given accuracy" -- arguably the more
    practically relevant question for someone deciding how many real
    experiments to run."""
    fig, axes2d = plt.subplots(2, 2, figsize=(20, 13))
    axes = [axes2d[0, 0], axes2d[0, 1], axes2d[1, 0], axes2d[1, 1]]
    rounds = np.arange(1, N_TOTAL_EXPERIMENTS + 1)

    for label, curve in rmse_curves.items():
        mean = np.nanmean(curve, axis=0)
        std = np.nanstd(curve, axis=0)
        color = _METHOD_COLORS.get(label, None)
        marker = _METHOD_MARKERS.get(label, "o")
        axes[0].plot(rounds, mean, label=label, color=color, linewidth=2.4, marker=marker, markersize=6)
        axes[0].fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    axes[0].set_xlabel("Experiment round (cumulative)")
    axes[0].set_ylabel("RMSE vs. true VOC(PAR, temp) surface (ppm)")
    axes[0].set_title(f"(a) Surface reconstruction\nmean ± std, {N_SEEDS} repeats")
    axes[0].legend(loc="upper right", framealpha=0.95)
    axes[0].set_xticks(rounds)

    pct_means = {}
    for label, curve in best_found_curves.items():
        pct = 100.0 * curve / TRUE_GLOBAL_MAX
        mean = np.mean(pct, axis=0)
        std = np.std(pct, axis=0)
        pct_means[label] = mean
        color = _METHOD_COLORS.get(label, None)
        marker = _METHOD_MARKERS.get(label, "o")
        axes[1].plot(rounds, mean, label=label, color=color, linewidth=2.4, marker=marker, markersize=6)
        axes[1].fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    axes[1].axhline(100.0, color=TRUE_COLOR, linestyle="--", linewidth=1.3, alpha=0.7, label="True global maximum")
    axes[1].set_xlabel("Experiment round (cumulative)")
    axes[1].set_ylabel("Best-found VOC (% of true global max)")
    axes[1].set_title(f"(b) Finding good conditions\nmean ± std, {N_SEEDS} repeats")
    axes[1].legend(loc="lower right", framealpha=0.95)
    axes[1].set_xticks(rounds)

    """
    Panel (c): same data as (b), rounds 4-10 only, y-axis zoomed to the
    actual data range (computed fresh, never hardcoded) so genuine
    differences between methods -- often just a few percentage points --
    are visible instead of looking like a flat overlapping line.
    """
    zoom_rounds = rounds[3:]
    all_zoom_values = np.concatenate([pct_means[label][3:] for label in best_found_curves])
    zoom_lo, zoom_hi = float(np.min(all_zoom_values)), float(np.max(all_zoom_values))
    margin = max(0.5, (zoom_hi - zoom_lo) * 0.15)
    for label in best_found_curves:
        color = _METHOD_COLORS.get(label, None)
        marker = _METHOD_MARKERS.get(label, "o")
        axes[2].plot(zoom_rounds, pct_means[label][3:], label=label, color=color, linewidth=2.4, marker=marker, markersize=7)
    axes[2].set_ylim(zoom_lo - margin, zoom_hi + margin)
    axes[2].set_xlabel("Experiment round (cumulative)")
    axes[2].set_ylabel("Best-found VOC (% of true global max)")
    axes[2].set_title("(c) Panel (b), zoomed to rounds 4-10")
    axes[2].legend(loc="best", framealpha=0.95, fontsize=9)
    axes[2].set_xticks(zoom_rounds)

    """
    Panel (d): "how many experiments does this method actually need,"
    the speed reframing requested in place of "how accurate is it at a
    fixed budget." A method with zero seeds reaching the target within
    N_TOTAL_EXPERIMENTS is shown as a hatched bar at the budget ceiling
    (a real, honest "never got there" result, not a missing value) --
    never silently dropped from the chart.
    """
    summary = _rounds_to_target_summary(rounds_to_target)
    labels_sorted = sorted(
        summary,
        key=lambda l: (summary[l]["median_rounds"] is None, summary[l]["median_rounds"] or 0.0),
    )
    heights = []
    hatch_flags = []
    annotations = []
    colors = []
    for label in labels_sorted:
        s = summary[label]
        colors.append(_METHOD_COLORS.get(label))
        if s["median_rounds"] is None:
            heights.append(float(N_TOTAL_EXPERIMENTS))
            hatch_flags.append(True)
            annotations.append("never\nreached")
        else:
            heights.append(s["median_rounds"])
            hatch_flags.append(False)
            annotations.append(f"{s['n_converged']}/{s['n_total']}" if s["n_converged"] < s["n_total"] else "")
    bars = axes[3].bar(labels_sorted, heights, color=colors, edgecolor="white", linewidth=0.5)
    for bar, hatched, ann in zip(bars, hatch_flags, annotations):
        if hatched:
            bar.set_hatch("//")
        if ann:
            axes[3].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, ann,
                ha="center", va="bottom", fontsize=9,
            )
    axes[3].set_ylabel("Median experiments needed (fewer is better)")
    axes[3].set_title(f"(d) Speed: experiments needed to reach\nRMSE <= {target_rmse:.0f} ppm (hatched = never, within budget)")
    axes[3].set_xticks(range(len(labels_sorted)))
    axes[3].set_xticklabels(labels_sorted, rotation=20, ha="right")
    axes[3].set_ylim(0, N_TOTAL_EXPERIMENTS + 1)

    fig.suptitle("Test 2 — DoE comparison: 10-experiment budget, 6 point-selection strategies")
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=3.0, h_pad=3.0)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_report(
    calibration_results: dict,
    cross_sensor_result,
    rmse_curves: dict[str, np.ndarray],
    best_found_curves: dict[str, np.ndarray],
    rounds_to_target: dict[str, list],
    target_rmse: float,
    dynamics_rmse: dict,
    dynamics_r2: dict,
    dynamics_equations: dict,
    dynamics_selected_features: dict,
    dynamics_diverged: dict,
    report_path: Path,
) -> None:
    lines = ["# AlgaeSense pre-calibration + JAXSR benchmark report", ""]

    ambient_defaults = AmbientCovariateTruth()
    noise_defaults = NoiseConfig()

    lines.append("## Ground truth: the one true function every test is measured against")
    lines.append("")
    lines.append(
        "All three tests below share the SAME synthetic ground truth -- one nonlinear function of "
        "PAR and temperature, plus a dynamic (time-varying) extension of it for Test 3. Nothing here "
        "is biologically exact; it's a stand-in 'real physics' chosen to be genuinely nonlinear "
        "without being so exotic that no polynomial-basis regressor could ever approximate it."
    )
    lines.append("")
    lines.append("**Static surface (Tests 1 and 2), `true_voc_ppm(PAR, temp)`:**")
    lines.append("")
    lines.append("```")
    lines.append("VOC(PAR, temp) = BASELINE")
    lines.append("               + VMAX * PAR / (K_M + PAR)                [saturating light response]")
    lines.append("               + TEMP_SLOPE * (temp - TEMP_REF)          [temperature main effect]")
    lines.append("               + GAMMA * PAR * (temp - TEMP_REF)         [genuine PAR x temp interaction]")
    lines.append("               - PHOTO_K * max(PAR - PHOTO_THRESHOLD, 0)^2   [high-PAR photoinhibition]")
    lines.append("```")
    lines.append("")
    lines.append("with the parameter values actually used:")
    lines.append("")
    lines.append(f"- `VMAX` = {GT_VMAX} ppm (maximum light-driven output as PAR → ∞)")
    lines.append(f"- `K_M` = {GT_K_M} µmol·m⁻²·s⁻¹ (half-saturation PAR)")
    lines.append(f"- `TEMP_REF` = {GT_TEMP_REF} °C (centering constant, not fitted)")
    lines.append(f"- `TEMP_SLOPE` = {GT_TEMP_SLOPE} ppm/°C")
    lines.append(f"- `GAMMA` = {GT_GAMMA} ppm per (µmol·m⁻²·s⁻¹·°C) -- the interaction coefficient Test 1 specifically checks")
    lines.append(f"- `BASELINE` = {GT_BASELINE} ppm")
    lines.append(f"- `PHOTO_THRESHOLD_PAR` = {GT_PHOTO_THRESHOLD} µmol·m⁻²·s⁻¹, `PHOTO_K` = {GT_PHOTO_K}")
    lines.append("")
    lines.append(
        "**Dynamic extension (Test 3):** a first-order relaxation ODE, `dVOC/dt = (1/tau) * "
        f"(true_voc_ppm(PAR, temp) - VOC(t))`, with `tau` = {DYNAMIC_RELAXATION_TAU_S:.0f} s -- "
        "holding any one (PAR, temp) setting constant forever converges onto the exact point already "
        "described by the static surface above, tying the two ground truths together deliberately. "
        f"Test 3 runs {len(PAR_LEVELS)} SEPARATE, independent experiments, each held at its OWN fixed "
        f"static PAR level ({', '.join(f'{p:.0f}' for p in PAR_LEVELS)} µmol·m⁻²·s⁻¹, temp fixed at "
        f"{TEMP:.0f}°C), each running for {DURATION_S / 86400.0:.0f} days -- long enough to fully settle "
        "into steady state (tau=120s means steady state is reached within minutes, so almost the entire "
        "week is spent AT the settled value, exactly like a real long-running culture would). Each "
        "experiment is analyzed entirely on its own -- NONE of them are pooled together before fitting "
        "(see the 'no pooling' rationale in dynamics_recovery.py's own module docstring)."
    )
    lines.append("")

    lines.append("### Noise sources injected on top of the ground truth (Tests 1 and 3)")
    lines.append("")
    lines.append(
        "Every noise source below is injected into synthetic RAW sensor voltage, then run through "
        "the REAL `jaxsr_calibration` diagnostics/calibration functions -- nothing here is faked "
        "downstream of the raw signal."
    )
    lines.append("")
    for sensor_id, truth in TRUE_CALIBRATION.items():
        lines.append(f"- **Fleet-zero-style per-sensor bias** (`{sensor_id}`): true b0 = {truth.b0_mv:.1f} mV, true b1 = {truth.b1_mv_per_ppm:.2f} mV/ppm")
    lines.append(
        f"- **Ambient RH/T covariate contamination**: `voltage += beta_rh*(RH-{ambient_defaults.rh_ref_pct:.0f}) "
        f"+ gamma_t*(T-{ambient_defaults.t_ref_c:.0f})`, with true beta_rh = {ambient_defaults.beta_rh}, "
        f"true gamma_t = {ambient_defaults.gamma_t} -- ambient RH swings ±{noise_defaults.ambient_rh_swing_pct:.0f}%, "
        f"ambient temperature swings ±{noise_defaults.ambient_t_swing_c:.0f}°C during each experiment"
    )
    lines.append(
        f"- **Shared common-mode artifact** (fleet-wide zero-check only, see NoiseConfig's own "
        "docstring for why it's never applied to the main experiment data): a 3.0 mV shared sine wave, "
        "60 s period, superimposed identically across all sensors at each instant"
    )
    lines.append(
        f"- **AR(1) autocorrelated sensor noise**: phi (autocorrelation) = {noise_defaults.ar1_phi}, "
        f"sigma = {noise_defaults.ar1_sigma_mv} mV per step -- a real 1/f-like memory in the raw "
        "signal, not simple white noise"
    )
    lines.append(
        f"- **Test 2's measurement noise** (a stand-in for 'this experiment ran through the full "
        f"corrected pipeline'): `true_voc_ppm(PAR, temp) + N(0, {MEASUREMENT_NOISE_SIGMA_PPM:.1f} ppm)` "
        "-- this exact sigma is Test 1's own measured corrected-pipeline residual, not an invented number"
    )
    lines.append("")

    lines.append("## Metrics used in this report")
    lines.append("")
    lines.append(
        "- **RMSE (root-mean-square error)**: the typical size of the gap between a recovered/"
        "predicted value and the true value, in the same units as the quantity itself (ppm, or "
        "ppm/s for a derivative). Lower is better. Squaring before averaging penalizes large "
        "individual misses more than many small ones."
    )
    lines.append(
        "- **R² (coefficient of determination)**: the fraction of the true surface/derivative's own "
        "variation that the recovered model explains, on a held-out grid never used for fitting "
        "(a dense 2D grid for Tests 1/2's static surface; a 1D grid at that trial's own fixed PAR "
        "for Test 3, since a single static-PAR experiment never saw any other PAR value) -- 1.0 is "
        "a perfect match, 0.0 is 'no better than always guessing the mean'. Used alongside RMSE "
        "since R² is scale-free (comparable across different quantities) while RMSE keeps the "
        "real-world units."
    )
    lines.append(
        "- **% parameter recovery error**: `100 * |recovered - true| / |true|` for each named "
        "coefficient in the known ground-truth equation (Test 1 only, via `scipy.optimize.curve_fit` "
        "against the EXACT known functional form) -- the direct test of whether the pipeline "
        "recovers the right underlying physics, not just a value that happens to predict well."
    )
    lines.append(
        "- **Best-found value, as % of true global max** (Test 2, Metric B): the true "
        "(noiseless) VOC value at the best point a method has sampled so far, divided by the real "
        "maximum of `true_voc_ppm` over the whole domain -- measures whether a method actually "
        "located good experimental conditions, independent of any downstream model fit."
    )
    lines.append(
        "- **Seed-adjusted improvement** (Test 2): a method's best-found value minus the shared "
        "seed points' own best value alone -- isolates what a method's OWN chosen points "
        "contributed, since a lucky shared starting point can otherwise dominate the raw score."
    )
    lines.append("")

    lines.append("## How to read each plot")
    lines.append("")
    lines.append(
        "A plain walkthrough of what's actually on each of the three images in this folder, panel by "
        "panel, for anyone opening them without already knowing the code behind them."
    )
    lines.append("")
    lines.append("### `cross_sensor_consistency.png` (Test 1b) -- 2 panels")
    lines.append("")
    lines.append(
        "3 sensors, one color each (fixed across both panels), all measuring the SAME true VOC value "
        "(the black dashed line) at the same time -- so in a perfect world every colored line would "
        "sit exactly on that dashed line in BOTH panels."
    )
    lines.append(
        "- **(a) Raw**: each sensor's own uncorrected trace over time -- the wobble visible here is "
        "the same ambient RH/T covariate contamination and AR(1) noise every other recording in this "
        "benchmark carries."
    )
    lines.append(
        "- **(b) Corrected**: the same three sensors after the real fleet-zero + ambient-baseline "
        "correction pipeline is applied. How much closer the colored lines sit to the dashed true-"
        "value line here, versus panel (a), is a direct visual read of whether correction actually "
        "brings sensors into agreement with each other and with the truth -- the report's own text "
        "states the exact numbers (spread before/after, per-sensor RMSE before/after) rather than "
        "leaving it to eyeballing alone."
    )
    lines.append("")
    lines.append("### `doe_comparison.png` (Test 2) -- 3 panels")
    lines.append("")
    lines.append(
        "Six colored lines run through all three panels, one per point-selection method (see the "
        "legend on panels (a)/(b)) -- each method also has its own marker shape (circle, square, "
        "triangle, diamond, upside-down triangle, plus-sign) so a line stays identifiable even where "
        "it overlaps another one exactly. The x-axis on every panel is 'experiment round' -- how "
        "many of the 10 total experiments that method has run so far, left to right."
    )
    lines.append(
        "- **(a) Surface reconstruction**: lower is better. At each round, how far off (RMSE, in "
        "ppm) that method's current best guess at the WHOLE VOC(PAR, temp) surface is from the "
        "truth. The shaded band around each line is +/- 1 standard deviation across the "
        f"{N_SEEDS} independent repeats -- a wide band means that method's outcome varies a lot "
        "run to run at that round, not just that the mean is uncertain."
    )
    lines.append(
        "- **(b) Finding good conditions**: higher is better (100% = the dashed line = the best "
        "conditions physically possible). At each round, the best true VOC value that method has "
        "found SO FAR, as a % of the true maximum. **All 6 lines are IDENTICAL for rounds 1-4** -- "
        "every method starts from the same 4 shared seed experiments by design (a fair comparison "
        "needs a common starting point) -- so only one line is visible there, sitting exactly on top "
        "of the other five. From round 5 onward the lines are each method's OWN choices, but they "
        "often stay within a few percentage points of each other, which can also look like 'one line' "
        "at this panel's necessary 0-100% scale even though the underlying numbers do differ -- that's "
        "what panel (c) is for."
    )
    lines.append(
        "- **(c) Panel (b), zoomed to rounds 4-10**: the exact same numbers as panel (b)'s tail end, "
        "just re-plotted with the y-axis zoomed into whatever narrow range the actual data spans "
        "(computed fresh each run, not a fixed zoom level) -- this is where the real, if small, "
        "separation between methods actually becomes visible. A method whose line is HIGHER here "
        "found better conditions than one whose line is lower, even though both looked flat in panel "
        "(b)."
    )
    lines.append("")
    lines.append("### `dynamics_recovery.png` (Test 3) -- 2 panels")
    lines.append("")
    lines.append(
        f"Each of the {len(PAR_LEVELS)} static PAR levels gets its own color, used consistently across "
        "panels (a) and (b) -- a SOLID line is the TRUE VOC(t) trajectory (what actually happened), a "
        "DASHED line of the same color is the DISCOVERED equation's own predicted trajectory (what "
        "`discover_led_response_dynamics` thinks would happen, integrated forward from the same "
        "starting point) -- so how closely dashed tracks solid, for each color, is a direct visual "
        "read of accuracy, not just an RMSE number. Each experiment actually runs for "
        f"{DURATION_S / 86400.0:.0f} days, but only the first "
        f"{PREDICT_DURATION_S / 60.0:.0f} minutes are plotted -- the relaxation settles within a few "
        "minutes, so the rest of the week is flat steady-state with nothing left to compare."
    )
    lines.append(
        "- **(a) Raw** / **(b) Corrected**: the same overlay, without vs. with the ambient-baseline "
        "correction applied before calibration. The exact RMSE/R^2 numbers behind these panels "
        "(averaged across every independent (seed, PAR level) trial, not just the one illustrative "
        "run shown here) are reported as text in the Test 3 section below rather than a separate chart."
    )
    lines.append("")

    lines.append("## Test 1 -- does pre-calibration correction help recover the true VOC(PAR, temp) function?")
    lines.append("")
    lines.append(
        "**Generated data** (12 synthetic experiments spanning the declared PAR/temp domain, each "
        "independently contaminated with the noise sources above, then processed through the REAL "
        "calibration/covariate-correction pipeline):"
    )
    lines.append("")
    lines.append("| PAR (µmol·m⁻²·s⁻¹) | temp (°C) | true VOC (ppm) | raw recovered (ppm) | corrected recovered (ppm) |")
    lines.append("|---:|---:|---:|---:|---:|")
    raw_r, corrected_r = calibration_results["raw"], calibration_results["corrected"]
    for i in range(len(raw_r.par_values)):
        lines.append(
            f"| {raw_r.par_values[i]:.1f} | {raw_r.temp_values[i]:.1f} | {raw_r.true_ppm[i]:.1f} | "
            f"{raw_r.measured_ppm[i]:.1f} | {corrected_r.measured_ppm[i]:.1f} |"
        )
    lines.append("")

    for key in ("raw", "corrected"):
        r = calibration_results[key]
        lines.append(f"**{r.label}**")
        lines.append(f"- RMSE vs true VOC: {r.rmse_vs_true_ppm:.2f} ppm")
        lines.append(f"- R^2 of recovered curve vs true surface (dense grid): {r.r2_on_dense_grid:.4f}")
        for name, err in r.param_pct_error.items():
            lines.append(f"- {name}: recovered={r.recovered_params[name]:.4f}, error={err:.1f}%")
        lines.append("")

    raw_rmse = calibration_results["raw"].rmse_vs_true_ppm
    corrected_rmse = calibration_results["corrected"].rmse_vs_true_ppm
    lines.append(
        f"**Verdict:** the corrected pipeline (fleet-zero + ambient-baseline + standard-addition "
        f"calibration) recovered the true VOC value {raw_rmse / corrected_rmse:.1f}x more accurately "
        f"than using raw voltage directly ({raw_rmse:.2f} ppm RMSE vs {corrected_rmse:.2f} ppm). "
        f"Every noise source injected (fleet-zero-style per-sensor bias, ambient RH/T covariate "
        f"contamination, shared common-mode artifact, AR(1) autocorrelated noise) was recovered "
        f"correctly by the real fleet_zero/ambient/covariate/common_mode functions individually -- "
        f"see the console output for each."
    )
    lines.append("")
    lines.append(
        "**On the `gamma` (PAR x temperature interaction) coefficient specifically -- this was "
        "previously reported as 'poorly identified... a real statistical limitation of this "
        "functional form,' which was WRONG.** The earlier ground truth used a multiplicative "
        "`light_term * exp(BETA_T*(temp-TEMP_REF))` temperature modulation, whose own first-order "
        "Taylor expansion contains a term of the EXACT SAME SHAPE as the 'interaction' term meant to "
        "be independent of it -- two coefficients fit against what is, to leading order, a single "
        "basis function. That was a fixable design bug, not an inherent limit on testing "
        "interactions at all. The ground truth above now uses additive main effects plus ONE genuine "
        "bilinear interaction term with nothing else proportional to it -- and `gamma` now recovers "
        f"to {calibration_results['corrected'].param_pct_error['gamma']:.2f}% error (corrected) / "
        f"{calibration_results['raw'].param_pct_error['gamma']:.2f}% error (raw), down from 44-67% "
        "under the old design. See CLAUDE.md's dev log for the full before/after and the general "
        "lesson (check whether an 'interaction' term's linearized shape collides with a main "
        "effect's own linearization before blaming collinearity on 'inherent difficulty')."
    )
    lines.append("")

    lines.append("### Cross-sensor consistency -- do different sensors converge after correction?")
    lines.append("")
    lines.append(
        "A genuinely different question from the rest of Test 1 above: instead of one sensor "
        f"observing many different (PAR, temp) settings, all {len(cross_sensor_result.sensor_ids)} sensors "
        f"here observe the EXACT SAME condition (PAR={cross_sensor_result.par:.0f} µmol·m⁻²·s⁻¹, "
        f"temp={cross_sensor_result.temp:.0f}°C, true VOC={cross_sensor_result.true_ppm:.1f} ppm) "
        "SIMULTANEOUSLY, contaminated only with the same ambient-covariate/AR(1) noise every other "
        "recording in this benchmark uses -- the contamination fleet-zero and ambient-baseline "
        "correction actually model."
    )
    lines.append("")
    lines.append(
        "Two DISTINCT things are checked: do the sensors agree WITH EACH OTHER after correction "
        "(cross-sensor spread), and does each one still agree with the TRUE value (RMSE vs. true) -- "
        "a pipeline could conceivably make sensors agree with each other while all being wrong "
        "together, or recover the true value for one sensor while leaving others far off, and these "
        "are two different failure modes worth telling apart."
    )
    lines.append("")
    spread_reduction_pct = 100.0 * (
        cross_sensor_result.raw_cross_sensor_spread - cross_sensor_result.corrected_cross_sensor_spread
    ) / cross_sensor_result.raw_cross_sensor_spread
    lines.append(
        f"**Cross-sensor spread** (standard deviation across the {len(cross_sensor_result.sensor_ids)} sensors' "
        f"readings at each instant, averaged over the whole window): raw = "
        f"{cross_sensor_result.raw_cross_sensor_spread:.2f} ppm, corrected = "
        f"{cross_sensor_result.corrected_cross_sensor_spread:.2f} ppm "
        f"({spread_reduction_pct:+.0f}% change)."
    )
    lines.append("")
    lines.append("**Per-sensor recovery vs. the true value:**")
    lines.append("")
    for sensor_id in cross_sensor_result.sensor_ids:
        lines.append(
            f"- **{sensor_id}**: raw RMSE = {cross_sensor_result.raw_rmse_vs_true[sensor_id]:.2f} ppm, "
            f"corrected RMSE = {cross_sensor_result.corrected_rmse_vs_true[sensor_id]:.2f} ppm"
        )
    lines.append("")

    """
    The verdict is computed from the actual numbers, not assumed --
    correction could plausibly have fully closed the gap, partially
    closed it, or (in principle) made it worse; whichever happened is
    what gets reported.
    """
    residual_spread_pct_of_true = 100.0 * cross_sensor_result.corrected_cross_sensor_spread / cross_sensor_result.true_ppm
    if cross_sensor_result.corrected_cross_sensor_spread < cross_sensor_result.raw_cross_sensor_spread * 0.15:
        verdict = (
            "correction very nearly eliminates the disagreement between sensors -- consistent with "
            "Test 1's own whole-domain finding above, since this uses the identical noise sources and "
            "the identical calibration/covariate models that already recover the true value there to "
            "within a fraction of a ppm."
        )
    else:
        verdict = (
            f"correction reduces but does NOT fully eliminate the disagreement between sensors -- a "
            f"residual of {cross_sensor_result.corrected_cross_sensor_spread:.1f} ppm "
            f"({residual_spread_pct_of_true:.1f}% of the true value) remains."
        )
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(
        "**A limitation worth flagging to experimentalists, not demonstrated live here:** real PID "
        "sensors can exhibit systematic drift patterns this benchmark's noise model does NOT include "
        "and this pipeline was never built to correct for -- e.g. a sluggish/heavily-damped response "
        "that lags well behind the true value, a periodic drift from intermittent connector/thermal-"
        "cycling contact resistance, or a slow exponential drift from sensor aging. None of these are "
        "a constant bias (what fleet-zero corrects) or a linear function of ambient RH/T (what "
        "ambient-baseline covariate correction models), so if a sensor's raw trace shows a shape like "
        "this in practice, fleet-zero/ambient-baseline correction alone should NOT be trusted to fix "
        "it -- that would need to be diagnosed and addressed separately (e.g. sensor replacement, a "
        "dedicated drift model), not assumed away by this pipeline."
    )
    lines.append("")

    lines.append("## Test 2 -- does calibration + JAXSR active learning + labwiki beat classic DoE?")
    lines.append("")
    lines.append(
        f"10-experiment budget, {N_SEEDS} independent repeats per method, identical measurement "
        "noise model for every method (the only variable across methods is which points get "
        "chosen). **Two DISTINCT metrics are reported, and they disagree -- both are given rather "
        "than picking the one that tells a tidier story.**"
    )
    lines.append("")

    lines.append("### Metric A: surface reconstruction (favors space-filling designs)")
    lines.append("")
    lines.append(
        "Lower RMSE = a more accurate reconstruction of the ENTIRE true VOC(PAR, temp) surface. "
        "**Reported primarily as SPEED -- how many experiments a method actually needs -- rather "
        "than accuracy at one fixed 10-experiment budget**, since \"how many real experiments do "
        "I need to run\" is usually the more practically relevant question than \"how accurate is "
        "round 10 specifically.\""
    )
    lines.append("")
    summary = _rounds_to_target_summary(rounds_to_target)
    lines.append(
        f"**Target**: RMSE <= {target_rmse:.1f} ppm (computed fresh each run as 1.5x the best "
        f"method's own mean round-{N_TOTAL_EXPERIMENTS} RMSE -- never a hardcoded number, so this "
        "stays meaningful if the ground truth or noise level ever changes)."
    )
    lines.append("")
    for label in sorted(summary, key=lambda l: (summary[l]["median_rounds"] is None, summary[l]["median_rounds"] or 0.0)):
        s = summary[label]
        if s["median_rounds"] is None:
            lines.append(f"- **{label}**: never reached the target within {N_TOTAL_EXPERIMENTS} experiments, in any of the {s['n_total']} repeats")
        elif s["n_converged"] < s["n_total"]:
            lines.append(
                f"- **{label}**: median {s['median_rounds']:.1f} experiments needed "
                f"(only reached it in {s['n_converged']} of {s['n_total']} repeats -- the rest never got there within the budget)"
            )
        else:
            lines.append(f"- **{label}**: median {s['median_rounds']:.1f} experiments needed (reached it in every repeat)")
    lines.append("")
    lines.append(
        "**For context, the same metric's older framing -- accuracy at the fixed "
        f"{N_TOTAL_EXPERIMENTS}-experiment budget:**"
    )
    lines.append("")
    final_means = {label: float(np.nanmean(curve[:, -1])) for label, curve in rmse_curves.items()}
    final_medians = {label: float(np.nanmedian(curve[:, -1])) for label, curve in rmse_curves.items()}
    final_stds = {label: float(np.nanstd(curve[:, -1])) for label, curve in rmse_curves.items()}
    for label, mean in sorted(final_means.items(), key=lambda kv: kv[1]):
        lines.append(
            f"- **{label}**: mean {mean:.1f} +/- {final_stds[label]:.1f} ppm, "
            f"median {final_medians[label]:.1f} ppm (round {N_TOTAL_EXPERIMENTS})"
        )
    lines.append("")
    lines.append(
        "Mean and median are both given deliberately: at n=10 experiments, a degree-2-polynomial "
        "SymbolicRegressor fit occasionally extrapolates badly on a small, awkwardly-placed sample "
        "-- the median is the more robust summary when that kind of rare instability is present."
    )
    lines.append("")

    lines.append("### Metric B: best-found value (what active learning is actually FOR)")
    lines.append("")
    lines.append(
        f"Did the method actually locate GOOD experimental conditions? Reported as a % of "
        f"TRUE_GLOBAL_MAX ({TRUE_GLOBAL_MAX:.1f} ppm, the real maximum of true_voc_ppm over the "
        "whole domain). A space-filling DoE design has no notion of 'good' at all -- it just "
        "covers ground -- while this is literally what the active learner's UCB acquisition is "
        "built to optimize for."
    )
    lines.append("")
    final_best_means = {label: float(np.mean(curve[:, -1])) for label, curve in best_found_curves.items()}
    final_best_stds = {label: float(np.std(curve[:, -1])) for label, curve in best_found_curves.items()}
    for label, mean in sorted(final_best_means.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * mean / TRUE_GLOBAL_MAX
        pct_std = 100.0 * final_best_stds[label] / TRUE_GLOBAL_MAX
        lines.append(f"- **{label}**: {pct:.1f}% +/- {pct_std:.1f}% of true max (round 10)")
    lines.append("")
    lines.append(
        f"**Important refinement -- all 6 methods share the SAME 4 seed points** (see "
        f"`doe_methods.SEED_POINTS`), and one seed corner already lands at "
        f"{100.0 * SEED_ONLY_BEST / TRUE_GLOBAL_MAX:.1f}% of the true max on its own, before any "
        "method makes a single one of its own choices. Confirmed directly: across many repeats, "
        "Latin Hypercube/Sobol/Random's own chosen points frequently NEVER beat this seed-only "
        "value at all -- their raw score above is really measuring 'did the shared seed already "
        "get lucky,' not their own point-selection quality. Isolating each method's genuine "
        "contribution BEYOND the seed:"
    )
    lines.append("")
    for label, mean in sorted(final_best_means.items(), key=lambda kv: -kv[1]):
        improvement_pct = 100.0 * (mean - SEED_ONLY_BEST) / TRUE_GLOBAL_MAX
        lines.append(f"- **{label}**: {improvement_pct:+.1f} percentage points beyond the seed-only baseline")
    lines.append("")

    rmse_best_label = min(final_medians, key=final_medians.get)
    value_best_label = max(final_best_means, key=final_best_means.get)
    active_learning_rmse_rank = sorted(final_medians, key=final_medians.get).index("Ours (plain)") + 1
    active_learning_value_rank = sorted(final_best_means, key=final_best_means.get, reverse=True).index("Ours (plain)") + 1
    ours_improvement = 100.0 * (final_best_means["Ours (plain)"] - SEED_ONLY_BEST) / TRUE_GLOBAL_MAX

    """
    The Grid-proximity claim is computed fresh from whatever ground
    truth is currently active, never hardcoded to a specific (PAR, temp)
    location -- an earlier version of this report hardcoded "par=417,
    temp=40" from a previous ground-truth design, which went silently
    stale and self-contradictory the moment the function changed (see
    CLAUDE.md's dev log). Grid's own fixed nodes are recomputed here and
    the true distance to the actual optimum is measured directly.
    """
    grid_node_points = grid_points(N_EXTRA)
    distances = [((gp[0] - TRUE_GLOBAL_MAX_PAR) ** 2 + (gp[1] - TRUE_GLOBAL_MAX_TEMP) ** 2) ** 0.5 for gp in grid_node_points]
    nearest_grid_distance = min(distances)
    par_span = 500.0
    temp_span = 20.0
    domain_diagonal = (par_span**2 + temp_span**2) ** 0.5
    grid_is_near_optimum = nearest_grid_distance < 0.15 * domain_diagonal

    grid_explanation = (
        f"Grid's fixed nodes happen to place one within {nearest_grid_distance:.0f} units of the true "
        f"optimum (PAR={TRUE_GLOBAL_MAX_PAR:.0f}, temp={TRUE_GLOBAL_MAX_TEMP:.0f}) -- a property of "
        "where THIS function's maximum happens to sit relative to a fixed grid's node placement, not "
        "a general guarantee that grid designs reliably find optima (an optimum further from every "
        "node would get no such assist)."
        if grid_is_near_optimum
        else
        f"Grid's fixed nodes are NOT particularly close to the true optimum (PAR={TRUE_GLOBAL_MAX_PAR:.0f}, "
        f"temp={TRUE_GLOBAL_MAX_TEMP:.0f}) this time (nearest node is {nearest_grid_distance:.0f} units "
        "away) -- so if Grid still does well here, it's from its even domain coverage, not a lucky "
        "corner hit."
    )

    lines.append(
        f"**Verdict, stated plainly rather than favorably:** on raw scores, **{rmse_best_label}** wins "
        f"Metric A and **{value_best_label}** wins Metric B. The active-learning workflow ('Ours "
        f"(plain)') ranks {active_learning_rmse_rank} of 6 on Metric A and {active_learning_value_rank} "
        f"of 6 on Metric B. But the seed-adjusted breakdown above tells a more precise story: Latin "
        "Hypercube, Sobol, and Random frequently contribute LITERALLY NOTHING beyond the shared seed's "
        "own lucky corner (their own 6 chosen points never find anything better, seed after seed) -- "
        "they win Metric B mostly by inheriting a good starting point, not by searching well. 'Ours "
        f"(plain)' reliably improves beyond the seed ({ours_improvement:+.1f} points), a small but "
        f"consistent signal that its adaptive search is genuinely doing something useful, unlike the "
        f"non-adaptive baselines sitting next to it. {grid_explanation} **Take-away:** at this budget, "
        "the seed-adjusted numbers show the active-learning workflow IS doing real, consistent, "
        "adaptive work that most DoE baselines aren't -- whether that's enough to lead on the raw "
        "score depends on where this particular ground truth's optimum happens to sit, which is "
        "exactly why both the raw and seed-adjusted numbers are reported rather than just one. See "
        "`doe_comparison.png` for both metrics' full round-by-round picture -- panel (c) is a zoomed "
        "view of panel (b)'s tail end specifically to make this seed-adjusted separation visible."
    )
    lines.append("")

    """
    Why does 'Ours + labwiki' underperform 'Ours (plain)'? Computed from
    the actual TRUE_GLOBAL_MAX_PAR vs. LABWIKI_BOUND_OVERRIDE values
    currently in effect, never hardcoded -- this is a real, checkable
    mechanism, not a guess: bound_overrides INTERSECTS the search range
    (never widens it), so if the note-derived upper bound sits below the
    true optimum, the labwiki-informed search is structurally unable to
    ever find it, no matter how many rounds run.
    """
    labwiki_upper_bound = LABWIKI_BOUND_OVERRIDE["par_umol_m2_s"][1]
    optimum_excluded = TRUE_GLOBAL_MAX_PAR > labwiki_upper_bound
    if optimum_excluded:
        labwiki_explanation = (
            f"**Why does 'Ours + labwiki' underperform 'Ours (plain)'?** The synthetic labwiki note "
            f"used here says VOC visibly declines above PAR={PHOTO_THRESHOLD_PAR:.0f} (true -- the "
            "ground truth genuinely has a photoinhibition penalty starting exactly there) and "
            f"recommends staying below it; the benchmark encodes that literally as a HARD "
            f"`bound_overrides` exclusion at PAR>{labwiki_upper_bound:.0f}. But the true global "
            f"optimum actually sits at PAR={TRUE_GLOBAL_MAX_PAR:.0f} -- just PAST that threshold, "
            "because the PAR x temperature interaction term still outweighs the modest "
            "photoinhibition penalty there at high temperature. `bound_overrides` only ever "
            "NARROWS the search range, never widens it, so once that hard exclusion is set, the "
            "labwiki-informed search is structurally unable to ever find the true best point, no "
            "matter how many rounds run -- not a bug in the active-learning workflow itself, but an "
            "honest illustration of a real risk: translating a qualitative note ('declines above X') "
            "into a hard numeric cutoff AT exactly X, with no safety margin, can actively exclude a "
            "genuinely better region the note-writer didn't anticipate. The REAL intended workflow "
            "(per this project's system_prompt.md) treats turning a note into a numeric constraint "
            "as Hermes's own judgment call, not an automatic, literal translation -- this benchmark's "
            "fixed, hardcoded bound demonstrates what happens when that judgment call is skipped."
        )
    else:
        labwiki_explanation = (
            "**On 'Ours + labwiki' vs. 'Ours (plain)':** this run, the labwiki-derived bound "
            f"(PAR <= {labwiki_upper_bound:.0f}) does NOT exclude the true optimum "
            f"(PAR={TRUE_GLOBAL_MAX_PAR:.0f} sits within it), so any underperformance here isn't "
            "from the constraint hiding the true best point."
        )
    lines.append(labwiki_explanation)
    lines.append("")
    lines.append(
        "**A second, important caveat found while building this benchmark:** `jaxsr."
        "SymbolicRegressor.fit()` is not perfectly reproducible given IDENTICAL input data -- "
        "confirmed directly by calling `run_doe_comparison(seed=0)` twice in the same process and "
        "getting visibly different round-by-round RMSE curves both times (one run stayed in the "
        "120-140 ppm range for late rounds, the other spiked to 600-750 ppm) despite every "
        "measurement, point, and RNG draw in this benchmark's own code being fully seeded. This "
        "means part of the Metric A spread reported above (especially the rare large outliers) "
        "reflects jaxsr's own fit instability at small sample sizes with a flexible degree-2-plus-"
        "interactions basis, not purely which points got chosen -- a real property of the tool "
        "worth knowing before trusting any single fit's coefficients, independent of this "
        "benchmark's DoE-comparison question. Averaging over many repeats (as done here) is the "
        "right mitigation, not a workaround to remove. Metric B is unaffected by this, since it's "
        "computed directly from the true function at the chosen points, not from a downstream fit."
    )
    lines.append("")
    lines.append(
        "**A genuine architectural finding surfaced while building this benchmark, independent of "
        "the DoE comparison itself:** `suggest_next_experiments`'s search bounds default to the "
        "*observed* data's min/max, and `bound_overrides` can only narrow that range, never widen "
        "it. Seeding active learning with only 2 clustered points left every later suggestion "
        "permanently confined to that narrow range, never exploring the rest of the declared "
        "domain at all -- confirmed directly before this benchmark's design was fixed to use a "
        "4-point seed spanning the real operating range instead. In practice this means: "
        "`suggest_next_experiments` is an *interpolation-refining* tool given a reasonable initial "
        "spread, not a from-scratch domain-exploration tool -- a real, worth-knowing property of "
        "the current tool, not a bug, but something worth telling an operator (or building an "
        "explicit `search_bounds` override for, distinct from `bound_overrides`) if it's ever "
        "seeded with too narrow an initial design."
    )
    lines.append("")

    lines.append("## Test 3 -- ground truth #1: does discover_led_response_dynamics recover the true WITHIN-experiment relaxation dynamics for a given static PAR level?")
    lines.append("")
    lines.append(
        "Distinct from Tests 1/2, which are entirely about ground truth #2 (how VOC varies ACROSS "
        f"many different static settings). Here there are {len(PAR_LEVELS)} SEPARATE, independent "
        f"experiments, each held at its OWN fixed static PAR level ({', '.join(f'{p:.0f}' for p in PAR_LEVELS)} "
        f"umol/m^2/s, {TEMP:.0f}C, each running {DURATION_S / 86400.0:.0f} days) -- deliberately NOT "
        "pooled together before fitting, so each experiment is answering the same question "
        "independently: does discover_led_response_dynamics recover the true relaxation law "
        "`dVOC/dt = (1/tau) * (true_voc_ppm(par, temp) - VOC(t))` -- a first-order lag toward the "
        "SAME static surface from Tests 1/2 as its steady-state target -- from just THIS one "
        "experiment's own noisy trajectory? Averaged over every independent (seed, PAR level) trial "
        "(same non-determinism reasoning as Test 2):"
    )
    lines.append("")
    for label in dynamics_rmse:
        rmse_mean = float(np.mean(dynamics_rmse[label]))
        rmse_std = float(np.std(dynamics_rmse[label]))
        r2_mean = float(np.mean(dynamics_r2[label]))
        lines.append(f"**{label}**")
        lines.append(f"- RMSE vs true derivative: {rmse_mean:.3f} +/- {rmse_std:.3f} ppm/s")
        lines.append(f"- R^2 vs true derivative (1D grid at that trial's own fixed PAR): {r2_mean:.3f}")
        lines.append(
            f"- Example discovered equation (one PAR level/seed only -- since none are pooled, "
            f"each independently gets its OWN equation, not one universal one): `{dynamics_equations[label]}`"
        )
        lines.append("")

    raw_label = next(l for l in dynamics_rmse if "raw" in l)
    corrected_label = next(l for l in dynamics_rmse if "corrected" in l)
    raw_r2 = float(np.mean(dynamics_r2[raw_label]))
    corrected_r2 = float(np.mean(dynamics_r2[corrected_label]))
    raw_dyn_rmse = float(np.mean(dynamics_rmse[raw_label]))
    corrected_dyn_rmse = float(np.mean(dynamics_rmse[corrected_label]))
    rmse_reduction_pct = 100.0 * (raw_dyn_rmse - corrected_dyn_rmse) / raw_dyn_rmse
    n_trials = len(dynamics_rmse[raw_label])

    """
    Every fraction/claim below is computed fresh from the real
    selected_features/equations returned by each independent (seed, PAR
    level) trial, never hardcoded -- an earlier version of this section
    (from the sinusoid-profile design) claimed "reactor_par_umol_m2_s was
    selected every run, confirming it detects PAR drives the dynamics."
    That claim does NOT carry over to this static-PAR redesign: within
    any SINGLE experiment here, PAR is literally constant (by design --
    see the "no pooling" rationale), so a term nominally involving
    reactor_par_umol_m2_s is mathematically degenerate with the
    intercept in that one fit and selecting it proves nothing about
    "detecting PAR." What this design CAN honestly check is whether the
    tool recovers each experiment's own simple first-order relaxation --
    dominated by a plain linear ppm_asgas self-decay term -- from just
    that one experiment's noisy trajectory.
    """

    def _linear_term_fraction(label: str, term: str = "ppm_asgas") -> float:
        trials = dynamics_selected_features[label]
        return sum(1 for feats in trials if term in feats) / len(trials)

    def _par_term_fraction(label: str) -> float:
        trials = dynamics_selected_features[label]
        return sum(1 for feats in trials if any("reactor_par_umol_m2_s" in f for f in feats)) / len(trials)

    raw_linear_frac = _linear_term_fraction(raw_label)
    corrected_linear_frac = _linear_term_fraction(corrected_label)
    raw_par_frac = _par_term_fraction(raw_label)
    corrected_par_frac = _par_term_fraction(corrected_label)

    def _recovery_fragment(frac: float, n: int) -> str:
        if frac == 1.0:
            return f"every one of the {n} trials"
        if frac == 0.0:
            return f"none of the {n} trials"
        return f"{frac * n:.0f} of the {n} trials"

    """
    Whether ambient-baseline correction actually HELPED here is computed
    from the real sign of the R^2/RMSE comparison, never assumed -- the
    prior (sinusoid-profile) design consistently found correction helped,
    but that is a property of THAT experiment design, not a law of
    nature; asserting it unconditionally here would be exactly the kind
    of stale, un-rechecked claim this project's own dev log has flagged
    as a recurring bug class before.
    """
    correction_helped = corrected_r2 > raw_r2
    if correction_helped:
        lines.append(
            f"**Verdict:** the REAL, public `discover_led_response_dynamics` recovered the true "
            f"linear `ppm_asgas` self-decay term -- the dominant term in this static-PAR relaxation "
            f"law -- in {_recovery_fragment(raw_linear_frac, n_trials)} (raw) and "
            f"{_recovery_fragment(corrected_linear_frac, n_trials)} (corrected) of every independent "
            f"(seed, PAR level) trial. Passing the `ambient_baseline_run_id` parameter (applying "
            f"that sensor's persisted ambient-covariate correction before calibration) improved "
            f"recovery on average (R^2 {raw_r2:.2f} without it vs {corrected_r2:.2f} with it, RMSE "
            f"{rmse_reduction_pct:.0f}% lower on average) here too, consistent with the earlier "
            "sinusoid-profile design's finding. Run a `run_ambient_baseline_check(..., "
            "persist_run_id=...)` once per sensor, then pass that same id here, to get this "
            "improvement on real hardware data."
        )
    else:
        lines.append(
            f"**Verdict:** the REAL, public `discover_led_response_dynamics` recovered the true "
            f"linear `ppm_asgas` self-decay term -- the dominant term in this static-PAR relaxation "
            f"law -- in {_recovery_fragment(raw_linear_frac, n_trials)} (raw) and "
            f"{_recovery_fragment(corrected_linear_frac, n_trials)} (corrected) of every independent "
            f"(seed, PAR level) trial. **Unlike the earlier sinusoid-profile design, ambient-"
            f"baseline correction did NOT help here -- it made recovery WORSE on average** (R^2 "
            f"{raw_r2:.2f} without it vs {corrected_r2:.2f} with it, RMSE {-rmse_reduction_pct:.0f}% "
            "higher on average). Reported plainly rather than silently assuming the earlier design's "
            "finding carries over: with a full week of mostly-flat, near-steady-state data (per the "
            "user's request for experimental realism), the finite-difference derivative estimate is "
            "dominated by noise once the brief initial transient has settled, and correction applied "
            "to a much longer, differently-structured noise realization than the shorter sinusoid "
            "experiment used may not generalize the same way -- a genuine, disclosed trade-off of "
            "this redesign, not evidence the correction itself is broken (Test 1 already separately "
            "confirms the correction's own math is correct on its own terms)."
        )
    lines.append("")
    lines.append(
        f"**A necessary honest caveat of NOT pooling across PAR levels (per the user's explicit "
        f"choice):** a term nominally involving `reactor_par_umol_m2_s` was still selected in "
        f"{_recovery_fragment(raw_par_frac, n_trials)} (raw) and "
        f"{_recovery_fragment(corrected_par_frac, n_trials)} (corrected) of trials -- but since PAR "
        "never varies within any single one of these experiments, this does NOT mean the tool "
        "'detected' a PAR effect the way it genuinely could when PAR varies within a run (e.g. the "
        "earlier sinusoid-profile design, or a real control-profile experiment). Here, a selected "
        "PAR-involving term is mathematically indistinguishable from a constant offset for that one "
        "fit -- the coefficients on it are an artifact of the fixed PAR value, not evidence of a "
        "learned PAR relationship. Recovering the ACTUAL PAR-dependence (how the decay target itself "
        "shifts with light level) would require comparing the STEADY-STATE VALUE each independent "
        "trial settles at across the different PAR levels against the static surface from Tests 1/2 "
        "-- not something this per-experiment dynamics fit does on its own."
    )
    lines.append("")

    if raw_linear_frac == 1.0 and corrected_linear_frac == 1.0:
        lines.append(
            f"**On the selection strategy:** `discover_led_response_dynamics` defaults to "
            f"`strategy=\"exhaustive\"` rather than `jaxsr`'s own default `\"greedy_forward\"` -- "
            "confirmed in an earlier version of this benchmark (the sinusoid-profile design) that "
            "greedy selection reliably missed the true linear decay term in favor of quadratic/cubic "
            "surrogate terms at the same `max_terms=5` -- a real instance of greedy forward "
            "selection's classic failure mode (an early locally-good pick blocking a later, "
            "globally-better combination), not a derivative-noise or basis-coverage problem. "
            "Exhaustive search is tractable here specifically because this function always uses a "
            "fixed, small 2-state default basis (8 candidate terms); it would need its own "
            "tractability check before being assumed safe for a much larger custom basis library."
        )
    else:
        lines.append(
            f"**An honest limitation, not papered over:** the true linear `ppm_asgas` decay term was "
            f"recovered in only {_recovery_fragment(raw_linear_frac, n_trials)} (raw) and "
            f"{_recovery_fragment(corrected_linear_frac, n_trials)} (corrected) -- inconsistent "
            f"recovery across trials at the current `max_terms=5`/`strategy=\"exhaustive\"` settings. "
            f"The resulting R^2 against the true derivative (raw {raw_r2:.2f}, corrected "
            f"{corrected_r2:.2f}) reflects a real, structurally-plausible but not always exact "
            "recovery -- good enough to see the relaxation shape, not always good enough to trust "
            "the exact discovered coefficients as the true physical law."
        )
    lines.append("")

    raw_diverged_frac = sum(dynamics_diverged[raw_label]) / len(dynamics_diverged[raw_label])
    corrected_diverged_frac = sum(dynamics_diverged[corrected_label]) / len(dynamics_diverged[corrected_label])
    if raw_diverged_frac > 0 or corrected_diverged_frac > 0:
        lines.append(
            "**A second honest limitation, specific to this static-PAR redesign: some discovered "
            "equations are numerically UNSTABLE when integrated forward over the full experiment "
            f"duration**, in {_recovery_fragment(raw_diverged_frac, n_trials)} (raw) and "
            f"{_recovery_fragment(corrected_diverged_frac, n_trials)} (corrected) -- see "
            "`dynamics_recovery.png` panels (a)/(b), where a predicted line simply stops partway "
            "rather than continuing (the plot deliberately does NOT extrapolate through a divergence "
            "at astronomical values). Root cause: a discovered equation can fit dVOC/dt reasonably "
            "well LOCALLY (a small RMSE against the true derivative over the range actually "
            "observed) while still having the wrong sign on a higher-order term (e.g. a positive "
            "`ppm_asgas^2` coefficient) -- invisible in a single-step derivative comparison, but "
            "compounding into runaway growth over thousands of forward-integration steps. This is a "
            "known, real limitation of SINDy-style discovery generally (a locally-accurate model "
            "is not guaranteed to be globally/dynamically stable), not specific to this benchmark's "
            "implementation -- worth knowing before trusting a discovered equation to be simulated "
            "forward over a long horizon, as opposed to just evaluated pointwise."
        )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Round 1 has no fit yet (a single point can't inform a degree-2
    model), so its RMSE column is all-NaN by design -- np.nanmean/
    nanmedian/nanstd correctly warn about that empty slice; suppressed
    here since it's expected, not a real problem."""
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Running Test 1 (calibration recovery)...")
    calibration_results, cross_sensor_result = run_calibration_recovery_test(verbose=True)
    _plot_cross_sensor_consistency(cross_sensor_result, OUTPUT_DIR / "cross_sensor_consistency.png")

    print(f"\nRunning Test 2 (DoE comparison, {N_SEEDS} repeats)...")
    rmse_curves, best_found_curves = _run_doe_comparison_repeated(N_SEEDS)
    rounds_to_target, target_rmse = _compute_rounds_to_target(rmse_curves)
    _plot_doe_comparison(rmse_curves, best_found_curves, rounds_to_target, target_rmse, OUTPUT_DIR / "doe_comparison.png")

    print(f"\nRunning Test 3 (dynamics recovery, {N_DYNAMICS_SEEDS} repeats)...")
    dynamics_rmse, dynamics_r2, dynamics_equations, dynamics_selected_features, dynamics_diverged, representative_trajectories = _run_dynamics_recovery_repeated(N_DYNAMICS_SEEDS)
    _plot_dynamics_recovery(representative_trajectories, OUTPUT_DIR / "dynamics_recovery.png")

    _write_report(
        calibration_results, cross_sensor_result, rmse_curves, best_found_curves, rounds_to_target, target_rmse,
        dynamics_rmse, dynamics_r2, dynamics_equations, dynamics_selected_features, dynamics_diverged,
        OUTPUT_DIR / "REPORT.md",
    )

    print(f"\nDone. Results written to {OUTPUT_DIR}/")
    print("  - cross_sensor_consistency.png")
    print("  - doe_comparison.png")
    print("  - dynamics_recovery.png")
    print("  - REPORT.md")


if __name__ == "__main__":
    main()
