"""How many experiments does each method need before it discovers the
true equation?

Every method is trying to find the same relationship, which is the one
`ground_truth.true_voc_ppm` implements:

    VOC = baseline
        + a light response that saturates
        + a temperature effect
        + a light x temperature interaction
        - a penalty once the light gets too strong

Each method chooses experiments its own way, runs them, and fits an
equation from a fixed menu of candidate building blocks. The menu holds
the true shapes plus three plausible-but-wrong decoys, and every method
fits the identical menu -- so expressiveness is a constant and the only
thing that varies is which experiments got run.

"Discovered" is judged strictly: the fit must select exactly the terms the
truth is made of, with no decoy and no spurious extra. A fit carrying a
wrong term is not the true relationship, however well it predicts.

There is no experiment budget. Each method keeps running experiments until
it discovers the structure and holds it, and the number it took is the
answer. A hard cap exists only so a method that never converges cannot run
forever; those are reported as not converged rather than quietly dropped.

Run:  python packages/jaxsr-calibration/benchmarks/discovery_speed.py
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jaxsr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from doe_methods import (
    DECLARED_SEARCH_BOUNDS,
    SEED_POINTS,
    TRAIN_PAR_BOUNDS,
    latin_hypercube_points,
    random_points,
    run_active_learning_campaign,
    sobol_points,
)
from ground_truth import (
    K_M,
    PHOTO_THRESHOLD_PAR,
    TEMP_BOUNDS,
    TEMP_REF,
    true_voc_ppm,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

MEASUREMENT_NOISE_PPM = 1.2
"""The residual noise Part 1 demonstrates the real correction pipeline
actually delivers, so this test operates on data of exactly the quality
that pipeline is measured to produce."""

MAX_EXPERIMENTS = 45
"""Not a budget -- a stop so a non-converging method terminates. Methods
that reach it are reported as not converged."""

MIN_EXPERIMENTS = 8
"""The truth needs five terms; fewer points than this cannot determine
them, so starting lower would only burn time fitting hopeless cases."""

HOLD_ROUNDS = 2
"""Discovery must survive one more experiment before it counts. jaxsr's
fit is not perfectly reproducible even on identical data, so a single
correct fit can be luck; requiring it to hold measures convergence
instead."""

PAR_MID = 220.0

MAX_TERMS = 6
"""One slot more than the truth needs. Enough rope to add a wrong term --
which is exactly what strict discovery is meant to catch."""


TRUE_TERM_SET = frozenset({"1", "sat_par", "x1", "par_x_dtemp", "photo_inhib"})
"""The truth, written in the basis it is actually fitted in: a constant, a
saturating light term, a linear temperature term, the light x temperature
interaction, and the photoinhibition penalty."""


def build_basis_library() -> jaxsr.BasisLibrary:
    """Feature 0 is light (PAR), feature 1 is temperature."""

    """
    Two things here were got wrong first and fixed on evidence, both of
    which would have made the test measure the menu rather than the
    methods.

    The interaction term is `par * (temp - TEMP_REF)`, matching how the
    truth is actually written. An earlier version used plain `par * temp`,
    which is the same thing plus a multiple of linear PAR -- so the truth
    could only be expressed by ALSO selecting linear PAR, and the "true
    term set" silently gained a term that is not in the truth.

    The decoys are centred. Raw `temp^2` correlates 0.996 with plain
    temperature across a 20-40 C range and raw `temp/(20+temp)` correlates
    0.995 -- neither is a decoy, both are near-duplicates of a term the
    truth genuinely contains, so no method could ever have avoided them
    and strict discovery would have been impossible by construction.
    Measured, not assumed: the centred replacements sit at 0.61 and 0.46.

    Linear PAR is deliberately KEPT despite correlating 0.95 with the true
    saturating term. That one is not an artifact -- it is the real
    experimental question. A campaign that samples a narrow light band
    genuinely cannot tell a saturating response from a straight line, and
    only one that spans the range can. Selecting it instead counts as not
    discovered, which is the honest verdict.
    """
    return (
        jaxsr.BasisLibrary(n_features=2)
        .add_constant()
        .add_linear()
        .add_custom("sat_par", lambda X: X[:, 0] / (K_M + X[:, 0]), complexity=3, feature_indices=(0,))
        .add_custom(
            "photo_inhib",
            lambda X: np.maximum(X[:, 0] - PHOTO_THRESHOLD_PAR, 0.0) ** 2,
            complexity=4,
            feature_indices=(0,),
        )
        .add_custom(
            "par_x_dtemp", lambda X: X[:, 0] * (X[:, 1] - TEMP_REF), complexity=3, feature_indices=(0, 1)
        )
        .add_custom("par_hump", lambda X: (X[:, 0] - PAR_MID) ** 2, complexity=3, feature_indices=(0,))
        .add_custom("temp_hump", lambda X: (X[:, 1] - TEMP_REF) ** 2, complexity=3, feature_indices=(1,))
        .add_custom("inv_par", lambda X: 1.0 / (1.0 + X[:, 0]), complexity=3, feature_indices=(0,))
    )


STAGE_SWITCH_ROUND = 8
"""When a staged policy stops asking "which form is right" and starts
asking "how sharp are its coefficients". Early on several forms still fit
the handful of points equally well, so discriminating between them is the
binding uncertainty; once one has emerged, sharpening it is."""


def _staged(round_num: int) -> str:
    return "model_discrimination" if round_num < STAGE_SWITCH_ROUND else "d_optimal"


def _staged_reversed(round_num: int) -> str:
    """The opposite order, and arguably the better-motivated one.

    Discriminating first assumes several candidate forms are worth telling
    apart -- but after a handful of points every candidate is poor, so
    "where do they disagree" is mostly noise. Sharpening first buys
    coverage and sane coefficients, which is what makes the later
    discrimination question meaningful.
    """
    return "d_optimal" if round_num < STAGE_SWITCH_ROUND else "model_discrimination"


ACQUISITION_BY_METHOD = {
    "UCB": "ucb",
    "D-optimal": "d_optimal",
    "Model-discrimination": "model_discrimination",
    "Blend (D-opt + discrim)": "0.5*d_optimal+0.5*model_discrimination",
    "Staged (discrim then D-opt)": _staged,
    "Staged (D-opt then discrim)": _staged_reversed,
}
"""Every 'ours' variant is run twice, with and without labwiki, so the
acquisition and the knowledge base can be told apart. The blend is
normalised before weighting -- raw D-optimal and discrimination scores
differ by orders of magnitude, so an unnormalised 50/50 would silently be
whichever component happens to be larger."""

CLASSICAL_METHODS = ["Latin Hypercube", "Sobol", "Grid", "Random"]

ADAPTIVE_METHODS = {
    f"{name}{suffix}"
    for name in ACQUISITION_BY_METHOD
    for suffix in ("", " + labwiki")
}

METHOD_NAMES = CLASSICAL_METHODS + [
    f"{name}{suffix}" for name in ACQUISITION_BY_METHOD for suffix in ("", " + labwiki")
]

LABWIKI_NOTE_ROUND = 2
LABWIKI_NOTE_TEXT = (
    "Prior campaigns: VOC output falls off above roughly 380 umol/m^2/s "
    "(photoinhibition). Very low light (<40) produced negligible VOC."
)
LABWIKI_BOUND_OVERRIDE = {"par_umol_m2_s": (40.0, 380.0)}


SEED_VARIANTS = {
    "clustered": [(190.0, 28.0), (230.0, 28.0), (190.0, 31.0), (230.0, 31.0)],
    "spread": [(40.0, 22.0), (400.0, 22.0), (40.0, 38.0), (400.0, 38.0)],
    "latin": [(87.5, 27.5), (182.5, 37.5), (277.5, 22.5), (372.5, 32.5)],
}
"""How a campaign's first four runs are laid out before any method takes
over. The clustered variant is what a cautious operator actually does --
a few runs near a setting they already trust. The spread variant is four
runs at the corners of the declared envelope.

The `latin` variant is the hypothesis this study exists to test: four
points that span both factors like `spread` does, but placed on a Latin
square so they cover the INTERIOR rather than sitting on the corners.
Corners tell you the extremes and nothing about the shape between them,
which is a plausible reason the corners-only seed left D-optimal
converging in only half its runs while Latin Hypercube and Sobol -- which
cover the interior by construction -- converged every time.

This is not a cosmetic difference for the adaptive methods. Their search
range defaults to the range of data observed so far, so a clustered seed
boxes them in at PAR 190-230 until something explicitly widens it, while
a spread seed hands them the full range from the start -- which may make
labwiki's declared bounds redundant rather than essential.
"""


def set_seed_points(variant: str) -> None:
    """Point every method at the same starting layout.

    Patches the shared module the campaign runner reads too, so the
    adaptive and fixed designs cannot silently disagree about where the
    campaign began -- which would confound the seed with the method.
    """
    import doe_methods

    points = SEED_VARIANTS[variant]
    doe_methods.SEED_POINTS[:] = points
    global SEED_POINTS
    SEED_POINTS = doe_methods.SEED_POINTS


def measure(par: float, temp: float, rng: np.random.Generator) -> float:
    """One experiment: run this condition, get back its VOC plateau."""
    return float(true_voc_ppm(par, temp) + rng.normal(0.0, MEASUREMENT_NOISE_PPM))


def _fit(points: np.ndarray, values: np.ndarray):
    model = jaxsr.SymbolicRegressor(basis_library=build_basis_library(), max_terms=MAX_TERMS)
    model.fit(points, values)
    return model


def _discovered(model) -> bool:
    """Strict: exactly the true terms, nothing else."""
    return frozenset(getattr(model, "selected_features_", ())) == TRUE_TERM_SET


def _found_all_true(model) -> bool:
    """The forgiving reading, tracked alongside: every true term present,
    decoys tolerated. Reported so a method that finds the real physics but
    keeps picking up junk is distinguishable from one that finds neither."""
    return TRUE_TERM_SET <= frozenset(getattr(model, "selected_features_", ()))


def _grid_design(n_extra: int) -> np.ndarray:
    """A full-factorial grid sized for the budget available.

    `doe_methods.grid_points` is fixed at the 3x2 layout Test 2's budget
    called for, but a run-until-converged comparison has to ask for a grid
    of any size. The factorisation is kept as square as possible, which is
    what an experimentalist laying out a two-factor grid would do: spend
    the runs evenly on both factors rather than resolving one finely and
    the other barely at all.
    """
    rows = max(2, int(round(np.sqrt(n_extra))))
    cols = max(2, int(np.ceil(n_extra / rows)))
    par = np.linspace(TRAIN_PAR_BOUNDS[0], TRAIN_PAR_BOUNDS[1], cols)
    temp = np.linspace(TEMP_BOUNDS[0], TEMP_BOUNDS[1], rows)
    pp, tt = np.meshgrid(par, temp)
    return np.column_stack([pp.ravel(), tt.ravel()])[:n_extra]


def _fixed_design(method: str, n: int, seed: int) -> np.ndarray:
    """A non-adaptive design of size n.

    Regenerated per size rather than extended, because that is genuinely
    how these are used: you commit to how many runs you can afford, lay
    out the design, and run it. So "how many experiments did Latin
    Hypercube need" means the smallest design that discovers the truth.
    """
    n_extra = max(n - len(SEED_POINTS), 1)
    if method == "Latin Hypercube":
        extra = latin_hypercube_points(n_extra, seed=seed)
    elif method == "Sobol":
        extra = sobol_points(n_extra, seed=seed)
    elif method == "Grid":
        extra = _grid_design(n_extra)
    elif method == "Random":
        extra = random_points(n_extra, seed=seed)
    else:
        raise ValueError(f"unknown fixed design {method!r}")
    return np.vstack([np.array(SEED_POINTS, dtype=float), np.asarray(extra, dtype=float)])


def _adaptive_sequence(method: str, seed: int, rng: np.random.Generator) -> np.ndarray:
    """Run the real active-learning campaign once, to the cap, and keep the
    order it chose. Evaluating prefixes of that order is equivalent to
    stopping it early, and costs one campaign instead of fifty."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        wiki_root = data_dir / "labwiki"

        def measure_fn(par: float, temp: float) -> tuple[float, float]:
            """The campaign runner records a plateau and a relaxation time;
            only the plateau matters here, so tau is passed through as a
            constant rather than simulated."""
            return measure(par, temp, rng), 12.0

        with_labwiki = method.endswith(" + labwiki")
        base = method[: -len(" + labwiki")] if with_labwiki else method
        points = run_active_learning_campaign(
            measure_fn,
            n_extra=MAX_EXPERIMENTS - len(SEED_POINTS),
            data_dir=data_dir,
            campaign_id=f"disc_{abs(hash(method)) % 9999}_{seed}",
            use_labwiki=with_labwiki,
            wiki_root=wiki_root,
            labwiki_note_round=LABWIKI_NOTE_ROUND if with_labwiki else None,
            labwiki_note_text=LABWIKI_NOTE_TEXT,
            search_bounds=DECLARED_SEARCH_BOUNDS if with_labwiki else None,
            acquisition=ACQUISITION_BY_METHOD[base],
        )
    return np.array(points, dtype=float)


@dataclass
class MethodResult:
    method: str
    experiments_to_discovery: int | None
    experiments_to_all_true: int | None
    surface_rmse_at_end: float
    extrapolation_rmse_at_end: float
    discovered_by: dict[int, bool] = field(default_factory=dict)


def _dense_grid(par_lo: float, par_hi: float, n: int = 40) -> np.ndarray:
    par = np.linspace(par_lo, par_hi, n)
    temp = np.linspace(TEMP_BOUNDS[0], TEMP_BOUNDS[1], n)
    pp, tt = np.meshgrid(par, temp)
    return np.column_stack([pp.ravel(), tt.ravel()])


def _prediction_errors(model) -> tuple[float, float]:
    """The two secondary questions: can it predict across the range it was
    trained on, and does it still hold in a high-light band deliberately
    withheld from every method."""
    inside = _dense_grid(*TRAIN_PAR_BOUNDS)
    outside = _dense_grid(TRAIN_PAR_BOUNDS[1], 500.0)

    def rmse(X: np.ndarray) -> float:
        truth = true_voc_ppm(X[:, 0], X[:, 1])
        return float(np.sqrt(np.mean((np.asarray(model.predict(X)).ravel() - truth) ** 2)))

    return rmse(inside), rmse(outside)


def run_method(method: str, seed: int, verbose: bool = True) -> MethodResult:
    rng = np.random.default_rng(seed)

    sequence = (
        _adaptive_sequence(method, seed, rng) if method in ADAPTIVE_METHODS else None
    )

    discovery_n: int | None = None
    all_true_n: int | None = None
    consecutive = 0
    consecutive_all_true = 0
    discovered_by: dict[int, bool] = {}
    model = None

    for n in range(MIN_EXPERIMENTS, MAX_EXPERIMENTS + 1):
        if sequence is not None:
            points = sequence[:n]
        else:
            points = _fixed_design(method, n, seed)
        values = np.array([measure(p, t, rng) for p, t in points])

        model = _fit(points, values)
        hit = _discovered(model)
        discovered_by[n] = hit

        consecutive = consecutive + 1 if hit else 0
        if discovery_n is None and consecutive >= HOLD_ROUNDS:
            discovery_n = n - HOLD_ROUNDS + 1

        consecutive_all_true = consecutive_all_true + 1 if _found_all_true(model) else 0
        if all_true_n is None and consecutive_all_true >= HOLD_ROUNDS:
            all_true_n = n - HOLD_ROUNDS + 1

        if discovery_n is not None:
            break

    surface, extrapolation = _prediction_errors(model)
    if verbose:
        found = f"{discovery_n}" if discovery_n else f">{MAX_EXPERIMENTS}"
        print(
            f"  {method:32s} discovered at {found:>4s} experiments   "
            f"(all true terms at {all_true_n or '-'})"
        )

    return MethodResult(
        method=method,
        experiments_to_discovery=discovery_n,
        experiments_to_all_true=all_true_n,
        surface_rmse_at_end=surface,
        extrapolation_rmse_at_end=extrapolation,
        discovered_by=discovered_by,
    )


def run(seeds: int = 8, verbose: bool = True) -> dict[str, list[MethodResult]]:
    results: dict[str, list[MethodResult]] = {m: [] for m in METHOD_NAMES}
    for seed in range(seeds):
        if verbose:
            print(f"\nseed {seed}:")
        for method in METHOD_NAMES:
            results[method].append(run_method(method, seed, verbose=verbose))
    _save_raw(results)
    _report(results, seeds)
    _plot(results, seeds)
    return results


def _save_raw(results: dict[str, list[MethodResult]]) -> None:
    """Write every individual run's outcome, not just the summary.

    Learned the hard way: a head-to-head run was launched with verbose
    off, which suppressed the per-seed lines, and the summary alone cannot
    say whether a two-experiment gap between two methods is real. The
    spread is the whole answer when the differences are this small, so it
    gets persisted rather than printed.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["method,seed,experiments_to_discovery,surface_rmse,extrapolation_rmse"]
    for method, runs in results.items():
        for seed, run in enumerate(runs):
            found = "" if run.experiments_to_discovery is None else run.experiments_to_discovery
            lines.append(
                f"{method},{seed},{found},{run.surface_rmse_at_end:.4f},{run.extrapolation_rmse_at_end:.4f}"
            )
    (RESULTS_DIR / "discovery_speed_runs.csv").write_text("\n".join(lines) + "\n")


def _summarise(runs: list[MethodResult]) -> tuple[float | None, int, int]:
    """Median experiments to discovery over the runs that converged, plus
    how many did. The median deliberately ignores non-converging runs --
    they have no number -- so the count is reported beside it and neither
    is readable alone."""
    hits = [r.experiments_to_discovery for r in runs if r.experiments_to_discovery is not None]
    median = float(np.median(hits)) if hits else None
    return median, len(hits), len(runs)


def _report(results: dict[str, list[MethodResult]], seeds: int) -> None:
    print("\n" + "=" * 78)
    print(f"Experiments needed to discover the true equation ({seeds} repeats)")
    print("=" * 78)
    ordered = sorted(
        list(results),
        key=lambda m: (_summarise(results[m])[0] is None, _summarise(results[m])[0] or 1e9),
    )
    for method in ordered:
        median, n_hit, n_total = _summarise(results[method])
        runs = results[method]
        found = [r.experiments_to_discovery for r in runs if r.experiments_to_discovery is not None]
        spread = (
            f"sd {float(np.std(found, ddof=1)):4.1f}" if len(found) > 1 else "sd    -"
        )
        surface = float(np.median([r.surface_rmse_at_end for r in runs]))
        extrap = float(np.median([r.extrapolation_rmse_at_end for r in runs]))
        shown = f"{median:.0f}" if median is not None else "never"
        print(
            f"  {method:40s} med {shown:>5s}  mean {np.mean(found) if found else float('nan'):5.1f}  "
            f"{spread}  converged {n_hit}/{n_total}   "
            f"surface {surface:5.2f}   extrap {extrap:6.2f} ppm"
        )


CLASSICAL_COLOURS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]
ACQUISITION_COLOURS = ["#9467bd", "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
CLASSICAL_MARKERS = ["o", "s", "^", "D"]
ACQUISITION_MARKERS = ["v", "P", "X", "*", "h", "<"]

"""
Colour carries the METHOD and line style carries whether labwiki was
used -- solid with, dotted without. Built programmatically rather than as
a hand-written list, because the last hand-written one silently ran two
entries short of the method list and only failed once the list grew.
"""
STYLE = {}
for index, name in enumerate(CLASSICAL_METHODS):
    STYLE[name] = {
        "color": CLASSICAL_COLOURS[index],
        "linestyle": "-",
        "marker": CLASSICAL_MARKERS[index],
    }
for index, name in enumerate(ACQUISITION_BY_METHOD):
    for suffix, style in ((" + labwiki", "-"), ("", ":")):
        STYLE[f"{name}{suffix}"] = {
            "color": ACQUISITION_COLOURS[index],
            "linestyle": style,
            "marker": ACQUISITION_MARKERS[index],
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


def _plot(results: dict[str, list[MethodResult]], seeds: int) -> None:
    _apply_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    """
    The headline: what fraction of repeats had discovered the truth by
    each number of experiments. A survival curve rather than a bar chart
    because it shows speed AND reliability at once -- a method that always
    gets there at 30 experiments and one that gets there at 15 half the
    time are genuinely different, and a single median hides that.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(MIN_EXPERIMENTS, MAX_EXPERIMENTS + 1)
    for method in METHOD_NAMES:
        found = [r.experiments_to_discovery for r in results[method]]
        fraction = [np.mean([f is not None and f <= x for f in found]) for x in xs]
        ax.plot(xs, fraction, label=method, linewidth=2.0, markevery=6, markersize=7, **STYLE[method])
    ax.set_xlabel("Experiments run")
    ax.set_ylabel("Fraction of repeats that discovered the true equation")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Speed to discovering the true VOC(light, temperature) equation ({seeds} repeats)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "discovery_speed.png", bbox_inches="tight")
    plt.close(fig)

    fig, (ax_surface, ax_extrap) = plt.subplots(1, 2, figsize=(14, 5.5))
    positions = np.arange(len(METHOD_NAMES))
    for ax, attr, title, ylabel in (
        (ax_surface, "surface_rmse_at_end", "Can it predict?", "RMSE vs truth (ppm)"),
        (
            ax_extrap,
            "extrapolation_rmse_at_end",
            "Does it hold outside the tested light range?",
            "RMSE on withheld band (ppm)",
        ),
    ):
        values = [float(np.median([getattr(r, attr) for r in results[m]])) for m in METHOD_NAMES]
        ax.bar(positions, values, color=[STYLE[m]["color"] for m in METHOD_NAMES], width=0.65)
        ax.set_xticks(positions)
        ax.set_xticklabels(METHOD_NAMES, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle(
        "Prediction quality at the point each method stopped -- the secondary questions",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "discovery_prediction_quality.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots written to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
