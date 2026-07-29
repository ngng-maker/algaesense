"""Does adaptive experiment design earn its keep once the space is big?

Every two-factor result says no: adaptive design matched Sobol on speed
across three different seed layouts, and only ever won on extrapolation.
The standing explanation is that two factors is too small a space for the
CHOICE of points to matter -- covering it evenly is nearly optimal, and
covering evenly is precisely what a space-filling design does for free.

This is the test of that. Four factors: light, temperature, pH, nitrate.
Covering four dimensions evenly is not cheap, so if choosing points is
ever worth anything, it should be worth something here.

Deliberately self-contained rather than a generalisation of
`doe_methods`: that module is hardcoded to two factors and three other
benchmarks depend on its exact behaviour, so widening it in place would
risk their results to save duplicating one loop. The loop below still
calls the REAL `suggest_next_experiments`, which is the part that matters.

See FOUR_FACTOR_NOTES.md for what is already established and the traps
this benchmark family has already hit.

Run:  python packages/jaxsr-calibration/benchmarks/discovery_speed_4d.py
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
import polars as pl
from scipy.stats import qmc

from algaesense_agent.mcp_pipeline.pipeline import suggest_next_experiments
from ground_truth import (
    K_M,
    NUTRIENT_BOUNDS,
    NUTRIENT_K_M,
    PH_BOUNDS,
    PH_OPT,
    HALDANE_K_I,
    TEMP_BOUNDS,
    TEMP_REF,
    VMAX,
    haldane_light,
    true_voc_ppm_4d_smooth,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"

FACTORS = ["par_umol_m2_s", "mean_sample_t_c", "mean_ph", "mean_nitrate_mg_l"]
BOUNDS = [(20.0, 500.0), TEMP_BOUNDS, PH_BOUNDS, NUTRIENT_BOUNDS]
TARGET_COLUMN = "mean_voc_ppm_asgas"

MEASUREMENT_NOISE_PPM = 1.2
"""The residual noise Part 1 measures the real correction pipeline
delivering, so this operates on data of the quality that pipeline
actually produces."""

MIN_EXPERIMENTS = 20
MAX_EXPERIMENTS = 180
EVAL_STEP = 4
"""Discovery is checked every fourth experiment rather than every one.
At ~150 experiments a resolution of four is immaterial, and it cuts the
scoring fits by 4x -- which is the difference between this run taking
hours and taking most of a day."""

HOLD_ROUNDS = 2
MAX_TERMS = 8
"""One slot more than the seven the truth needs, so a wrong term is
possible and strict discovery can catch it."""

PAR_MID = 220.0
NUT_MID = 15.0

TRUE_TERM_SET = frozenset(
    {"1", "haldane_par", "x1", "par_x_dtemp", "ph_hump", "sat_nut"}
)


def build_basis_library() -> jaxsr.BasisLibrary:
    """Features in FACTORS order: light, temperature, pH, nitrate.

    Every distractor here was screened for collinearity against the true
    terms over the real operating ranges before being included -- see
    FOUR_FACTOR_NOTES.md. `ph_x_temp` (0.995 against linear temperature)
    and `log_nut` (0.986 against the true nitrate term) were rejected as
    near-duplicates rather than distractors: a collinear decoy makes
    strict discovery impossible by construction and fails silently, by
    making every method look equally bad.

    Linear PAR is kept despite correlating 0.953 with the saturating light
    term, for the same reason as the two-factor study: telling a
    saturating response from a straight line is the genuine experimental
    question, and a campaign that never samples low light cannot answer
    it.
    """
    return (
        jaxsr.BasisLibrary(n_features=4)
        .add_constant()
        .add_linear()
        .add_custom(
            "haldane_par",
            lambda X: X[:, 0] / (K_M + X[:, 0] + X[:, 0] ** 2 / HALDANE_K_I),
            complexity=4,
            feature_indices=(0,),
        )
        .add_custom("sat_par", lambda X: X[:, 0] / (K_M + X[:, 0]), complexity=3, feature_indices=(0,))
        .add_custom(
            "par_x_dtemp", lambda X: X[:, 0] * (X[:, 1] - TEMP_REF), complexity=3, feature_indices=(0, 1)
        )
        .add_custom("ph_hump", lambda X: (X[:, 2] - PH_OPT) ** 2, complexity=3, feature_indices=(2,))
        .add_custom(
            "sat_nut", lambda X: X[:, 3] / (NUTRIENT_K_M + X[:, 3]), complexity=3, feature_indices=(3,)
        )
        .add_custom("par_hump", lambda X: (X[:, 0] - PAR_MID) ** 2, complexity=3, feature_indices=(0,))
        .add_custom("temp_hump", lambda X: (X[:, 1] - TEMP_REF) ** 2, complexity=3, feature_indices=(1,))
        .add_custom("nut_hump", lambda X: (X[:, 3] - NUT_MID) ** 2, complexity=3, feature_indices=(3,))
        .add_custom("inv_par", lambda X: 1.0 / (1.0 + X[:, 0]), complexity=3, feature_indices=(0,))
    )


N_SEED = 8
"""Four factors need more than the two-factor study's four starting runs
before a fit means anything."""


def seed_points() -> np.ndarray:
    """A Latin-hypercube seed across all four factors.

    Latin rather than clustered or corners, on evidence: the two-factor
    study found a corners-only seed halved the adaptive convergence rate
    (12/24 against 24/24) because corner points describe the extremes and
    say nothing about the shape between them, while a clustered seed
    confines the search range outright.
    """
    sample = qmc.LatinHypercube(d=4, seed=12345).random(N_SEED)
    return qmc.scale(sample, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])


DECLARED_SEARCH_BOUNDS = {name: bounds for name, bounds in zip(FACTORS, BOUNDS)}
"""What labwiki supplies: the known safe operating envelope. Always given
to the adaptive arms -- the two-factor study found 0/24 convergence
without it, under every seed layout tried."""


def measure(point: np.ndarray, rng: np.random.Generator) -> float:
    return float(true_voc_ppm_4d_smooth(*point) + rng.normal(0.0, MEASUREMENT_NOISE_PPM))


def _write_row(data_dir: Path, campaign_id: str, index: int, point: np.ndarray, value: float) -> None:
    campaign_dir = data_dir / "derived" / "features" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"exp_{index:03d}"
    row = {
        "experiment_id": experiment_id,
        "campaign_id": campaign_id,
        "reactor_id": "R01",
        "sensor_id": "PID01",
        TARGET_COLUMN: value,
        **{name: float(v) for name, v in zip(FACTORS, point)},
    }
    pl.DataFrame([row]).write_parquet(campaign_dir / f"{experiment_id}.parquet")


SOBOL_WARMUP = 24
"""How many experiments the hybrid arm spends on Sobol before handing over
to the active learner.

Aimed squarely at the mechanism the two-factor study identified: D-optimal
chooses the point that best sharpens the model it CURRENTLY believes, so
while that model is still wrong it is sharpening the wrong thing. A
space-filling warm-up buys a model worth optimising against before
adaptivity starts. 24 is three times the shared seed and well short of the
~100 where discovery becomes reachable, so the handover happens while
there is still most of the campaign left for adaptivity to matter."""


def adaptive_sequence(
    acquisition: str, seed: int, rng: np.random.Generator, sobol_warmup: int = 0
) -> np.ndarray:
    """Run the real active-learning loop to the cap and keep its order.

    Evaluating prefixes of that order is equivalent to stopping it early
    and costs one campaign instead of forty.

    With `sobol_warmup` set, the first that-many experiments come from a
    Sobol sequence instead of the learner, and the learner takes over
    afterwards with those results already in hand. The shared seed is
    unchanged either way, so the only thing that differs between this and
    the plain adaptive arm is what happens between the seed and the
    handover.
    """
    points = list(seed_points())
    warmup_extra = max(sobol_warmup - N_SEED, 0)
    if warmup_extra:
        lo = [b[0] for b in BOUNDS]
        hi = [b[1] for b in BOUNDS]
        points += list(
            qmc.scale(qmc.Sobol(d=4, scramble=True, seed=seed).random(warmup_extra), lo, hi)
        )
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        campaign_id = f"disc4d_{abs(hash(acquisition)) % 9999}_{seed}"
        for index, point in enumerate(points):
            _write_row(data_dir, campaign_id, index, point, measure(point, rng))

        for index in range(len(points), MAX_EXPERIMENTS):
            result = suggest_next_experiments(
                campaign_id,
                data_dir=data_dir,
                target=TARGET_COLUMN,
                feature_columns=FACTORS,
                n_points=1,
                acquisition=acquisition,
                max_terms=6,
                search_bounds=DECLARED_SEARCH_BOUNDS,
            )
            point = np.array([result.points[0][name] for name in FACTORS])
            _write_row(data_dir, campaign_id, index, point, measure(point, rng))
            points.append(point)
    return np.array(points)


def fixed_design(method: str, n: int, seed: int) -> np.ndarray:
    """A non-adaptive design of size n, regenerated per size -- which is
    how these are genuinely used: commit to a budget, lay out the design,
    run it."""
    n_extra = max(n - N_SEED, 1)
    lo = [b[0] for b in BOUNDS]
    hi = [b[1] for b in BOUNDS]

    if method == "Latin Hypercube":
        extra = qmc.scale(qmc.LatinHypercube(d=4, seed=seed).random(n_extra), lo, hi)
    elif method == "Sobol":
        extra = qmc.scale(qmc.Sobol(d=4, scramble=True, seed=seed).random(n_extra), lo, hi)
    elif method == "Random":
        rng = np.random.default_rng(seed)
        extra = np.column_stack([rng.uniform(a, b, n_extra) for a, b in BOUNDS])
    elif method == "Grid":
        """
        A full factorial over four factors grows as the fourth power, so
        the per-factor resolution is whatever the budget allows -- three
        levels each is already 81 runs. That coarseness is not a flaw in
        the implementation, it is the actual cost of gridding four
        factors, and part of what this study is measuring.
        """
        per_axis = max(2, int(round(n_extra ** 0.25)))
        axes = [np.linspace(a, b, per_axis) for a, b in BOUNDS]
        mesh = np.meshgrid(*axes, indexing="ij")
        extra = np.column_stack([m.ravel() for m in mesh])[:n_extra]
    else:
        raise ValueError(f"unknown fixed design {method!r}")

    return np.vstack([seed_points(), np.asarray(extra, dtype=float)])


METHOD_NAMES = [
    "Latin Hypercube",
    "Sobol",
    "Grid",
    "Random",
    "D-optimal + labwiki",
    "Model-discrimination + labwiki",
    "Sobol warm-up then D-optimal",
]

ACQUISITION_BY_METHOD = {
    "D-optimal + labwiki": "d_optimal",
    "Model-discrimination + labwiki": "model_discrimination",
    "Sobol warm-up then D-optimal": "d_optimal",
}

WARMUP_BY_METHOD = {"Sobol warm-up then D-optimal": SOBOL_WARMUP}
"""Only the hybrid arm warms up. Everything else starts adapting straight
after the shared seed, so the comparison isolates the warm-up rather than
confounding it with the acquisition."""


@dataclass
class MethodResult:
    method: str
    experiments_to_discovery: int | None
    surface_rmse: float
    extrapolation_rmse: float


def _fit(points: np.ndarray, values: np.ndarray):
    model = jaxsr.SymbolicRegressor(basis_library=build_basis_library(), max_terms=MAX_TERMS)
    model.fit(points, values)
    return model


def _discovered(model) -> bool:
    return frozenset(getattr(model, "selected_features_", ())) == TRUE_TERM_SET


def _errors(model, rng: np.random.Generator) -> tuple[float, float]:
    """Prediction quality inside the sampled envelope, and in a high-light
    band withheld from every method."""

    def rmse(par_lo: float, par_hi: float) -> float:
        n = 4000
        X = np.column_stack(
            [
                rng.uniform(par_lo, par_hi, n),
                rng.uniform(*TEMP_BOUNDS, n),
                rng.uniform(*PH_BOUNDS, n),
                rng.uniform(*NUTRIENT_BOUNDS, n),
            ]
        )
        truth = true_voc_ppm_4d_smooth(*X.T)
        return float(np.sqrt(np.mean((np.asarray(model.predict(X)).ravel() - truth) ** 2)))

    return rmse(20.0, 500.0), rmse(500.0, 600.0)


def run_method(method: str, seed: int, verbose: bool = True) -> MethodResult:
    rng = np.random.default_rng(seed)
    sequence = (
        adaptive_sequence(
            ACQUISITION_BY_METHOD[method], seed, rng, WARMUP_BY_METHOD.get(method, 0)
        )
        if method in ACQUISITION_BY_METHOD
        else None
    )

    discovery_n: int | None = None
    consecutive = 0
    model = None

    for n in range(MIN_EXPERIMENTS, MAX_EXPERIMENTS + 1, EVAL_STEP):
        points = sequence[:n] if sequence is not None else fixed_design(method, n, seed)
        values = np.array([measure(p, rng) for p in points])
        model = _fit(points, values)

        consecutive = consecutive + 1 if _discovered(model) else 0
        if consecutive >= HOLD_ROUNDS:
            discovery_n = n - (HOLD_ROUNDS - 1) * EVAL_STEP
            break

    surface, extrapolation = _errors(model, rng)
    if verbose:
        print(
            f"  {method:34s} discovered at {discovery_n or '>' + str(MAX_EXPERIMENTS):>5}   "
            f"surface {surface:6.2f}  extrap {extrapolation:7.2f} ppm"
        )
    return MethodResult(method, discovery_n, surface, extrapolation)


def run(seeds: int = 6, verbose: bool = True) -> dict[str, list[MethodResult]]:
    results: dict[str, list[MethodResult]] = {m: [] for m in METHOD_NAMES}
    for seed in range(seeds):
        if verbose:
            print(f"\nseed {seed}:")
        for method in METHOD_NAMES:
            results[method].append(run_method(method, seed, verbose))
    _save_raw(results)
    _report(results, seeds)
    _plot(results, seeds)
    return results


def _save_raw(results: dict[str, list[MethodResult]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["method,seed,experiments_to_discovery,surface_rmse,extrapolation_rmse"]
    for method, runs in results.items():
        for seed, run_result in enumerate(runs):
            found = "" if run_result.experiments_to_discovery is None else run_result.experiments_to_discovery
            lines.append(
                f"{method},{seed},{found},{run_result.surface_rmse:.4f},{run_result.extrapolation_rmse:.4f}"
            )
    (RESULTS_DIR / "discovery_4d_runs.csv").write_text("\n".join(lines) + "\n")


def _report(results: dict[str, list[MethodResult]], seeds: int) -> None:
    print("\n" + "=" * 86)
    print(f"FOUR FACTORS: experiments needed to discover the true equation ({seeds} repeats)")
    print("=" * 86)
    print("Convergence count first -- a mean over only the runs that succeeded is not")
    print("comparable to a mean over all of them.\n")
    for method in METHOD_NAMES:
        runs = results[method]
        found = [r.experiments_to_discovery for r in runs if r.experiments_to_discovery is not None]
        shown = f"{np.mean(found):.1f}" if found else "never"
        sd = f"{np.std(found, ddof=1):4.1f}" if len(found) > 1 else "   -"
        print(
            f"  {method:34s} converged {len(found):2d}/{len(runs):2d}   mean {shown:>6s}  sd {sd}   "
            f"surface {np.median([r.surface_rmse for r in runs]):6.2f}   "
            f"extrap {np.median([r.extrapolation_rmse for r in runs]):7.2f} ppm"
        )


PALETTE = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#17becf", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "P", "X", "*"]


def _plot(results: dict[str, list[MethodResult]], seeds: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )

    fig, (ax_curve, ax_extrap) = plt.subplots(1, 2, figsize=(15, 6))
    xs = np.arange(MIN_EXPERIMENTS, MAX_EXPERIMENTS + 1, EVAL_STEP)
    for index, method in enumerate(METHOD_NAMES):
        found = [r.experiments_to_discovery for r in results[method]]
        fraction = [np.mean([f is not None and f <= x for f in found]) for x in xs]
        ax_curve.plot(
            xs, fraction, label=method, color=PALETTE[index], marker=MARKERS[index],
            markevery=6, linewidth=2.0, markersize=7,
        )
    ax_curve.set_xlabel("Experiments run")
    ax_curve.set_ylabel("Fraction of repeats that discovered the equation")
    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.set_title(f"Four factors, {seeds} repeats")
    ax_curve.legend(loc="upper left", fontsize=9)

    positions = np.arange(len(METHOD_NAMES))
    values = [float(np.median([r.extrapolation_rmse for r in results[m]])) for m in METHOD_NAMES]
    ax_extrap.bar(positions, values, color=PALETTE, width=0.65)
    ax_extrap.set_xticks(positions)
    ax_extrap.set_xticklabels(METHOD_NAMES, rotation=30, ha="right", fontsize=9)
    ax_extrap.set_ylabel("RMSE on withheld high-light band (ppm)")
    ax_extrap.set_title("Does the equation hold outside the sampled range?")

    fig.suptitle(
        "Four factors: does choosing your experiments start to pay?", fontsize=15, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "discovery_4d.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlots written to {RESULTS_DIR}")


if __name__ == "__main__":
    run()
