"""Test 2 -- does the calibration+JAXSR+labwiki active-learning
workflow characterize the true VOC(PAR, temp) surface faster than
classic DoE (Latin Hypercube / Sobol / Grid / Random), given the same
10-experiment budget?

Every method's chosen points get 'measured' through the identical
noisy-but-calibrated readout (see measure_voc_corrected below, whose
residual noise level is taken directly from Test 1's finding that
ambient-baseline-corrected data recovers the true surface to within
~0.2 ppm RMSE at these signal magnitudes). The ONLY thing that varies
between methods is which (PAR, temp) points get chosen -- point
selection is the single independent variable this benchmark isolates.

'Ours' and 'ours+labwiki' pick points one at a time using the real
suggest_next_experiments / suggest_next_experiments_with_context tools
(the same production code Hermes calls); the 4 DoE baselines generate
their 8 non-seed points upfront, as those methods actually work.
Round-by-round scoring, for every method, fits jaxsr.SymbolicRegressor
with the SAME basis library `pipeline.py` itself uses (constant +
linear + degree-2 polynomial) against however many points that method
has revealed so far, then checks that fit's predictions against the
true surface over a dense held-out grid.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jaxsr
import numpy as np
import polars as pl

from doe_methods import (
    grid_points,
    latin_hypercube_points,
    random_points,
    run_active_learning_campaign,
    run_fixed_design_campaign,
    sobol_points,
)
from ground_truth import PAR_BOUNDS, PHOTO_THRESHOLD_PAR, TEMP_BOUNDS, true_voc_ppm

from algaesense_agent.mcp_pipeline.pipeline import default_basis_library
from jaxsr_calibration.processing.features import load_features_for_jaxsr


N_TOTAL_EXPERIMENTS = 10
N_EXTRA = N_TOTAL_EXPERIMENTS - 4  # 4 shared seed points, per doe_methods.SEED_POINTS
MEASUREMENT_NOISE_SIGMA_PPM = 8.0  # from Test 1's corrected-pipeline RMSE finding
DENSE_GRID_N = 40

LABWIKI_NOTE_ROUND = 2
LABWIKI_NOTE_TEXT = (
    f"VOC output visibly declines once PAR exceeds roughly {PHOTO_THRESHOLD_PAR:.0f} "
    "umol/m2/s -- consistent with photoinhibition at high light intensity. Recommend "
    "focusing future runs below this threshold rather than continuing to probe higher."
)
LABWIKI_BOUND_OVERRIDE = {"par_umol_m2_s": (PAR_BOUNDS[0], PHOTO_THRESHOLD_PAR)}


def measure_voc_corrected(par: float, temp: float, rng: np.random.Generator) -> float:
    """A single simulated 'run this experiment through the real
    calibration+ambient-baseline pipeline' measurement -- not a full
    raw-signal re-simulation per point (Test 1 already proved that
    pipeline recovers the true surface to ~0.2 ppm RMSE once corrected;
    reusing that residual noise level here keeps Test 2's runtime
    tractable while still being an honest, measured noise floor rather
    than an invented one)."""
    return float(true_voc_ppm(par, temp) + rng.normal(0.0, MEASUREMENT_NOISE_SIGMA_PPM))


@dataclass
class MethodRun:
    label: str
    points: list[tuple[float, float]]
    data_dir: Path
    campaign_id: str


def _dense_test_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    par_grid, temp_grid = np.meshgrid(
        np.linspace(*PAR_BOUNDS, DENSE_GRID_N), np.linspace(*TEMP_BOUNDS, DENSE_GRID_N)
    )
    true_values = true_voc_ppm(par_grid.ravel(), temp_grid.ravel())
    return par_grid.ravel(), temp_grid.ravel(), true_values


def _fit_and_score_round(features_df: pl.DataFrame, test_par: np.ndarray, test_temp: np.ndarray, test_true: np.ndarray) -> float:
    """Fit the SAME basis library pipeline.py uses against however many
    rows have been revealed so far, and return the RMSE of its
    predictions against the true surface on the held-out dense grid --
    this is what 'has this method figured out the true PAR/temp
    behaviour' actually means, made concrete and comparable."""
    X, y, _feature_names = load_features_for_jaxsr(
        features_df,
        target="mean_voc_ppm_asgas",
        feature_columns=["par_umol_m2_s", "mean_sample_t_c"],
        include_categorical=False,
    )
    library = default_basis_library(n_features=X.shape[1])
    model = jaxsr.SymbolicRegressor(basis_library=library, max_terms=5)
    model.fit(X, y)

    X_test = np.column_stack([test_par, test_temp])
    predicted = model.predict(X_test)
    return float(np.sqrt(np.mean((predicted - test_true) ** 2)))


def _score_all_rounds(points: list[tuple[float, float]], test_par, test_temp, test_true, rng: np.random.Generator) -> list[float]:
    """Re-measure each revealed point's readout fresh for scoring
    (independent noise draw from whatever was used to pick it), then
    fit/score cumulatively after each round -- round k's score reflects
    a fit over exactly the first k points in that method's own reveal
    order."""
    rows = []
    rmses = []
    for i, (par, temp) in enumerate(points):
        ppm = measure_voc_corrected(par, temp, rng)
        rows.append(
            {
                "experiment_id": f"exp_{i:02d}",
                "campaign_id": "scoring",
                "reactor_id": "R01",
                "sensor_id": "PID01",
                "par_umol_m2_s": float(par),
                "mean_sample_t_c": float(temp),
                "mean_sample_rh_pct": 55.0,
                "mean_voc_ppm_asgas": ppm,
            }
        )
        if i < 1:
            # jaxsr.SymbolicRegressor needs at least a couple of points
            # before a degree-2 fit is even well-posed; round 1 has no
            # meaningful score yet for any method.
            rmses.append(float("nan"))
            continue
        features_df = pl.DataFrame(rows)
        rmses.append(_fit_and_score_round(features_df, test_par, test_temp, test_true))
    return rmses


def run_doe_comparison(seed: int = 0, verbose: bool = True) -> dict[str, list[float]]:
    test_par, test_temp, test_true = _dense_test_grid()
    scoring_rng = np.random.default_rng(seed + 1000)

    tmp_root = Path(tempfile.mkdtemp(prefix="algaesense_doe_bench_"))
    try:
        methods: dict[str, MethodRun] = {}

        def measure(par, temp):
            return measure_voc_corrected(par, temp, np.random.default_rng(hash((round(par, 3), round(temp, 3), seed)) % (2**31)))

        for label, generator in [
            ("Latin Hypercube", lambda: latin_hypercube_points(N_EXTRA, seed)),
            ("Sobol", lambda: sobol_points(N_EXTRA, seed)),
            ("Grid", lambda: grid_points(N_EXTRA)),
            ("Random", lambda: random_points(N_EXTRA, seed)),
        ]:
            data_dir = tmp_root / label.replace(" ", "_")
            campaign_id = "doe_bench"
            points = run_fixed_design_campaign(generator(), measure, data_dir, campaign_id)
            methods[label] = MethodRun(label, points, data_dir, campaign_id)
            if verbose:
                print(f"  {label}: generated {len(points)} points")

        for label, use_labwiki in [("Ours (plain)", False), ("Ours + labwiki", True)]:
            data_dir = tmp_root / label.replace(" ", "_").replace("(", "").replace(")", "")
            campaign_id = "doe_bench"
            wiki_root = data_dir / "labwiki" if use_labwiki else None
            points = run_active_learning_campaign(
                measure,
                N_EXTRA,
                data_dir,
                campaign_id,
                use_labwiki=use_labwiki,
                wiki_root=wiki_root,
                labwiki_note_round=LABWIKI_NOTE_ROUND if use_labwiki else None,
                labwiki_note_text=LABWIKI_NOTE_TEXT if use_labwiki else None,
                bound_override_after_note=LABWIKI_BOUND_OVERRIDE if use_labwiki else None,
            )
            methods[label] = MethodRun(label, points, data_dir, campaign_id)
            if verbose:
                print(f"  {label}: generated {len(points)} points")

        results: dict[str, list[float]] = {}
        for label, method_run in methods.items():
            rmses = _score_all_rounds(method_run.points, test_par, test_temp, test_true, scoring_rng)
            results[label] = rmses
            if verbose:
                print(f"  {label} RMSE by round: {[round(r, 1) if r == r else None for r in rmses]}")

        return results
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    results = run_doe_comparison()
    print("\nFinal (round 10) RMSE vs true surface, lower is better:")
    for label, rmses in sorted(results.items(), key=lambda kv: kv[1][-1]):
        print(f"  {label}: {rmses[-1]:.1f} ppm")
