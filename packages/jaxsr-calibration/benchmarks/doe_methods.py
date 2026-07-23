"""Point-selection strategies compared in Test 2: five ways of choosing
which (PAR, temp) conditions to try next, given a fixed budget of 10
experiments total. Every method starts from the SAME two seed points
(so all 6 strategies genuinely spend the same 10-experiment budget),
then either generates the remaining 8 points upfront (the 4 classic DoE
baselines) or picks them one at a time using the real
suggest_next_experiments/_with_context active-learning tools (ours).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import qmc

from ground_truth import PAR_BOUNDS, TEMP_BOUNDS

from algaesense_agent.labwiki.models import ExperimentResult
from algaesense_agent.labwiki.wiki import ingest_experiment_result
from algaesense_agent.mcp_pipeline.pipeline import (
    suggest_next_experiments,
    suggest_next_experiments_with_context,
)


"""
suggest_next_experiments's search bounds default to the OBSERVED data's
min/max (bound_overrides can only narrow that, never widen it -- see
mcp_pipeline/pipeline.py's _apply_bound_overrides) -- confirmed
directly while building this benchmark: seeding with only 2 clustered
points left every subsequent active-learning suggestion permanently
confined to that narrow observed range, never exploring the rest of
the declared domain at all. A real practitioner wouldn't seed active
learning with 2 arbitrary clustered points either -- they'd start from
a small design that already spans the physical operating range, then
let active learning refine within/around it. Four points near (not
exactly on) the domain's corners give every method, including the DoE
baselines that see all 10 points upfront, the same realistic starting
information.
"""
SEED_POINTS: list[tuple[float, float]] = [
    (40.0, 22.0),
    (460.0, 22.0),
    (40.0, 38.0),
    (460.0, 38.0),
]

REACTOR_ID = "R01"
SENSOR_ID = "PID01"


def _scale_unit_points(unit_points: np.ndarray) -> list[tuple[float, float]]:
    par = PAR_BOUNDS[0] + unit_points[:, 0] * (PAR_BOUNDS[1] - PAR_BOUNDS[0])
    temp = TEMP_BOUNDS[0] + unit_points[:, 1] * (TEMP_BOUNDS[1] - TEMP_BOUNDS[0])
    return list(zip(par.tolist(), temp.tolist()))


def latin_hypercube_points(n_extra: int, seed: int) -> list[tuple[float, float]]:
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    return _scale_unit_points(sampler.random(n_extra))


def sobol_points(n_extra: int, seed: int) -> list[tuple[float, float]]:
    """scipy's Sobol sampler works best at power-of-2 sample counts, but
    `.random(n)` still returns a valid (if slightly less balanced)
    low-discrepancy sequence for an arbitrary n -- fine for our n_extra=8."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    return _scale_unit_points(sampler.random(n_extra))


def grid_points(n_extra: int) -> list[tuple[float, float]]:
    """A plain regular grid -- the classic non-adaptive DoE baseline.
    Hardcoded to a 3x2 layout since n_extra=6 is fixed by this
    benchmark's 10-experiment budget (4 seed + 6 extra)."""
    if n_extra != 6:
        raise ValueError("grid_points is hardcoded for n_extra=6 (a 3x2 layout)")
    par_levels = np.linspace(PAR_BOUNDS[0], PAR_BOUNDS[1], 3)
    temp_levels = np.linspace(TEMP_BOUNDS[0], TEMP_BOUNDS[1], 2)
    return [(float(par), float(temp)) for par in par_levels for temp in temp_levels]


def random_points(n_extra: int, seed: int) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    par = rng.uniform(PAR_BOUNDS[0], PAR_BOUNDS[1], size=n_extra)
    temp = rng.uniform(TEMP_BOUNDS[0], TEMP_BOUNDS[1], size=n_extra)
    return list(zip(par.tolist(), temp.tolist()))


def _write_campaign_row(
    data_dir: Path, campaign_id: str, experiment_index: int, par: float, temp: float, ppm: float
) -> None:
    campaign_dir = data_dir / "derived" / "features" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = f"exp_{experiment_index:02d}"
    row = {
        "experiment_id": experiment_id,
        "campaign_id": campaign_id,
        "reactor_id": REACTOR_ID,
        "sensor_id": SENSOR_ID,
        "par_umol_m2_s": float(par),
        "mean_sample_t_c": float(temp),
        "mean_sample_rh_pct": 55.0,
        "mean_voc_ppm_asgas": float(ppm),
    }
    pl.DataFrame([row]).write_parquet(campaign_dir / f"{experiment_id}.parquet")


def run_fixed_design_campaign(
    points: list[tuple[float, float]],
    measure_fn: Callable[[float, float], float],
    data_dir: Path,
    campaign_id: str,
) -> list[tuple[float, float]]:
    """Write a non-adaptive method's points (seed + its own upfront
    design) in order -- 'round k' for these methods is simply the first
    k points revealed in this order, the standard way to score a
    non-adaptive design's incremental convergence."""
    all_points = SEED_POINTS + points
    for i, (par, temp) in enumerate(all_points):
        ppm = measure_fn(par, temp)
        _write_campaign_row(data_dir, campaign_id, i, par, temp, ppm)
    return all_points


def run_active_learning_campaign(
    measure_fn: Callable[[float, float], float],
    n_extra: int,
    data_dir: Path,
    campaign_id: str,
    use_labwiki: bool = False,
    wiki_root: Path | None = None,
    labwiki_note_round: int | None = None,
    labwiki_note_text: str | None = None,
    bound_override_after_note: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """'Ours': the real suggest_next_experiments/_with_context tools
    pick one new point per round, informed by every experiment run so
    far. When use_labwiki=True, a genuine operator note (a documented,
    physically-grounded finding, not a made-up one -- see
    ground_truth.py's photoinhibition term) gets ingested partway
    through, and every suggestion from that round onward passes the
    corresponding bound_overrides -- mirroring the real two-step
    Hermes workflow this project's system_prompt.md describes, just
    with the 'read the note and decide the constraint' step already
    resolved up front since this is a scripted benchmark, not a live
    chat."""
    all_points: list[tuple[float, float]] = list(SEED_POINTS)
    for i, (par, temp) in enumerate(SEED_POINTS):
        ppm = measure_fn(par, temp)
        _write_campaign_row(data_dir, campaign_id, i, par, temp, ppm)

    note_ingested = False
    for round_num in range(n_extra):
        experiment_index = len(SEED_POINTS) + round_num

        bound_overrides = None
        if use_labwiki and labwiki_note_round is not None and round_num >= labwiki_note_round:
            if not note_ingested:
                ingest_experiment_result(
                    ExperimentResult(
                        experiment_id=f"exp_{experiment_index - 1:02d}",
                        campaign_id=campaign_id,
                        reactor_id=REACTOR_ID,
                        sensor_id=SENSOR_ID,
                        conditions={"par_umol_m2_s": all_points[-1][0]},
                        target_metrics={"mean_voc_ppm_asgas": 0.0},
                        operator_notes=[labwiki_note_text],
                    ),
                    wiki_root=wiki_root,
                )
                note_ingested = True
            bound_overrides = bound_override_after_note

        if use_labwiki:
            result = suggest_next_experiments_with_context(
                campaign_id,
                data_dir=data_dir,
                wiki_root=wiki_root,
                target="mean_voc_ppm_asgas",
                feature_columns=["par_umol_m2_s", "mean_sample_t_c"],
                n_points=1,
                kappa=2.0,
                max_terms=5,
                bound_overrides=bound_overrides,
            ).suggestion
        else:
            result = suggest_next_experiments(
                campaign_id,
                data_dir=data_dir,
                target="mean_voc_ppm_asgas",
                feature_columns=["par_umol_m2_s", "mean_sample_t_c"],
                n_points=1,
                kappa=2.0,
                max_terms=5,
            )

        point = result.points[0]
        par, temp = point["par_umol_m2_s"], point["mean_sample_t_c"]
        ppm = measure_fn(par, temp)
        _write_campaign_row(data_dir, campaign_id, experiment_index, par, temp, ppm)
        all_points.append((par, temp))

    return all_points
