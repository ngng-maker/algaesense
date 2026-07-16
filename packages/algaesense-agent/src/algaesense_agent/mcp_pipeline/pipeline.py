"""Read/compute-only bridge from derived experiment features to JAXSR fits
and next-experiment suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jaxsr
import numpy as np
import polars as pl

from jaxsr_calibration.processing.features import load_features_for_jaxsr


"""
This module is the actual logic behind the `mcp_pipeline` MCP server --
kept separate from server.py (the thin FastMCP tool-registration layer) so
every function here is a plain, directly-testable Python call with no MCP
protocol involved, same split already used throughout jaxsr_calibration
(e.g. calibration/apply.py vs cli.py). Nothing in this module has a side
effect: it only reads already-written Parquet files and returns computed
results.
"""


class CampaignNotFoundError(FileNotFoundError):
    """Raised when a campaign has no derived-features Parquet files yet."""


def load_campaign_features(campaign_id: str, data_dir: Path) -> pl.DataFrame:
    """Load and concatenate every experiment's derived-features file for
    one campaign."""

    """
    Per the derived-feature layout
    (`data/derived/features/{campaign_id}/{experiment_id}.parquet`), one
    file per experiment run under that campaign -- concatenating them is
    exactly "compare summarized outcomes across many experiments", the
    `load_features_for_jaxsr` use case.
    """
    campaign_dir = Path(data_dir) / "derived" / "features" / campaign_id
    paths = sorted(campaign_dir.glob("*.parquet")) if campaign_dir.exists() else []

    if not paths:
        raise CampaignNotFoundError(
            f"No derived-features Parquet files found for campaign {campaign_id!r} "
            f"under {campaign_dir}"
        )

    return pl.concat([pl.read_parquet(p) for p in paths])


@dataclass
class FitResult:
    """A fitted symbolic model plus enough context to run active learning
    against it later."""

    expression: str
    coefficients: list[float]
    selected_features: list[str]
    complexity: int
    metrics: dict
    feature_names: list[str]
    feature_bounds: list[tuple[float, float]]


def default_basis_library(n_features: int) -> jaxsr.BasisLibrary:
    """Build a generic, no-assumptions basis library for a first-pass fit."""

    """
    Constant + linear + degree-2 polynomial terms is a reasonable
    general-purpose starting library for "we don't yet know the functional
    form" -- the same spirit as jaxsr_calibration.processing.config's
    BasisConfig default (polynomial_degree=2), but this is a fresh,
    independent library rather than reused from there: that one is scoped
    to per-experiment covariate correction, this one is campaign-level
    design-of-experiments, a different question entirely.
    """
    return (
        jaxsr.BasisLibrary(n_features=n_features)
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=2)
    )


def fit_symbolic_model(
    campaign_id: str,
    data_dir: Path,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    max_terms: int = 5,
    include_categorical: bool = True,
) -> FitResult:
    """Fit a symbolic-regression model over one campaign's experiments."""

    features_df = load_campaign_features(campaign_id, data_dir)

    X, y, feature_names = load_features_for_jaxsr(
        features_df,
        target=target,
        feature_columns=feature_columns,
        include_categorical=include_categorical,
    )

    library = default_basis_library(n_features=X.shape[1])
    model = jaxsr.SymbolicRegressor(basis_library=library, max_terms=max_terms)
    model.fit(X, y)

    """
    Bounds are taken from the observed data's own min/max per feature --
    the natural "safe to suggest within" range for the active-learning step
    below, rather than an arbitrarily chosen range that might not match
    what this reactor/sensor combination has ever actually run at.
    """
    feature_bounds = [(float(np.min(X[:, i])), float(np.max(X[:, i]))) for i in range(X.shape[1])]

    return FitResult(
        expression=model.expression_,
        coefficients=[float(c) for c in model.coefficients_],
        selected_features=list(model.selected_features_),
        complexity=int(model.complexity_),
        metrics={k: float(v) for k, v in model.metrics_.items()},
        feature_names=feature_names,
        feature_bounds=feature_bounds,
    )


@dataclass
class SuggestionResult:
    """Next experimental conditions JAXSR's active learner recommends
    trying, plus the fit they were derived from."""

    points: list[dict[str, float]]
    scores: list[float]
    acquisition: str
    fit: FitResult


def suggest_next_experiments(
    campaign_id: str,
    data_dir: Path,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    n_points: int = 3,
    kappa: float = 2.0,
    max_terms: int = 5,
) -> SuggestionResult:
    """Fit a model over one campaign, then suggest the next `n_points`
    experimental conditions to try."""

    """
    `include_categorical=False` here specifically: active learning needs a
    continuous, bounded input space to generate candidate points in --
    a one-hot categorical dummy column (e.g. sensor_id_PID01) has no
    meaningful "in-between value" to suggest, unlike a numeric condition
    like PAR or temperature. This is a separate fit from
    `fit_symbolic_model`'s general-purpose one (which is fine to include
    categoricals in), not a reuse of the same result.
    """
    fit = fit_symbolic_model(
        campaign_id,
        data_dir,
        target=target,
        feature_columns=feature_columns,
        max_terms=max_terms,
        include_categorical=False,
    )

    features_df = load_campaign_features(campaign_id, data_dir)
    X, y, _ = load_features_for_jaxsr(
        features_df, target=target, feature_columns=feature_columns, include_categorical=False
    )

    library = default_basis_library(n_features=X.shape[1])
    model = jaxsr.SymbolicRegressor(basis_library=library, max_terms=max_terms)
    model.fit(X, y)

    learner = jaxsr.ActiveLearner(
        model=model,
        bounds=fit.feature_bounds,
        acquisition=jaxsr.UCB(kappa=kappa),
    )
    result = learner.suggest(n_points=n_points)

    """
    `result.points` is a `(n_points, n_features)` array in the same
    feature order as `fit.feature_names` -- zipping each row against that
    name list turns it into a self-describing dict (e.g. {"par_umol_m2_s":
    250.0, "reactor_temp_c": 30.0}) instead of a bare positional array,
    which is what an MCP tool caller (and a human reading the agent's
    response in Slack) actually needs to act on it.
    """
    points = [
        dict(zip(fit.feature_names, (float(v) for v in row)))
        for row in np.asarray(result.points)
    ]

    return SuggestionResult(
        points=points,
        scores=[float(s) for s in result.scores],
        acquisition=result.acquisition,
        fit=fit,
    )
