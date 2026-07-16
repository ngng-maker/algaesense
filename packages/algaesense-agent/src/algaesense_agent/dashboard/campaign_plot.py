"""Ties campaign data loading, fitting, and plotting together into one
on-demand "show me the current fit" call.
"""

from __future__ import annotations

from pathlib import Path

import jaxsr
import numpy as np

from algaesense_agent.dashboard.plots import render_fit_curve_png
from algaesense_agent.mcp_pipeline.pipeline import default_basis_library, load_campaign_features
from jaxsr_calibration.processing.features import load_features_for_jaxsr


def render_campaign_fit_plot(
    campaign_id: str,
    data_dir: Path,
    feature_column: str,
    target: str = "mean_voc_ppm_asgas",
    max_terms: int = 5,
) -> bytes:
    """Fit a single-feature model over one campaign and render observed
    data against the fitted curve as a PNG."""

    """
    Deliberately a single named `feature_column` (not the general
    multi-feature `feature_columns` list mcp_pipeline's fit tools accept)
    -- a 2-D scatter-plus-curve plot only makes visual sense against one
    controllable variable at a time, e.g. "PAR vs VOC emission", the
    plan's own worked example.
    """
    features_df = load_campaign_features(campaign_id, data_dir)

    X, y, _ = load_features_for_jaxsr(
        features_df,
        target=target,
        feature_columns=[feature_column],
        include_categorical=False,
    )
    x = X[:, 0]

    library = default_basis_library(n_features=1)
    model = jaxsr.SymbolicRegressor(basis_library=library, max_terms=max_terms)
    model.fit(X, y)

    return render_fit_curve_png(
        x=x,
        y=y,
        predict_fn=lambda x_grid: np.asarray(model.predict(x_grid.reshape(-1, 1))),
        x_label=feature_column,
        y_label=target,
        title=f"{campaign_id}: {model.expression_}",
    )
