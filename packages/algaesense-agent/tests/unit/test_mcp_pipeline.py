"""Unit tests for algaesense_agent.mcp_pipeline: pure-Python pipeline logic
plus a check that the MCP tool wrappers call through to it correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from algaesense_agent.mcp_pipeline.pipeline import (
    CampaignNotFoundError,
    fit_symbolic_model,
    load_campaign_features,
    suggest_next_experiments,
)


def _write_synthetic_campaign(data_dir: Path, campaign_id: str, n_experiments: int = 8) -> None:
    """Write one derived-features Parquet file per synthetic experiment,
    with a known linear relationship between PAR and VOC output."""

    """
    `mean_voc_ppm_asgas = 2.0 * par_umol_m2_s + 5.0`, no noise -- a
    trivially recoverable relationship, so a successful fit here is a real
    signal that `fit_symbolic_model`/`suggest_next_experiments` wired
    `load_features_for_jaxsr` -> `jaxsr.SymbolicRegressor` ->
    `jaxsr.ActiveLearner` together correctly, not that the model happened
    to get lucky on noisy data.
    """
    campaign_dir = data_dir / "derived" / "features" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    par_values = np.linspace(100.0, 500.0, n_experiments)

    for i, par in enumerate(par_values):
        experiment_id = f"exp_{i:02d}"
        row = {
            "experiment_id": experiment_id,
            "campaign_id": campaign_id,
            "reactor_id": "R01",
            "sensor_id": "PID01",
            "par_umol_m2_s": float(par),
            "mean_sample_t_c": 30.0,
            "mean_sample_rh_pct": 60.0,
            "mean_voc_ppm_asgas": float(2.0 * par + 5.0),
        }
        pl.DataFrame([row]).write_parquet(campaign_dir / f"{experiment_id}.parquet")


def test_load_campaign_features_concatenates_every_experiment(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01", n_experiments=5)

    features_df = load_campaign_features("camp_01", tmp_path)

    assert features_df.height == 5
    assert set(features_df["experiment_id"].to_list()) == {f"exp_0{i}" for i in range(5)}


def test_load_campaign_features_raises_for_unknown_campaign(tmp_path: Path) -> None:
    with pytest.raises(CampaignNotFoundError):
        load_campaign_features("does_not_exist", tmp_path)


def test_fit_symbolic_model_recovers_known_linear_relationship(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")

    """
    `include_categorical=False`: this synthetic fixture only has one
    sensor_id/reactor_id value, so the one-hot dummy columns
    `load_features_for_jaxsr` would otherwise add are just noise for what
    this test is actually checking (the par_umol_m2_s -> VOC
    relationship).
    """
    result = fit_symbolic_model(
        "camp_01",
        data_dir=tmp_path,
        target="mean_voc_ppm_asgas",
        feature_columns=["par_umol_m2_s"],
        include_categorical=False,
    )

    assert result.feature_names == ["par_umol_m2_s"]
    # A clean linear relationship should fit essentially perfectly.
    assert result.metrics["r2"] > 0.99


def test_suggest_next_experiments_returns_points_within_observed_bounds(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")

    result = suggest_next_experiments(
        "camp_01",
        data_dir=tmp_path,
        target="mean_voc_ppm_asgas",
        feature_columns=["par_umol_m2_s"],
        n_points=3,
    )

    assert len(result.points) == 3
    lo, hi = result.fit.feature_bounds[0]
    for point in result.points:
        assert lo <= point["par_umol_m2_s"] <= hi


async def test_mcp_server_fit_tool_matches_direct_pipeline_call(tmp_path: Path, monkeypatch) -> None:
    """The FastMCP tool wrapper should return the same result as calling
    the pipeline function directly -- confirms the server wiring (argument
    passing, dataclass -> dict conversion) doesn't silently change behavior."""

    _write_synthetic_campaign(tmp_path, "camp_01")
    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path))

    """
    Imported after setting the env var, and via importlib.reload if
    already imported by an earlier test in this session, so `_data_dir()`
    picks up this test's `tmp_path` rather than whatever value an earlier
    import might have already read.
    """
    import importlib

    from algaesense_agent.mcp_pipeline import server as server_module

    importlib.reload(server_module)

    direct_result = fit_symbolic_model(
        "camp_01", data_dir=tmp_path, target="mean_voc_ppm_asgas", feature_columns=["par_umol_m2_s"]
    )

    tool_result = await server_module.mcp.call_tool(
        "fit_campaign_model",
        {
            "campaign_id": "camp_01",
            "target": "mean_voc_ppm_asgas",
            "feature_columns": ["par_umol_m2_s"],
        },
    )

    """
    `call_tool` returns a list of MCP content blocks, not the raw Python
    return value -- a dict-returning tool comes back as a single
    `TextContent` block whose `.text` is the JSON-serialized dict, so we
    parse that back into a dict rather than asserting on wire-format
    objects directly.
    """
    import json

    structured = json.loads(tool_result[0].text)

    assert structured["expression"] == direct_result.expression
    assert structured["feature_names"] == direct_result.feature_names
