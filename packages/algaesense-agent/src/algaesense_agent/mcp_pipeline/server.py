"""MCP server exposing the jaxsr-calibration pipeline as tools Hermes can call."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from algaesense_agent.mcp_pipeline.pipeline import fit_symbolic_model, suggest_next_experiments


"""
Every tool here is read/compute-only -- it loads already-written derived
feature Parquet files and returns a fit or a suggestion, never writes
anything or drives hardware. That's a deliberate split from
`mcp_actuators` (the only server in this package allowed to cause a
side effect), so Hermes's own instructions can treat "call mcp_pipeline
tools freely" and "call mcp_actuators tools only after confirmation"
as two clearly different trust levels.
"""

mcp = FastMCP("algaesense-pipeline")


def _data_dir() -> Path:
    """Where derived feature Parquet files live."""

    """
    `ALGAESENSE_DATA_DIR` lets whoever configures Hermes's
    `~/.hermes/config.yaml` point this server at the actual data
    directory for their installation, without editing code -- defaults to
    `./data` (matching the layout `jaxsr_calibration`'s own CLI and specs
    assume) for local development.
    """
    return Path(os.environ.get("ALGAESENSE_DATA_DIR", "data"))


@mcp.tool()
def fit_campaign_model(
    campaign_id: str,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    max_terms: int = 5,
) -> dict:
    """Fit a symbolic-regression model over one campaign's completed
    experiments and return the discovered expression."""
    result = fit_symbolic_model(
        campaign_id,
        data_dir=_data_dir(),
        target=target,
        feature_columns=feature_columns,
        max_terms=max_terms,
    )
    return asdict(result)


@mcp.tool()
def suggest_next_experiment_conditions(
    campaign_id: str,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    n_points: int = 3,
    kappa: float = 2.0,
    max_terms: int = 5,
) -> dict:
    """Suggest the next experimental conditions to run for a campaign,
    using active learning over the current fit."""
    result = suggest_next_experiments(
        campaign_id,
        data_dir=_data_dir(),
        target=target,
        feature_columns=feature_columns,
        n_points=n_points,
        kappa=kappa,
        max_terms=max_terms,
    )
    return {
        "points": result.points,
        "scores": result.scores,
        "acquisition": result.acquisition,
        "fit": asdict(result.fit),
    }


def main() -> None:
    """Entry point for the `algaesense-mcp-pipeline` console script."""

    """
    `transport="stdio"` is the default and what Hermes's
    `~/.hermes/config.yaml` `mcp_servers:` stdio entries expect -- Hermes
    launches this as a subprocess and talks to it over stdin/stdout, not a
    network socket.
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
