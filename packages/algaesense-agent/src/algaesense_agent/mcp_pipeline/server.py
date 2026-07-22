"""MCP server exposing the jaxsr-calibration pipeline as tools Hermes can call."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from algaesense_agent.mcp_pipeline.pipeline import (
    fit_symbolic_model,
    suggest_next_experiments,
    suggest_next_experiments_with_context,
)
from algaesense_agent.mcp_pipeline.pipeline import discover_led_response_dynamics as _discover_led_response_dynamics


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


def _wiki_root() -> Path:
    """Where labwiki pages live -- same `ALGAESENSE_LABWIKI_ROOT` env var
    `mcp_labwiki`'s own server already uses, so both point at the same
    archive without needing a second, separately-configured path."""
    return Path(os.environ.get("ALGAESENSE_LABWIKI_ROOT", "data/labwiki"))


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


@mcp.tool()
def suggest_next_experiment_conditions_with_context(
    campaign_id: str,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    n_points: int = 3,
    kappa: float = 2.0,
    max_terms: int = 5,
    extra_topics: list[str] | None = None,
) -> dict:
    """Same as suggest_next_experiment_conditions, plus relevant labwiki
    findings for the campaign and every condition/target involved --
    prior operator notes, fit expressions, and active-learning proposals
    already recorded for these same things, surfaced alongside the new
    quantitative suggestion so you don't have to separately call
    query_labwiki_topic to check. Prefer this over the plain
    suggest_next_experiment_conditions when discussing what to try next
    with the operator, since it gives them the fuller picture in one
    response; the plain version is still fine for a quick numeric-only
    check."""
    result = suggest_next_experiments_with_context(
        campaign_id,
        data_dir=_data_dir(),
        wiki_root=_wiki_root(),
        target=target,
        feature_columns=feature_columns,
        n_points=n_points,
        kappa=kappa,
        max_terms=max_terms,
        extra_topics=extra_topics,
    )
    return {
        "suggestion": {
            "points": result.suggestion.points,
            "scores": result.suggestion.scores,
            "acquisition": result.suggestion.acquisition,
            "fit": asdict(result.suggestion.fit),
        },
        "labwiki_context": [
            {
                "topic": context.topic,
                "findings": [
                    {"path": str(finding.path), "matching_lines": finding.matching_lines, "full_content": finding.full_content}
                    for finding in context.findings
                ],
            }
            for context in result.labwiki_context
        ],
    }


@mcp.tool()
def discover_led_response_dynamics(
    experiment_id: str,
    reactor_id: str,
    sensor_id: str,
    calibration_run_id: str,
    since: str | None = None,
    until: str | None = None,
    max_terms: int = 5,
) -> dict:
    """Discover how a reactor's VOC output dynamically responds to its
    LED's actual light history over one experiment -- meant to be run over
    data collected during a ramp/sinusoid/step control-profile run, where
    PAR genuinely varies within the run (a static setpoint gives it no
    within-run trend to discover). `since`/`until` are optional ISO
    datetime strings to scope to part of the experiment; omit both to use
    the whole thing."""
    result = _discover_led_response_dynamics(
        experiment_id,
        reactor_id,
        sensor_id,
        calibration_run_id,
        data_dir=_data_dir(),
        since=dt.datetime.fromisoformat(since) if since is not None else None,
        until=dt.datetime.fromisoformat(until) if until is not None else None,
        max_terms=max_terms,
    )
    return asdict(result)


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
