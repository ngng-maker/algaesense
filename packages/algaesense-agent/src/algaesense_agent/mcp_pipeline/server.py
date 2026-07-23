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


def _convert_bounds_dict(raw: dict[str, list[float]] | None) -> dict[str, tuple[float, float]] | None:
    """MCP tool arguments are plain JSON -- a `[lo, hi]` list, not a
    Python tuple -- converted here once rather than at every call site.
    Used for both `search_bounds` and `bound_overrides`, since both have
    the identical `{feature_name: [lo, hi]}` JSON shape."""
    if raw is None:
        return None
    return {name: (float(bounds[0]), float(bounds[1])) for name, bounds in raw.items()}


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
    search_bounds: dict[str, list[float]] | None = None,
    bound_overrides: dict[str, list[float]] | None = None,
) -> dict:
    """Suggest the next experimental conditions to run for a campaign,
    using active learning over the current fit.

    By default, the range searched is always the OBSERVED data's own
    min/max for each feature -- meaning a campaign with only a few
    clustered experiments so far can never be suggested a point outside
    that narrow range, no matter how many rounds run. `search_bounds`
    (e.g. `{"par_umol_m2_s": [0.0, 500.0]}`) fixes that: it REPLACES the
    default range for named features with one you declare -- use this
    once, early in a campaign, whenever you know the true physical/safety
    range to search (e.g. the reactor's configured max-PAR ceiling) but
    the campaign's own observed data doesn't span it yet. Unlike
    `bound_overrides` below, this CAN widen past the observed data.

    `bound_overrides` (e.g. `{"par_umol_m2_s": [0.0, 300.0]}`) narrows
    whichever range results (search_bounds if given, else the observed
    data range) for named features -- only narrows, never widens. Use
    this when a genuine finding (from labwiki, or from your own
    reasoning over the fit/data) implies a real constraint on where the
    next experiment should land; don't use it just to nudge the
    suggestion toward a value you'd prefer. The two can be combined:
    search_bounds sets the starting range, bound_overrides narrows it
    further."""
    result = suggest_next_experiments(
        campaign_id,
        data_dir=_data_dir(),
        target=target,
        feature_columns=feature_columns,
        n_points=n_points,
        kappa=kappa,
        max_terms=max_terms,
        search_bounds=_convert_bounds_dict(search_bounds),
        bound_overrides=_convert_bounds_dict(bound_overrides),
    )
    return {
        "points": result.points,
        "scores": result.scores,
        "acquisition": result.acquisition,
        "fit": asdict(result.fit),
        "search_bounds": result.search_bounds,
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
    search_bounds: dict[str, list[float]] | None = None,
    bound_overrides: dict[str, list[float]] | None = None,
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
    check.

    `search_bounds` (e.g. `{"par_umol_m2_s": [0.0, 500.0]}`) replaces the
    default observed-data-derived range for named features, letting the
    active learner explore beyond whatever's been tried so far -- see
    `suggest_next_experiment_conditions`'s own docstring for when to use
    it (a campaign seeded with only a few clustered experiments so far,
    but where you know the true physical/safety range to search).

    Recommended two-step workflow when you actually want labwiki findings
    to change the suggestion, not just sit alongside it: (1) call this
    once with no `bound_overrides` to see JAXSR's baseline suggestion and
    the labwiki context together; (2) if something in the fit's own
    trends or the labwiki findings' full_content implies a genuine,
    concrete constraint (a documented problem above/below some value --
    not vague speculation), call this again *with* `bound_overrides`
    reflecting it, and present both results to the operator along with
    your reasoning for the adjustment, rather than only showing the
    adjusted one."""
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
        search_bounds=_convert_bounds_dict(search_bounds),
        bound_overrides=_convert_bounds_dict(bound_overrides),
    )
    return {
        "suggestion": {
            "points": result.suggestion.points,
            "scores": result.suggestion.scores,
            "acquisition": result.suggestion.acquisition,
            "fit": asdict(result.suggestion.fit),
            "search_bounds": result.suggestion.search_bounds,
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
    ambient_baseline_run_id: str | None = None,
) -> dict:
    """Discover how a reactor's VOC output dynamically responds to its
    LED's actual light history over one experiment -- meant to be run over
    data collected during a ramp/sinusoid/step control-profile run, where
    PAR genuinely varies within the run (a static setpoint gives it no
    within-run trend to discover). `since`/`until` are optional ISO
    datetime strings to scope to part of the experiment; omit both to use
    the whole thing. `ambient_baseline_run_id` (optional): the id from a
    prior `run_ambient_baseline_check(..., persist_run_id=...)` call --
    applies that sensor's ambient-covariate correction before calibration,
    which a synthetic benchmark with known ground truth confirmed
    meaningfully improves recovery. Prefer supplying this whenever an
    ambient-baseline check has already been run for this sensor."""
    result = _discover_led_response_dynamics(
        experiment_id,
        reactor_id,
        sensor_id,
        calibration_run_id,
        data_dir=_data_dir(),
        since=dt.datetime.fromisoformat(since) if since is not None else None,
        until=dt.datetime.fromisoformat(until) if until is not None else None,
        max_terms=max_terms,
        ambient_baseline_run_id=ambient_baseline_run_id,
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
