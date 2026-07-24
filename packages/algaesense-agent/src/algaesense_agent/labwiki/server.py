"""MCP server exposing labwiki ingestion, search, and consistency checks."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from algaesense_agent.labwiki.lint import lint_labwiki
from algaesense_agent.labwiki.models import ExperimentResult
from algaesense_agent.labwiki.wiki import ingest_experiment_result, query_labwiki, render_summary_page


"""
`apply_ingest_experiment` runs once `mcp_pipeline`'s fit/suggest tools
complete for an experiment and the operator has confirmed the result --
it is idempotent (see wiki.py's entity/index update logic), so calling it
again for the same experiment_id updates rather than duplicates.
`propose_ingest_experiment` (below) gives the operator something concrete
to review first: labwiki entries are meant to hold a vetted analysis, not
whatever the agent drafted on the first pass, so this follows the same
propose/apply split every hardware-touching tool in this package already
uses, even though writing a markdown file has no hardware safety risk --
the risk here is a wrong or premature conclusion getting baked into the
knowledge base other experiments will later be reasoned over.
"""

mcp = FastMCP("algaesense-labwiki")


def _wiki_root() -> Path:
    """Where the labwiki's per-campaign directories live."""

    """
    `ALGAESENSE_LABWIKI_ROOT` mirrors the same env-var-configuration
    pattern as `ALGAESENSE_DATA_DIR`/`ALGAESENSE_EDGE_BASE_URL` in the
    other servers in this package -- defaults to the layout the plan
    describes (`data/labwiki/{campaign_id}/...`).
    """
    return Path(os.environ.get("ALGAESENSE_LABWIKI_ROOT", "data/labwiki"))


def _build_result(
    experiment_id: str,
    campaign_id: str,
    reactor_id: str,
    sensor_id: str,
    conditions: dict[str, float],
    target_metrics: dict[str, float],
    fit_expression: str | None,
    active_learning_proposal: dict | None,
    operator_notes: list[str] | None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        campaign_id=campaign_id,
        reactor_id=reactor_id,
        sensor_id=sensor_id,
        conditions=conditions,
        target_metrics=target_metrics,
        fit_expression=fit_expression,
        active_learning_proposal=active_learning_proposal,
        operator_notes=operator_notes or [],
    )


@mcp.tool()
def propose_ingest_experiment(
    experiment_id: str,
    campaign_id: str,
    reactor_id: str,
    sensor_id: str,
    conditions: dict[str, float],
    target_metrics: dict[str, float],
    fit_expression: str | None = None,
    active_learning_proposal: dict | None = None,
    operator_notes: list[str] | None = None,
) -> dict:
    """Preview the labwiki summary page this experiment would produce --
    writes NOTHING yet. Show `preview_summary_markdown` to the operator for
    review/edits (especially `fit_expression`/`operator_notes`, the
    AI-drafted parts); once they confirm, call `apply_ingest_experiment`
    with the same (or edited) arguments to actually save it."""
    result = _build_result(
        experiment_id, campaign_id, reactor_id, sensor_id, conditions, target_metrics,
        fit_expression, active_learning_proposal, operator_notes,
    )
    return {
        "preview_summary_markdown": render_summary_page(result),
        "note": "Not yet saved to labwiki -- confirm with the operator, then call apply_ingest_experiment with the same (or edited) arguments.",
    }


@mcp.tool()
def apply_ingest_experiment(
    experiment_id: str,
    campaign_id: str,
    reactor_id: str,
    sensor_id: str,
    conditions: dict[str, float],
    target_metrics: dict[str, float],
    fit_expression: str | None = None,
    active_learning_proposal: dict | None = None,
    operator_notes: list[str] | None = None,
) -> dict:
    """Record one completed experiment's result into the labwiki: writes
    its raw source and updates the summary, entity, index, and log pages.
    Only call this after the operator has reviewed `propose_ingest_experiment`'s
    preview and confirmed (or edited) it."""
    result = _build_result(
        experiment_id, campaign_id, reactor_id, sensor_id, conditions, target_metrics,
        fit_expression, active_learning_proposal, operator_notes,
    )
    summary = ingest_experiment_result(result, wiki_root=_wiki_root())
    return {
        "raw_path": str(summary.raw_path),
        "summary_path": str(summary.summary_path),
        "entity_paths": [str(p) for p in summary.entity_paths],
        "index_path": str(summary.index_path),
        "log_path": str(summary.log_path),
    }


@mcp.tool()
def query_labwiki_topic(campaign_id: str, topic: str) -> list[dict]:
    """Search a campaign's labwiki for pages mentioning `topic`, e.g. "what
    have we learned about PAR so far?" -> topic="PAR"."""
    matches = query_labwiki(campaign_id, topic, wiki_root=_wiki_root())
    return [{"path": str(m.path), "matching_lines": m.matching_lines} for m in matches]


@mcp.tool()
def lint_labwiki_consistency(campaign_id: str) -> list[str]:
    """Check one campaign's labwiki for orphaned pages and entity pages
    that are missing a reference to an experiment that named them."""
    return lint_labwiki(campaign_id, wiki_root=_wiki_root())


def main() -> None:
    """Entry point for the `algaesense-mcp-labwiki` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
