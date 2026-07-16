"""MCP server exposing labwiki ingestion, search, and consistency checks."""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from algaesense_agent.labwiki.lint import lint_labwiki
from algaesense_agent.labwiki.models import ExperimentResult
from algaesense_agent.labwiki.wiki import ingest_experiment_result, query_labwiki


"""
`ingest_experiment_result` is meant to be called automatically right after
`mcp_pipeline`'s fit/suggest tools complete for an experiment (per the
plan's Phase 2c design) -- it is idempotent (see wiki.py's entity/index
update logic), so calling it again for the same experiment_id updates
rather than duplicates.
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


@mcp.tool()
def ingest_experiment(
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
    its raw source and updates the summary, entity, index, and log
    pages."""
    result = ExperimentResult(
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
