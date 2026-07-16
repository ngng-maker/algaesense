"""Data shape for one experiment's result, as handed to the labwiki for
ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentResult:
    """Everything about one completed experiment worth remembering in the
    labwiki."""

    """
    Deliberately a plain, JSON/YAML-serializable dataclass (no jax arrays,
    no live model objects) -- this is what actually gets written to
    `raw/{experiment_id}.yaml` as the immutable source record the wiki
    pages are built from, per the Karpathy LLM-wiki pattern's "raw sources
    are immutable, derived pages are rebuilt/updated from them" split.
    """

    experiment_id: str
    campaign_id: str
    reactor_id: str
    sensor_id: str
    conditions: dict[str, float]
    target_metrics: dict[str, float]
    fit_expression: str | None = None
    active_learning_proposal: dict | None = None
    operator_notes: list[str] = field(default_factory=list)
