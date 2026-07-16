"""Ingest experiment results into a durable, human-readable markdown
knowledge base, and search it back.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from algaesense_agent.labwiki.models import ExperimentResult


"""
Follows the Karpathy LLM-wiki pattern (see the package's own SCHEMA.md):
raw sources are immutable and never rewritten once written; the wiki
layer (index.md, log.md, summaries/, entities/) is derived from them and
updated incrementally as new experiments come in. Everything in this
module is deterministic template-filling, not LLM-authored prose -- v1's
"knowledge base" is durable and testable first; richer synthesized
`concepts/` pages are for the agent itself to write later, using its own
file-editing tools, once there's enough raw material to synthesize from.
"""


def _campaign_dir(wiki_root: Path, campaign_id: str) -> Path:
    return Path(wiki_root) / campaign_id


def _raw_dir(wiki_root: Path, campaign_id: str) -> Path:
    return _campaign_dir(wiki_root, campaign_id) / "raw"


def _wiki_dir(wiki_root: Path, campaign_id: str) -> Path:
    return _campaign_dir(wiki_root, campaign_id) / "wiki"


@dataclass
class IngestSummary:
    """Which files an ingest call created or updated -- handed back so a
    caller (or a test) can confirm exactly what happened, not just that
    "ingestion succeeded"."""

    raw_path: Path
    summary_path: Path
    entity_paths: list[Path]
    index_path: Path
    log_path: Path


def ingest_experiment_result(
    result: ExperimentResult, wiki_root: Path, now: dt.datetime | None = None
) -> IngestSummary:
    """Record one experiment's result: write its immutable raw source,
    then create/update the summary, entity, index, and log pages that
    reference it."""

    """
    `now` is injectable (rather than always `dt.datetime.now(...)`) so
    tests get fully deterministic log.md content, same pattern already
    used by jaxsr_calibration.diagnostics.weekly's `today` parameter.
    """
    now = now or dt.datetime.now(dt.timezone.utc)

    raw_path = _write_raw_source(result, wiki_root)
    summary_path = _write_summary_page(result, wiki_root)
    entity_paths = [
        _update_entity_page(entity_id, result, wiki_root)
        for entity_id in (result.reactor_id, result.sensor_id)
    ]
    index_path = _update_index(result, wiki_root)
    log_path = _append_log(result, wiki_root, now)

    return IngestSummary(
        raw_path=raw_path,
        summary_path=summary_path,
        entity_paths=entity_paths,
        index_path=index_path,
        log_path=log_path,
    )


def _write_raw_source(result: ExperimentResult, wiki_root: Path) -> Path:
    raw_dir = _raw_dir(wiki_root, result.campaign_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{result.experiment_id}.yaml"

    """
    `sort_keys=False` preserves ExperimentResult's own field order in the
    written file (experiment_id, campaign_id, reactor_id, ... in the order
    the dataclass declares them) rather than yaml's default alphabetical
    re-sort -- reads more naturally for a human opening this file, same
    reasoning as jaxsr_calibration.camera.calibration's persistence
    function.
    """
    path.write_text(yaml.safe_dump(asdict(result), sort_keys=False), encoding="utf-8")

    return path


def _write_summary_page(result: ExperimentResult, wiki_root: Path) -> Path:
    summaries_dir = _wiki_dir(wiki_root, result.campaign_id) / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    path = summaries_dir / f"{result.experiment_id}.md"

    conditions_lines = "\n".join(f"- {k}: {v}" for k, v in result.conditions.items()) or "- (none recorded)"
    results_lines = "\n".join(f"- {k}: {v}" for k, v in result.target_metrics.items()) or "- (none recorded)"
    notes_lines = "\n".join(f"- {note}" for note in result.operator_notes) or "- (none)"
    fit_line = result.fit_expression or "(not yet fit)"

    """
    `[[wikilinks]]` (double square brackets around campaign_id/reactor_id/
    sensor_id) follow the gist's own cross-referencing convention -- any
    markdown viewer/tool that understands wikilinks (or a future labwiki
    lint/query pass) can follow "which page does this reference" without
    needing a separate link database.
    """
    path.write_text(
        f"""# Experiment {result.experiment_id}

- Campaign: [[{result.campaign_id}]]
- Reactor: [[{result.reactor_id}]]
- Sensor: [[{result.sensor_id}]]

## Conditions

{conditions_lines}

## Results

{results_lines}

## Fit

{fit_line}

## Notes

{notes_lines}
""",
        encoding="utf-8",
    )

    return path


def _update_entity_page(entity_id: str, result: ExperimentResult, wiki_root: Path) -> Path:
    entities_dir = _wiki_dir(wiki_root, result.campaign_id) / "entities"
    entities_dir.mkdir(parents=True, exist_ok=True)
    path = entities_dir / f"{entity_id}.md"

    metrics_summary = ", ".join(f"{k}={v}" for k, v in result.target_metrics.items())
    entry_line = f"- [[{result.experiment_id}]]: {metrics_summary}"

    if not path.exists():
        path.write_text(f"# Entity: {entity_id}\n\n## Experiments\n\n{entry_line}\n", encoding="utf-8")
        return path

    existing = path.read_text(encoding="utf-8")

    """
    Idempotent: re-ingesting the same experiment (e.g. after correcting
    a value) replaces its one line rather than appending a duplicate --
    every entry line is uniquely identified by its `[[experiment_id]]`
    wikilink, so finding "does a line for this experiment already exist"
    is a simple substring check.
    """
    if f"[[{result.experiment_id}]]" in existing:
        lines = existing.splitlines()
        lines = [entry_line if f"[[{result.experiment_id}]]" in line else line for line in lines]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text(existing.rstrip("\n") + "\n" + entry_line + "\n", encoding="utf-8")

    return path


def _update_index(result: ExperimentResult, wiki_root: Path) -> Path:
    wiki_dir = _wiki_dir(wiki_root, result.campaign_id)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    path = wiki_dir / "index.md"

    entry_line = f"- [[{result.experiment_id}]] ({result.reactor_id}/{result.sensor_id})"

    if not path.exists():
        path.write_text(
            f"# Campaign {result.campaign_id}\n\n## Experiments\n\n{entry_line}\n", encoding="utf-8"
        )
        return path

    existing = path.read_text(encoding="utf-8")
    if entry_line not in existing:
        path.write_text(existing.rstrip("\n") + "\n" + entry_line + "\n", encoding="utf-8")

    return path


def _append_log(result: ExperimentResult, wiki_root: Path, now: dt.datetime) -> Path:
    wiki_dir = _wiki_dir(wiki_root, result.campaign_id)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    path = wiki_dir / "log.md"

    """
    Append-only, one line per ingest event -- never edited or reordered
    after being written, so `log.md` is always a truthful chronological
    record of what was ingested when, independent of whatever the
    derived pages (index/summaries/entities) currently look like.
    """
    line = f"- {now.isoformat()} ingested [[{result.experiment_id}]] ({result.campaign_id})\n"

    with path.open("a", encoding="utf-8") as f:
        f.write(line)

    return path


@dataclass
class QueryMatch:
    """One markdown page that matched a labwiki search, with the specific
    lines that matched."""

    path: Path
    matching_lines: list[str] = field(default_factory=list)


def query_labwiki(campaign_id: str, topic: str, wiki_root: Path) -> list[QueryMatch]:
    """Find wiki pages mentioning `topic`, e.g. "what have we learned
    about PAR so far?" -> topic="PAR"."""

    """
    A plain case-insensitive substring search across every markdown page
    under this campaign's wiki directory -- deliberately not
    embeddings/RAG, per the gist's own philosophy that a small,
    well-organized directory of markdown pages doesn't need a vector
    database to be searchable; the agent (or a human) reads the matched
    pages directly rather than getting a synthesized answer from this
    function.
    """
    wiki_dir = _wiki_dir(wiki_root, campaign_id)
    if not wiki_dir.exists():
        return []

    topic_lower = topic.lower()
    matches: list[QueryMatch] = []

    for path in sorted(wiki_dir.rglob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        matching_lines = [line for line in lines if topic_lower in line.lower()]
        if matching_lines:
            matches.append(QueryMatch(path=path, matching_lines=matching_lines))

    return matches
