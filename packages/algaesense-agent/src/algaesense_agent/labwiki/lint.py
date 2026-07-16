"""Mechanical consistency checks over one campaign's labwiki."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from algaesense_agent.labwiki.wiki import _raw_dir, _wiki_dir


"""
Only checks the two things a mechanical pass can actually verify without
understanding the content: every page is reachable via a wikilink from
somewhere else, and every ingested experiment's reactor/sensor entity page
actually references it. Recognizing *contradictory* findings across
`concepts/` pages needs judgment, not pattern matching -- see
SCHEMA.md's "Lint pass" section for why that's deliberately left to the
agent reading whatever this flags, not attempted here.
"""

_WIKILINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")


def _all_wikilinks(wiki_dir: Path) -> set[str]:
    links: set[str] = set()
    for path in wiki_dir.rglob("*.md"):
        links.update(_WIKILINK_PATTERN.findall(path.read_text(encoding="utf-8")))
    return links


def _find_orphaned_pages(wiki_dir: Path) -> list[str]:
    if not wiki_dir.exists():
        return []

    linked = _all_wikilinks(wiki_dir)
    warnings: list[str] = []

    for subdir_name in ("summaries", "entities", "concepts"):
        subdir = wiki_dir / subdir_name
        if not subdir.exists():
            continue
        for path in sorted(subdir.glob("*.md")):
            if path.stem not in linked:
                warnings.append(
                    f"orphaned page: {path.relative_to(wiki_dir)} is not linked from any other page"
                )

    return warnings


def _find_stale_entity_pages(campaign_id: str, wiki_root: Path) -> list[str]:
    raw_dir = _raw_dir(wiki_root, campaign_id)
    wiki_dir = _wiki_dir(wiki_root, campaign_id)
    warnings: list[str] = []

    if not raw_dir.exists():
        return warnings

    for raw_path in sorted(raw_dir.glob("*.yaml")):
        result = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
        experiment_id = result["experiment_id"]

        for entity_id in (result["reactor_id"], result["sensor_id"]):
            entity_path = wiki_dir / "entities" / f"{entity_id}.md"
            referenced = entity_path.exists() and f"[[{experiment_id}]]" in entity_path.read_text(encoding="utf-8")
            if not referenced:
                warnings.append(
                    f"stale entity page: entities/{entity_id}.md does not reference "
                    f"experiment [[{experiment_id}]]"
                )

    return warnings


def lint_labwiki(campaign_id: str, wiki_root: Path) -> list[str]:
    """Return a list of human-readable warnings; an empty list means the
    campaign's labwiki is internally consistent by these two checks."""
    wiki_dir = _wiki_dir(wiki_root, campaign_id)
    return _find_orphaned_pages(wiki_dir) + _find_stale_entity_pages(campaign_id, wiki_root)
