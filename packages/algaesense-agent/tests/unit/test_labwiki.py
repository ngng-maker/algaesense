"""Unit tests for algaesense_agent.labwiki: ingestion produces real,
cross-linked markdown content (not just files existing), search finds it,
and the lint pass catches manufactured inconsistencies.

Mirrors the Phase 2c DoD: running the ingest scenario twice produces a
wiki with two summary pages, an updated entity page, an index listing
both, and a two-line log.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from algaesense_agent.labwiki.lint import lint_labwiki
from algaesense_agent.labwiki.models import ExperimentResult
from algaesense_agent.labwiki.wiki import ingest_experiment_result, query_labwiki


def _result(experiment_id: str, par: float, voc: float) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        campaign_id="camp_01",
        reactor_id="R01",
        sensor_id="PID01",
        conditions={"par_umol_m2_s": par},
        target_metrics={"mean_voc_ppm_asgas": voc},
        fit_expression="y = 2*x + 5" if experiment_id == "exp_02" else None,
        operator_notes=["clean run"] if experiment_id == "exp_01" else [],
    )


def test_ingesting_two_experiments_produces_two_summaries_one_entity_and_a_two_line_log(
    tmp_path: Path,
) -> None:
    now = dt.datetime(2026, 7, 16, 9, 0, 0, tzinfo=dt.timezone.utc)

    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path, now=now)
    ingest_experiment_result(
        _result("exp_02", 300.0, 605.0), wiki_root=tmp_path, now=now + dt.timedelta(hours=1)
    )

    wiki_dir = tmp_path / "camp_01" / "wiki"

    summaries = sorted((wiki_dir / "summaries").glob("*.md"))
    assert [p.stem for p in summaries] == ["exp_01", "exp_02"]

    """
    The entity page for R01/PID01 should now reference BOTH experiments --
    confirms entity updates append (not overwrite) across ingest calls.
    """
    entity_text = (wiki_dir / "entities" / "R01.md").read_text(encoding="utf-8")
    assert "[[exp_01]]" in entity_text
    assert "[[exp_02]]" in entity_text

    index_text = (wiki_dir / "index.md").read_text(encoding="utf-8")
    assert "[[exp_01]]" in index_text
    assert "[[exp_02]]" in index_text

    log_lines = (wiki_dir / "log.md").read_text(encoding="utf-8").strip().splitlines()
    assert len(log_lines) == 2
    assert "exp_01" in log_lines[0]
    assert "exp_02" in log_lines[1]


def test_summary_page_content_reflects_conditions_results_fit_and_notes(tmp_path: Path) -> None:
    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path)

    summary_text = (tmp_path / "camp_01" / "wiki" / "summaries" / "exp_01.md").read_text(encoding="utf-8")

    assert "par_umol_m2_s: 200.0" in summary_text
    assert "mean_voc_ppm_asgas: 405.0" in summary_text
    assert "clean run" in summary_text
    assert "[[R01]]" in summary_text
    assert "[[PID01]]" in summary_text


def test_reingesting_the_same_experiment_updates_rather_than_duplicates(tmp_path: Path) -> None:
    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path)
    ingest_experiment_result(_result("exp_01", 200.0, 999.0), wiki_root=tmp_path)

    entity_text = (tmp_path / "camp_01" / "wiki" / "entities" / "R01.md").read_text(encoding="utf-8")
    index_text = (tmp_path / "camp_01" / "wiki" / "index.md").read_text(encoding="utf-8")

    assert entity_text.count("[[exp_01]]") == 1
    assert "999.0" in entity_text
    assert index_text.count("[[exp_01]]") == 1


def test_query_labwiki_finds_pages_mentioning_a_topic(tmp_path: Path) -> None:
    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path)
    ingest_experiment_result(_result("exp_02", 300.0, 605.0), wiki_root=tmp_path)

    matches = query_labwiki("camp_01", "par_umol_m2_s", wiki_root=tmp_path)

    matched_stems = {m.path.stem for m in matches}
    assert "exp_01" in matched_stems
    assert "exp_02" in matched_stems


def test_query_labwiki_returns_empty_for_unknown_campaign(tmp_path: Path) -> None:
    assert query_labwiki("no_such_campaign", "PAR", wiki_root=tmp_path) == []


def test_lint_labwiki_reports_no_warnings_after_normal_ingestion(tmp_path: Path) -> None:
    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path)
    ingest_experiment_result(_result("exp_02", 300.0, 605.0), wiki_root=tmp_path)

    assert lint_labwiki("camp_01", wiki_root=tmp_path) == []


def test_lint_labwiki_flags_a_stale_entity_page(tmp_path: Path) -> None:
    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path)

    """
    Manually corrupt the entity page to simulate a partially-failed
    ingest or a manual edit that dropped the reference -- the lint pass
    should catch that the raw source for exp_01 names R01, but R01's
    entity page no longer says so.
    """
    entity_path = tmp_path / "camp_01" / "wiki" / "entities" / "R01.md"
    entity_path.write_text("# Entity: R01\n\n## Experiments\n", encoding="utf-8")

    warnings = lint_labwiki("camp_01", wiki_root=tmp_path)

    assert any("R01" in w and "exp_01" in w for w in warnings)


def test_lint_labwiki_flags_an_orphaned_page(tmp_path: Path) -> None:
    ingest_experiment_result(_result("exp_01", 200.0, 405.0), wiki_root=tmp_path)

    """
    A hand-written concept page that nothing links to yet -- exactly the
    "orphaned page" case the lint pass exists to catch.
    """
    concepts_dir = tmp_path / "camp_01" / "wiki" / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    (concepts_dir / "par_findings.md").write_text("# PAR findings\n\nNothing links here yet.\n", encoding="utf-8")

    warnings = lint_labwiki("camp_01", wiki_root=tmp_path)

    assert any("par_findings" in w for w in warnings)
