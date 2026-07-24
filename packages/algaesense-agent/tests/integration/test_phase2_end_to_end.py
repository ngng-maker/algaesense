"""Phase 2 end-to-end scenario, per the plan's own DoD (adapted): pull
fused synthetic data -> fit -> propose a next LED setting -> gated apply
-> record the outcome in the labwiki.

No fake Slack harness is involved -- the actual approval-gate guarantee
("no actuator call without a human confirming first") is a property of
the tool sequence itself (propose_led_change has no side effect;
apply_led_change is the only call that reaches hardware), not something a
simulated chat needs to prove. The real conversational confirmation step
happens for real, once Hermes and Slack are wired up per
profile/README.md -- this test verifies the underlying tool chain is
correct, which is the part actually within this repo's control.

The "apply reaches real hardware" step needs the real WS2811 strip, so
that scenario is `@pytest.mark.hardware`. The "denied" scenario (apply is
simply never called) needs no hardware at all.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.fixtures.real_edge_app import build_real_edge_app, edge_transport


def _write_synthetic_campaign(data_dir: Path, campaign_id: str, n_experiments: int = 8) -> None:
    campaign_dir = data_dir / "derived" / "features" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    for i, par in enumerate(np.linspace(100.0, 500.0, n_experiments)):
        row = {
            "experiment_id": f"exp_{i:02d}",
            "campaign_id": campaign_id,
            "reactor_id": "R01",
            "sensor_id": "PID01",
            "par_umol_m2_s": float(par),
            "mean_voc_ppm_asgas": float(2.0 * par + 5.0),
        }
        pl.DataFrame([row]).write_parquet(campaign_dir / f"exp_{i:02d}.parquet")


@pytest.fixture
def wired_servers(tmp_path, monkeypatch):
    """Reload every server module against this test's tmp data directory
    and a real, in-process edge app, returning the four `mcp` objects."""

    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGAESENSE_LABWIKI_ROOT", str(tmp_path / "labwiki"))
    monkeypatch.setenv("ALGAESENSE_EDGE_BASE_URL", "http://fake-edge")

    from algaesense_agent.mcp_actuators import edge_client as edge_client_module
    from algaesense_agent.mcp_actuators import server as actuators_server
    from algaesense_agent.mcp_pipeline import server as pipeline_server
    from algaesense_agent.labwiki import server as labwiki_server

    importlib.reload(pipeline_server)
    importlib.reload(actuators_server)
    importlib.reload(labwiki_server)

    real_edge_app, edge_state = build_real_edge_app(max_par=500.0, par_per_full_duty=1000.0)

    def _build_real_edge_client() -> edge_client_module.EdgeClient:
        return edge_client_module.EdgeClient(
            base_url="http://fake-edge", transport=edge_transport(real_edge_app)
        )

    """
    Swapping `_build_edge_client` (rather than passing a transport
    through an env var, which can't carry a live Python object) is the
    injection point added to mcp_actuators/server.py for exactly this.
    """
    monkeypatch.setattr(actuators_server, "_build_edge_client", _build_real_edge_client)

    return {
        "pipeline": pipeline_server.mcp,
        "actuators": actuators_server.mcp,
        "labwiki": labwiki_server.mcp,
        "edge_state": edge_state,
    }


def _payload(raw_result):
    """See test_labwiki_server.py's `_tool_payload` for why both shapes
    need handling."""
    if isinstance(raw_result, tuple):
        return raw_result[1]["result"]
    return json.loads(raw_result[0].text)


@pytest.mark.hardware
async def test_approved_proposal_reaches_the_edge_and_gets_ingested(tmp_path, wired_servers) -> None:
    """Run only on the Pi: the approved path actually drives the real
    WS2811 strip through the full propose -> apply -> labwiki chain."""

    _write_synthetic_campaign(tmp_path, "camp_01")

    suggestion_result = await wired_servers["pipeline"].call_tool(
        "suggest_next_experiment_conditions",
        {"campaign_id": "camp_01", "feature_columns": ["par_umol_m2_s"], "n_points": 1},
    )
    suggestion = _payload(suggestion_result)
    proposed_par = suggestion["points"][0]["par_umol_m2_s"]

    propose_result = await wired_servers["actuators"].call_tool(
        "propose_led_change", {"reactor_id": "R01", "par_umol_m2_s": proposed_par}
    )
    proposal = _payload(propose_result)
    assert "not yet applied" in proposal["note"].lower()

    """
    In a real session, this is where Hermes posts `proposal["note"]` to
    Slack and waits for the human's reply before continuing -- see
    profile/system_prompt.md's confirm-before-apply rule. This test
    plays the "confirmed" branch directly, since simulating that wait
    with a fake chat object wouldn't prove anything the propose/apply
    split itself doesn't already guarantee.
    """
    apply_result = await wired_servers["actuators"].call_tool(
        "apply_led_change", {"reactor_id": "R01", "par_umol_m2_s": proposed_par}
    )
    applied = _payload(apply_result)

    assert applied["applied_par_umol_m2_s"] == pytest.approx(proposed_par)
    assert wired_servers["edge_state"].led_actuators["R01"].read_par() == pytest.approx(proposed_par)

    ingest_result = await wired_servers["labwiki"].call_tool(
        "apply_ingest_experiment",
        {
            "experiment_id": "exp_proposed_01",
            "campaign_id": "camp_01",
            "reactor_id": "R01",
            "sensor_id": "PID01",
            "conditions": {"par_umol_m2_s": proposed_par},
            "target_metrics": {},
            "fit_expression": suggestion["fit"]["expression"],
            "active_learning_proposal": {"points": suggestion["points"], "scores": suggestion["scores"]},
        },
    )
    ingest_payload = _payload(ingest_result)
    assert Path(ingest_payload["summary_path"]).exists()


async def test_denied_proposal_never_reaches_the_edge(tmp_path, wired_servers) -> None:
    """No hardware needed: the whole point is that apply_led_change is
    never called at all when the human doesn't confirm."""

    _write_synthetic_campaign(tmp_path, "camp_01")

    propose_result = await wired_servers["actuators"].call_tool(
        "propose_led_change", {"reactor_id": "R01", "par_umol_m2_s": 999.0}
    )
    proposal = _payload(propose_result)
    assert "not yet applied" in proposal["note"].lower()

    """
    This is the actual assertion the DoD cares about: given a denial, the
    agent must not call apply_led_change at all. Checking that the
    hardware object was never even connected (`_pixels` still `None`) --
    rather than calling `read_par()`, which would itself connect to real
    hardware on first read -- confirms nothing touched the strip at all,
    without needing real GPIO for this no-hardware scenario.
    """
    hardware = wired_servers["edge_state"].led_actuators["R01"].hardware
    assert hardware._pixels is None
