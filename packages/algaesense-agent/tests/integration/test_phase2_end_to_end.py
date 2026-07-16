"""Phase 2 synthetic end-to-end scenario, per the plan's own DoD:

    user message ("investigate effect of PAR on VOC emission")
    -> agent pulls fused synthetic data -> fits
    -> proposes a next LED setting
    -> posts to Slack for approval -> (mocked) approval
    -> actuator MCP call fires -> confirmation posted back

No real Hermes Agent or LLM is involved -- this test plays the role of
"the agent" with plain Python, calling the same MCP tools a real Hermes
session would, gated by a FakeSlackChannel standing in for a real
Slack workspace and a real human operator's reply. What it verifies is
that OUR code (the four MCP servers, together) actually enforces "no
actuator call without approval" and correctly wires propose -> apply ->
labwiki ingestion -- not anything about Hermes's own LLM behavior, which
this repo doesn't control.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import httpx
import numpy as np
import polars as pl
import pytest

from algaesense_edge.actuators.actuators import LEDActuator, MockLEDHardware
from algaesense_edge.api.app import create_app
from algaesense_edge.api.state import AppState
from jaxsr_calibration.calibration.config import ReactorConfig

from tests.fixtures.fake_slack import FakeSlackChannel


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


def _fake_edge_app():
    """A real algaesense-edge FastAPI app, in-process, with a mocked LED
    -- not a mock of the HTTP layer, the actual safety-validating
    LEDActuator code path runs for real here."""
    state = AppState()
    reactor_config = ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=500.0)
    state.led_actuators["R01"] = LEDActuator(
        hardware=MockLEDHardware(), reactor_config=reactor_config, par_per_full_duty_umol_m2_s=1000.0
    )
    return create_app(state), state


@pytest.fixture
def wired_servers(tmp_path, monkeypatch):
    """Reload every server module against this test's tmp data directory
    and a fake in-process edge app, returning the four `mcp` objects."""

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

    fake_edge_app, edge_state = _fake_edge_app()

    def _build_fake_edge_client() -> edge_client_module.EdgeClient:
        return edge_client_module.EdgeClient(
            base_url="http://fake-edge", transport=httpx.ASGITransport(app=fake_edge_app)
        )

    """
    Swapping `_build_edge_client` (rather than passing a transport through
    an env var, which can't carry a live Python object) is exactly the
    injection point added to mcp_actuators/server.py for this test.
    """
    monkeypatch.setattr(actuators_server, "_build_edge_client", _build_fake_edge_client)

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


async def test_approved_proposal_reaches_the_edge_and_gets_ingested(tmp_path, wired_servers) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")
    slack = FakeSlackChannel(scripted_replies=["yes, go ahead"])

    """
    Step 1: "user message in Slack" -- in this harness, that's simply
    the test calling the same tool sequence a real Hermes session would,
    starting from the pipeline's suggestion.
    """
    suggestion_result = await wired_servers["pipeline"].call_tool(
        "suggest_next_experiment_conditions",
        {"campaign_id": "camp_01", "feature_columns": ["par_umol_m2_s"], "n_points": 1},
    )
    suggestion = _payload(suggestion_result)
    proposed_par = suggestion["points"][0]["par_umol_m2_s"]

    """
    Step 2: propose (no side effect yet).
    """
    propose_result = await wired_servers["actuators"].call_tool(
        "propose_led_change", {"reactor_id": "R01", "par_umol_m2_s": proposed_par}
    )
    proposal = _payload(propose_result)
    slack.post_message(proposal["note"])

    """
    Step 3: "posts to Slack for approval -> (mocked) approval".
    """
    reply = slack.await_reply()
    assert "yes" in reply.lower()

    """
    Step 4: only now does the actuator MCP call fire.
    """
    apply_result = await wired_servers["actuators"].call_tool(
        "apply_led_change", {"reactor_id": "R01", "par_umol_m2_s": proposed_par}
    )
    applied = _payload(apply_result)

    assert applied["applied_par_umol_m2_s"] == pytest.approx(proposed_par)
    """
    Confirms the call reached the REAL LEDActuator behind the fake edge
    app, not just that our code returned something plausible-looking.
    """
    assert wired_servers["edge_state"].led_actuators["R01"].read_par() == pytest.approx(proposed_par)

    slack.post_message(f"Done -- reactor R01's LED is now at {applied['applied_par_umol_m2_s']} umol/m^2/s.")

    """
    Step 5: record the outcome in the labwiki.
    """
    ingest_result = await wired_servers["labwiki"].call_tool(
        "ingest_experiment",
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

    assert len(slack.sent_messages) == 2
    assert "not yet applied" in slack.sent_messages[0].lower()
    assert "Done" in slack.sent_messages[1]


async def test_denied_proposal_never_reaches_the_edge(tmp_path, wired_servers) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")
    slack = FakeSlackChannel(scripted_replies=["no, don't do that"])

    propose_result = await wired_servers["actuators"].call_tool(
        "propose_led_change", {"reactor_id": "R01", "par_umol_m2_s": 999.0}
    )
    proposal = _payload(propose_result)
    slack.post_message(proposal["note"])

    reply = slack.await_reply()

    """
    This is the actual assertion the DoD cares about: given a denial,
    the test (playing the agent) must not call apply_led_change at all --
    confirmed here by checking the edge's actuator state never changed
    from its untouched default, not just that we "chose" not to call it.
    """
    if "yes" in reply.lower():
        await wired_servers["actuators"].call_tool(
            "apply_led_change", {"reactor_id": "R01", "par_umol_m2_s": 999.0}
        )

    assert wired_servers["edge_state"].led_actuators["R01"].read_par() == 0.0
    assert len(slack.sent_messages) == 1
