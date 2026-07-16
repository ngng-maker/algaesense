"""Unit test for the labwiki FastMCP server's tool wrappers."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def _tool_payload(raw_result):
    """Extract a tool call's actual Python return value.

    `FastMCP.call_tool` returns either a bare `list[ContentBlock]` (for
    tools with no separately-inferred output schema, e.g. a plain dict
    return -- exactly one content block, parse its JSON text) or a
    `(blocks, {"result": ...})` tuple (when a structured output schema
    was inferred, e.g. a `list[...]`-typed return, where each list item
    gets its own content block) -- in the tuple case, the metadata dict's
    "result" key already holds the real parsed Python value, so use that
    directly rather than trying to reassemble it from possibly-multiple
    text blocks.
    """
    if isinstance(raw_result, tuple):
        return raw_result[1]["result"]
    return json.loads(raw_result[0].text)


async def test_ingest_and_query_tools_round_trip_through_the_server(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALGAESENSE_LABWIKI_ROOT", str(tmp_path))

    from algaesense_agent.labwiki import server as server_module

    importlib.reload(server_module)

    ingest_result = await server_module.mcp.call_tool(
        "ingest_experiment",
        {
            "experiment_id": "exp_01",
            "campaign_id": "camp_01",
            "reactor_id": "R01",
            "sensor_id": "PID01",
            "conditions": {"par_umol_m2_s": 200.0},
            "target_metrics": {"mean_voc_ppm_asgas": 405.0},
        },
    )
    ingest_payload = _tool_payload(ingest_result)
    assert Path(ingest_payload["summary_path"]).exists()

    query_result = await server_module.mcp.call_tool(
        "query_labwiki_topic", {"campaign_id": "camp_01", "topic": "par_umol_m2_s"}
    )
    query_payload = _tool_payload(query_result)
    assert any("exp_01" in match["path"] for match in query_payload)

    lint_result = await server_module.mcp.call_tool("lint_labwiki_consistency", {"campaign_id": "camp_01"})
    assert _tool_payload(lint_result) == []
