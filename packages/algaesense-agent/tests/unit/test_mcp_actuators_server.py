"""Unit tests for the mcp_actuators FastMCP server's tool wrappers.

Only covers the tools that need no network access (propose_led_change,
the not-implemented stubs) -- apply_led_change's server-level wiring
against a real edge service is covered by the Phase 2 end-to-end test
instead, since it needs a full fake-edge harness.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from algaesense_agent.mcp_actuators.server import mcp


async def test_propose_led_change_tool_returns_a_structured_proposal() -> None:
    result = await mcp.call_tool("propose_led_change", {"reactor_id": "R01", "par_umol_m2_s": 250.0})

    payload = json.loads(result[0].text)

    assert payload["reactor_id"] == "R01"
    assert payload["requested_value"] == 250.0


async def test_propose_led_profile_change_tool_returns_a_structured_proposal() -> None:
    profile = {"shape": "constant", "par_umol_m2_s": 100.0}

    result = await mcp.call_tool("propose_led_profile_change", {"reactor_id": "R01", "profile": profile})

    payload = json.loads(result[0].text)
    assert payload["reactor_id"] == "R01"
    assert payload["profile"] == profile


async def test_propose_temperature_change_tool_reports_not_implemented() -> None:
    """
    FastMCP re-raises a tool's exception as a `ToolError` rather than
    returning it as a normal successful result -- confirmed by running
    this test before assuming otherwise. Checking for the explanatory
    text in the raised error is what matters here.
    """
    with pytest.raises(ToolError, match="no temperature-control hardware"):
        await mcp.call_tool("propose_temperature_change", {"reactor_id": "R01", "temperature_c": 30.0})
