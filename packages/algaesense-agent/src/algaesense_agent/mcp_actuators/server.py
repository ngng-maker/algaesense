"""MCP server exposing algaesense-edge's actuator API as propose/apply tools."""

from __future__ import annotations

import os
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from algaesense_agent.mcp_actuators.actuators import (
    apply_led_setpoint,
    propose_led_setpoint,
    propose_stirring_setpoint,
    propose_temperature_setpoint,
)
from algaesense_agent.mcp_actuators.edge_client import EdgeClient


"""
Only `apply_led_setpoint` below is capable of a side effect (it reaches a
real Raspberry Pi over the network). Every other tool in this server is
safe to call freely -- see actuators.py's module docstring for the full
reasoning behind the propose/apply split.
"""

mcp = FastMCP("algaesense-actuators")


def _edge_base_url() -> str:
    """Which algaesense-edge instance this server proxies to."""

    """
    `ALGAESENSE_EDGE_BASE_URL` lets Hermes's `~/.hermes/config.yaml`
    point this server at the actual reactor's edge service (e.g.
    `http://192.168.1.42:8000`) without editing code -- defaults to
    localhost for local development against a mock-hardware edge
    instance.
    """
    return os.environ.get("ALGAESENSE_EDGE_BASE_URL", "http://localhost:8000")


def _build_edge_client() -> EdgeClient:
    """Construct the EdgeClient `apply_led_change` talks through."""

    """
    A separate, overridable function (rather than inlining
    `EdgeClient(_edge_base_url())` directly in the tool below) so an
    end-to-end test can monkeypatch this one function to return a client
    backed by an in-process fake edge service (`httpx.ASGITransport`
    against algaesense_edge's real FastAPI app), exercising the real MCP
    tool call instead of only the plain function underneath it.
    """
    return EdgeClient(_edge_base_url())


@mcp.tool()
def propose_led_change(reactor_id: str, par_umol_m2_s: float) -> dict:
    """Describe a proposed LED setpoint change for a reactor, without
    applying it. Always call this before apply_led_change and show the
    result to the user for confirmation."""
    return asdict(propose_led_setpoint(reactor_id, par_umol_m2_s))


@mcp.tool()
async def apply_led_change(reactor_id: str, par_umol_m2_s: float) -> dict:
    """Actually apply an LED setpoint change on a reactor. Only call this
    after the user has explicitly confirmed the corresponding
    propose_led_change result."""
    edge = _build_edge_client()
    try:
        return await apply_led_setpoint(edge, reactor_id, par_umol_m2_s)
    finally:
        await edge.close()


@mcp.tool()
def propose_temperature_change(reactor_id: str, temperature_c: float) -> dict:
    """Not implemented -- no temperature-control hardware exists yet."""

    """
    Raises ActuatorNotImplementedError, which FastMCP converts into a
    normal MCP tool-error response for the caller -- no special handling
    needed here, same as any other tool exception.
    """
    return asdict(propose_temperature_setpoint(reactor_id, temperature_c))


@mcp.tool()
def propose_stirring_change(reactor_id: str, speed_rpm: float) -> dict:
    """Not implemented -- no stirring-control hardware exists yet."""
    return asdict(propose_stirring_setpoint(reactor_id, speed_rpm))


def main() -> None:
    """Entry point for the `algaesense-mcp-actuators` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
