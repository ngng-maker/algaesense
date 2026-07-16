"""Top-level package for algaesense-agent."""

"""
Runs on the "brain" server, not the Raspberry Pi. Hosts two MCP servers
(`mcp_pipeline`, `mcp_actuators`) that a separately-installed Hermes Agent
connects to as a client, plus the labwiki knowledge base and dashboard
plotting tool those servers use. This package never talks to hardware
directly -- `mcp_actuators` talks to algaesense-edge's network API, and
`mcp_pipeline` talks to jaxsr-calibration's own functions in-process.
"""

__version__ = "0.1.0"
