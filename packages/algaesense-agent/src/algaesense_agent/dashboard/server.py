"""MCP server exposing on-demand plot generation as a tool Hermes can call
and relay straight into Slack.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

from algaesense_agent.dashboard.campaign_plot import render_campaign_fit_plot


"""
Returning an `Image` (not a file path string) lets an MCP client render
the plot inline in the conversation directly -- Hermes doesn't need a
separate "upload this file to Slack" step, the image data is already part
of the tool's response.
"""

mcp = FastMCP("algaesense-dashboard")


def _data_dir() -> Path:
    return Path(os.environ.get("ALGAESENSE_DATA_DIR", "data"))


@mcp.tool()
def plot_campaign_fit(
    campaign_id: str,
    feature_column: str,
    target: str = "mean_voc_ppm_asgas",
    max_terms: int = 5,
) -> Image:
    """Render a plot of observed data against the current fitted model for
    one campaign and one controllable variable (e.g. PAR vs VOC output)."""
    png_bytes = render_campaign_fit_plot(
        campaign_id,
        data_dir=_data_dir(),
        feature_column=feature_column,
        target=target,
        max_terms=max_terms,
    )
    return Image(data=png_bytes, format="png")


def main() -> None:
    """Entry point for the `algaesense-mcp-dashboard` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
