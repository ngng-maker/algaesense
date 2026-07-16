"""Unit tests for algaesense_agent.dashboard: plot rendering and the
campaign-fit orchestration that feeds it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from algaesense_agent.dashboard.campaign_plot import render_campaign_fit_plot
from algaesense_agent.dashboard.plots import render_fit_curve_png


def test_render_fit_curve_png_produces_a_valid_png() -> None:
    x = np.linspace(0.0, 10.0, 20)
    y = 2.0 * x + 1.0

    png_bytes = render_fit_curve_png(
        x=x,
        y=y,
        predict_fn=lambda x_grid: 2.0 * x_grid + 1.0,
        x_label="x",
        y_label="y",
        title="test plot",
    )

    """
    A PNG file's first 8 bytes are a fixed "magic number" every valid PNG
    starts with -- checking it is a cheap, reliable way to confirm this
    actually rendered a real image rather than, say, an empty buffer or a
    matplotlib error silently producing garbage bytes.
    """
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png_bytes) > 100


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


def test_render_campaign_fit_plot_produces_a_valid_png(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")

    png_bytes = render_campaign_fit_plot(
        "camp_01",
        data_dir=tmp_path,
        feature_column="par_umol_m2_s",
        target="mean_voc_ppm_asgas",
    )

    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


async def test_plot_campaign_fit_tool_returns_an_image(tmp_path: Path, monkeypatch) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")
    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path))

    import importlib

    from algaesense_agent.dashboard import server as server_module

    importlib.reload(server_module)

    result = await server_module.mcp.call_tool(
        "plot_campaign_fit", {"campaign_id": "camp_01", "feature_column": "par_umol_m2_s"}
    )

    """
    An `Image`-returning tool comes back as an `ImageContent` block (not
    `TextContent`, unlike the dict-returning tools elsewhere in this
    package) -- checking for the PNG magic number in its base64-decoded
    data confirms the whole path (env var -> data loading -> fit ->
    render -> MCP image content) actually works end to end.
    """
    import base64

    image_block = result[0]
    decoded = base64.b64decode(image_block.data)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
