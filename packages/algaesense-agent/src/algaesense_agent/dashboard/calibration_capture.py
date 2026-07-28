"""Reading live sensor values off the edge service for the calibration wizard.

Split out of the Streamlit page rather than living inside it because a
`streamlit run` script cannot be imported and patched by a test the way an
ordinary module can. Keeping the two network calls here gives the wizard's
tests a real seam to substitute a transport at, the same reason the MCP
servers build their clients through a small factory instead of inline.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import polars as pl

from algaesense_agent.mcp_actuators.edge_client import EdgeClient
from jaxsr_calibration.calibration.apply import apply_calibration


def _build_edge_client(base_url: str) -> EdgeClient:
    return EdgeClient(base_url)


def _recent(base_url: str, limit: int) -> list[dict]:
    async def _go() -> list[dict]:
        client = _build_edge_client(base_url)
        try:
            return await client.recent_voc_readings(limit=limit)
        finally:
            await client.close()

    return asyncio.run(_go())


def fetch_latest_voltage_mv(base_url: str) -> float:
    """The newest buffered PID voltage.

    One value at a time: the operator watches the bench and decides when the
    sensor has settled, so the wizard captures on their click rather than
    averaging a window it chose itself.
    """
    readings = _recent(base_url, limit=1)
    if not readings:
        raise RuntimeError("The edge service has no VOC readings buffered yet.")
    return float(readings[-1]["pid_voltage_mv"])


def fetch_recent_ppm(
    base_url: str, sensor_id: str, calibration_run_id: str, data_dir: Path, limit: int = 60
) -> list[float]:
    """Recent readings converted through a saved calibration.

    This is what closes the loop for the operator: the same sensor they just
    calibrated, reported in concentration rather than millivolts, so a bad
    calibration is visible immediately instead of at analysis time weeks
    later.
    """
    rows = _recent(base_url, limit=limit)
    if not rows:
        return []

    frame = pl.DataFrame(rows)
    height = frame.height
    t_c = frame["sample_t_c"] if "sample_t_c" in frame.columns else pl.Series([None] * height)
    rh = frame["sample_rh_pct"] if "sample_rh_pct" in frame.columns else pl.Series([None] * height)

    ppm, _, _ = apply_calibration(
        frame["pid_voltage_mv"],
        sensor_id,
        t_c,
        rh,
        calibration_run_id,
        data_dir=Path(data_dir) / "derived" / "calibrations" / "standard_addition",
    )
    return [float(v) for v in ppm]
