"""MCP server wrapping jaxsr-calibration's sensor-health diagnostics
(fleet-zero, ambient baseline, swap-pilot, weekly audit) over already-
collected raw Parquet data. Read-only, no hardware or live experiments
touched.
"""

from algaesense_agent.mcp_diagnostics.diagnostics import (
    NoRawReadingsFoundError,
    ambient_baseline_check,
    fleet_zero_check,
    load_raw_voc_readings,
    swap_pilot_check,
    weekly_audit_check,
)

__all__ = [
    "load_raw_voc_readings",
    "fleet_zero_check",
    "ambient_baseline_check",
    "swap_pilot_check",
    "weekly_audit_check",
    "NoRawReadingsFoundError",
]
