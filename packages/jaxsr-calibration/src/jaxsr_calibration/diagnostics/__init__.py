"""Sensor-health diagnostics: fleet-zero, ambient baseline, swap pilot, weekly audit."""

"""
Every diagnostic function that analyzes sensor data (fleet_zero, ambient,
swap_pilot) currently requires already-collected data via a keyword-only
`readings=` argument rather than acquiring it live -- see
jaxsr_calibration.errors for why (live acquisition is algaesense_edge, a
Raspberry-Pi-side package planned for a later phase). I2C bus scanning
(scan_i2c) is not here -- it's real hardware I/O and lives in
algaesense_edge instead, not in this hardware-agnostic analysis package.
"""

from jaxsr_calibration.diagnostics.ambient import run_ambient_baseline
from jaxsr_calibration.diagnostics.fleet_zero import run_fleet_zero
from jaxsr_calibration.diagnostics.models import (
    AmbientBaselineResult,
    FleetZeroResult,
    SwapPilotResult,
    WeeklyAuditResult,
)
from jaxsr_calibration.diagnostics.swap_pilot import run_swap_pilot
from jaxsr_calibration.diagnostics.weekly import run_weekly_audit

__all__ = [
    "run_fleet_zero",
    "run_ambient_baseline",
    "run_swap_pilot",
    "run_weekly_audit",
    "FleetZeroResult",
    "AmbientBaselineResult",
    "SwapPilotResult",
    "WeeklyAuditResult",
]
