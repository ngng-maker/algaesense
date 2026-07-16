"""Dataclasses returned by the diagnostics subpackage (spec §34)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from jaxsr_calibration.processing.covariate import CovariateModel


"""
FleetZeroResult, AmbientBaselineResult, and SwapPilotResult field shapes
are taken directly from the spec. WeeklyAuditResult's fields are NOT given
explicitly in the spec (only its function signature,
`run_weekly_audit(output_markdown=None) -> WeeklyAuditResult`, is) -- the
shape below is a reasonable design covering the three checks the spec's
prose names for the weekly audit (§23: sensor-id variance-share drift,
backup currency, lamp-cleaning due dates).
"""


"""
A three-tier status shared across every diagnostic's summary_status
field -- defined once here rather than repeated as a literal string union
in each dataclass below, so all diagnostics agree on the exact same three
spellings.
"""
DiagnosticStatus = Literal["GREEN", "YELLOW", "RED"]

"""
Per-sensor status is a separate, finer-grained vocabulary (matches the
PASS/SUSPECT/FAIL status already used for calibration results elsewhere in
the spec) -- deliberately distinct from the three-tier summary above so
"one sensor's status" and "the whole fleet's status" are never confused
for the same kind of value.
"""
SensorStatus = Literal["PASS", "SUSPECT", "FAIL"]


@dataclass
class FleetZeroResult:
    """Result of run_fleet_zero: every sensor's baseline noise
    characteristics on clean air, plus an overall pass/fail summary."""

    """
    `dict[str, dict]` per the spec's own type -- in practice each inner
    dict has the keys "mean_mv", "std_mv", "slope_mv_per_min", "status".
    """
    per_sensor: dict[str, dict]

    summary_status: DiagnosticStatus


@dataclass
class AmbientBaselineResult:
    """Result of run_ambient_baseline: a fitted CovariateModel per sensor,
    ready to be handed to apply_covariate_correction."""

    covariate_models: dict[str, CovariateModel]

    r_squared_per_sensor: dict[str, float]


@dataclass
class SwapPilotResult:
    """Result of run_swap_pilot: how much of the total signal variance is
    explained by sensor identity vs. reactor identity vs. unexplained
    residual."""

    """
    e.g. {"sensor_id": 0.18, "reactor_id": 0.12, "residual": 0.70} -- the
    three values are designed to sum to (approximately) 1.0.
    """
    variance_share: dict[str, float]

    """
    The full human-readable statsmodels MixedLM summary text, kept around
    for an operator to read directly rather than only seeing the three
    extracted numbers above.
    """
    mixedlm_summary: str


@dataclass
class WeeklyAuditResult:
    """Result of run_weekly_audit."""

    """
    Not given an explicit field list in the spec -- this shape covers the
    three checks the spec's prose (§23, §27) names: sensor-id
    variance-share drift, backup currency, and which sensors are due for
    lamp cleaning.
    """

    summary_status: DiagnosticStatus

    """
    None if this is the first audit on record (nothing to compare against
    yet) -- otherwise the change in sensor-id variance share since the
    previous swap-pilot result, e.g. +0.05 meaning it got 5 percentage
    points worse.
    """
    sensor_variance_share_delta: float | None

    sensors_due_for_cleaning: list[str] = field(default_factory=list)

    backup_current: bool | None = None

    report_path: Path | None = None
