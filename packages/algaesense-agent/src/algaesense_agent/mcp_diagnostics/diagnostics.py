"""Read-only bridge from raw VOC Parquet files to jaxsr_calibration's
sensor-health diagnostics.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import yaml
from jaxsr_calibration.calibration.config import SensorConfig
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


"""
Every diagnostic function in jaxsr_calibration expects already-collected
data (a `readings=` DataFrame) -- see jaxsr_calibration.errors for why
(they raise LiveAcquisitionNotAvailableError without it). This module's
whole job is loading that DataFrame from the real raw Parquet files
algaesense-edge already wrote, then calling straight through to the real
diagnostic; none of the actual health-check math lives here.
"""


class NoRawReadingsFoundError(FileNotFoundError):
    """Raised when no raw VOC Parquet files exist yet for the requested
    experiment."""


def load_raw_voc_readings(data_dir: Path, experiment_id: str) -> pl.DataFrame:
    """Load and concatenate every sensor's raw VOC readings for one
    experiment -- the "clean-air" or "ambient" diagnostic run, not a
    growth experiment."""

    """
    Globs `{partition}=*` (every sensor that reported during this
    experiment) rather than requiring the caller to name sensors up
    front -- a fleet-zero/ambient check is inherently about the whole
    fleet, per-sensor breakdown included, not one sensor at a time.
    Read via polars, not raw pyarrow -- this directory layout's own
    `sensor_id=PID01`-style partition naming is exactly what triggers the
    Hive-partitioning false positive documented in CLAUDE.md/writer.py;
    polars has no such auto-detection.
    """
    experiment_dir = Path(data_dir) / "raw" / "experiments" / experiment_id
    paths = sorted(experiment_dir.glob("sensor_id=*/hour=*.parquet")) if experiment_dir.exists() else []

    if not paths:
        raise NoRawReadingsFoundError(
            f"No raw VOC Parquet files found for experiment {experiment_id!r} under {experiment_dir}"
        )

    return pl.concat([pl.read_parquet(p) for p in paths])


def fleet_zero_check(data_dir: Path, experiment_id: str, duration_min: int = 60) -> FleetZeroResult:
    """Run the real fleet-zero diagnostic over one experiment's already-
    collected clean-air readings."""
    readings = load_raw_voc_readings(data_dir, experiment_id)
    return run_fleet_zero(duration_min=duration_min, readings=readings)


def ambient_baseline_check(
    data_dir: Path, experiment_id: str, duration_h: int = 12, method: str = "ols"
) -> AmbientBaselineResult:
    """Run the real ambient-baseline diagnostic over one experiment's
    already-collected ambient-air readings."""
    readings = load_raw_voc_readings(data_dir, experiment_id)
    return run_ambient_baseline(duration_h=duration_h, method=method, readings=readings)


def swap_pilot_check(data_dir: Path, experiment_id: str, n_blocks: int = 4) -> SwapPilotResult:
    """Run the real swap-pilot diagnostic over one experiment's already-
    collected sensor/reactor rotation readings."""
    readings = load_raw_voc_readings(data_dir, experiment_id)
    return run_swap_pilot(n_blocks=n_blocks, readings=readings)


def _load_sensor_configs(sensors_yaml_path: Path) -> list[SensorConfig]:
    """Load `configs/sensors.yaml` (a top-level `sensors:` list, one
    entry per sensor, matching SensorConfig's fields) into real
    SensorConfig models."""
    raw = yaml.safe_load(Path(sensors_yaml_path).read_text(encoding="utf-8"))
    return [SensorConfig(**entry) for entry in raw["sensors"]]


def weekly_audit_check(
    swap_pilot_variance_shares: list[dict[str, float]] | None = None,
    sensors_yaml_path: Path | None = None,
    backup_current: bool | None = None,
    lamp_cleaning_age_days: int = 180,
    today: dt.date | None = None,
) -> WeeklyAuditResult:
    """Compose the real weekly audit rollup from already-computed
    swap-pilot results (oldest first) and the sensor fleet's config."""

    """
    `swap_pilot_variance_shares` are plain dicts (e.g. from a previous
    `swap_pilot_check` call's `variance_share` field, or a prior week's
    saved result) rather than requiring the caller to reconstruct a full
    SwapPilotResult -- `run_weekly_audit` only ever reads `.variance_share`
    off each one, so `mixedlm_summary` is left blank here; it isn't used
    for anything this function computes.
    """
    swap_pilot_results = (
        [SwapPilotResult(variance_share=share, mixedlm_summary="") for share in swap_pilot_variance_shares]
        if swap_pilot_variance_shares
        else None
    )
    sensor_configs = _load_sensor_configs(sensors_yaml_path) if sensors_yaml_path is not None else None

    return run_weekly_audit(
        swap_pilot_results=swap_pilot_results,
        sensor_configs=sensor_configs,
        backup_current=backup_current,
        lamp_cleaning_age_days=lamp_cleaning_age_days,
        today=today,
    )
