"""Loading raw, per-experiment Parquet files algaesense-edge already
wrote -- shared by any `algaesense-agent` module that needs an
experiment's un-derived sensor data (mcp_diagnostics, mcp_pipeline).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


class NoRawReadingsFoundError(FileNotFoundError):
    """Raised when no raw Parquet files exist yet for the requested
    experiment."""


def load_raw_voc_readings(data_dir: Path, experiment_id: str) -> pl.DataFrame:
    """Load and concatenate every sensor's raw VOC readings for one
    experiment."""

    """
    Globs `sensor_id=*` (every sensor that reported during this
    experiment) rather than requiring the caller to name sensors up
    front. Read via polars, not raw pyarrow -- this directory layout's
    own `sensor_id=PID01`-style partition naming is exactly what triggers
    the Hive-partitioning false positive documented in
    CLAUDE.md/writer.py; polars has no such auto-detection.
    """
    experiment_dir = Path(data_dir) / "raw" / "experiments" / experiment_id
    paths = sorted(experiment_dir.glob("sensor_id=*/hour=*.parquet")) if experiment_dir.exists() else []

    if not paths:
        raise NoRawReadingsFoundError(
            f"No raw VOC Parquet files found for experiment {experiment_id!r} under {experiment_dir}"
        )

    return pl.concat([pl.read_parquet(p) for p in paths])


def load_raw_camera_readings(data_dir: Path, experiment_id: str) -> pl.DataFrame:
    """Load and concatenate every camera's raw readings for one experiment
    -- same shape/reasoning as `load_raw_voc_readings`, just for the
    `camera_id=*` partition instead of `sensor_id=*`."""
    experiment_dir = Path(data_dir) / "raw" / "experiments" / experiment_id
    paths = sorted(experiment_dir.glob("camera_id=*/hour=*.parquet")) if experiment_dir.exists() else []

    if not paths:
        raise NoRawReadingsFoundError(
            f"No raw camera Parquet files found for experiment {experiment_id!r} under {experiment_dir}"
        )

    return pl.concat([pl.read_parquet(p) for p in paths])


def list_raw_experiment_ids(data_dir: Path) -> list[str]:
    """List every experiment_id that has raw data under `data_dir` --
    just the subdirectory names under `raw/experiments/`, sorted so a
    caller building a picker gets a stable, predictable order."""
    experiments_dir = Path(data_dir) / "raw" / "experiments"
    if not experiments_dir.exists():
        return []
    return sorted(p.name for p in experiments_dir.iterdir() if p.is_dir())
