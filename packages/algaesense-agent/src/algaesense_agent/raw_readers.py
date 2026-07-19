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
