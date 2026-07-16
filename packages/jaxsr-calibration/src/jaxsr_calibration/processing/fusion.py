"""Dual-rate sensor fusion: joins the fast VOC PID stream (~1 Hz) with the
slow camera/biomass stream (~hourly) so every VOC reading is paired with
the most recent available biomass estimate.
"""

from __future__ import annotations

import polars as pl


"""
New in this project, not part of the original spec (which only covered a
single VOC sensor stream).
"""


def fuse_multirate(voc_timeseries: pl.DataFrame, camera_timeseries: pl.DataFrame) -> pl.DataFrame:
    """Attach the most recent camera-derived biomass reading to every VOC
    row."""

    """
    Via a backward "as-of" join on timestamp within each (experiment_id,
    reactor_id) group.

    `camera_timeseries` is expected to already carry a `biomass_signal_arb`
    column (the output of
    jaxsr_calibration.camera.calibration.apply_biomass_calibration for each
    captured frame) -- this function only handles the *temporal* join, not
    the calibration math itself.

    Adds a `biomass_reading_age_s` column: how many seconds old the
    attached biomass reading was at the time of the VOC reading it's paired
    with -- since the camera only samples hourly, this could be anywhere
    from ~0 up to just under one capture interval, and downstream consumers
    (including the eventual agent) can use it to judge how stale the paired
    value is.
    """

    """
    `join_asof` requires both frames sorted by the "on" column (within each
    "by" group) -- polars will raise if they aren't, so we sort explicitly
    here rather than relying on the caller to have already done so.
    """
    voc_sorted = voc_timeseries.sort("timestamp")

    """
    Renaming the camera frame's timestamp column avoids a naming collision
    with the VOC frame's own "timestamp" column -- after the join, both the
    VOC reading's timestamp and the camera reading's timestamp are
    available as two distinctly-named columns, which is exactly what we
    need to compute `biomass_reading_age_s` below.
    """
    camera_sorted = camera_timeseries.sort("timestamp").rename({"timestamp": "camera_timestamp"})

    fused = voc_sorted.join_asof(
        camera_sorted,
        left_on="timestamp",
        right_on="camera_timestamp",
        by=["experiment_id", "reactor_id"],
        strategy="backward",
    )

    """
    `(a - b).dt.total_seconds()` on two polars Datetime columns gives the
    difference as a plain numeric column (seconds) rather than a polars
    Duration object -- easier for downstream code (and for writing this
    straight into a Parquet feature file) to work with directly.
    """
    fused = fused.with_columns(
        (pl.col("timestamp") - pl.col("camera_timestamp"))
        .dt.total_seconds()
        .alias("biomass_reading_age_s")
    )

    return fused
