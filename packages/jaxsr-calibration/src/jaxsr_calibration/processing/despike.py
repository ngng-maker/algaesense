"""Separate real transient VOC events from per-instrument glitches, using
agreement between simultaneous sensors.

A PID trace regularly contains sharp excursions. Some are genuine -- a
disturbance, a feed, a door opening -- and some are the instrument's own
electrical noise or a dropout. On a single trace the two are frequently
indistinguishable: same duration, same shape, same amplitude range.
Filtering blindly therefore risks deleting exactly the transient biology
an experiment exists to capture.

What distinguishes them is physical rather than statistical. A real
emission is in the GAS, so every sensor sampling that gas registers it at
the same moment. A glitch is in ONE instrument's electronics, so no other
sensor sees it. Coincidence across independent sensors is the signature of
something real, and it is the reason to run several sensors on one reactor
rather than one.

This module never deletes data. It adds columns describing what it found
and, separately, a despiked copy of the signal -- so a caller can inspect
the classification, disagree with it, and still reach the original values.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from jaxsr_calibration.validation import require_columns


_REQUIRED_COLUMNS = frozenset({"timestamp", "sensor_id"})

_MAD_TO_SIGMA = 1.4826
"""Scales a median absolute deviation to a standard-deviation equivalent
for normally distributed data. Using MAD rather than the plain standard
deviation matters here: the spikes we are trying to find would themselves
inflate an ordinary sigma, making a large excursion look unremarkable
against the noise floor it created."""


def _within_run(mask: np.ndarray, min_length: int) -> np.ndarray:
    """True only where `mask` is part of an unbroken run of at least
    `min_length` samples."""
    if min_length <= 1:
        return mask.copy()

    out = np.zeros_like(mask)
    start = 0
    while start < mask.size:
        if not mask[start]:
            start += 1
            continue
        stop = start
        while stop < mask.size and mask[stop]:
            stop += 1
        if stop - start >= min_length:
            out[start:stop] = True
        start = stop
    return out


def flag_glitches_across_sensors(
    df: pl.DataFrame,
    *,
    value_column: str = "pid_voltage_mv",
    window: int = 11,
    z_threshold: float = 6.0,
    min_coincident_sensors: int = 2,
    require_matching_sign: bool = True,
    min_event_samples: int = 1,
) -> pl.DataFrame:
    """Classify sharp excursions as real events or per-sensor glitches.

    Returns `df` with four added columns: `local_residual` (deviation from
    the sensor's own local level), `is_outlier` (this sensor saw an
    excursion), `n_coincident_sensors` (how many sensors saw one at that
    same timestamp), and `is_glitch` (an excursion no other sensor
    corroborates). A despiked copy of the signal is added as
    `{value_column}_despiked`, with glitches -- and only glitches --
    replaced by the local level.

    Real events are left untouched in the despiked column, because they
    are signal.

    Two further tests sharpen what counts as corroboration, because bare
    simultaneity is weak evidence -- two instruments glitching in the same
    sample by chance looks identical to a shared event, and over a long run
    that will happen.

    `require_matching_sign` only counts a neighbouring sensor as
    corroborating when its excursion goes the SAME direction. A shared gas
    event cannot push one sensor up while pushing another down, so an
    opposite-signed neighbour is not evidence of anything. On by default:
    it is a statement about physics rather than a tuning choice.

    `min_event_samples` additionally requires the corroboration to persist
    for that many consecutive samples. Gas has to mix and a PID has a
    response time, so a real transient spans several samples at any
    sensible sampling rate, while electrical glitches are usually one.
    Defaults to 1 (off), because whether "several" means two samples or
    twenty depends entirely on the sampling interval, and switching it on
    by default would start deleting single-sample excursions this function
    has always kept.
    """

    require_columns(df, _REQUIRED_COLUMNS | {value_column}, "flag_glitches_across_sensors")

    n_sensors = df["sensor_id"].n_unique()

    """
    With too few sensors to corroborate anything, coincidence is simply
    undecidable -- and the dangerous failure is silent: every real event
    would be flagged as a glitch and imputed away, quietly deleting the
    transient biology this function exists to protect. So the safe answer
    is to detect outliers, report them, and classify NONE of them as
    glitches. The caller still gets `is_outlier` to work with.
    """
    coincidence_decidable = n_sensors >= min_coincident_sensors

    frames = []
    for (sensor_id,), sensor_df in df.partition_by("sensor_id", as_dict=True).items():
        sensor_df = sensor_df.sort("timestamp")
        values = sensor_df[value_column].to_numpy().astype(float)

        local_level = _rolling_median(values, window)
        residual = values - local_level

        """
        The scale is estimated from the residual, not the raw signal, so a
        genuine slow trend (a culture climbing toward its plateau, a
        thermal drift) does not inflate it and mask the spikes riding on
        top.
        """
        mad = float(np.median(np.abs(residual - np.median(residual))))
        sigma = _MAD_TO_SIGMA * mad

        if sigma <= 0.0:
            """
            A perfectly flat residual gives a zero scale, which would make
            every non-zero deviation infinitely significant. Nothing is an
            outlier against a signal with no variation to speak of.
            """
            is_outlier = np.zeros(values.size, dtype=bool)
        else:
            is_outlier = np.abs(residual) > z_threshold * sigma

        frames.append(
            sensor_df.with_columns(
                pl.Series("local_level", local_level),
                pl.Series("local_residual", residual),
                pl.Series("is_outlier", is_outlier),
            )
        )

    flagged = pl.concat(frames)

    """
    Coincidence is counted on exact timestamp equality. That holds for
    sensors logged on a shared acquisition tick, which is how this
    project's edge service records them. Sensors on independent clocks
    would need a tolerance window first -- resampling onto a common grid
    before calling this is the straightforward fix, and doing it here
    would hide a real assumption inside a filter.
    """
    flagged = flagged.with_columns(
        pl.when(pl.col("is_outlier"))
        .then(pl.col("local_residual").sign())
        .otherwise(0.0)
        .alias("outlier_direction")
    )

    coincidence = flagged.group_by("timestamp").agg(
        pl.col("is_outlier").sum().alias("n_outliers"),
        (pl.col("outlier_direction") > 0).sum().alias("n_up"),
        (pl.col("outlier_direction") < 0).sum().alias("n_down"),
    )
    flagged = flagged.join(coincidence, on="timestamp", how="left")

    if require_matching_sign:
        n_coincident = (
            pl.when(pl.col("outlier_direction") > 0)
            .then(pl.col("n_up"))
            .when(pl.col("outlier_direction") < 0)
            .then(pl.col("n_down"))
            .otherwise(0)
        )
    else:
        n_coincident = pl.col("n_outliers")

    flagged = flagged.with_columns(n_coincident.alias("n_coincident_sensors")).drop(
        "n_outliers", "n_up", "n_down"
    )

    if coincidence_decidable:
        corroborated = pl.col("is_outlier") & (
            pl.col("n_coincident_sensors") >= min_coincident_sensors
        )
    else:
        corroborated = pl.col("is_outlier")

    flagged = flagged.with_columns(corroborated.alias("_corroborated"))

    """
    The duration test runs per sensor over its own ordered samples, so a
    run is a genuinely consecutive stretch of that instrument's readings
    rather than an artifact of how the frame happens to be sorted.
    """
    sustained_frames = []
    for (_,), sensor_df in flagged.partition_by("sensor_id", as_dict=True).items():
        sensor_df = sensor_df.sort("timestamp")
        sustained = _within_run(
            sensor_df["_corroborated"].to_numpy().astype(bool), min_event_samples
        )
        sustained_frames.append(sensor_df.with_columns(pl.Series("_sustained", sustained)))
    flagged = pl.concat(sustained_frames)

    if coincidence_decidable:
        is_glitch = pl.col("is_outlier") & ~pl.col("_sustained")
    else:
        is_glitch = pl.lit(False)

    flagged = flagged.with_columns(is_glitch.alias("is_glitch")).drop(
        "_corroborated", "_sustained"
    )

    """
    Only glitches are replaced, and they are replaced by the sensor's own
    local level rather than dropped -- keeping the series on a regular
    time grid, which every downstream consumer here (covariate
    correction, derivative estimation, resampling) assumes.
    """
    despiked = (
        pl.when(pl.col("is_glitch"))
        .then(pl.col("local_level"))
        .otherwise(pl.col(value_column))
        .alias(f"{value_column}_despiked")
    )

    return flagged.with_columns(despiked).drop("local_level").sort(["sensor_id", "timestamp"])


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median, with the window shrinking at the edges so
    the output is the same length as the input and the first and last
    samples are still given a local level."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    half = window // 2
    out = np.empty(values.size, dtype=float)
    for i in range(values.size):
        lo = max(i - half, 0)
        hi = min(i + half + 1, values.size)
        out[i] = np.median(values[lo:hi])
    return out
