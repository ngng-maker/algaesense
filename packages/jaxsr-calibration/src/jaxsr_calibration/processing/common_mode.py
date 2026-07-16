"""Common-mode subtraction: remove whatever signal all sensors share in
common at each instant (room-wide VOC drift, a shared HVAC cycle, etc.),
leaving each sensor's *distinctive* response.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from jaxsr_calibration.processing.errors import CommonModeInsufficientFleetError

CommonModeMethod = Literal["median", "trimmed_mean"]

"""
Fraction of readings trimmed from each end for method="trimmed_mean" --
not spec-mandated (the spec's config only exposes outlier_std_threshold and
min_healthy_fraction, not a trim proportion), a conventional middle-ground
choice (trim the most extreme 10% from each tail before averaging).
"""
_TRIM_PROPORTION = 0.1


def subtract_common_mode(
    df: pl.DataFrame,
    method: CommonModeMethod = "median",
    outlier_std_threshold: float = 3.0,
    min_healthy_fraction: float = 0.75,
    value_column: str = "pid_voltage_mv",
) -> pl.DataFrame:
    """At every timestamp, subtract the fleet's common-mode signal from
    every sensor's reading, excluding outliers from the common-mode
    estimate itself."""

    """
    At every distinct timestamp, exclude sensors whose reading is more than
    `outlier_std_threshold` standard deviations from that timestamp's
    cross-sensor mean, compute the "common mode" (median or trimmed mean) of
    the remaining healthy sensors, and subtract it from every sensor's
    reading at that timestamp (healthy or not -- an excluded sensor still
    gets the correction applied, it just didn't get a *vote* in computing it).

    `min_healthy_fraction` is checked against the FULL configured fleet size
    (`df["sensor_id"].n_unique()`), not just however many sensors happen to
    have a row at a given timestamp -- a sensor missing entirely from a
    timestamp counts against the healthy fraction too, since "sensor didn't
    report" is at least as concerning as "sensor reported something odd".

    Raises CommonModeInsufficientFleetError at the first timestamp where
    the healthy fraction drops below the threshold.

    Known limitation, confirmed while writing this function's tests: with a
    SMALL fleet (this project's hardware protocol recommends as few as 4
    units), a single badly-offset sensor inflates the group's own std enough
    that its z-score can stay surprisingly low -- e.g. a +50 mV outlier
    among 4 otherwise-agreeing sensors only reaches z ~= 1.7, under the
    default outlier_std_threshold=3.0, so it would NOT be excluded at
    default settings ("masking", a well-known weakness of plain std-based
    outlier detection on small samples). A more robust alternative would use
    a median-absolute-deviation-based scale estimate instead of the standard
    deviation, which resists masking much better -- left as a follow-up
    rather than done here, since it changes the numeric meaning of
    `outlier_std_threshold` and should be a deliberate choice, not a quiet
    swap. In the meantime, a smaller `outlier_std_threshold` (e.g. 1.5-2.0)
    is recommended for small fleets.
    """

    n_sensors_total = df["sensor_id"].n_unique()

    corrected_frames = []

    """
    Partitioning by exact timestamp assumes sensors are logging on a
    shared, synchronized clock (all sensors' readings at "the same instant"
    share the exact same timestamp value) -- true for this project's ~1 Hz
    VOC acquisition design, per the raw schema in
    jaxsr_calibration.logging_.schema.
    """
    for (timestamp,), group in df.partition_by("timestamp", as_dict=True).items():
        values = group[value_column].to_numpy()
        mean_v = float(np.mean(values))
        std_v = float(np.std(values))

        if std_v == 0.0:
            """
            Every sensor reported exactly the same value -- trivially,
            nothing is an outlier (avoids a divide-by-zero in the z-score
            comparison below).
            """
            healthy_mask = np.ones_like(values, dtype=bool)
        else:
            healthy_mask = np.abs(values - mean_v) <= outlier_std_threshold * std_v

        healthy_fraction = healthy_mask.sum() / n_sensors_total
        if healthy_fraction < min_healthy_fraction:
            raise CommonModeInsufficientFleetError(
                f"at {timestamp}: only {healthy_mask.sum()}/{n_sensors_total} sensors "
                f"({healthy_fraction:.0%}) survived outlier exclusion, below the "
                f"required {min_healthy_fraction:.0%}"
            )

        healthy_values = values[healthy_mask]

        if method == "median":
            common_value = float(np.median(healthy_values))
        elif method == "trimmed_mean":
            """
            `scipy.stats.trim_mean(a, proportiontocut)` sorts `a`, discards
            the lowest and highest `proportiontocut` fraction of values from
            each end, and averages what's left -- a mean that's more robust
            to remaining extreme values than a plain average, but (unlike
            the median) still uses more than just the single middle value.
            """
            common_value = float(scipy_stats.trim_mean(healthy_values, _TRIM_PROPORTION))
        else:
            raise ValueError(f"Unknown method: {method!r}")

        corrected_column = pl.Series(f"{value_column}_common_mode_subtracted", values - common_value)
        corrected_frames.append(group.with_columns(corrected_column))

    return pl.concat(corrected_frames)
