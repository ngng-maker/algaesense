"""Reference-jar cross-sensor drift tracking: a shared permeation source
every sensor visits in turn, so their readings can be compared to each
other without a full standard-addition recalibration each time.
"""

from __future__ import annotations

import polars as pl

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.calibration.models import CalibrationGas


"""
Same split as standard_addition.py: `compute_fleet_ratios` is real, pure
math; `run_reference_jar_rotation` (physically disconnecting/reconnecting
each sensor to the jar, live) is a stub -- see LiveAcquisitionNotAvailableError
(jaxsr_calibration.errors) for why.
"""


def run_reference_jar_rotation(
    sensors: list[str] | str = "all",
    dwell_min: int = 10,
    reference_gas: CalibrationGas | None = None,
) -> pl.DataFrame:
    """Drive the weekly reference-jar rotation: prompt the operator to
    disconnect one sensor at a time, connect it to the reference jar, dwell,
    record, then reconnect it -- repeated for every sensor in `sensors`."""

    """
    Same "needs live hardware + a human present" situation as
    run_standard_addition; see that function's technical block for the
    general reasoning. If you already have reference-jar readings as a
    DataFrame (columns: sensor_id, pid_voltage_mv), call
    compute_fleet_ratios(readings) directly instead.
    """
    raise LiveAcquisitionNotAvailableError(
        "run_reference_jar_rotation needs to drive a live, interactive "
        "disconnect/dwell/reconnect procedure per sensor (needs "
        "algaesense-edge, a later phase). If you already have reference-jar "
        "readings as a DataFrame, call compute_fleet_ratios(readings) directly."
    )


def compute_fleet_ratios(readings: pl.DataFrame, value_column: str = "pid_voltage_mv") -> dict[str, float]:
    """Given reference-jar readings for one or more sensors, return each
    sensor's ratio to the fleet median."""

    """
    A ratio of 1.0 means "reads exactly like the fleet's typical sensor";
    the experimentalist protocol (§21) flags any sensor drifting past ±20%
    (i.e. outside roughly 0.8-1.2) between weekly checks as due for lamp
    cleaning -- that threshold check itself is left to the caller, this
    function only computes the ratios.
    """

    if readings.height == 0:
        raise ValueError("compute_fleet_ratios: readings is empty")

    """
    `.group_by("sensor_id").agg(...)` collapses possibly-multiple rows per
    sensor (several readings during its dwell window) down to one mean
    value per sensor, the same shape a ratio calculation needs.
    """
    per_sensor = readings.group_by("sensor_id").agg(pl.col(value_column).mean().alias("mean_value"))
    fleet_median = per_sensor["mean_value"].median()

    if fleet_median == 0:
        raise ValueError(
            "compute_fleet_ratios: fleet median reading is exactly 0, can't compute ratios "
            "(every sensor read zero at the reference jar -- check the jar/permeation source)."
        )

    return {
        row["sensor_id"]: row["mean_value"] / fleet_median
        for row in per_sensor.iter_rows(named=True)
    }
