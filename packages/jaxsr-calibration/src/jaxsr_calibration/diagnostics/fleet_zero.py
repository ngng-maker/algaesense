"""Fleet-zero diagnostic: every sensor should read ~0 with low noise and no
drift on clean air (spec §20).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.diagnostics.models import DiagnosticStatus, FleetZeroResult, SensorStatus
from jaxsr_calibration.processing.config import DiagnosticThresholds, FleetZeroThresholds


def _classify_sensor(
    mean_mv: float, std_mv: float, slope_mv_per_min: float, limits: FleetZeroThresholds
) -> SensorStatus:
    """PASS if every metric is within its limit; FAIL if any metric is over
    `limits.fail_multiplier` times its limit; SUSPECT for anything in
    between."""

    """
    This three-tier scheme isn't spelled out numerically in the spec
    (which only says the tool "prints a per-sensor pass/fail summary" and
    describes a human retest-then-tag-SUSPECT workflow in the
    experimentalist protocol) -- it's a reasonable single-run
    approximation of that same idea: PASS/SUSPECT/FAIL per sensor,
    worst-case-wins across the three metrics.
    """

    """
    Comparing the *worst* (largest) of the three ratios-to-limit against
    1.0 and against fail_multiplier lets one bad metric drive the whole
    verdict, rather than needing all three to individually fail before we
    notice.
    """
    ratios = [
        abs(mean_mv) / limits.max_mean_mv,
        std_mv / limits.max_std_mv,
        abs(slope_mv_per_min) / limits.max_abs_slope_mv_per_min,
    ]
    worst_ratio = max(ratios)

    if worst_ratio <= 1.0:
        return "PASS"
    if worst_ratio <= limits.fail_multiplier:
        return "SUSPECT"
    return "FAIL"


def _summarize(statuses: list[SensorStatus]) -> DiagnosticStatus:
    if any(s == "FAIL" for s in statuses):
        return "RED"
    if any(s == "SUSPECT" for s in statuses):
        return "YELLOW"
    return "GREEN"


def run_fleet_zero(
    duration_min: int,
    thresholds: DiagnosticThresholds | None = None,
    output_dir: Path | None = None,
    *,
    readings: pl.DataFrame | None = None,
) -> FleetZeroResult:
    """Analyze a clean-air logging window and report each sensor's baseline
    noise characteristics plus an overall pass/fail summary."""

    """
    `duration_min` documents how long the clean-air window *should* be
    (per the spec's live-acquisition-driven design), but since live
    acquisition doesn't exist yet in this package (see algaesense-edge),
    this function does not itself collect data -- callers must pass
    already-collected data via the keyword-only `readings` argument (e.g.
    loaded from a Parquet file, or a synthetic fixture in tests). Omitting
    `readings` raises LiveAcquisitionNotAvailableError.
    """

    if readings is None:
        raise LiveAcquisitionNotAvailableError(
            "run_fleet_zero has no live-acquisition backend yet; pass "
            "readings=<a DataFrame of already-collected clean-air data> instead."
        )

    limits = (thresholds or DiagnosticThresholds()).fleet_zero

    per_sensor: dict[str, dict] = {}
    statuses: list[SensorStatus] = []

    """
    `pl.DataFrame.partition_by("sensor_id", as_dict=True)` splits one big
    DataFrame into a dict of smaller DataFrames, one per distinct
    sensor_id value -- equivalent in spirit to pandas' `.groupby
    ("sensor_id")` but returning actual separate DataFrames rather than a
    lazy GroupBy object, which is convenient here since we want to loop
    over each sensor's data and compute several different statistics from
    it.
    """
    for (sensor_id,), sensor_df in readings.partition_by("sensor_id", as_dict=True).items():
        voltage = sensor_df["pid_voltage_mv"].to_numpy()
        mean_mv = float(np.mean(voltage))
        std_mv = float(np.std(voltage))

        """
        Linear drift: fit voltage = intercept + slope * (minutes since
        first sample), and keep only the slope. `np.polyfit(x, y, 1)`
        returns `[slope, intercept]` for a degree-1 (straight-line)
        least-squares fit.
        """
        timestamps = sensor_df["timestamp"].to_list()
        t0 = min(timestamps)
        minutes_elapsed = np.array([(t - t0).total_seconds() / 60.0 for t in timestamps])
        if len(minutes_elapsed) >= 2 and minutes_elapsed.max() > 0:
            slope_mv_per_min = float(np.polyfit(minutes_elapsed, voltage, 1)[0])
        else:
            """
            Can't estimate a slope from fewer than 2 distinct time points
            -- report zero drift rather than raising, since mean/std are
            still meaningful even with very little data.
            """
            slope_mv_per_min = 0.0

        status = _classify_sensor(mean_mv, std_mv, slope_mv_per_min, limits)
        statuses.append(status)
        per_sensor[sensor_id] = {
            "mean_mv": mean_mv,
            "std_mv": std_mv,
            "slope_mv_per_min": slope_mv_per_min,
            "status": status,
        }

    result = FleetZeroResult(
        per_sensor=per_sensor,
        summary_status=_summarize(statuses),
    )

    if output_dir is not None:
        _write_result(result, output_dir)

    return result


def _write_result(result: FleetZeroResult, output_dir: Path) -> Path:
    """Persist a fleet-zero result to `{output_dir}/{run_id}.parquet`, one
    row per sensor."""

    """
    Per spec §20's "writes results to
    data/derived/diagnostics/{diagnostic_name}/{run_id}.parquet"
    convention.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    """
    A run_id derived from the current UTC time, filesystem-safe (colons
    aren't valid in Windows filenames, hence replacing them with "-").
    """
    run_id = "fleet_zero_" + dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

    rows = [
        {"sensor_id": sensor_id, **stats} for sensor_id, stats in result.per_sensor.items()
    ]
    table = pl.DataFrame(rows)
    out_path = output_dir / f"{run_id}.parquet"
    table.write_parquet(out_path)

    return out_path
