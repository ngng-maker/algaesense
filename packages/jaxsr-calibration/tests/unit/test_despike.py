"""Unit tests for coincidence-gated despiking.

The property that matters is asymmetric: removing a glitch is a
convenience, but removing a REAL event destroys the transient biology an
experiment exists to capture. So the tests below check event preservation
at least as hard as glitch removal, including the degenerate
single-sensor case where coincidence cannot be judged at all.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.processing.despike import flag_glitches_across_sensors


BASE_TIME = dt.datetime(2026, 7, 27, 8, 0, 0, tzinfo=dt.timezone.utc)
N = 200


def _readings(
    sensor_ids: list[str],
    *,
    event_index: int | None = None,
    glitch: dict[str, int] | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """A flat baseline per sensor, optionally with one SHARED event (all
    sensors, same index) and per-sensor glitches at given indices."""
    rng = np.random.default_rng(seed)
    timestamps = [BASE_TIME + dt.timedelta(seconds=300 * i) for i in range(N)]

    frames = []
    for offset, sensor_id in enumerate(sensor_ids):
        voltage = 300.0 + 10.0 * offset + rng.normal(0.0, 1.0, size=N)
        if event_index is not None:
            voltage[event_index] += 120.0
        if glitch and sensor_id in glitch:
            voltage[glitch[sensor_id]] += 150.0
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": [sensor_id] * N,
                    "pid_voltage_mv": voltage,
                }
            )
        )
    return pl.concat(frames)


def test_a_shared_event_is_kept_not_despiked() -> None:
    """An excursion every sensor sees at the same instant is real signal."""
    df = _readings(["PID01", "PID02", "PID03"], event_index=100)

    out = flag_glitches_across_sensors(df)
    at_event = out.filter(pl.col("timestamp") == BASE_TIME + dt.timedelta(seconds=300 * 100))

    assert at_event["is_outlier"].all()
    assert not at_event["is_glitch"].any()

    """The despiked signal must still contain the event, unchanged."""
    assert at_event["pid_voltage_mv"].to_list() == at_event["pid_voltage_mv_despiked"].to_list()


def test_a_single_sensor_glitch_is_flagged_and_removed() -> None:
    """An excursion no other sensor corroborates is that instrument's own."""
    df = _readings(["PID01", "PID02", "PID03"], glitch={"PID02": 50})

    out = flag_glitches_across_sensors(df)
    at_glitch = out.filter(pl.col("timestamp") == BASE_TIME + dt.timedelta(seconds=300 * 50))

    glitched = at_glitch.filter(pl.col("sensor_id") == "PID02")
    assert glitched["is_glitch"].item() is True

    """Replaced by the local level, so it no longer stands 150 mV proud."""
    assert abs(glitched["pid_voltage_mv_despiked"].item() - 300.0 - 10.0) < 10.0

    others = at_glitch.filter(pl.col("sensor_id") != "PID02")
    assert not others["is_glitch"].any()


def test_events_and_glitches_are_separated_in_the_same_recording() -> None:
    """The realistic case: both kinds present, only one kind removed."""
    df = _readings(["PID01", "PID02", "PID03"], event_index=100, glitch={"PID01": 40})

    out = flag_glitches_across_sensors(df)

    event_rows = out.filter(pl.col("timestamp") == BASE_TIME + dt.timedelta(seconds=300 * 100))
    glitch_rows = out.filter(
        (pl.col("timestamp") == BASE_TIME + dt.timedelta(seconds=300 * 40))
        & (pl.col("sensor_id") == "PID01")
    )

    assert not event_rows["is_glitch"].any()
    assert glitch_rows["is_glitch"].item() is True


def test_a_single_sensor_recording_never_classifies_anything_as_a_glitch() -> None:
    """With one sensor coincidence is undecidable, and the unsafe failure
    is silent: every real event would be imputed away. Outliers are still
    reported, but nothing is deleted."""
    df = _readings(["PID01"], event_index=100)

    out = flag_glitches_across_sensors(df)

    assert out["is_outlier"].any(), "the excursion should still be detected"
    assert not out["is_glitch"].any(), "but nothing may be removed without corroboration"
    assert out["pid_voltage_mv"].to_list() == out["pid_voltage_mv_despiked"].to_list()


def test_a_clean_recording_is_left_alone() -> None:
    """No excursions means no changes -- a despiker that quietly rewrites
    ordinary noise would be worse than none."""
    df = _readings(["PID01", "PID02", "PID03"])

    out = flag_glitches_across_sensors(df)

    assert not out["is_glitch"].any()
    assert out["pid_voltage_mv"].to_list() == pytest.approx(
        out["pid_voltage_mv_despiked"].to_list()
    )


def test_missing_required_column_is_rejected_clearly() -> None:
    df = _readings(["PID01", "PID02"]).drop("sensor_id")

    with pytest.raises(Exception) as exc:
        flag_glitches_across_sensors(df)
    assert "sensor_id" in str(exc.value)
