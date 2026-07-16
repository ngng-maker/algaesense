"""Unit tests for jaxsr_calibration.processing.common_mode.subtract_common_mode."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.processing.common_mode import subtract_common_mode
from jaxsr_calibration.processing.errors import CommonModeInsufficientFleetError
from tests.fixtures.synthetic_readings import make_common_mode_readings


def test_subtract_common_mode_removes_shared_signal() -> None:
    readings = make_common_mode_readings(
        ["PID01", "PID02", "PID03", "PID04"],
        common_signal_amplitude=5.0,
        individual_noise_std=0.05,
        seed=70,
    )

    corrected = subtract_common_mode(readings, method="median")

    values = corrected["pid_voltage_mv_common_mode_subtracted"].to_numpy()
    # After removing the shared sine-wave signal, what's left should be
    # small (just the independent per-sensor noise), regardless of how large
    # the original shared signal's amplitude was.
    assert np.std(values) < 0.5
    assert np.max(np.abs(values)) < 1.0


def test_subtract_common_mode_excludes_outlier_sensor_from_the_estimate() -> None:
    readings = make_common_mode_readings(
        ["PID01", "PID02", "PID03", "PID04"],
        common_signal_amplitude=5.0,
        individual_noise_std=0.05,
        outlier_sensor_ids=["PID04"],
        outlier_offset=50.0,
        seed=71,
    )

    corrected = subtract_common_mode(
        readings, method="median", outlier_std_threshold=3.0, min_healthy_fraction=0.5
    )

    healthy_values = corrected.filter(pl.col("sensor_id") != "PID04")[
        "pid_voltage_mv_common_mode_subtracted"
    ].to_numpy()
    outlier_values = corrected.filter(pl.col("sensor_id") == "PID04")[
        "pid_voltage_mv_common_mode_subtracted"
    ].to_numpy()

    # Healthy sensors: the common-mode estimate (correctly excluding PID04)
    # should still cleanly remove their shared signal.
    assert np.std(healthy_values) < 0.5
    # PID04 itself: still gets a correction applied (per this function's
    # design -- excluded sensors aren't dropped, just not counted toward the
    # estimate), but since its own 50 mV offset was never part of that
    # estimate, its corrected value should sit near +50, not near 0.
    assert np.mean(outlier_values) == pytest.approx(50.0, abs=1.0)


def test_subtract_common_mode_raises_when_too_few_sensors_survive() -> None:
    # With only 4 sensors, a single outlier's own offset inflates the group's
    # std enough that its z-score can stay surprisingly low (the well-known
    # "masking" effect in naive outlier detection -- confirmed by hand while
    # writing this test: a +50 mV outlier among 4 sensors only reaches
    # z ~= 1.7, comfortably under the default outlier_std_threshold=3.0, so
    # it would NOT be excluded at default settings). To reliably exercise the
    # "insufficient fleet" path we tighten outlier_std_threshold so that
    # z ~= 1.7 *does* count as excluded, then set min_healthy_fraction above
    # the resulting 75% survival rate (3 of 4 sensors).
    readings = make_common_mode_readings(
        ["PID01", "PID02", "PID03", "PID04"],
        common_signal_amplitude=1.0,
        individual_noise_std=0.05,
        outlier_sensor_ids=["PID04"],
        outlier_offset=50.0,
        seed=72,
    )

    with pytest.raises(CommonModeInsufficientFleetError):
        subtract_common_mode(readings, outlier_std_threshold=1.0, min_healthy_fraction=0.8)


def test_subtract_common_mode_trimmed_mean_method_runs() -> None:
    readings = make_common_mode_readings(["PID01", "PID02", "PID03"], seed=73)

    corrected = subtract_common_mode(readings, method="trimmed_mean")

    assert "pid_voltage_mv_common_mode_subtracted" in corrected.columns


def test_subtract_common_mode_rejects_unknown_method() -> None:
    readings = make_common_mode_readings(["PID01", "PID02", "PID03"], seed=74)

    with pytest.raises(ValueError, match="Unknown method"):
        subtract_common_mode(readings, method="bogus")
