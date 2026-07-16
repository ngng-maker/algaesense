"""Unit tests for jaxsr_calibration.calibration.apply.apply_calibration."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from jaxsr_calibration.calibration.apply import apply_calibration, persist_calibration
from jaxsr_calibration.calibration.models import CalibrationUnitUnavailableError
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from tests.fixtures.synthetic_readings import make_standard_addition_readings


@pytest.fixture
def calibrated_run(tmp_path: Path) -> tuple[str, Path]:
    """Fit and persist one real calibration run, reused by several tests
    below via pytest's fixture mechanism (a function that pytest calls once
    per test that names it as a parameter, injecting the return value)."""
    df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 2.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.02}},
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        calibration_compound="isoprene",
        response_factor=0.63,
        seed=50,
    )
    models = fit_sensitivity_per_sensor(df)
    out_dir = tmp_path / "calibrations"
    persist_calibration(models, "cal_test_run", "exp_test", out_dir)
    return "cal_test_run", out_dir


def test_apply_calibration_inverts_voltage_to_known_ppm(calibrated_run: tuple[str, Path]) -> None:
    run_id, data_dir = calibrated_run
    # True line was voltage = 2.0 + 4.0*ppm, so voltage=22.0 should invert to
    # ppm=(22.0-2.0)/4.0=5.0.
    voltage = pl.Series([22.0])
    t = pl.Series([32.0])
    rh = pl.Series([55.0])

    ppm, stderr, unit = apply_calibration(
        voltage, "PID01", t, rh, run_id, data_dir=data_dir
    )

    assert ppm[0] == pytest.approx(5.0, abs=0.5)
    assert unit == "ppm_asgas:isoprene"
    assert stderr[0] >= 0.0


def test_apply_calibration_unknown_sensor_raises_key_error(calibrated_run: tuple[str, Path]) -> None:
    run_id, data_dir = calibrated_run
    with pytest.raises(KeyError):
        apply_calibration(
            pl.Series([10.0]), "PID_NOT_CALIBRATED", pl.Series([32.0]), pl.Series([55.0]),
            run_id, data_dir=data_dir,
        )


def test_apply_calibration_clip_floors_negative_ppm_at_zero(calibrated_run: tuple[str, Path]) -> None:
    run_id, data_dir = calibrated_run
    # voltage below baseline (b0=2.0) inverts to a negative ppm.
    voltage = pl.Series([0.0])

    ppm, _, _ = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        extrapolation_policy="clip", data_dir=data_dir,
    )

    assert ppm[0] == 0.0


def test_apply_calibration_linear_policy_keeps_negative_ppm(calibrated_run: tuple[str, Path]) -> None:
    run_id, data_dir = calibrated_run
    voltage = pl.Series([0.0])

    ppm, _, _ = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        extrapolation_policy="linear", data_dir=data_dir,
    )

    assert ppm[0] < 0.0


def test_apply_calibration_nan_policy_replaces_negative_ppm(calibrated_run: tuple[str, Path]) -> None:
    run_id, data_dir = calibrated_run
    voltage = pl.Series([0.0])

    ppm, _, _ = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        extrapolation_policy="nan", data_dir=data_dir,
    )

    assert math.isnan(ppm[0])


def test_apply_calibration_isobutylene_equiv_scales_by_response_factor(
    calibrated_run: tuple[str, Path]
) -> None:
    run_id, data_dir = calibrated_run
    voltage = pl.Series([22.0])  # -> ppm_asgas ~= 5.0

    ppm_iso, _, unit = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        output_unit="isobutylene_equiv", data_dir=data_dir,
    )

    assert unit == "ppm_isobutylene_equiv"
    # 5.0 ppm_asgas * RF(0.63) = 3.15
    assert ppm_iso[0] == pytest.approx(5.0 * 0.63, abs=0.1)


def test_apply_calibration_both_returns_dataframe_with_two_columns(
    calibrated_run: tuple[str, Path]
) -> None:
    run_id, data_dir = calibrated_run
    voltage = pl.Series([22.0])

    values, stderrs, unit = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        output_unit="both", data_dir=data_dir,
    )

    assert isinstance(values, pl.DataFrame)
    assert set(values.columns) == {"ppm_asgas", "ppm_isobutylene_equiv"}
    assert set(stderrs.columns) == {"ppm_asgas_stderr", "ppm_isobutylene_equiv_stderr"}


def test_apply_calibration_isobutylene_equiv_raises_when_rf_unknown(tmp_path: Path) -> None:
    df = make_standard_addition_readings(
        {"PID02": {"b0_mv": 1.0, "b1_mv_per_ppm": 3.0, "noise_std": 0.02}},
        spike_ppm_list=[0.0, 2.0, 10.0],
        calibration_compound="my_unlisted_voc",
        response_factor=None,
        seed=51,
    )
    models = fit_sensitivity_per_sensor(df)
    data_dir = tmp_path / "calibrations"
    persist_calibration(models, "cal_no_rf", "exp_test", data_dir)

    with pytest.raises(CalibrationUnitUnavailableError):
        apply_calibration(
            pl.Series([10.0]), "PID02", pl.Series([32.0]), pl.Series([55.0]),
            "cal_no_rf", output_unit="isobutylene_equiv", data_dir=data_dir,
        )


def test_apply_calibration_both_nan_fills_iso_equiv_when_rf_unknown(tmp_path: Path) -> None:
    df = make_standard_addition_readings(
        {"PID02": {"b0_mv": 1.0, "b1_mv_per_ppm": 3.0, "noise_std": 0.02}},
        spike_ppm_list=[0.0, 2.0, 10.0],
        calibration_compound="my_unlisted_voc",
        response_factor=None,
        seed=52,
    )
    models = fit_sensitivity_per_sensor(df)
    data_dir = tmp_path / "calibrations"
    persist_calibration(models, "cal_no_rf2", "exp_test", data_dir)

    values, _, _ = apply_calibration(
        pl.Series([10.0]), "PID02", pl.Series([32.0]), pl.Series([55.0]),
        "cal_no_rf2", output_unit="both", data_dir=data_dir,
    )

    assert math.isnan(values["ppm_isobutylene_equiv"][0])


def test_apply_calibration_voltage_stderr_increases_reported_uncertainty(
    calibrated_run: tuple[str, Path]
) -> None:
    run_id, data_dir = calibrated_run
    voltage = pl.Series([22.0])

    _, stderr_low, _ = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        data_dir=data_dir, voltage_stderr=0.0,
    )
    _, stderr_high, _ = apply_calibration(
        voltage, "PID01", pl.Series([32.0]), pl.Series([55.0]), run_id,
        data_dir=data_dir, voltage_stderr=5.0,
    )

    assert stderr_high[0] > stderr_low[0]
