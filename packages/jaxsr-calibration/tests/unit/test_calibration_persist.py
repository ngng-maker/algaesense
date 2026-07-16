"""Unit tests for jaxsr_calibration.calibration.apply's persist_calibration
/ load_calibration round-trip (persist/load logic, as opposed to
test_calibration_apply.py's apply_calibration tests -- both now live in the
same apply.py source module, per the calibration lifecycle merge)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jaxsr_calibration.calibration.apply import load_calibration, persist_calibration
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from tests.fixtures.synthetic_readings import make_standard_addition_readings


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    df = make_standard_addition_readings(
        {
            "PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.05},
            "PID02": {"b0_mv": -0.5, "b1_mv_per_ppm": 6.0, "noise_std": 0.05},
        },
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        calibration_compound="isoprene",
        response_factor=0.63,
        seed=40,
    )
    models = fit_sensitivity_per_sensor(df)

    out_dir = tmp_path / "calibrations" / "standard_addition"
    parquet_path = persist_calibration(
        models, calibration_run_id="cal_2026-07-15_pre", experiment_id="exp_test", out_dir=out_dir
    )

    assert parquet_path.exists()
    assert (out_dir / "cal_2026-07-15_pre.yaml").exists()

    loaded = load_calibration("cal_2026-07-15_pre", out_dir)

    assert set(loaded.keys()) == {"PID01", "PID02"}
    for sensor_id, original in models.items():
        reloaded = loaded[sensor_id]
        assert reloaded.b0_mv == pytest.approx(original.b0_mv)
        assert reloaded.b1_mv_per_ppm_asgas == pytest.approx(original.b1_mv_per_ppm_asgas)
        assert reloaded.b1_mv_per_ppm_iso_equiv == pytest.approx(original.b1_mv_per_ppm_iso_equiv)
        assert reloaded.status == original.status
        assert reloaded.calibration_gas.name == "isoprene"
        assert reloaded.calibration_gas.response_factor == pytest.approx(0.63)
        assert reloaded.calibration_gas.has_rf is True


def test_persist_calibration_rejects_mixed_compounds_across_run(tmp_path: Path) -> None:
    df_isoprene = make_standard_addition_readings(
        {"PID01": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0}},
        spike_ppm_list=[0.0, 5.0],
        calibration_compound="isoprene",
        seed=41,
    )
    df_acetone = make_standard_addition_readings(
        {"PID02": {"b0_mv": 1.0, "b1_mv_per_ppm": 4.0}},
        spike_ppm_list=[0.0, 5.0],
        calibration_compound="acetone",
        seed=42,
    )
    models = {
        **fit_sensitivity_per_sensor(df_isoprene),
        **fit_sensitivity_per_sensor(df_acetone),
    }

    with pytest.raises(ValueError, match="same compound"):
        persist_calibration(models, "cal_bad", "exp_test", tmp_path)


def test_load_calibration_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_calibration("does_not_exist", tmp_path)
