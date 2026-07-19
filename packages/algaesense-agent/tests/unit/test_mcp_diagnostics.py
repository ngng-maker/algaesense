"""Unit tests for algaesense_agent.mcp_diagnostics: dispatching real raw
VOC Parquet data (written via algaesense-edge's real writer, not
hand-rolled) to jaxsr-calibration's real diagnostics. The loader itself
(`load_raw_voc_readings`) has its own tests in test_raw_readers.py.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import yaml
from algaesense_edge.acquisition.writer import PartitionedParquetWriter
from jaxsr_calibration.logging_.schema import VOC_RAW_SCHEMA

from algaesense_agent.mcp_diagnostics.diagnostics import (
    ambient_baseline_check,
    fleet_zero_check,
    swap_pilot_check,
    weekly_audit_check,
)

_START = dt.datetime(2026, 7, 17, 8, 0, 0, tzinfo=dt.timezone.utc)


def _row(**overrides) -> dict:
    row = {
        "timestamp": _START,
        "experiment_id": "exp_diag_test",
        "sensor_id": "PID01",
        "reactor_id": "R01",
        "pid_voltage_mv": 0.0,
        "sample_t_c": 25.0,
        "sample_rh_pct": 50.0,
        "sample_flow_sccm": None,
        "pump_pwm": None,
        "lamp_hours": 10.0,
        "reactor_par_umol_m2_s": None,
        "reactor_temp_c": None,
        "reactor_od": None,
        "reactor_ph": None,
        "light_state": "on",
        "room_t_c": None,
        "room_rh_pct": None,
        "acquisition_status": "OK",
    }
    row.update(overrides)
    return row


def _write_rows(data_dir: Path, experiment_id: str, sensor_id: str, rows: list[dict]) -> None:
    writer = PartitionedParquetWriter(
        base_dir=data_dir / "raw",
        experiment_id=experiment_id,
        partition_key="sensor_id",
        partition_value=sensor_id,
        schema=VOC_RAW_SCHEMA,
    )
    for row in rows:
        writer.write_row(row)
    writer.close()


def test_fleet_zero_check_flags_a_healthy_fleet_as_green(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    for sensor_id in ("PID01", "PID02", "PID03"):
        rows = [
            _row(
                sensor_id=sensor_id,
                timestamp=_START + dt.timedelta(minutes=i),
                pid_voltage_mv=float(rng.normal(0.0, 0.1)),
            )
            for i in range(20)
        ]
        _write_rows(tmp_path, "exp_fleet_zero", sensor_id, rows)

    result = fleet_zero_check(tmp_path, "exp_fleet_zero", duration_min=20)

    assert result.summary_status == "GREEN"
    assert set(result.per_sensor.keys()) == {"PID01", "PID02", "PID03"}


def test_ambient_baseline_check_fits_a_real_covariate_model(tmp_path: Path) -> None:
    """
    voltage = 2.0 + 0.05*RH, RH swept across a 40-percentage-point range --
    comfortably over fit_covariate_model's own min_rh_range_pct
    requirement (20), so this should fit cleanly rather than raising
    TrainingWindowInsufficientError. `sample_t_c` is given its own
    independent variation (not tied to the RH sweep) -- holding it exactly
    constant makes the [RH, T, RH*T] design matrix rank-deficient (T and
    RH*T both become scalar multiples of other columns), which is a
    degenerate test setup, not a real bug in fit_covariate_model.
    """
    rng = np.random.default_rng(2)
    rows = [
        _row(
            timestamp=_START + dt.timedelta(minutes=i),
            sample_rh_pct=float(rh),
            sample_t_c=float(rng.uniform(20.0, 30.0)),
            pid_voltage_mv=2.0 + 0.05 * rh,
        )
        for i, rh in enumerate(np.linspace(30.0, 70.0, 20))
    ]
    _write_rows(tmp_path, "exp_ambient", "PID01", rows)

    result = ambient_baseline_check(tmp_path, "exp_ambient", duration_h=1)

    assert "PID01" in result.covariate_models
    assert result.r_squared_per_sensor["PID01"] > 0.9


def test_swap_pilot_check_recovers_the_dominant_variance_source(tmp_path: Path) -> None:
    """
    Same structural-correctness approach as jaxsr-calibration's own
    swap-pilot tests (variance estimates from a handful of levels are
    known to be noisy) -- reactor effect is huge, sensor effect is tiny,
    so reactor_id's fitted share should clearly dominate.
    """
    rng = np.random.default_rng(1)
    sensors = ["PID01", "PID02", "PID03"]
    reactors = ["R01", "R02", "R03"]
    sensor_effects = {s: rng.normal(0.0, 0.1) for s in sensors}
    reactor_effects = {r: rng.normal(0.0, 10.0) for r in reactors}

    rows = []
    i = 0
    for sensor_id in sensors:
        for reactor_id in reactors:
            for _ in range(5):
                rows.append(
                    _row(
                        sensor_id=sensor_id,
                        reactor_id=reactor_id,
                        timestamp=_START + dt.timedelta(minutes=i),
                        pid_voltage_mv=50.0
                        + sensor_effects[sensor_id]
                        + reactor_effects[reactor_id]
                        + float(rng.normal(0.0, 1.0)),
                    )
                )
                i += 1

    """
    All three sensors' rows are written under whichever partition
    matches their own sensor_id -- swap_pilot_check loads across all of
    them via load_raw_voc_readings, same as any other diagnostic here.
    """
    for sensor_id in sensors:
        _write_rows(
            tmp_path, "exp_swap_pilot", sensor_id, [r for r in rows if r["sensor_id"] == sensor_id]
        )

    result = swap_pilot_check(tmp_path, "exp_swap_pilot", n_blocks=3)

    assert result.variance_share["reactor_id"] > result.variance_share["sensor_id"]
    assert sum(result.variance_share.values()) == pytest.approx(1.0, abs=1e-6)


def test_weekly_audit_check_composes_swap_pilot_history_and_sensor_configs(tmp_path: Path) -> None:
    sensors_yaml_path = tmp_path / "sensors.yaml"
    sensors_yaml_path.write_text(
        yaml.safe_dump(
            {
                "sensors": [
                    {
                        "id": "PID01",
                        "model": "Alphasense PID",
                        "serial": "SN001",
                        "lamp_install_date": dt.date(2026, 1, 1),
                        "lamp_hours_at_install": 0.0,
                        "calibration_gas": "isoprene",
                        "factory_sensitivity_mV_per_ppm": 30.0,
                        "associated_rh_sensor": "RH01",
                        "associated_reactor": "R01",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = weekly_audit_check(
        swap_pilot_variance_shares=[
            {"sensor_id": 0.10, "reactor_id": 0.20, "residual": 0.70},
            {"sensor_id": 0.35, "reactor_id": 0.20, "residual": 0.45},
        ],
        sensors_yaml_path=sensors_yaml_path,
        backup_current=True,
        today=dt.date(2026, 7, 17),
    )

    # Sensor-id variance share jumped from 0.10 to 0.35 -- a real, sizeable
    # drift the composed rollup should reflect, not just "no crash".
    assert result.sensor_variance_share_delta == pytest.approx(0.25)
