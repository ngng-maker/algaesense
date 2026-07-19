"""Unit tests for algaesense_agent.raw_readers: loading real raw VOC
Parquet files (written via algaesense-edge's real writer, not hand-rolled).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from algaesense_edge.acquisition.writer import PartitionedParquetWriter
from jaxsr_calibration.logging_.schema import VOC_RAW_SCHEMA

from algaesense_agent.raw_readers import NoRawReadingsFoundError, load_raw_voc_readings

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


def test_load_raw_voc_readings_concatenates_every_sensor(tmp_path: Path) -> None:
    _write_rows(tmp_path, "exp_diag_test", "PID01", [_row(sensor_id="PID01", pid_voltage_mv=0.1)])
    _write_rows(tmp_path, "exp_diag_test", "PID02", [_row(sensor_id="PID02", pid_voltage_mv=0.2)])

    readings = load_raw_voc_readings(tmp_path, "exp_diag_test")

    assert readings.height == 2
    assert set(readings["sensor_id"].to_list()) == {"PID01", "PID02"}


def test_load_raw_voc_readings_raises_for_unknown_experiment(tmp_path: Path) -> None:
    with pytest.raises(NoRawReadingsFoundError):
        load_raw_voc_readings(tmp_path, "does_not_exist")
