"""Unit tests for algaesense_edge.acquisition.writer.PartitionedParquetWriter.

Uses jaxsr_calibration's real VOC_RAW_SCHEMA rather than a made-up test
schema, so these tests also double-check the writer actually produces files
that match the schema the rest of the pipeline (jaxsr-calibration) expects.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from jaxsr_calibration.logging_.schema import VOC_RAW_SCHEMA

from algaesense_edge.acquisition.writer import PartitionedParquetWriter


def _voc_row(timestamp: dt.datetime, pid_voltage_mv: float = 1.0) -> dict:
    return {
        "timestamp": timestamp,
        "experiment_id": "exp_test",
        "sensor_id": "PID01",
        "reactor_id": "R01",
        "pid_voltage_mv": pid_voltage_mv,
        "sample_t_c": 32.0,
        "sample_rh_pct": 55.0,
        "sample_flow_sccm": None,
        "pump_pwm": None,
        "lamp_hours": 12.0,
        "reactor_par_umol_m2_s": 200.0,
        "reactor_temp_c": 32.0,
        "reactor_od": None,
        "reactor_ph": None,
        "light_state": "on",
        "room_t_c": 22.0,
        "room_rh_pct": 45.0,
        "acquisition_status": "OK",
    }


def _writer(tmp_path: Path) -> PartitionedParquetWriter:
    return PartitionedParquetWriter(
        base_dir=tmp_path,
        experiment_id="exp_test",
        partition_key="sensor_id",
        partition_value="PID01",
        schema=VOC_RAW_SCHEMA,
    )


def test_write_row_does_not_touch_disk_until_flush(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc)))

    # Nothing should be written yet -- still buffered in memory.
    assert not (tmp_path / "experiments").exists()


def test_flush_writes_to_the_expected_hour_partition_path(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 9, 30, 0, tzinfo=dt.timezone.utc)))
    writer.flush()

    expected_path = (
        tmp_path / "experiments" / "exp_test" / "sensor_id=PID01" / "hour=2026-07-15T09.parquet"
    )
    assert expected_path.exists()
    table = pq.read_table(expected_path)
    assert table.num_rows == 1


def test_crossing_an_hour_boundary_auto_flushes_the_previous_hour(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 9, 59, 0, tzinfo=dt.timezone.utc)))
    # This row belongs to the NEXT hour -- writing it should trigger an
    # automatic flush of the 09:00 hour's buffered row first.
    writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 10, 0, 0, tzinfo=dt.timezone.utc)))

    hour_09_path = (
        tmp_path / "experiments" / "exp_test" / "sensor_id=PID01" / "hour=2026-07-15T09.parquet"
    )
    assert hour_09_path.exists()
    # The 10:00 row is still only buffered, not yet on disk.
    hour_10_path = (
        tmp_path / "experiments" / "exp_test" / "sensor_id=PID01" / "hour=2026-07-15T10.parquet"
    )
    assert not hour_10_path.exists()


def test_close_flushes_the_final_partial_hour(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 11, 0, 0, tzinfo=dt.timezone.utc)))
    writer.close()

    hour_11_path = (
        tmp_path / "experiments" / "exp_test" / "sensor_id=PID01" / "hour=2026-07-15T11.parquet"
    )
    assert hour_11_path.exists()


def test_restarting_mid_hour_appends_rather_than_overwrites(tmp_path: Path) -> None:
    # Simulates a process restart: a first writer flushes one row, then a
    # brand-new writer instance (as if the process crashed and came back)
    # writes another row to the SAME hour.
    first_writer = _writer(tmp_path)
    first_writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc), pid_voltage_mv=1.0))
    first_writer.close()

    second_writer = _writer(tmp_path)
    second_writer.write_row(_voc_row(dt.datetime(2026, 7, 15, 9, 30, 0, tzinfo=dt.timezone.utc), pid_voltage_mv=2.0))
    second_writer.close()

    hour_path = (
        tmp_path / "experiments" / "exp_test" / "sensor_id=PID01" / "hour=2026-07-15T09.parquet"
    )
    table = pq.read_table(hour_path)
    assert table.num_rows == 2
    assert sorted(table.column("pid_voltage_mv").to_pylist()) == [1.0, 2.0]


def test_flush_with_nothing_buffered_is_a_safe_no_op(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    writer.flush()  # nothing written yet, should not raise or create files

    assert not (tmp_path / "experiments").exists()
