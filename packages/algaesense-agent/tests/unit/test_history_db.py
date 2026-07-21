"""Unit tests for algaesense_agent.dashboard.history_db: the dashboard's
local SQLite archive of past experiments' raw readings.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from algaesense_edge.acquisition.writer import PartitionedParquetWriter
from jaxsr_calibration.logging_.schema import CAMERA_RAW_SCHEMA, VOC_RAW_SCHEMA

from algaesense_agent.dashboard.history_db import (
    ingest_all_experiments,
    ingest_experiment,
    list_experiments,
    load_experiment_camera_readings,
    load_experiment_voc_readings,
)

_START = dt.datetime(2026, 7, 25, 8, 0, 0, tzinfo=dt.timezone.utc)


def _write_voc_rows(data_dir: Path, experiment_id: str, reactor_id: str, sensor_id: str, n: int, start: dt.datetime) -> None:
    writer = PartitionedParquetWriter(
        base_dir=data_dir / "raw",
        experiment_id=experiment_id,
        partition_key="sensor_id",
        partition_value=sensor_id,
        schema=VOC_RAW_SCHEMA,
    )
    for i in range(n):
        writer.write_row(
            {
                "timestamp": start + dt.timedelta(seconds=i),
                "experiment_id": experiment_id,
                "sensor_id": sensor_id,
                "reactor_id": reactor_id,
                "pid_voltage_mv": 300.0 + i,
                "sample_t_c": None,
                "sample_rh_pct": None,
                "sample_flow_sccm": None,
                "pump_pwm": None,
                "lamp_hours": 10.0,
                "reactor_par_umol_m2_s": 100.0,
                "reactor_temp_c": None,
                "reactor_od": None,
                "reactor_ph": None,
                "light_state": "on",
                "room_t_c": None,
                "room_rh_pct": None,
                "acquisition_status": "OK",
            }
        )
    writer.close()


def _write_camera_rows(data_dir: Path, experiment_id: str, reactor_id: str, camera_id: str, n: int, start: dt.datetime) -> None:
    writer = PartitionedParquetWriter(
        base_dir=data_dir / "raw",
        experiment_id=experiment_id,
        partition_key="camera_id",
        partition_value=camera_id,
        schema=CAMERA_RAW_SCHEMA,
    )
    for i in range(n):
        writer.write_row(
            {
                "timestamp": start + dt.timedelta(hours=i),
                "experiment_id": experiment_id,
                "reactor_id": reactor_id,
                "camera_id": camera_id,
                "video_path": None,
                "capture_duration_s": 10.0,
                "frame_rate_fps": 10.0,
                "frame_count": 100,
                "image_feature_vector": [120.0 + i, 125.0, 110.0],
                "exposure_us": None,
                "gain": None,
                "light_state": "on",
                "acquisition_status": "OK",
            }
        )
    writer.close()


def test_ingest_experiment_loads_both_streams(tmp_path: Path) -> None:
    _write_voc_rows(tmp_path, "exp_01", "R01", "PID01", n=5, start=_START)
    _write_camera_rows(tmp_path, "exp_01", "R01", "CAM01", n=2, start=_START)
    db_path = tmp_path / "history.db"

    result = ingest_experiment(db_path, tmp_path, "exp_01")

    assert result == {"experiment_id": "exp_01", "voc_rows": 5, "camera_rows": 2}

    voc = load_experiment_voc_readings(db_path, "exp_01")
    assert len(voc) == 5
    assert voc[0]["pid_voltage_mv"] == pytest.approx(300.0)
    assert voc[-1]["pid_voltage_mv"] == pytest.approx(304.0)

    camera = load_experiment_camera_readings(db_path, "exp_01")
    assert len(camera) == 2
    assert camera[0]["image_feature_vector"] == [120.0, 125.0, 110.0]


def test_ingest_experiment_with_only_voc_leaves_camera_empty(tmp_path: Path) -> None:
    _write_voc_rows(tmp_path, "exp_voc_only", "R01", "PID01", n=3, start=_START)
    db_path = tmp_path / "history.db"

    result = ingest_experiment(db_path, tmp_path, "exp_voc_only")

    assert result == {"experiment_id": "exp_voc_only", "voc_rows": 3, "camera_rows": 0}
    assert load_experiment_camera_readings(db_path, "exp_voc_only") == []


def test_re_ingesting_the_same_experiment_does_not_duplicate_rows(tmp_path: Path) -> None:
    """Regression-shaped test: re-running ingestion against unchanged raw
    files (e.g. running the sync CLI twice) must not leave two copies of
    the same rows sitting in the SQLite archive."""
    _write_voc_rows(tmp_path, "exp_01", "R01", "PID01", n=3, start=_START)
    db_path = tmp_path / "history.db"

    ingest_experiment(db_path, tmp_path, "exp_01")
    ingest_experiment(db_path, tmp_path, "exp_01")

    voc = load_experiment_voc_readings(db_path, "exp_01")
    assert len(voc) == 3


def test_list_experiments_returns_every_ingested_experiment_newest_first(tmp_path: Path) -> None:
    _write_voc_rows(tmp_path, "exp_older", "R01", "PID01", n=2, start=_START)
    _write_voc_rows(tmp_path, "exp_newer", "R01", "PID01", n=2, start=_START + dt.timedelta(days=1))
    db_path = tmp_path / "history.db"
    ingest_experiment(db_path, tmp_path, "exp_older")
    ingest_experiment(db_path, tmp_path, "exp_newer")

    experiments = list_experiments(db_path)

    assert [e["experiment_id"] for e in experiments] == ["exp_newer", "exp_older"]
    assert experiments[0]["voc_row_count"] == 2


def test_ingest_all_experiments_finds_every_experiment_under_data_dir(tmp_path: Path) -> None:
    _write_voc_rows(tmp_path, "exp_a", "R01", "PID01", n=1, start=_START)
    _write_voc_rows(tmp_path, "exp_b", "R01", "PID01", n=1, start=_START)
    db_path = tmp_path / "history.db"

    results = ingest_all_experiments(db_path, tmp_path)

    assert {r["experiment_id"] for r in results} == {"exp_a", "exp_b"}
    assert {e["experiment_id"] for e in list_experiments(db_path)} == {"exp_a", "exp_b"}
