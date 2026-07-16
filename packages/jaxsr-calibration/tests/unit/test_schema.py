"""Unit tests for jaxsr_calibration.logging_.schema: the two pyarrow raw-record
schemas (VOC_RAW_SCHEMA, CAMERA_RAW_SCHEMA) and the ExperimentMeta pydantic
model + its load_metadata() YAML loader.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pytest
import yaml

from jaxsr_calibration.logging_.schema import (
    CAMERA_RAW_SCHEMA,
    VOC_RAW_SCHEMA,
    ExperimentMeta,
    load_metadata,
)


def test_voc_raw_schema_accepts_one_valid_row() -> None:
    # `pa.Table.from_pylist` builds an in-memory Arrow table from a list of
    # row-dicts, validated against the given schema. If any column were
    # missing, mistyped, or a non-nullable field were left out, this call
    # would raise -- so a successful call here *is* the assertion.
    row = {
        "timestamp": dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc),
        "experiment_id": "exp_2026-07-15_batch03",
        "sensor_id": "PID01",
        "reactor_id": "R01",
        "pid_voltage_mv": 123.4,
        "sample_t_c": 32.1,
        "sample_rh_pct": 55.0,
        "sample_flow_sccm": None,
        "pump_pwm": 0.5,
        "lamp_hours": 12.0,
        "reactor_par_umol_m2_s": 200.0,
        "reactor_temp_c": 32.0,
        "reactor_od": 0.6,
        "reactor_ph": None,
        "light_state": "on",
        "room_t_c": 22.0,
        "room_rh_pct": 45.0,
        "acquisition_status": "OK",
    }
    table = pa.Table.from_pylist([row], schema=VOC_RAW_SCHEMA)
    assert table.num_rows == 1
    assert table.schema.equals(VOC_RAW_SCHEMA)


def test_voc_raw_schema_nullable_false_is_descriptive_not_enforced() -> None:
    # Surprising pyarrow behavior worth pinning down with a test: `nullable=False`
    # in a pa.field(...) declaration is *documentation* about what a well-formed
    # file should contain -- it is NOT automatically enforced when building a
    # table. A row missing "pid_voltage_mv" (declared non-nullable) still
    # builds successfully, with that cell silently filled as null, and even
    # `table.validate(full=True)` (pyarrow's own strict validation pass)
    # raises nothing. Real enforcement has to be written explicitly -- that's
    # deferred to the writer code in a later milestone, not this schema module.
    row = {
        "timestamp": dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc),
        "experiment_id": "exp_2026-07-15_batch03",
        "sensor_id": "PID01",
        "reactor_id": "R01",
        # pid_voltage_mv intentionally omitted even though nullable=False.
        "lamp_hours": 12.0,
        "light_state": "on",
        "acquisition_status": "OK",
    }
    table = pa.Table.from_pylist([row], schema=VOC_RAW_SCHEMA)
    table.validate(full=True)  # does not raise -- see comment above
    assert table.column("pid_voltage_mv").null_count == 1


def test_camera_raw_schema_accepts_feature_vector_row() -> None:
    row = {
        "timestamp": dt.datetime(2026, 7, 15, 9, 0, 0, tzinfo=dt.timezone.utc),
        "experiment_id": "exp_2026-07-15_batch03",
        "reactor_id": "R01",
        "camera_id": "CAM01",
        "video_path": "clips/CAM01_2026-07-15T09-00.mp4",
        "capture_duration_s": 10.0,
        "frame_rate_fps": 10.0,
        "frame_count": 100,
        # A list value in a single cell -- this is exactly what
        # pa.list_(pa.float32()) exists to represent. Here it's the feature
        # vector already averaged across all 100 frames of the clip above.
        "image_feature_vector": [0.12, 0.30, 0.18],
        "exposure_us": 10000.0,
        "gain": 1.0,
        "light_state": "on",
        "acquisition_status": "OK",
    }
    table = pa.Table.from_pylist([row], schema=CAMERA_RAW_SCHEMA)
    assert table.num_rows == 1
    # Reading the list-typed column back out gives you a native Python list.
    assert table.column("image_feature_vector")[0].as_py() == [
        pytest.approx(0.12),
        pytest.approx(0.30),
        pytest.approx(0.18),
    ]


def test_experiment_meta_parses_spec_example() -> None:
    # Transcribed from the spec's own meta.yaml example (§17).
    raw = {
        "experiment_id": "exp_2026-07-15_batch03",
        "started_at": "2026-07-15T09:00:00Z",
        "ended_at": "2026-07-15T21:00:00Z",
        "operator": "name-or-id",
        "campaign_id": "2026-Q3-spirulina-voc",
        "proposed_by": {
            "tool": "modeling",
            "acquisition_run": "acq_2026-07-14_003",
            "point_index": 2,
        },
        "conditions": {
            "R01": {"par_umol_m2_s": 200, "reactor_temp_c": 32, "n_nano3_g_l": 2.5},
            "R02": {"par_umol_m2_s": 100, "reactor_temp_c": 30, "n_nano3_g_l": 1.5},
        },
        "sensor_assignment": {"PID01": "R01", "PID02": "R02"},
        "calibration_run": "cal_2026-07-15_pre",
        "excluded_sensors": [],
        "notes": [{"t": "2026-07-15T14:22Z", "text": "R03 LED flicker, replaced 14:30"}],
    }
    meta = ExperimentMeta(**raw)
    assert meta.experiment_id == "exp_2026-07-15_batch03"
    assert meta.proposed_by is not None
    assert meta.proposed_by.point_index == 2
    assert meta.conditions["R01"]["par_umol_m2_s"] == 200
    assert meta.notes[0].text.startswith("R03 LED flicker")


def test_experiment_meta_ended_at_optional_for_aborted_run() -> None:
    # A still-running (or aborted-with-unknown-end-time) experiment should be
    # representable with ended_at left out entirely.
    raw = {
        "experiment_id": "exp_2026-07-16_batch01",
        "started_at": "2026-07-16T09:00:00Z",
        "operator": "op1",
        "campaign_id": "2026-Q3-spirulina-voc",
        "conditions": {"R01": {"par_umol_m2_s": 150}},
        "sensor_assignment": {"PID01": "R01"},
        "calibration_run": "cal_2026-07-16_pre",
    }
    meta = ExperimentMeta(**raw)
    assert meta.ended_at is None
    # Fields with default_factory=list should default to an empty list, not
    # error out for being absent from the YAML.
    assert meta.excluded_sensors == []
    assert meta.notes == []


def test_load_metadata_reads_yaml_file_from_disk(tmp_path: Path) -> None:
    # `tmp_path` is a built-in pytest fixture: pytest creates a fresh, unique
    # temporary directory for this test and hands us its Path, then cleans it
    # up afterwards -- we never touch the real filesystem outside of it.
    meta_dict = {
        "experiment_id": "exp_2026-07-17_batch02",
        "started_at": "2026-07-17T09:00:00Z",
        "operator": "op1",
        "campaign_id": "2026-Q3-spirulina-voc",
        "conditions": {"R01": {"par_umol_m2_s": 150}},
        "sensor_assignment": {"PID01": "R01"},
        "calibration_run": "cal_2026-07-17_pre",
    }
    meta_path = tmp_path / "meta.yaml"
    meta_path.write_text(yaml.safe_dump(meta_dict), encoding="utf-8")

    meta = load_metadata(meta_path)

    assert meta.experiment_id == "exp_2026-07-17_batch02"
