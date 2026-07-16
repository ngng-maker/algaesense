"""Raw-record schemas for both sensor streams, plus a camera schema this
project adds on top of the original spec.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import yaml
from pydantic import BaseModel, Field


"""
Two different kinds of "schema" show up in this file, for two different jobs:

1. `pyarrow.Schema` objects (VOC_RAW_SCHEMA, CAMERA_RAW_SCHEMA) describe the
   *columns of a Parquet file* -- they are what pyarrow/polars use to
   validate and write the actual per-row sensor data efficiently on disk.
2. `pydantic.BaseModel` subclasses (ExperimentMeta and its nested models)
   describe the *shape of a YAML file* -- they validate and give you
   autocomplete/type-checking over the experiment's metadata (who ran it,
   what conditions were assigned, etc).

`from __future__ import annotations` (must be the first real statement in
the file) makes every type hint below lazily-evaluated text rather than a
live Python object at import time. It's what lets us write modern syntax
like `list[str] | None` further down and have it work even if this file
were ever imported on Python 3.9/3.10 (where `X | None` union syntax and
bare generic `list[...]` didn't exist yet) -- Pydantic still reads and
validates against the *text* of the annotation correctly. Since this
project already requires Python >= 3.11 it's not strictly required, but
it's a cheap habit that avoids a class of forward-reference bugs, so we
keep it.
"""


"""
1. pyarrow schemas for the two raw Parquet record types

pa.schema(...) takes a list of pa.field(name, type, nullable=...) entries
and returns an immutable Schema object. This is the same object you'd hand
to pyarrow.parquet.write_table(..., schema=VOC_RAW_SCHEMA) or that polars
uses under the hood when you tell it which dtypes a column must have --
getting this right up front means every writer (Pi acquisition code,
synthetic test fixtures) and every reader (the fusion/processing code)
agree on column names, types, and nullability without needing to re-derive
it from a DataFrame's inferred types (which can silently drift between
runs).

Field-by-field notes for VOC_RAW_SCHEMA below:
- `timestamp` uses `pa.timestamp("ns", tz="UTC")`: nanosecond-precision
  instant in time, always stored/interpreted as UTC -- this avoids the
  classic timezone bug where a bare "naive" timestamp is ambiguous about
  which timezone it means. `nullable=False` means every row MUST have one.
- `sample_t_c` through `room_rh_pct` are nullable because not every rig has
  every ancillary sensor (e.g. a flow meter is optional per the hardware
  spec), or a reading can be legitimately missing for one tick (a T/RH
  sensor timeout shouldn't force the whole row to be dropped).
"""
VOC_RAW_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("sensor_id", pa.string(), nullable=False),
        pa.field("reactor_id", pa.string(), nullable=False),
        pa.field("pid_voltage_mv", pa.float64(), nullable=False),
        pa.field("sample_t_c", pa.float64(), nullable=True),
        pa.field("sample_rh_pct", pa.float64(), nullable=True),
        pa.field("sample_flow_sccm", pa.float64(), nullable=True),
        pa.field("pump_pwm", pa.float32(), nullable=True),
        pa.field("lamp_hours", pa.float32(), nullable=False),
        pa.field("reactor_par_umol_m2_s", pa.float32(), nullable=True),
        pa.field("reactor_temp_c", pa.float32(), nullable=True),
        pa.field("reactor_od", pa.float32(), nullable=True),
        pa.field("reactor_ph", pa.float32(), nullable=True),
        pa.field("light_state", pa.string(), nullable=False),
        pa.field("room_t_c", pa.float32(), nullable=True),
        pa.field("room_rh_pct", pa.float32(), nullable=True),
        pa.field("acquisition_status", pa.string(), nullable=False),
    ]
)


"""
New in this project (not part of the original spec, which only covered VOC
sensors): the raw schema for a single hourly camera capture. One row per
capture EVENT, not per video frame -- each capture event records a short
clip (CameraConfig.capture_duration_s at CameraConfig.frame_rate_fps), but
still lands as exactly one row here, with image_feature_vector holding the
already-averaged-across-the-clip's-frames feature summary. The camera
still samples far less often than the PID (one clip per hour, not one row
per second), which is exactly the "different sampling frequency" the
fusion step later has to account for (see `fuse_multirate`).

Field-by-field notes for CAMERA_RAW_SCHEMA below:
- `video_path`: we store a *path* to the recorded clip rather than
  embedding raw video bytes in the Parquet file -- Parquet is a columnar
  format optimized for numeric/string columns, not large binary blobs, so
  keeping clips as separate files (referenced by path) and only putting
  small numeric summaries in Parquet keeps these files fast to query and
  small enough to sync/back up easily.
- `frame_count`: how many actual video frames this capture contains --
  normally capture_duration_s * frame_rate_fps, but stored explicitly
  rather than only implied, since a dropped-frame capture would have fewer
  frames than that product suggests, and downstream code (deciding how
  much weight to give this capture's averaged feature vector) may care
  about that.
- `image_feature_vector` uses `pa.list_(pa.float32())`, pyarrow's way of
  saying "a variable-length list of float32 values" -- i.e. one column can
  hold an array-typed value per row rather than a single scalar. Fixed at
  exactly 3 elements, always in this order: [mean_red, mean_green,
  mean_blue] channel intensity, MEAN-averaged across every frame of the
  clip. This is no longer an arbitrary/generic feature vector -- biomass
  detection specifically compares GREENNESS (see
  jaxsr_calibration.camera.calibration.greenness_index), which needs to
  know which position is which channel.
- Deliberately no lamp_hours field here -- that's the PID sensor's UV lamp
  aging metric (spec's VOC schema, used by run_weekly_audit's "due for
  cleaning" check). It says nothing about the camera itself, and nothing
  in this codebase reads it off a camera row, so it was removed rather
  than kept "for consistency" with VOC_RAW_SCHEMA -- every field here
  should earn its place.
- `light_state` is replicated here just like the VOC schema, since image
  brightness/color depends heavily on whether the LED is on, off, or
  ramping -- this lets the fusion step (or a human reviewing the data)
  tell whether a dim reading means "low biomass" or just "LED was off".
"""
CAMERA_RAW_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("experiment_id", pa.string(), nullable=False),
        pa.field("reactor_id", pa.string(), nullable=False),
        pa.field("camera_id", pa.string(), nullable=False),
        pa.field("video_path", pa.string(), nullable=True),
        pa.field("capture_duration_s", pa.float32(), nullable=True),
        pa.field("frame_rate_fps", pa.float32(), nullable=True),
        pa.field("frame_count", pa.int32(), nullable=True),
        pa.field("image_feature_vector", pa.list_(pa.float32()), nullable=True),
        pa.field("exposure_us", pa.float32(), nullable=True),
        pa.field("gain", pa.float32(), nullable=True),
        pa.field("light_state", pa.string(), nullable=False),
        pa.field("acquisition_status", pa.string(), nullable=False),
    ]
)


"""
2. ExperimentMeta -- pydantic model for meta.yaml (spec §17)
"""


class ProposedBy(BaseModel):
    """Present only if this experiment's conditions came from JAXSR's own
    active-learning suggestion rather than being manually chosen."""

    tool: str

    acquisition_run: str

    point_index: int


class RunNote(BaseModel):
    """One freeform operator annotation, e.g. logged via `jaxsr-cal note`."""

    """
    `t` uses pydantic's built-in datetime parsing: pydantic will accept an
    ISO-8601 string like "2026-07-15T14:22Z" straight out of YAML and
    convert it into a real `datetime.datetime` object automatically -- we
    don't have to write any parsing code ourselves.
    """
    t: dt.datetime

    text: str


class ExperimentMeta(BaseModel):
    """Validates data/raw/experiments/{experiment_id}/meta.yaml."""

    """
    Every field here mirrors the YAML example in the spec (§17) 1:1. Using
    a pydantic model instead of a plain dict means a malformed meta.yaml
    (typo'd key, wrong type, missing required field) fails loudly and
    specifically at load time, rather than producing a confusing error
    deep inside the processing pipeline much later.
    """

    experiment_id: str

    started_at: dt.datetime

    """
    `dt.datetime | None` (a "union type") means this field accepts either a
    real datetime OR the Python value None. `= None` gives it a default, so
    callers constructing this model don't have to pass ended_at explicitly
    for a still-running experiment -- exactly the "null if aborted mid-run"
    case the spec describes.
    """
    ended_at: dt.datetime | None = None

    operator: str

    campaign_id: str

    """
    Optional nested model: only present if this run's conditions were
    proposed by the active-learning loop rather than chosen by a human.
    """
    proposed_by: ProposedBy | None = None

    """
    `dict[str, dict[str, float]]` reads as "a dictionary whose keys are
    reactor IDs (strings) and whose values are themselves dictionaries
    mapping a condition name to a numeric value" -- e.g. {"R01":
    {"par_umol_m2_s": 200, "reactor_temp_c": 32}}. Pydantic validates every
    level of this nesting, not just the outer dict.
    """
    conditions: dict[str, dict[str, float]]

    sensor_assignment: dict[str, str]

    calibration_run: str

    """
    `Field(default_factory=list)` is the correct way to give a pydantic
    field a default empty list. Writing `excluded_sensors: list[str] = []`
    directly would be a classic Python mutable-default-argument bug if this
    were a plain function/dataclass (every instance would share the *same*
    list object) -- pydantic actually guards against this specific mistake,
    but default_factory is the idiomatic, unambiguous way to express "make
    me a fresh empty list per instance" regardless.
    """
    excluded_sensors: list[str] = Field(default_factory=list)

    notes: list[RunNote] = Field(default_factory=list)


def load_metadata(path: Path) -> ExperimentMeta:
    """Read and validate a meta.yaml file into an ExperimentMeta
    instance."""

    """
    This is intentionally the *only* function in this schema module that
    touches the filesystem -- everything else here is pure data
    definitions.
    """

    """
    `Path.open()` is a context manager: the `with` block guarantees the
    file handle is closed automatically when the block exits, even if
    yaml.safe_load raises an exception partway through.
    """
    with path.open("r", encoding="utf-8") as f:
        """
        yaml.safe_load parses YAML text into plain Python objects (dicts,
        lists, strings, numbers) -- "safe" because, unlike yaml.load, it
        refuses to construct arbitrary Python objects from the file, which
        matters since this file could in principle come from an untrusted
        source (a shared/synced data directory).
        """
        raw = yaml.safe_load(f)

    """
    `ExperimentMeta(**raw)` unpacks the parsed dict as keyword arguments --
    pydantic then validates every field against the types declared above
    and raises a `pydantic.ValidationError` (with a precise, per-field
    message) if anything doesn't match.
    """
    return ExperimentMeta(**raw)
