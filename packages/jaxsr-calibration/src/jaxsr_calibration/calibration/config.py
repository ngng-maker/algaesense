"""Pydantic schemas for configs/sensors.yaml, configs/reactors.yaml, and
configs/rotation_schedule.yaml.
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


"""
`sensors.yaml`'s exact fields are given verbatim as a worked example in the
spec (§12), so SensorConfig matches it field-for-field. `reactors.yaml` and
`rotation_schedule.yaml` are only described in prose in the spec (no literal
YAML example), so ReactorConfig/RotationSchedule below are a reasonable
design filling that gap -- unlike SensorConfig's fields, these are NOT
locked by the spec and can be extended later without it being a breaking
change to anything the spec itself promises.
"""


class SensorConfig(BaseModel):
    """One entry per PID (or other) sensor in configs/sensors.yaml."""

    """
    Field names/types here are taken directly from the spec's own example
    (§12) -- this is intentionally a literal transcription, not a redesign.
    """

    id: str

    model: str

    serial: str

    """
    `dt.date` (not `dt.datetime`) because the YAML value "2026-01-15" is a
    calendar date with no time-of-day component -- pydantic parses an
    ISO-8601 date string straight into a real `datetime.date` object.
    """
    lamp_install_date: dt.date

    lamp_hours_at_install: float

    calibration_gas: str

    """
    The YAML key itself is "factory_sensitivity_mV_per_ppm" (mixed case) --
    Python naming convention prefers snake_case attribute names, so we
    declare the attribute as lowercase and tell pydantic the YAML key it
    should actually read from via `Field(alias=...)`.
    """
    factory_sensitivity_mv_per_ppm: float = Field(alias="factory_sensitivity_mV_per_ppm")

    associated_rh_sensor: str

    """
    Which reactor this sensor is currently plumbed to. This is deliberately
    NOT permanent -- the spec's whole point of the rotation schedule (below)
    is that this assignment changes week to week so a bad sensor and a bad
    reactor can be told apart.
    """
    associated_reactor: str

    """
    `model_config` is pydantic v2's replacement for the old nested `class
    Config:` block. Without `populate_by_name=True`, pydantic would only
    accept the exact snake_case attribute name
    ("factory_sensitivity_mv_per_ppm") as a constructor keyword, and reject
    a dict loaded straight from YAML that still has the original mixed-case
    key. This setting allows *either* spelling, so
    `SensorConfig(**yaml.safe_load(...))` just works.
    """
    model_config = ConfigDict(populate_by_name=True)


class ReactorConfig(BaseModel):
    """One entry per physical Pioreactor vessel in configs/reactors.yaml."""

    """
    Not given a literal example in the spec, so this is an inferred minimal
    shape covering what the rest of the pipeline actually needs to know about
    a reactor: its identity and the physical/safety bounds that
    jaxsr_calibration.camera and the actuator layer will validate against.
    Extend this model rather than working around its absence.
    """

    id: str

    """
    e.g. "pioreactor_20mL" -- kept as a free string rather than an enum
    since the spec doesn't commit to a fixed set of supported models.
    """
    model: str

    """
    Hard safety bounds for this specific vessel. These are the numbers
    algaesense_edge's actuator code clamps against independently of
    whatever an upstream caller asks for.
    """
    max_par_umol_m2_s: float = 15000.0

    max_reactor_temp_c: float = 40.0

    min_reactor_temp_c: float = 4.0


class RotationSchedule(BaseModel):
    """configs/rotation_schedule.yaml -- which sensor is plumbed to which
    reactor for a given period."""

    """
    Per the Latin-square rotation the spec describes (§12, and
    experimentalist protocol §12). `assignments` deliberately mirrors the
    shape of ExperimentMeta's own `sensor_assignment` field in
    jaxsr_calibration.logging_.schema (a plain dict[sensor_id, reactor_id])
    so the two stay easy to compare/diff.
    """

    """
    An identifier for *which* rotation period this file describes, e.g.
    "2026-W29" -- lets the operator (and the tool) confirm they're reading
    the schedule for the correct week before a run.
    """
    period_id: str

    assignments: dict[str, str]
