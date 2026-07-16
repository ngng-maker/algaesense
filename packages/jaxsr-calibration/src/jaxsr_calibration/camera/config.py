"""Pydantic schema for configs/camera.yaml."""

from __future__ import annotations

from pydantic import BaseModel, Field


"""
New in this project, not part of the original spec (which only covered VOC
PID sensors). Mirrors the style of calibration/config.py's SensorConfig
(one entry per physical sensor unit) so the two config families feel
consistent.
"""


class CameraConfig(BaseModel):
    """One entry per Pi camera in configs/camera.yaml."""

    id: str

    associated_reactor: str

    """
    Hourly by default, per the requirement that the camera samples far less
    often than the VOC sensor. Stored in minutes (rather than seconds like
    the VOC sensor's ~1 Hz) since an hourly-ish cadence is more naturally
    expressed and edited by a human in minutes.
    """
    capture_interval_min: int = 60

    """
    How long each capture records for, in seconds. Each scheduled capture
    (every `capture_interval_min`) records a short VIDEO CLIP of this
    length, not a single still photo -- e.g. the default 10s/60min means
    "record a 10-second clip once an hour". Not spec-mandated; 10s is a
    starting default, easy to tune once real hardware is in hand.
    """
    capture_duration_s: float = 10.0

    """
    Recording frame rate for each clip. Combined with capture_duration_s,
    this determines how many individual video frames get MEAN-averaged (per
    feature dimension) into that capture's single aggregated feature vector
    (see jaxsr_calibration.camera.calibration's module docstring) -- e.g.
    10s * 10 fps = 100 frames averaged per hourly capture.
    """
    frame_rate_fps: float = 10.0

    """
    `tuple[int, int]` fixes the length of the tuple to exactly two elements
    at the type level (unlike `list[int]`, which could hold any number of
    elements) -- appropriate here since a resolution is always exactly
    (width, height), never more or fewer values.
    """
    resolution_wh: tuple[int, int] = (1280, 720)

    """
    `float | None = None`: if not set, the camera driver should use its own
    auto-exposure/auto-gain behavior rather than a fixed value. Locking
    these down (non-None) matters for the zero-point calibration, where a
    *consistent* exposure/gain between the blank reference capture and
    later experiment captures is what makes the background-subtraction
    calibration meaningful at all.
    """
    exposure_us: float | None = None

    gain: float | None = None

    """
    Set once run_biomass_zero_calibration has produced a
    calibration_run_id for this camera; left as None until then.
    """
    blank_calibration_run_id: str | None = Field(default=None)
