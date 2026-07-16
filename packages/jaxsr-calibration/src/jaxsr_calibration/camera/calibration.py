"""Camera zero-point (blank) biomass calibration.

Captures the camera's response to clear, cell-free medium and uses that as
a background to subtract from later readings, producing a relative
biomass signal based on how GREEN a reading is (not how bright it is).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError


"""
Not part of the original spec (which only covered VOC PID sensors) -- added
for this project's camera/biomass requirement. This is a *zero-point/blank*
calibration only (analogous to jaxsr_calibration.diagnostics.fleet_zero's
"establish a clean baseline" idea, not a full multi-point standard-addition
curve like the VOC calibration). The result is a RELATIVE biomass index
("more/less green than the blank"), not an absolute concentration, unless a
second reference point (Pioreactor's own OD sensor, dry weight, or a
benchtop spectrophotometer) is added later.

Each "capture" here is a short VIDEO CLIP (CameraConfig.capture_duration_s
at CameraConfig.frame_rate_fps), not a single still photo -- the feature
vector for one capture is already the MEAN, per color channel, across every
frame in that clip, before it reaches this module. That per-clip
frame-averaging happens on the Pi, in algaesense-edge, not here. Naming
below reflects this: "n_captures"/"min_captures" count capture EVENTS
(clips), not raw video frames.

Biomass detection is GREEN-color-based, not generic brightness, because
Arthrospira/Spirulina is a blue-green cyanobacterium: a denser culture is
visibly greener, not just darker or lighter, and overall brightness alone is
easily confounded by exposure/lighting drift even when exposure/gain are
held fixed (see CameraConfig for why that consistency matters).
`greenness_index` below computes the "Excess Green Index" (2*G - R - B), a
standard formula from RGB-camera vegetation/algae imaging, not something
invented for this project.
"""


"""
Named indices for the fixed [red, green, blue] feature-vector convention
used throughout this module and in CAMERA_RAW_SCHEMA -- writing `rgb[_GREEN]`
below is self-explanatory without needing a comment at every use site, unlike
a bare `rgb[1]`.
"""
_RED, _GREEN, _BLUE = 0, 1, 2


def greenness_index(rgb: list[float] | np.ndarray) -> float:
    """Measure how green an RGB reading is, independent of its brightness."""

    """
    Implements the "Excess Green Index": 2*G - R - B, for one [red, green,
    blue] triple. Subtracting R and B (rather than reading G alone) cancels
    out most of the effect of overall brightness -- a uniformly darker or
    brighter image (same color, different exposure) shifts R, G, and B
    together and mostly cancels out of this formula, while an image that's
    specifically greener (chlorophyll-driven, not lighting-driven) does not.
    """

    """
    `np.asarray(rgb, dtype=float)` converts whatever was passed in (a plain
    Python list, or already a numpy array) into a numpy array of floats, so
    the indexing/arithmetic below works the same way regardless of which one
    the caller passed.
    """
    rgb = np.asarray(rgb, dtype=float)

    """
    `rgb.shape != (3,)` checks the array has exactly one dimension with
    exactly 3 elements. Rejecting anything else here (wrong length, or an
    accidentally 2-D array) with a clear error is safer than letting the
    formula below silently compute something meaningless from bad input.
    """
    if rgb.shape != (3,):
        raise ValueError(
            f"greenness_index expects exactly 3 values [red, green, blue], got shape {rgb.shape}"
        )

    """
    `float(...)` converts the numpy scalar this expression produces into a
    plain Python float, so callers always get a real `float` back, not a
    numpy-specific scalar type that behaves *almost* like one.
    """
    return float(2 * rgb[_GREEN] - rgb[_RED] - rgb[_BLUE])


@dataclass
class BiomassCameraModel:
    """A camera's calibrated zero-point (blank) baseline."""

    """
    Stores the mean and standard deviation of a camera's RGB feature vector
    when pointed at cell-free (algae-free) medium. `apply_biomass_calibration`
    compares a later reading against `blank_baseline_mean` to compute a
    greenness-based biomass signal.
    """

    camera_id: str

    calibration_run_id: str

    """
    Always exactly 3 elements, [mean_red, mean_green, mean_blue], matching
    CAMERA_RAW_SCHEMA's `image_feature_vector` column
    (jaxsr_calibration.logging_.schema). Kept as the full RGB triple, rather
    than already-collapsed into one greenness number, so
    `compute_blank_baseline`'s per-channel stability check below can still
    catch a single stuck/noisy channel -- a pre-collapsed number would hide
    that.
    """
    blank_baseline_mean: list[float]

    blank_baseline_std: list[float]

    n_captures: int

    """
    One of "PASS", "SUSPECT", or "FAIL" -- the same three-tier vocabulary
    SensitivityModel (VOC calibration) uses, so both calibration kinds read
    the same way.
    """
    status: str


def run_biomass_zero_calibration(
    camera_id: str,
    experiment_id: str,
    n_captures: int = 30,
) -> BiomassCameraModel:
    """Record fresh video clips against clear medium and calibrate from them."""

    """
    Not implemented: actually operating a camera requires algaesense-edge (a
    Raspberry-Pi-side package, built in a later phase), the same situation as
    jaxsr_calibration.calibration.standard_addition.run_standard_addition. If
    you already have captured clip feature vectors (e.g. from a test fixture),
    call compute_blank_baseline(feature_vectors) directly instead -- that part
    is fully implemented.
    """
    raise LiveAcquisitionNotAvailableError(
        "run_biomass_zero_calibration needs to drive live camera capture "
        "against a cell-free reference reactor (needs algaesense-edge, a "
        "later phase). If you already have captured clip feature vectors, "
        "call compute_blank_baseline(feature_vectors) directly instead."
    )


def compute_blank_baseline(
    feature_vectors: list[list[float]],
    camera_id: str,
    calibration_run_id: str,
    min_captures: int = 10,
    max_relative_std: float = 0.10,
) -> BiomassCameraModel:
    """Calibrate a camera's blank baseline from already-captured clips."""

    """
    Each entry in `feature_vectors` is one clip's already-averaged-across-
    its-own-frames [red, green, blue] reading. `max_relative_std` (std /
    |mean|, not spec-mandated -- a reasonable default, same style as
    fleet_zero's thresholds) flags the calibration as SUSPECT if the blank
    readings are noisier than expected across captures, e.g. from
    inconsistent lighting or camera settings between clips.
    """

    """
    Refuse to calibrate from too few captures -- a baseline estimated from
    only a handful of clips could easily just be noise rather than a
    trustworthy blank reading.
    """
    if len(feature_vectors) < min_captures:
        raise ValueError(
            f"compute_blank_baseline: need at least {min_captures} captures, got {len(feature_vectors)}"
        )

    """
    `np.asarray(feature_vectors)` stacks the list-of-lists into a proper
    (n_captures, 3) 2-D array. `axis=0` in the mean/std calls below then
    means "average down the rows (across captures), separately for each
    color channel" -- exactly "one mean/std per RGB channel".
    """
    array = np.asarray(feature_vectors, dtype=float)
    mean = array.mean(axis=0)
    std = array.std(axis=0)

    """
    Guards against dividing by (near) zero when a channel's true mean is
    ~0 -- relative std would blow up meaninglessly there, so any such
    channel is instead treated as automatically "not stable" (`np.inf`)
    rather than crashing or silently producing an invalid result.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_std = np.where(np.abs(mean) > 1e-9, std / np.abs(mean), np.inf)

    """
    The worst (largest) relative std across all 3 channels decides the
    overall status -- one noisy channel is enough to flag the whole
    calibration, even if the other two channels look fine.
    """
    worst_relative_std = float(np.max(relative_std))
    if worst_relative_std <= max_relative_std:
        status = "PASS"
    elif worst_relative_std <= max_relative_std * 2:
        status = "SUSPECT"
    else:
        status = "FAIL"

    return BiomassCameraModel(
        camera_id=camera_id,
        calibration_run_id=calibration_run_id,
        blank_baseline_mean=mean.tolist(),
        blank_baseline_std=std.tolist(),
        n_captures=len(feature_vectors),
        status=status,
    )


def apply_biomass_calibration(clip_features: list[float], model: BiomassCameraModel) -> float:
    """Turn a new camera reading into a biomass signal, relative to the blank."""

    """
    Sign convention: positive means the reading is *greener* than the blank
    (more chlorophyll-driven green signal), which corresponds to more
    biomass. This replaced an earlier "dimmer = more biomass" brightness-only
    convention -- brightness alone is confounded by exposure/lighting drift
    even when exposure/gain are nominally fixed, while greenness specifically
    tracks the pigment-driven color change denser culture actually causes.
    """

    reading = np.asarray(clip_features, dtype=float)
    baseline = np.asarray(model.blank_baseline_mean, dtype=float)

    """
    Confirms the new reading has the same number of color channels the model
    was calibrated with, before comparing them -- catches a caller passing a
    malformed reading with a clear error instead of a confusing shape
    mismatch deeper inside greenness_index.
    """
    if reading.shape != baseline.shape:
        raise ValueError(
            f"apply_biomass_calibration: clip_features has {reading.shape[0]} dimensions, "
            f"but model {model.calibration_run_id!r} was calibrated with {baseline.shape[0]}"
        )

    """
    Compares GREENNESS (a color ratio, brightness-independent) rather than
    averaging the raw per-channel differences -- see greenness_index's own
    docstring for why that distinction matters.
    """
    return greenness_index(reading) - greenness_index(baseline)


def persist_biomass_calibration(model: BiomassCameraModel, out_dir: Path) -> Path:
    """Save a camera's blank calibration to disk."""

    """
    Writes to `{out_dir}/{calibration_run_id}.yaml`. Unlike VOC calibration
    (Parquet, one row per sensor), a single camera's blank baseline is small
    enough that a single YAML file is simpler and just as adequate -- no
    per-row table structure is needed for one camera's one baseline.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model.calibration_run_id}.yaml"

    with path.open("w", encoding="utf-8") as f:
        """
        `sort_keys=False` preserves this dict's own field order in the
        written file, rather than yaml's default alphabetical re-sort --
        reads more naturally for a human opening this file directly.
        """
        yaml.safe_dump(
            {
                "camera_id": model.camera_id,
                "calibration_run_id": model.calibration_run_id,
                "blank_baseline_mean": model.blank_baseline_mean,
                "blank_baseline_std": model.blank_baseline_std,
                "n_captures": model.n_captures,
                "status": model.status,
            },
            f,
            sort_keys=False,
        )

    return path


def load_biomass_calibration(calibration_run_id: str, data_dir: Path) -> BiomassCameraModel:
    """Load a previously-saved camera blank calibration back from disk."""

    path = data_dir / f"{calibration_run_id}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"No biomass calibration file at {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    """
    `BiomassCameraModel(**raw)` unpacks the parsed YAML dict as keyword
    arguments -- this only works because the dict's keys (camera_id,
    calibration_run_id, ...) exactly match the dataclass's field names,
    which is exactly what persist_biomass_calibration wrote above.
    """
    return BiomassCameraModel(**raw)
