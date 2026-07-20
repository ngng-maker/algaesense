"""Time-varying actuator control profiles: a small, fixed vocabulary of
shapes (ramp, sinusoid, step, constant) described as plain data, evaluated
by this project's own trusted code -- not arbitrary generated code
executed on the Pi. The shape math here has no LED dependency (it just
returns a target float) -- LED is the only actuator kind that exists
today, but the same profiles apply equally to any future actuator kind
that implements `ControlProfileActuator` (see actuators.py).
"""

from __future__ import annotations

import math


"""
This module exists specifically BECAUSE letting an LLM generate and run
arbitrary Python code on the Pi (the originally-considered design) would
be a materially larger trust boundary than anything else in this project
-- the Pi already needs root/sudo for GPIO access (see
docs/hardware_setup.md), so arbitrary code execution there means
arbitrary root-level access, not just "a bad light value." Confirmed with
the user 2026-07-17: the agent only ever supplies typed PARAMETERS for
one of the shapes below; this module's `evaluate_control_profile` is the
only code that ever runs, and every value it produces still passes
through `LEDActuator.set_par()`'s existing bounds-check on every single
tick, not just once at the start.

A profile is a plain dict (not a dataclass) deliberately -- it crosses
process boundaries twice (agent -> edge API -> this function), and a
plain JSON-shaped dict is what both those boundaries already speak
natively, without needing a parsing/validation layer on both sides.
`validate_control_profile` is the one place that actually checks the
shape is well-formed.
"""


class UnknownProfileShapeError(ValueError):
    """Raised when a profile's `shape` isn't one of the supported ones."""


_KNOWN_SHAPES = {"constant", "ramp", "sinusoid", "step"}


def validate_control_profile(profile: dict) -> None:
    """Check a profile dict is well-formed before it's ever started --
    raises ValueError with a clear message if not."""

    shape = profile.get("shape")
    if shape not in _KNOWN_SHAPES:
        raise UnknownProfileShapeError(
            f"Unknown control profile shape {shape!r}; supported shapes are {sorted(_KNOWN_SHAPES)}"
        )

    if shape == "constant":
        _require_keys(profile, ["par_umol_m2_s"])
    elif shape == "ramp":
        _require_keys(profile, ["start_par_umol_m2_s", "end_par_umol_m2_s", "duration_s"])
        if profile["duration_s"] <= 0:
            raise ValueError("ramp profile: duration_s must be positive")
    elif shape == "sinusoid":
        _require_keys(profile, ["mean_par_umol_m2_s", "amplitude_par_umol_m2_s", "period_s"])
        if profile["period_s"] <= 0:
            raise ValueError("sinusoid profile: period_s must be positive")
    elif shape == "step":
        _require_keys(profile, ["segments"])
        if not profile["segments"]:
            raise ValueError("step profile: segments must not be empty")
        for segment in profile["segments"]:
            if "par_umol_m2_s" not in segment:
                raise ValueError("step profile: every segment must have par_umol_m2_s")
            if segment.get("duration_s", 0) <= 0:
                raise ValueError("step profile: every segment's duration_s must be positive")


def _require_keys(profile: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in profile]
    if missing:
        raise ValueError(f"{profile.get('shape')!r} profile is missing required keys: {missing}")


def evaluate_control_profile(profile: dict, elapsed_s: float) -> float:
    """Compute the target PAR (umol/m^2/s) a profile calls for at
    `elapsed_s` seconds since it started."""

    shape = profile["shape"]

    if shape == "constant":
        return float(profile["par_umol_m2_s"])

    if shape == "ramp":
        """
        Linear interpolation from start to end over duration_s, then
        holds at end_par_umol_m2_s indefinitely once duration_s has
        elapsed -- a ramp that "finished" is a completed step, not an
        error.
        """
        start = float(profile["start_par_umol_m2_s"])
        end = float(profile["end_par_umol_m2_s"])
        duration = float(profile["duration_s"])
        fraction = min(max(elapsed_s / duration, 0.0), 1.0)
        return start + (end - start) * fraction

    if shape == "sinusoid":
        """
        `mean + amplitude * sin(2*pi*(elapsed_s + phase_s) / period_s)` --
        the standard parameterization (mean, amplitude, period, phase)
        rather than raw angular-frequency terms, since those are the
        quantities a DoE recommendation would actually specify.
        """
        mean = float(profile["mean_par_umol_m2_s"])
        amplitude = float(profile["amplitude_par_umol_m2_s"])
        period = float(profile["period_s"])
        phase = float(profile.get("phase_s", 0.0))
        return mean + amplitude * math.sin(2 * math.pi * (elapsed_s + phase) / period)

    if shape == "step":
        """
        Walks forward through each segment's duration until `elapsed_s`
        falls inside one -- once past the last segment, holds at its
        value indefinitely, same "finished = hold at the end" behavior as
        ramp.
        """
        cursor = 0.0
        for segment in profile["segments"]:
            duration = float(segment["duration_s"])
            if elapsed_s < cursor + duration:
                return float(segment["par_umol_m2_s"])
            cursor += duration
        return float(profile["segments"][-1]["par_umol_m2_s"])

    raise UnknownProfileShapeError(f"Unknown control profile shape {shape!r}")
