"""Unit tests for algaesense_edge.actuators.control_profiles: pure,
hardware-independent evaluation of each supported profile shape.
"""

from __future__ import annotations

import math

import pytest

from algaesense_edge.actuators.control_profiles import (
    UnknownProfileShapeError,
    evaluate_control_profile,
    validate_control_profile,
)


def test_constant_profile_always_returns_the_same_value() -> None:
    profile = {"shape": "constant", "par_umol_m2_s": 150.0}

    assert evaluate_control_profile(profile, elapsed_s=0.0) == 150.0
    assert evaluate_control_profile(profile, elapsed_s=9999.0) == 150.0


def test_ramp_profile_interpolates_linearly_then_holds_at_the_end() -> None:
    profile = {"shape": "ramp", "start_par_umol_m2_s": 0.0, "end_par_umol_m2_s": 100.0, "duration_s": 100.0}

    assert evaluate_control_profile(profile, elapsed_s=0.0) == pytest.approx(0.0)
    assert evaluate_control_profile(profile, elapsed_s=50.0) == pytest.approx(50.0)
    assert evaluate_control_profile(profile, elapsed_s=100.0) == pytest.approx(100.0)
    # Past the ramp's own duration, it holds at the end value rather than
    # extrapolating past it or raising.
    assert evaluate_control_profile(profile, elapsed_s=500.0) == pytest.approx(100.0)


def test_sinusoid_profile_matches_the_documented_formula() -> None:
    profile = {"shape": "sinusoid", "mean_par_umol_m2_s": 200.0, "amplitude_par_umol_m2_s": 50.0, "period_s": 3600.0}

    # At t=0 (no phase offset), sin(0) == 0, so the value should be exactly the mean.
    assert evaluate_control_profile(profile, elapsed_s=0.0) == pytest.approx(200.0)
    # A quarter-period in, sin(pi/2) == 1, so the value should be mean + amplitude.
    assert evaluate_control_profile(profile, elapsed_s=900.0) == pytest.approx(250.0)


def test_step_profile_holds_each_segment_then_the_last_one_indefinitely() -> None:
    profile = {
        "shape": "step",
        "segments": [
            {"par_umol_m2_s": 50.0, "duration_s": 10.0},
            {"par_umol_m2_s": 150.0, "duration_s": 10.0},
        ],
    }

    assert evaluate_control_profile(profile, elapsed_s=0.0) == 50.0
    assert evaluate_control_profile(profile, elapsed_s=9.9) == 50.0
    assert evaluate_control_profile(profile, elapsed_s=10.0) == 150.0
    assert evaluate_control_profile(profile, elapsed_s=19.9) == 150.0
    # Past the last segment, holds at its value rather than erroring.
    assert evaluate_control_profile(profile, elapsed_s=1000.0) == 150.0


def test_evaluate_raises_for_unknown_shape() -> None:
    with pytest.raises(UnknownProfileShapeError):
        evaluate_control_profile({"shape": "spiral"}, elapsed_s=0.0)


def test_validate_rejects_unknown_shape() -> None:
    with pytest.raises(UnknownProfileShapeError):
        validate_control_profile({"shape": "spiral"})


def test_validate_rejects_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        validate_control_profile({"shape": "ramp", "start_par_umol_m2_s": 0.0})


def test_validate_rejects_non_positive_ramp_duration() -> None:
    with pytest.raises(ValueError, match="duration_s must be positive"):
        validate_control_profile(
            {"shape": "ramp", "start_par_umol_m2_s": 0.0, "end_par_umol_m2_s": 100.0, "duration_s": 0.0}
        )


def test_validate_rejects_empty_step_segments() -> None:
    with pytest.raises(ValueError, match="segments must not be empty"):
        validate_control_profile({"shape": "step", "segments": []})


def test_validate_accepts_every_well_formed_shape() -> None:
    validate_control_profile({"shape": "constant", "par_umol_m2_s": 100.0})
    validate_control_profile({"shape": "ramp", "start_par_umol_m2_s": 0.0, "end_par_umol_m2_s": 100.0, "duration_s": 60.0})
    validate_control_profile({"shape": "sinusoid", "mean_par_umol_m2_s": 100.0, "amplitude_par_umol_m2_s": 20.0, "period_s": 3600.0})
    validate_control_profile({"shape": "step", "segments": [{"par_umol_m2_s": 50.0, "duration_s": 10.0}]})
