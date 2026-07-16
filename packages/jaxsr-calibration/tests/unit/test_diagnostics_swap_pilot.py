"""Unit tests for jaxsr_calibration.diagnostics.swap_pilot.run_swap_pilot.

Note on tolerances: variance-component estimates from a mixed-effects model
fit over only a handful of sensor/reactor levels are known to be noisy (this
was confirmed by hand while prototyping this module -- with only 4 sensors
and 4 reactors, fitted variance components can be off from their true
generating values by a factor of 2-3x, even though the model is correctly
specified). So these tests check *directional/structural* correctness
(shares sum to ~1, the dominant true source of variance comes back as the
dominant fitted share) rather than tight numeric recovery of the exact true
variance values.
"""

from __future__ import annotations

import pytest

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.diagnostics.swap_pilot import run_swap_pilot
from tests.fixtures.synthetic_readings import make_swap_pilot_readings

_SENSORS = ["PID01", "PID02", "PID03", "PID04", "PID05", "PID06"]
_REACTORS = ["R01", "R02", "R03", "R04", "R05", "R06"]


def test_run_swap_pilot_raises_without_readings() -> None:
    with pytest.raises(LiveAcquisitionNotAvailableError):
        run_swap_pilot(n_blocks=4, block_hours=4)


def test_run_swap_pilot_variance_shares_sum_to_one() -> None:
    readings = make_swap_pilot_readings(
        _SENSORS, _REACTORS, sensor_effect_std=3.0, reactor_effect_std=2.0, residual_std=5.0, seed=20
    )

    result = run_swap_pilot(n_blocks=6, block_hours=4, readings=readings)

    assert set(result.variance_share.keys()) == {"sensor_id", "reactor_id", "residual"}
    total = sum(result.variance_share.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert "MixedLM" in result.mixedlm_summary


def test_run_swap_pilot_dominant_true_source_shows_up_as_dominant_share() -> None:
    # Reactor effect is huge, sensor effect is essentially zero, residual is
    # small -- reactor_id's fitted share should clearly dominate sensor_id's.
    readings = make_swap_pilot_readings(
        _SENSORS,
        _REACTORS,
        sensor_effect_std=0.1,
        reactor_effect_std=10.0,
        residual_std=1.0,
        seed=21,
    )

    result = run_swap_pilot(n_blocks=6, block_hours=4, readings=readings)

    assert result.variance_share["reactor_id"] > result.variance_share["sensor_id"]
    # With such a small sensor effect relative to reactor effect, the
    # sensor's share should stay comfortably under the spec's own "healthy"
    # bar of 0.30 (spec §22's worked example asserts exactly this).
    assert result.variance_share["sensor_id"] < 0.30


def test_run_swap_pilot_high_residual_noise_dominates_when_effects_are_tiny() -> None:
    # Both sensor and reactor effects are tiny compared to residual noise --
    # residual should end up as the clearly dominant share.
    readings = make_swap_pilot_readings(
        _SENSORS,
        _REACTORS,
        sensor_effect_std=0.1,
        reactor_effect_std=0.1,
        residual_std=10.0,
        seed=22,
    )

    result = run_swap_pilot(n_blocks=6, block_hours=4, readings=readings)

    assert result.variance_share["residual"] > 0.8
