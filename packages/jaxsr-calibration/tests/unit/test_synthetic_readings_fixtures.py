"""Regression tests for tests.fixtures.synthetic_readings' generators
themselves -- not the code under test, but the fixtures that generate
data FOR that code.

`make_swap_pilot_readings` and `make_standard_addition_readings` were
rewritten from Python row-loops to vectorized numpy construction (the
other four generators in this module were already vectorized). These
tests pin their exact output under a fixed seed against values captured
from the original loop-based implementation, before it was replaced --
confirming the rewrite preserves byte-for-byte generating semantics, not
just "produces plausible-looking data".
"""

from __future__ import annotations

import pytest

from tests.fixtures.synthetic_readings import make_standard_addition_readings, make_swap_pilot_readings


def test_make_swap_pilot_readings_matches_captured_reference_output() -> None:
    df = make_swap_pilot_readings(
        sensor_ids=["PID01", "PID02", "PID03"],
        reactor_ids=["R01", "R02", "R03"],
        sensor_effect_std=0.5,
        reactor_effect_std=2.0,
        residual_std=0.3,
        obs_per_block=4,
        seed=99,
    )

    assert df.shape == (36, 4)
    # Captured from the pre-vectorization loop-based implementation under
    # the same seed/params -- see this module's own docstring.
    assert df["pid_voltage_mv"][0] == pytest.approx(51.27635593972524)
    assert df["pid_voltage_mv"][17] == pytest.approx(53.17884098307124)
    assert df["pid_voltage_mv"][35] == pytest.approx(46.95307589518206)
    assert df["sensor_id"].to_list()[:4] == ["PID01", "PID01", "PID01", "PID01"]
    assert df["reactor_id"].to_list()[:8] == ["R01", "R01", "R01", "R01", "R02", "R02", "R02", "R02"]
    # Block 1 (rows 12-15) should rotate PID01 onto R02, not R01 -- the
    # Latin-square rotation the original nested loop implemented.
    assert df["reactor_id"].to_list()[12:16] == ["R02", "R02", "R02", "R02"]


def test_make_standard_addition_readings_matches_captured_reference_output() -> None:
    df = make_standard_addition_readings(
        {
            "PID01": {"b0_mv": 2.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.1},
            "PID02": {"b0_mv": 1.0, "b1_mv_per_ppm": 3.0},
        },
        spike_ppm_list=[0.0, 5.0],
        n_per_level=3,
        seed=77,
    )

    assert df.height == 12
    # Captured from the pre-vectorization loop-based implementation under
    # the same seed/params.
    assert df["pid_voltage_mv"][0] == pytest.approx(2.0427769964204727)
    assert df["pid_voltage_mv"][3] == pytest.approx(21.901532378991135)
    assert df["pid_voltage_mv"][11] == pytest.approx(16.177868870146213)
    assert df["sample_t_c"][0] == pytest.approx(31.942916244311355)
    assert df["sample_rh_pct"][0] == pytest.approx(56.32723034486505)
    assert df["spike_ppm_asgas"].to_list() == [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
