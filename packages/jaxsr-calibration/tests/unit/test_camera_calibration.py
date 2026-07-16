"""Unit tests for jaxsr_calibration.camera.calibration.

Every clip feature vector here is a 3-element [red, green, blue] triple --
that shape is now fixed and load-bearing (see greenness_index), not an
arbitrary-length vector like earlier versions of this module used.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxsr_calibration.camera.calibration import (
    apply_biomass_calibration,
    compute_blank_baseline,
    greenness_index,
    load_biomass_calibration,
    persist_biomass_calibration,
    run_biomass_zero_calibration,
)
from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError


def _synthetic_blank_clips(
    true_rgb: list[float], noise_std: float, n: int = 30, seed: int = 0
) -> list[list[float]]:
    # Each row stands in for one already-frame-averaged video clip's
    # [red, green, blue] feature vector (the per-frame averaging itself
    # happens on the Pi, not in this package).
    rng = np.random.default_rng(seed)
    mean_array = np.asarray(true_rgb)
    clips = mean_array + rng.normal(0.0, noise_std, size=(n, 3))
    return clips.tolist()


def test_greenness_index_matches_excess_green_formula() -> None:
    # 2*G - R - B for [red=10, green=50, blue=20] = 2*50 - 10 - 20 = 70.
    assert greenness_index([10.0, 50.0, 20.0]) == pytest.approx(70.0)


def test_greenness_index_is_brightness_independent() -> None:
    # Uniformly scaling all three channels (a pure brightness/exposure
    # change, same color) should leave greenness at/near zero if the base
    # color is neutral grey -- greenness responds to *color*, not brightness.
    grey_dim = [50.0, 50.0, 50.0]
    grey_bright = [150.0, 150.0, 150.0]
    assert greenness_index(grey_dim) == pytest.approx(0.0)
    assert greenness_index(grey_bright) == pytest.approx(0.0)


def test_greenness_index_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="exactly 3 values"):
        greenness_index([1.0, 2.0])


def test_run_biomass_zero_calibration_always_needs_live_acquisition() -> None:
    with pytest.raises(LiveAcquisitionNotAvailableError):
        run_biomass_zero_calibration(camera_id="CAM01", experiment_id="exp_test")


def test_compute_blank_baseline_recovers_known_rgb_mean() -> None:
    clips = _synthetic_blank_clips([100.0, 90.0, 80.0], noise_std=1.0, n=50, seed=1)

    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_01")

    assert model.blank_baseline_mean[0] == pytest.approx(100.0, abs=1.0)  # red
    assert model.blank_baseline_mean[1] == pytest.approx(90.0, abs=1.0)  # green
    assert model.blank_baseline_mean[2] == pytest.approx(80.0, abs=1.0)  # blue
    assert model.n_captures == 50
    assert model.status == "PASS"


def test_compute_blank_baseline_flags_noisy_capture_as_suspect_or_fail() -> None:
    # noise_std=25 against a channel mean of 100 is way above the default
    # 10% relative-std PASS bar and its 2x SUSPECT/FAIL boundary.
    clips = _synthetic_blank_clips([100.0, 100.0, 100.0], noise_std=25.0, n=50, seed=2)

    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_02")

    assert model.status in ("SUSPECT", "FAIL")


def test_compute_blank_baseline_rejects_too_few_captures() -> None:
    clips = _synthetic_blank_clips([100.0, 100.0, 100.0], noise_std=1.0, n=3, seed=3)

    with pytest.raises(ValueError, match="at least"):
        compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_03", min_captures=10)


def test_apply_biomass_calibration_returns_zero_for_reading_matching_baseline() -> None:
    clips = _synthetic_blank_clips([80.0, 90.0, 70.0], noise_std=0.5, n=30, seed=4)
    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_04")

    signal = apply_biomass_calibration(model.blank_baseline_mean, model)

    assert signal == pytest.approx(0.0, abs=1e-9)


def test_apply_biomass_calibration_positive_when_reading_is_greener_than_baseline() -> None:
    clips = _synthetic_blank_clips([80.0, 90.0, 70.0], noise_std=0.5, n=30, seed=5)
    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_05")

    # Same overall brightness as the baseline, but shifted toward green
    # (green up, red/blue down) -- greenness should read positive.
    greener_reading = [70.0, 110.0, 60.0]
    signal = apply_biomass_calibration(greener_reading, model)

    assert signal > 0.0


def test_apply_biomass_calibration_ignores_pure_brightness_change() -> None:
    # This is the key fix over the old brightness-based convention: a reading
    # that's simply DIMMER but the same color as the baseline should NOT
    # register as more biomass. noise_std=0 here (unlike other tests) so the
    # baseline mean is EXACTLY [100,100,100] -- otherwise finite-sample noise
    # in the estimated baseline would leave a tiny nonzero residual and this
    # exact-zero assertion would be testing sampling noise, not the logic.
    clips = _synthetic_blank_clips([100.0, 100.0, 100.0], noise_std=0.0, n=30, seed=6)
    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_06")

    dimmer_but_same_color = [60.0, 60.0, 60.0]
    signal = apply_biomass_calibration(dimmer_but_same_color, model)

    assert signal == pytest.approx(0.0, abs=1e-6)


def test_apply_biomass_calibration_rejects_wrong_length_reading() -> None:
    clips = _synthetic_blank_clips([100.0, 90.0, 80.0], noise_std=0.5, n=30, seed=7)
    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_07")

    with pytest.raises(ValueError, match="dimensions"):
        apply_biomass_calibration([80.0, 70.0], model)  # only 2 dimensions, needs 3


def test_persist_and_load_biomass_calibration_round_trip(tmp_path: Path) -> None:
    clips = _synthetic_blank_clips([100.0, 90.0, 80.0], noise_std=1.0, n=30, seed=8)
    model = compute_blank_baseline(clips, camera_id="CAM01", calibration_run_id="cam_cal_08")

    out_dir = tmp_path / "calibrations" / "camera_zero"
    path = persist_biomass_calibration(model, out_dir)

    assert path.exists()
    loaded = load_biomass_calibration("cam_cal_08", out_dir)

    assert loaded.camera_id == "CAM01"
    assert loaded.blank_baseline_mean == pytest.approx(model.blank_baseline_mean)
    assert loaded.status == model.status


def test_load_biomass_calibration_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_biomass_calibration("does_not_exist", tmp_path)
