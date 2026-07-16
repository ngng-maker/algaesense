"""Unit tests for jaxsr_calibration.processing.spectral."""

from __future__ import annotations

import numpy as np
import pytest

from jaxsr_calibration.processing.spectral import lomb_scargle, notch_filter_known_artifacts


def _sine_signal(freq_hz: float, amplitude: float = 3.0, duration_s: int = 2000, noise_std: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0)
    y = amplitude * np.sin(2 * np.pi * freq_hz * t) + rng.normal(0.0, noise_std, size=t.shape)
    return t, y


def test_lomb_scargle_finds_known_dominant_frequency() -> None:
    t, y = _sine_signal(freq_hz=0.05, seed=1)

    freqs_hz, power = lomb_scargle(t, y)

    peak_freq = freqs_hz[np.argmax(power)]
    assert peak_freq == pytest.approx(0.05, abs=0.002)


def test_lomb_scargle_respects_explicit_freq_range() -> None:
    t, y = _sine_signal(freq_hz=0.05, seed=2)

    freqs_hz, _ = lomb_scargle(t, y, freq_range=(0.01, 0.1))

    assert freqs_hz.min() == pytest.approx(0.01)
    assert freqs_hz.max() == pytest.approx(0.1)


def test_notch_filter_removes_known_artifact_frequency() -> None:
    t, y = _sine_signal(freq_hz=0.05, seed=3)

    filtered, flags = notch_filter_known_artifacts(t, y, artifact_freqs=[0.05], q=30.0)

    assert any("notched 0.05" in flag for flag in flags)
    # The dominant-frequency power at 0.05 Hz should drop sharply after
    # notching it out.
    _, power_before = lomb_scargle(t, y, freq_range=(0.04, 0.06))
    _, power_after = lomb_scargle(t, filtered, freq_range=(0.04, 0.06))
    assert power_after.max() < power_before.max() * 0.5


def test_notch_filter_leaves_unrelated_frequency_largely_intact() -> None:
    # Signal has energy at 0.05 Hz; we notch a completely different
    # frequency (0.2 Hz) -- the 0.05 Hz peak should survive essentially
    # unchanged.
    t, y = _sine_signal(freq_hz=0.05, seed=4)

    filtered, _ = notch_filter_known_artifacts(t, y, artifact_freqs=[0.2], q=30.0)

    _, power_before = lomb_scargle(t, y, freq_range=(0.04, 0.06))
    _, power_after = lomb_scargle(t, filtered, freq_range=(0.04, 0.06))
    assert power_after.max() == pytest.approx(power_before.max(), rel=0.2)


def test_notch_filter_skips_frequency_at_or_above_nyquist() -> None:
    t, y = _sine_signal(freq_hz=0.05, seed=5)  # sampled at 1 Hz -> Nyquist = 0.5 Hz

    filtered, flags = notch_filter_known_artifacts(t, y, artifact_freqs=[0.6], q=30.0)

    assert any("skipped 0.6" in flag for flag in flags)
    # Signal should be untouched since the only requested frequency was skipped.
    assert np.allclose(filtered, y)


def test_notch_filter_handles_multiple_frequencies_with_mixed_validity() -> None:
    t, y = _sine_signal(freq_hz=0.05, seed=6)

    _, flags = notch_filter_known_artifacts(t, y, artifact_freqs=[0.05, 0.9], q=30.0)

    assert len(flags) == 2
    assert "notched" in flags[0]
    assert "skipped" in flags[1]
