"""Spectral diagnostics and filtering: find periodic artifacts in a signal
and remove known ones with a notch filter.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


"""
Uses Lomb-Scargle rather than a plain FFT because Lomb-Scargle tolerates
irregularly spaced or gappy timestamps.
"""


def lomb_scargle(
    t: np.ndarray, y: np.ndarray, freq_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Lomb-Scargle periodogram of `y` sampled at times `t`."""

    """
    Returns (frequencies_hz, power) -- the strongest peak in `power`
    indicates the dominant periodic component's frequency (spec's own
    `dominant_freq_hz`/`dominant_freq_amp` feature columns, §19, are
    derived from exactly this).

    `freq_range` defaults to a sensible span derived from the data itself:
    the low end is 1/(observation duration) (you can't resolve a cycle
    slower than your whole recording), the high end is the Nyquist
    frequency implied by the median sample spacing (you can't resolve
    anything faster than twice your sampling rate).
    """

    if freq_range is None:
        duration_s = float(t[-1] - t[0])
        median_dt_s = float(np.median(np.diff(t)))
        f_min = 1.0 / duration_s
        f_max = 0.5 / median_dt_s
    else:
        f_min, f_max = freq_range

    freqs_hz = np.linspace(f_min, f_max, 2000)

    """
    `scipy.signal.lombscargle` wants ANGULAR frequencies (radians/second,
    i.e. 2*pi*f), not frequencies in Hz -- easy to get wrong, confirmed by
    hand-testing this against a signal with a known frequency before
    writing this function.
    """
    angular_freqs = 2 * np.pi * freqs_hz

    """
    Subtracting the mean before calling lombscargle is standard practice:
    the algorithm looks for periodic (sinusoidal) structure, and a nonzero
    mean/DC offset isn't periodic information -- leaving it in would just
    waste some of the frequency-0-ish end of the spectrum on something
    that isn't actually a cycle.
    """
    y_centered = y - np.mean(y)

    """
    `normalize=True` scales the output power onto a standardized [0,
    1]-ish scale (specifically a form of the "generalized Lomb-Scargle"
    normalization) rather than raw, hard-to-interpret units -- makes the
    spec's `min_amplitude_to_flag` threshold (PreprocessingConfig.spectral)
    meaningful as a fixed number across sensors/signals of different scale.
    """
    power = scipy_signal.lombscargle(t, y_centered, angular_freqs, normalize=True)

    return freqs_hz, power


def notch_filter_known_artifacts(
    t: np.ndarray, y: np.ndarray, artifact_freqs: list[float], q: float = 30.0
) -> tuple[np.ndarray, list[str]]:
    """Remove each frequency in `artifact_freqs` from `y` with a digital
    notch (band-stop) filter."""

    """
    Returns (filtered_signal, flags) where `flags` is one human-readable
    string per requested frequency describing what happened to it.

    `q` (the filter's "quality factor") controls how narrow the notch is --
    higher Q removes a narrower band around each target frequency, leaving
    more of the surrounding spectrum untouched.
    """

    median_dt_s = float(np.median(np.diff(t)))
    sample_rate_hz = 1.0 / median_dt_s
    nyquist_hz = sample_rate_hz / 2.0

    filtered = y.copy()
    flags: list[str] = []

    for freq_hz in artifact_freqs:
        if freq_hz <= 0 or freq_hz >= nyquist_hz:
            flags.append(
                f"skipped {freq_hz} Hz: outside the filterable range "
                f"(0, {nyquist_hz:.4g}) Hz for this sample rate ({sample_rate_hz:.4g} Hz)"
            )
            continue

        """
        `scipy.signal.iirnotch(freq, Q, fs)` designs a notch filter's
        coefficients (b = numerator, a = denominator of the filter's
        transfer function) -- it doesn't apply the filter itself yet.
        """
        b, a = scipy_signal.iirnotch(freq_hz, q, fs=sample_rate_hz)

        """
        `filtfilt` (as opposed to `lfilter`) applies the filter forwards
        AND backwards through the signal, which cancels out the phase
        shift a normal one-directional digital filter would otherwise
        introduce -- important here since we want to remove a frequency's
        *amplitude* without distorting the timing of what's left.
        """
        filtered = scipy_signal.filtfilt(b, a, filtered)
        flags.append(f"notched {freq_hz} Hz (Q={q})")

    return filtered, flags
