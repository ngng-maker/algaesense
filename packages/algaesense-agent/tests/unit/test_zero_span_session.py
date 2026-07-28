"""Tests for the zero-and-span calibration logic behind the Streamlit wizard.

Every test here works against a KNOWN generating line, so what is asserted is
that the procedure recovers the truth -- not merely that it returns numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algaesense_agent.dashboard.zero_span_session import (
    MIN_READINGS_PER_LEVEL,
    ZeroSpanSession,
    assess_quality,
    build_readings_frame,
    fit_zero_span,
    save_zero_span,
)
from jaxsr_calibration.calibration.apply import load_calibration


TRUE_B0 = 18.5
TRUE_B1 = 0.62
SPAN_PPM = 500.0


def _session(
    *,
    zero_noise: float = 0.2,
    span_noise: float = 0.2,
    n: int = 8,
    span_ppm: float = SPAN_PPM,
    seed: int = 0,
) -> ZeroSpanSession:
    """A completed bench recording of a sensor whose true response is the
    line above."""
    rng = np.random.default_rng(seed)
    session = ZeroSpanSession(
        reactor_id="R1",
        sensor_id="PID01",
        calibration_run_id="zerospan_test",
        experiment_id="exp_test",
        calibration_compound="isobutylene",
        mw_g_mol=56.11,
        response_factor=1.0,
        span_ppm=span_ppm,
    )
    session.zero_readings_mv = list(TRUE_B0 + rng.normal(0.0, zero_noise, size=n))
    session.span_readings_mv = list(
        TRUE_B0 + TRUE_B1 * span_ppm + rng.normal(0.0, span_noise, size=n)
    )
    return session


def test_a_two_point_calibration_recovers_the_true_line() -> None:
    """The whole point of the procedure: clean air plus one known gas is
    enough to pin down both the offset and the slope."""
    model = fit_zero_span(_session())

    assert model.b0_mv == pytest.approx(TRUE_B0, abs=0.3)
    assert model.b1_mv_per_ppm_asgas == pytest.approx(TRUE_B1, rel=0.01)


def test_a_two_point_calibration_never_produces_a_curvature_term() -> None:
    """Two concentrations cannot determine three coefficients. If a b2 ever
    appeared here it would be fitted to noise, and apply_calibration would
    then invert a curve that was never measured."""
    assert fit_zero_span(_session()).b2_mv_per_ppm2_asgas is None


def test_a_quadratic_fit_is_refused_with_a_reason() -> None:
    with pytest.raises(ValueError) as exc:
        fit_zero_span(_session(), method="polynomial_deg2")

    assert "three distinct concentrations" in str(exc.value)


def test_fitting_is_blocked_until_both_levels_have_enough_readings() -> None:
    session = _session(n=MIN_READINGS_PER_LEVEL)
    session.span_readings_mv = session.span_readings_mv[:1]

    assert not session.ready_to_fit
    with pytest.raises(ValueError) as exc:
        build_readings_frame(session)
    assert str(MIN_READINGS_PER_LEVEL) in str(exc.value)


def test_a_settled_recording_raises_no_warnings() -> None:
    quality = assess_quality(_session())

    assert quality.warnings == []
    assert quality.separation_mv == pytest.approx(TRUE_B1 * SPAN_PPM, rel=0.02)
    assert quality.curvature_untested is True


def test_an_unsettled_zero_is_caught_before_fitting() -> None:
    """A drifting baseline is the most common way a field calibration goes
    quietly wrong -- the fit still succeeds, it is just wrong."""
    quality = assess_quality(_session(zero_noise=8.0))

    assert any("Zero readings are unsettled" in w for w in quality.warnings)


def test_a_span_gas_too_weak_to_distinguish_is_caught() -> None:
    """With almost no separation between the two levels, the slope is
    determined by noise rather than by the gas."""
    quality = assess_quality(_session(span_ppm=5.0, zero_noise=0.2, span_noise=0.2))

    assert any("dominated by" in w for w in quality.warnings)


def test_a_span_reading_below_zero_is_called_out_specifically() -> None:
    """Usually means the span gas never actually reached the sensor. Worth
    its own message rather than a generic 'bad fit'."""
    session = _session()
    session.span_readings_mv = [v - 400.0 for v in session.span_readings_mv]

    quality = assess_quality(session)

    assert any("at or below zero" in w for w in quality.warnings)


def test_a_saved_calibration_reloads_with_the_same_coefficients(tmp_path: Path) -> None:
    session = _session()
    model = fit_zero_span(session)

    save_zero_span(session, model, tmp_path)
    reloaded = load_calibration(
        session.calibration_run_id,
        tmp_path / "derived" / "calibrations" / "standard_addition",
    )["PID01"]

    assert reloaded.b0_mv == pytest.approx(model.b0_mv)
    assert reloaded.b1_mv_per_ppm_asgas == pytest.approx(model.b1_mv_per_ppm_asgas)
