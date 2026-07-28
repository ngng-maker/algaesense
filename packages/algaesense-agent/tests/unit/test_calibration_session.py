"""Tests for the multi-point calibration logic behind the Streamlit wizard.

Every test works against a KNOWN generating curve, so what is asserted is
that the procedure recovers the truth -- not merely that it returns
numbers. Several use a deliberately CURVED sensor, because detecting
curvature is the whole reason this procedure records more than two levels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algaesense_agent.dashboard.calibration_session import (
    DEFAULT_LEVELS_PPM,
    MIN_LEVELS,
    MIN_READINGS_PER_LEVEL,
    CalibrationSession,
    assess_quality,
    build_readings_frame,
    build_session_levels,
    fit_calibration,
    save_calibration,
)
from jaxsr_calibration.calibration.apply import load_calibration
from jaxsr_calibration.calibration.models import CalibrationGas


TRUE_B0 = 18.5
TRUE_B1 = 62.0
"""mV per ppm. Steep, because this sensor's whole working range is 0-5 ppm
-- a slope sized for hundreds of ppm would put full scale inside the noise."""

TRUE_B2 = -1.8
"""Negative: the response compresses toward the top of the range, the usual
PID saturation direction, and exactly what two points cannot see."""


def _session(
    *,
    levels_ppm: list[float] | None = None,
    b2: float = 0.0,
    noise_mv: float = 0.15,
    n_per_level: int = 5,
    seed: int = 0,
) -> CalibrationSession:
    """A completed bench recording of a sensor whose true response is the
    curve above."""
    rng = np.random.default_rng(seed)
    session = CalibrationSession(
        reactor_id="R1",
        sensor_id="PID01",
        calibration_run_id="cal_test",
        experiment_id="exp_test",
        calibration_compound="isobutylene",
        mw_g_mol=56.11,
        response_factor=1.0,
        levels=build_session_levels(levels_ppm or DEFAULT_LEVELS_PPM),
    )
    for level in session.levels:
        true_mv = TRUE_B0 + TRUE_B1 * level.ppm + b2 * level.ppm**2
        level.readings_mv = list(true_mv + rng.normal(0.0, noise_mv, size=n_per_level))
    return session


def test_the_default_levels_are_zero_plus_three_across_the_range() -> None:
    """Clean air fixes the offset; three more spanning 0-5 ppm pin down the
    shape of the response across the range the reactor actually uses."""
    assert DEFAULT_LEVELS_PPM == [0.0, 1.0, 3.0, 5.0]
    assert len(DEFAULT_LEVELS_PPM) >= MIN_LEVELS
    assert 0.0 in DEFAULT_LEVELS_PPM


def test_levels_are_ordered_lowest_first() -> None:
    """The operator works upward, so a high concentration's residue never
    lingers in the line and inflates a lower reading taken after it."""
    levels = build_session_levels([5.0, 0.0, 3.0, 1.0])

    assert [level.ppm for level in levels] == [0.0, 1.0, 3.0, 5.0]


def test_a_linear_sensor_is_recovered_across_four_levels() -> None:
    model = fit_calibration(_session(), method="ols")

    assert model.b0_mv == pytest.approx(TRUE_B0, abs=0.3)
    assert model.b1_mv_per_ppm_asgas == pytest.approx(TRUE_B1, rel=0.02)


def test_a_curved_sensor_is_detected_and_fitted() -> None:
    """The capability the extra levels buy: a two-point calibration would
    have reported this sensor as a straight line."""
    session = _session(b2=TRUE_B2)

    quality = assess_quality(session)
    assert quality.suggests_quadratic, "curvature this size should be visible"
    assert any("bends" in w for w in quality.warnings)

    model = fit_calibration(session, method="polynomial_deg2")
    assert model.b2_mv_per_ppm2_asgas == pytest.approx(TRUE_B2, rel=0.2)


def test_a_straight_sensor_is_not_accused_of_bending() -> None:
    """The counterpart: noise alone must not raise a curvature warning, or
    it becomes something operators learn to ignore."""
    quality = assess_quality(_session(seed=3))

    assert not quality.suggests_quadratic
    assert quality.r_squared > 0.999


def test_r_squared_is_a_real_measure_here() -> None:
    """With four levels and two coefficients the fit can genuinely fail to
    describe the points -- unlike a two-point calibration, where R² sits
    near 1.0 whatever the sensor did."""
    session = _session()
    for level in session.levels:
        level.readings_mv = [TRUE_B0 + 5.0, TRUE_B0 - 5.0, TRUE_B0]

    assert assess_quality(session).r_squared < 0.5


def test_an_unsettled_level_is_caught_before_fitting() -> None:
    session = _session()
    session.levels[1].readings_mv = [100.0, 40.0, 75.0, 55.0, 90.0]

    assert any("unsettled" in w for w in assess_quality(session).warnings)


def test_a_non_monotonic_series_is_called_out() -> None:
    """A higher concentration reading lower than a lower one is physically
    impossible for a PID -- it means a procedural mistake, not noise."""
    session = _session()
    session.levels[2].readings_mv = [mv - 200.0 for mv in session.levels[2].readings_mv]

    assert any("reads no higher than" in w for w in assess_quality(session).warnings)


def test_fitting_is_blocked_until_every_level_is_recorded() -> None:
    session = _session()
    session.levels[-1].readings_mv = session.levels[-1].readings_mv[:1]

    assert not session.ready_to_fit
    with pytest.raises(ValueError) as exc:
        build_readings_frame(session)
    assert str(MIN_READINGS_PER_LEVEL) in str(exc.value)


def test_too_few_distinct_levels_is_refused() -> None:
    """Three points fit a quadratic exactly, leaving nothing left over to
    judge whether that curvature is real."""
    session = _session(levels_ppm=[0.0, 1.0, 5.0])

    assert session.distinct_level_count < MIN_LEVELS
    assert not session.ready_to_fit


def test_an_unsupported_fit_method_is_refused_with_the_alternatives() -> None:
    with pytest.raises(ValueError) as exc:
        fit_calibration(_session(), method="spline_deg5")

    assert "polynomial_deg2" in str(exc.value)


def test_weighted_least_squares_is_the_default() -> None:
    """PID scatter grows with concentration, so weighting each level by how
    tightly its own replicates clustered stops the noisiest end of the range
    dominating the fit."""
    session = _session()
    for level in session.levels:
        """Noise proportional to concentration, which is the real behaviour."""
        rng = np.random.default_rng(int(level.ppm * 10))
        true_mv = TRUE_B0 + TRUE_B1 * level.ppm
        level.readings_mv = list(true_mv + rng.normal(0.0, 0.1 + 0.6 * level.ppm, size=8))

    assert fit_calibration(session).fit_method == "wls"


def test_a_saved_calibration_reloads_with_the_same_curve(tmp_path: Path) -> None:
    """Guards a silent failure: if the quadratic term were dropped on save,
    apply_calibration would fall back to linear inversion and report biased
    concentrations with no error raised anywhere."""
    session = _session(b2=TRUE_B2)
    model = fit_calibration(session, method="polynomial_deg2")

    save_calibration(session, model, tmp_path)
    reloaded = load_calibration(
        session.calibration_run_id, tmp_path / "derived" / "calibrations" / "standard_addition"
    )["PID01"]

    assert reloaded.b0_mv == pytest.approx(model.b0_mv)
    assert reloaded.b1_mv_per_ppm_asgas == pytest.approx(model.b1_mv_per_ppm_asgas)
    assert reloaded.b2_mv_per_ppm2_asgas == pytest.approx(model.b2_mv_per_ppm2_asgas)


def test_the_response_factor_table_is_anchored_to_isobutylene() -> None:
    """The industrial reference: every other compound's response factor is
    meaningless without knowing what it is relative to."""
    gases = {gas.name: gas for gas in CalibrationGas.list_builtin()}

    assert gases["isobutylene"].response_factor == 1.00
    assert gases["isobutylene"].mw == pytest.approx(56.11)


def test_every_listed_gas_carries_the_numbers_a_calibration_needs() -> None:
    """The wizard fills molecular weight and response factor straight from
    this table, so a missing value there becomes a blank field the operator
    has to guess at."""
    gases = CalibrationGas.list_builtin()

    assert len(gases) >= 5
    for gas in gases:
        assert gas.mw > 0, f"{gas.name} has no molecular weight"
        assert gas.response_factor and gas.response_factor > 0, f"{gas.name} has no response factor"
        assert gas.source, f"{gas.name} does not say where its value came from"
        assert gas.is_builtin
