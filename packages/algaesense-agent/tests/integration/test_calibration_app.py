"""Tests that drive the real calibration Streamlit page.

`streamlit.testing.v1.AppTest` executes the actual app script and lets a
test set widget values and click buttons, so these exercise the genuine
page -- its step transitions, its guards, the calibration it writes --
rather than a re-implementation of it.

The two that matter most: walking the whole wizard must produce a
calibration that turns a raw recording it never saw into correct
concentrations, and that calibration must then change what the monitoring
page displays.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from algaesense_agent.dashboard import calibration_capture
from algaesense_agent.dashboard.calibration_session import DEFAULT_LEVELS_PPM
from jaxsr_calibration.calibration.apply import apply_calibration


_DASHBOARD_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "algaesense_agent" / "dashboard"
)
APP_PATH = str(_DASHBOARD_DIR / "calibration_app.py")
COMBINED_APP_PATH = str(_DASHBOARD_DIR / "app.py")

TRUE_B0 = 22.0
TRUE_B1 = 62.0
"""mV per ppm across this sensor's real 0-5 ppm working range."""


def _true_mv(ppm: float) -> float:
    return TRUE_B0 + TRUE_B1 * ppm


def _readings_at(ppm: float) -> list[float]:
    """Three settled readings, a tenth of a millivolt apart."""
    base = _true_mv(ppm)
    return [base - 0.1, base, base + 0.1]


def _app(tmp_path: Path) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["cal_data_dir"] = str(tmp_path)
    return at.run()


def _complete_setup(
    at: AppTest, levels: str | None = None, gas: str = "isobutylene"
) -> AppTest:
    """Fill in EVERY setup field, then start.

    Every one is set explicitly, including the levels box. A step
    transition reruns the script inside a single AppTest.run(), and the
    resulting element tree holds widgets from both passes -- so a setup
    widget the test never set is later read back from session state that
    Streamlit has already purged, and the next run raises KeyError.
    """
    at.text_input(key="cal_reactor_id").set_value("R1")
    at.text_input(key="cal_sensor_id").set_value("PID01")
    at.text_input(key="cal_experiment_id").set_value("exp_cal")
    at.selectbox(key="cal_gas_choice").set_value(gas)
    at.text_input(key="cal_levels_text").set_value(
        levels if levels is not None else ", ".join(f"{v:g}" for v in DEFAULT_LEVELS_PPM)
    )
    at.button(key="cal_start").click()
    return at.run()


def _record_current_level(at: AppTest, values: list[float]) -> AppTest:
    """Add readings one at a time through the real manual-entry widgets."""
    index = at.session_state["cal_level_index"]
    for value in values:
        at.number_input(key=f"cal_manual_{index}").set_value(value)
        at.button(key=f"cal_add_{index}").click()
        at = at.run()
    return at


def _walk_to_saved(tmp_path: Path) -> AppTest:
    at = _complete_setup(_app(tmp_path))

    for _ in DEFAULT_LEVELS_PPM:
        index = at.session_state["cal_level_index"]
        ppm = at.session_state["cal_session"].levels[index].ppm
        at = _record_current_level(at, _readings_at(ppm))
        at.button(key=f"cal_next_{index}").click()
        at = at.run()

    at.button(key="cal_fit").click()
    at = at.run()
    at.button(key="cal_save").click()
    return at.run()


def test_the_page_starts_on_setup_without_errors() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert at.session_state["cal_step"] == "Setup"


def test_the_gas_picker_fills_in_the_response_factor_table_values(tmp_path: Path) -> None:
    """The operator picks a compound; molecular weight and response factor
    come from the packaged table rather than being typed from memory."""
    at = _complete_setup(_app(tmp_path), gas="isoprene")
    session = at.session_state["cal_session"]

    assert session.calibration_compound == "isoprene"
    assert session.response_factor == pytest.approx(0.63)
    assert session.mw_g_mol == pytest.approx(68.12)


def test_setup_refuses_to_start_with_missing_fields(tmp_path: Path) -> None:
    at = _app(tmp_path)
    at.text_input(key="cal_reactor_id").set_value("R1")
    at.button(key="cal_start").click()
    at = at.run()

    assert at.session_state["cal_step"] == "Setup"
    assert "Sensor ID" in at.error[0].value


def test_setup_refuses_fewer_than_four_levels(tmp_path: Path) -> None:
    """Three points fit a quadratic exactly, leaving nothing left over to
    judge whether the curvature is real."""
    at = _complete_setup(_app(tmp_path), levels="0, 1, 5")

    assert at.session_state["cal_step"] == "Setup"
    assert "distinct concentrations" in at.error[0].value


def test_setup_refuses_a_series_without_clean_air(tmp_path: Path) -> None:
    """Without a zero there is nothing to fix the sensor's own baseline
    offset against."""
    at = _complete_setup(_app(tmp_path), levels="1, 2, 3, 5")

    assert at.session_state["cal_step"] == "Setup"
    assert "0 ppm" in at.error[0].value


def test_the_wizard_steps_through_every_level_in_ascending_order(tmp_path: Path) -> None:
    at = _complete_setup(_app(tmp_path))

    seen = []
    for _ in DEFAULT_LEVELS_PPM:
        index = at.session_state["cal_level_index"]
        seen.append(at.session_state["cal_session"].levels[index].ppm)
        at = _record_current_level(at, _readings_at(seen[-1]))
        at.button(key=f"cal_next_{index}").click()
        at = at.run()

    assert seen == sorted(DEFAULT_LEVELS_PPM)
    assert at.session_state["cal_step"] == "Review"


def test_a_level_blocks_continuing_before_enough_readings(tmp_path: Path) -> None:
    at = _complete_setup(_app(tmp_path))
    at = _record_current_level(at, [_true_mv(0.0)])

    assert at.warning
    assert not [b for b in at.button if b.key == "cal_next_0"]


def test_a_level_can_be_cleared_and_re_recorded(tmp_path: Path) -> None:
    """A capture taken before the gas settled has to be discardable, or the
    operator's only recourse is restarting the wizard."""
    at = _complete_setup(_app(tmp_path))
    at = _record_current_level(at, _readings_at(0.0))

    at.button(key="cal_clear_0").click()
    at = at.run()

    assert at.session_state["cal_session"].levels[0].readings_mv == []


def test_capture_from_sensor_pulls_a_live_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibration_capture, "fetch_latest_voltage_mv", lambda base_url: 21.7)

    at = _complete_setup(_app(tmp_path))
    at.button(key="cal_capture_0").click()
    at = at.run()

    assert at.session_state["cal_session"].levels[0].readings_mv == [21.7]


def test_a_failed_sensor_read_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(base_url: str) -> float:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(calibration_capture, "fetch_latest_voltage_mv", _boom)

    at = _complete_setup(_app(tmp_path))
    at.button(key="cal_capture_0").click()
    at = at.run()

    assert at.error and "connection refused" in at.error[0].value
    assert at.session_state["cal_session"].levels[0].readings_mv == []


def test_a_curved_response_is_surfaced_on_the_review_step(tmp_path: Path) -> None:
    """The reason this procedure records four levels rather than two: the
    review step has to tell the operator the sensor bends, while they can
    still act on it."""
    at = _complete_setup(_app(tmp_path))

    for _ in DEFAULT_LEVELS_PPM:
        index = at.session_state["cal_level_index"]
        ppm = at.session_state["cal_session"].levels[index].ppm
        curved = TRUE_B0 + TRUE_B1 * ppm - 1.8 * ppm**2
        at = _record_current_level(at, [curved - 0.1, curved, curved + 0.1])
        at.button(key=f"cal_next_{index}").click()
        at = at.run()

    assert any("bends" in w.value for w in at.warning)
    assert at.selectbox(key="cal_method").value == "polynomial_deg2"


def test_walking_the_whole_wizard_recovers_the_true_line(tmp_path: Path) -> None:
    at = _walk_to_saved(tmp_path)

    assert not at.exception
    assert at.session_state["cal_step"] == "Verify"

    model = at.session_state["cal_model"]
    assert model.b0_mv == pytest.approx(TRUE_B0, abs=0.2)
    assert model.b1_mv_per_ppm_asgas == pytest.approx(TRUE_B1, rel=0.01)
    assert Path(at.session_state["cal_saved_path"]).exists()


def test_the_saved_calibration_turns_a_raw_recording_into_correct_ppm(tmp_path: Path) -> None:
    """The point of the whole wizard.

    A fresh recording from the calibrated sensor -- voltages, with noise,
    that the operator never showed the wizard -- must come back out as the
    concentrations that actually produced them.
    """
    at = _walk_to_saved(tmp_path)
    run_id = at.session_state["cal_session"].calibration_run_id

    rng = np.random.default_rng(11)
    true_ppm = np.array([0.0, 0.4, 1.2, 2.5, 3.8, 4.9])
    voltage = TRUE_B0 + TRUE_B1 * true_ppm + rng.normal(0.0, 0.3, size=true_ppm.size)

    recovered, _, _ = apply_calibration(
        pl.Series(voltage),
        "PID01",
        pl.Series([30.0] * true_ppm.size),
        pl.Series([55.0] * true_ppm.size),
        run_id,
        data_dir=tmp_path / "derived" / "calibrations" / "standard_addition",
    )
    recovered = np.asarray(recovered, dtype=float)

    """
    Judged as an absolute error in ppm rather than a relative one: near
    zero a fraction of a millivolt is a large *percentage* of a tiny
    concentration while being physically negligible, so a relative
    tolerance would fail on the reading that matters least.
    """
    assert np.max(np.abs(recovered - true_ppm)) < 0.05


def test_the_monitoring_page_reports_real_ppm_from_this_calibration(tmp_path: Path) -> None:
    """Closes the loop the wizard exists to close: a calibration saved on
    the calibration page must change what the monitoring page displays,
    rather than leaving it on the labelled placeholder forever."""
    at = _walk_to_saved(tmp_path)
    run_id = at.session_state["cal_session"].calibration_run_id

    from algaesense_agent.dashboard import streamlit_app as monitoring

    true_ppm = 3.0
    rows = [
        {
            "sensor_id": "PID01",
            "pid_voltage_mv": _true_mv(true_ppm),
            "sample_t_c": None,
            "sample_rh_pct": None,
        }
    ]

    """
    _data_dir reads an environment variable at call time, so pointing it at
    the directory the wizard just wrote into exercises the real lookup path
    rather than bypassing it.
    """
    previous = os.environ.get("ALGAESENSE_DATA_DIR")
    os.environ["ALGAESENSE_DATA_DIR"] = str(tmp_path)
    try:
        recovered = monitoring._calibrated_voc_ppm(rows, run_id)
    finally:
        if previous is None:
            os.environ.pop("ALGAESENSE_DATA_DIR", None)
        else:
            os.environ["ALGAESENSE_DATA_DIR"] = previous

    assert recovered is not None, "the saved calibration should be found and applied"
    assert recovered[0] == pytest.approx(true_ppm, abs=0.05)

    """
    The placeholder maps this sensor's full 0-3300 mV span onto 0-5 ppm, so
    it would call this same voltage roughly 0.3 ppm. Asserting the two
    differ is what proves the real calibration is being used rather than a
    silent fallback.
    """
    assert abs(recovered[0] - monitoring._voc_ppm_placeholder(_true_mv(true_ppm))) > 1.0


def test_an_unusable_voc_calibration_falls_back_instead_of_crashing() -> None:
    """A mistyped run id should degrade to the labelled placeholder, not
    take down the live view mid-experiment."""
    from algaesense_agent.dashboard import streamlit_app as monitoring

    assert monitoring._calibrated_voc_ppm([{"sensor_id": "PID01", "pid_voltage_mv": 100.0}], "nope") is None


def test_restarting_clears_the_previous_calibration_from_the_page(tmp_path: Path) -> None:
    """Calibrating a second sensor must not inherit the first one's
    readings -- that would silently attribute one sensor's response to
    another."""
    at = _walk_to_saved(tmp_path)

    at.button(key="cal_restart").click()
    at = at.run()

    assert at.session_state["cal_step"] == "Setup"
    assert "cal_session" not in at.session_state
    assert "cal_model" not in at.session_state


def test_the_combined_dashboard_opens_on_monitoring() -> None:
    at = AppTest.from_file(COMBINED_APP_PATH, default_timeout=60).run()

    assert not at.exception
    assert any("AlgaeSense" in t.value for t in at.title)


def test_the_combined_dashboard_navigates_to_calibration() -> None:
    """Monitoring and calibration are two stages of one workflow, so they
    have to be reachable from each other -- not two servers on two ports
    the operator has to remember and start separately."""
    at = AppTest.from_file(COMBINED_APP_PATH, default_timeout=60).run()

    at.switch_page("calibration_app.py")
    at = at.run()

    assert not at.exception
    assert any("Multi-point calibration" in t.value for t in at.title)
    assert at.session_state["cal_step"] == "Setup"
