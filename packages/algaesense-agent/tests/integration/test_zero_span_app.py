"""Tests that drive the real zero/span Streamlit page.

`streamlit.testing.v1.AppTest` executes the actual app script and lets a test
set widget values and click buttons, so these exercise the genuine page --
its step transitions, its guards, the calibration it writes -- rather than a
re-implementation of it.

The last test is the one the wizard exists for: after walking the UI, a raw
recording from that sensor must come back out as correct concentrations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from streamlit.testing.v1 import AppTest

from algaesense_agent.dashboard import zero_span_capture
from jaxsr_calibration.calibration.apply import apply_calibration


_DASHBOARD_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "algaesense_agent" / "dashboard"
)
APP_PATH = str(_DASHBOARD_DIR / "zero_span_app.py")
COMBINED_APP_PATH = str(_DASHBOARD_DIR / "app.py")

TRUE_B0 = 22.0
TRUE_B1 = 0.58
SPAN_PPM = 500.0
N_READINGS = 3

ZERO_MV = [TRUE_B0 - 0.1, TRUE_B0, TRUE_B0 + 0.1]
SPAN_MV = [
    TRUE_B0 + TRUE_B1 * SPAN_PPM - 0.1,
    TRUE_B0 + TRUE_B1 * SPAN_PPM,
    TRUE_B0 + TRUE_B1 * SPAN_PPM + 0.1,
]


def _app(tmp_path: Path) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.session_state["zs_data_dir"] = str(tmp_path)
    return at.run()


def _complete_setup(at: AppTest) -> AppTest:
    at.text_input(key="zs_reactor_id").set_value("R1")
    at.text_input(key="zs_sensor_id").set_value("PID01")
    at.text_input(key="zs_experiment_id").set_value("exp_zerospan")
    at.text_input(key="zs_compound").set_value("isobutylene")
    at.number_input(key="zs_span_ppm").set_value(SPAN_PPM)
    at.number_input(key="zs_mw").set_value(56.11)
    at.number_input(key="zs_rf").set_value(1.0)
    at.button(key="zs_start").click()
    return at.run()


def _enter_readings(at: AppTest, bucket: str, values: list[float]) -> AppTest:
    """Add readings one at a time through the real manual-entry widgets."""
    for value in values:
        at.number_input(key=f"zs_manual_{bucket}").set_value(value)
        at.button(key=f"zs_add_{bucket}").click()
        at = at.run()
    return at


def _walk_to_saved(tmp_path: Path) -> AppTest:
    at = _complete_setup(_app(tmp_path))

    at = _enter_readings(at, "zero_readings_mv", ZERO_MV)
    at.button(key="zs_next_zero_readings_mv").click()
    at = at.run()

    at = _enter_readings(at, "span_readings_mv", SPAN_MV)
    at.button(key="zs_next_span_readings_mv").click()
    at = at.run()

    at.button(key="zs_fit").click()
    at = at.run()
    at.button(key="zs_save").click()
    return at.run()


def test_the_combined_dashboard_opens_on_monitoring() -> None:
    """Calibration happens once at the start of a campaign; monitoring is
    the daily view, so it is the one that should open by default."""
    at = AppTest.from_file(COMBINED_APP_PATH, default_timeout=60).run()

    assert not at.exception
    assert any("AlgaeSense" in t.value for t in at.title)


def test_the_combined_dashboard_navigates_to_calibration() -> None:
    """Monitoring and calibration are two stages of one workflow, so they
    have to be reachable from each other -- not two servers on two ports
    the operator has to remember and start separately."""
    at = AppTest.from_file(COMBINED_APP_PATH, default_timeout=60).run()

    at.switch_page("zero_span_app.py")
    at = at.run()

    assert not at.exception
    assert any("Zero & span calibration" in t.value for t in at.title)
    assert at.session_state["zs_step"] == "Setup"


def test_the_page_starts_on_setup_without_errors() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert at.session_state["zs_step"] == "Setup"


def test_setup_refuses_to_start_with_missing_fields(tmp_path: Path) -> None:
    """A calibration with no sensor id would be unattributable later, so the
    wizard must not advance past it."""
    at = _app(tmp_path)
    at.text_input(key="zs_reactor_id").set_value("R1")
    at.button(key="zs_start").click()
    at = at.run()

    assert at.session_state["zs_step"] == "Setup"
    assert at.error, "the operator should be told what is missing"
    assert "Sensor ID" in at.error[0].value


def test_setup_refuses_a_zero_concentration_span_gas(tmp_path: Path) -> None:
    """Zero ppm span gas is just a second zero -- there would be no slope."""
    at = _app(tmp_path)
    at.text_input(key="zs_reactor_id").set_value("R1")
    at.text_input(key="zs_sensor_id").set_value("PID01")
    at.text_input(key="zs_experiment_id").set_value("exp")
    at.text_input(key="zs_compound").set_value("isobutylene")
    at.number_input(key="zs_rf").set_value(1.0)
    at.button(key="zs_start").click()
    at = at.run()

    assert at.session_state["zs_step"] == "Setup"
    assert "greater than zero" in at.error[0].value


def test_the_zero_step_blocks_continuing_before_enough_readings(tmp_path: Path) -> None:
    at = _complete_setup(_app(tmp_path))
    at = _enter_readings(at, "zero_readings_mv", ZERO_MV[:1])

    assert at.session_state["zs_step"] == "Zero (clean air)"
    assert at.warning, "the operator should be told more readings are needed"
    assert not [b for b in at.button if b.key == "zs_next_zero_readings_mv"]


def test_captured_readings_can_be_cleared_and_re_recorded(tmp_path: Path) -> None:
    """A mistimed capture -- taken before the gas settled -- has to be
    discardable, or the operator's only recourse is restarting the wizard."""
    at = _complete_setup(_app(tmp_path))
    at = _enter_readings(at, "zero_readings_mv", ZERO_MV)

    at.button(key="zs_clear_zero_readings_mv").click()
    at = at.run()

    assert at.session_state["zs_session"].zero_readings_mv == []


def test_capture_from_sensor_pulls_a_live_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live-capture path, with the edge call substituted at the module
    seam the page imports it through."""
    monkeypatch.setattr(zero_span_capture, "fetch_latest_voltage_mv", lambda base_url: 21.7)

    at = _complete_setup(_app(tmp_path))
    at.button(key="zs_capture_zero_readings_mv").click()
    at = at.run()

    assert at.session_state["zs_session"].zero_readings_mv == [21.7]


def test_a_failed_sensor_read_is_reported_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the edge service is down, silently recording nothing would look
    identical to a click that did not register."""

    def _boom(base_url: str) -> float:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(zero_span_capture, "fetch_latest_voltage_mv", _boom)

    at = _complete_setup(_app(tmp_path))
    at.button(key="zs_capture_zero_readings_mv").click()
    at = at.run()

    assert at.error and "connection refused" in at.error[0].value
    assert at.session_state["zs_session"].zero_readings_mv == []


def test_an_unsettled_zero_is_surfaced_on_the_review_step(tmp_path: Path) -> None:
    """The review step has to show the operator a drifting baseline while
    they can still redo it."""
    at = _complete_setup(_app(tmp_path))
    at = _enter_readings(at, "zero_readings_mv", [10.0, 25.0, 40.0])
    at.button(key="zs_next_zero_readings_mv").click()
    at = at.run()
    at = _enter_readings(at, "span_readings_mv", SPAN_MV)
    at.button(key="zs_next_span_readings_mv").click()
    at = at.run()

    assert any("unsettled" in w.value for w in at.warning)


def test_walking_the_whole_wizard_recovers_the_true_line(tmp_path: Path) -> None:
    at = _walk_to_saved(tmp_path)

    assert not at.exception
    assert at.session_state["zs_step"] == "Verify"

    model = at.session_state["zs_model"]
    assert model.b0_mv == pytest.approx(TRUE_B0, abs=0.2)
    assert model.b1_mv_per_ppm_asgas == pytest.approx(TRUE_B1, rel=0.005)

    saved = Path(at.session_state["zs_saved_path"])
    assert saved.exists()


def test_the_saved_calibration_turns_a_raw_recording_into_correct_ppm(tmp_path: Path) -> None:
    """The point of the whole wizard.

    A fresh recording from the calibrated sensor -- voltages, with noise, that
    the operator never showed the wizard -- must come back out as the
    concentrations that actually produced them.
    """
    at = _walk_to_saved(tmp_path)
    run_id = at.session_state["zs_session"].calibration_run_id

    rng = np.random.default_rng(11)
    true_ppm = np.array([0.0, 25.0, 80.0, 150.0, 300.0, 480.0])
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
    Judged as an absolute error in ppm rather than a relative one: at the low
    end a fraction of a millivolt of noise is a large *percentage* of a near
    zero concentration while being physically negligible, so a relative
    tolerance would fail on the reading that matters least.
    """
    assert np.max(np.abs(recovered - true_ppm)) < 1.5


def test_the_monitoring_page_reports_real_ppm_from_the_wizards_calibration(tmp_path: Path) -> None:
    """Closes the loop the wizard exists to close: a calibration saved on
    the calibration page must change what the monitoring page displays,
    rather than leaving it on the labelled placeholder forever."""
    at = _walk_to_saved(tmp_path)
    run_id = at.session_state["zs_session"].calibration_run_id

    from algaesense_agent.dashboard import streamlit_app as monitoring

    true_ppm = 120.0
    voltage = TRUE_B0 + TRUE_B1 * true_ppm
    rows = [{"sensor_id": "PID01", "pid_voltage_mv": voltage, "sample_t_c": None, "sample_rh_pct": None}]

    monkeypatched_data_dir = tmp_path

    """
    _data_dir reads an environment variable at call time, so pointing it at
    the directory the wizard just wrote into is enough to exercise the real
    lookup path rather than bypassing it.
    """
    import os

    previous = os.environ.get("ALGAESENSE_DATA_DIR")
    os.environ["ALGAESENSE_DATA_DIR"] = str(monkeypatched_data_dir)
    try:
        recovered = monitoring._calibrated_voc_ppm(rows, run_id)
    finally:
        if previous is None:
            os.environ.pop("ALGAESENSE_DATA_DIR", None)
        else:
            os.environ["ALGAESENSE_DATA_DIR"] = previous

    assert recovered is not None, "the saved calibration should be found and applied"
    assert recovered[0] == pytest.approx(true_ppm, abs=1.5)

    """
    The placeholder would have reported this same voltage as roughly
    0.5 ppm -- two orders of magnitude out. Asserting they differ is what
    proves the real calibration is genuinely being used.
    """
    assert abs(recovered[0] - monitoring._voc_ppm_placeholder(voltage)) > 100.0


def test_an_unusable_voc_calibration_falls_back_instead_of_crashing() -> None:
    """A mistyped run id should degrade to the labelled placeholder, not
    take down the live view mid-experiment."""
    from algaesense_agent.dashboard import streamlit_app as monitoring

    rows = [{"sensor_id": "PID01", "pid_voltage_mv": 100.0}]

    assert monitoring._calibrated_voc_ppm(rows, "no_such_calibration_run") is None


def test_restarting_clears_the_previous_calibration_from_the_page(tmp_path: Path) -> None:
    """Calibrating a second sensor must not inherit the first one's readings
    -- that would silently attribute one sensor's response to another."""
    at = _walk_to_saved(tmp_path)

    at.button(key="zs_restart").click()
    at = at.run()

    assert at.session_state["zs_step"] == "Setup"
    assert "zs_session" not in at.session_state
    assert "zs_model" not in at.session_state
