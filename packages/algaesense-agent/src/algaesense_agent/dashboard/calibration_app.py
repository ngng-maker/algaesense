"""Guided multi-point gas calibration for one sensor-reactor pair.

One page of the multi-page app whose entry point is `app.py`.
"""

# NOTE: this file is run via `streamlit run`, so it uses plain `#` comments
# throughout rather than this project's usual triple-quoted rationale blocks.
# Streamlit's magic-commands feature renders every bare top-level string as
# page content, including inside a function body. See CLAUDE.md.

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from algaesense_agent.dashboard import calibration_capture
from algaesense_agent.dashboard.calibration_session import (
    DEFAULT_LEVELS_PPM,
    MIN_LEVELS,
    MIN_READINGS_PER_LEVEL,
    SUPPORTED_FIT_METHODS,
    CalibrationSession,
    assess_quality,
    build_session_levels,
    fit_calibration,
    save_calibration,
)
from jaxsr_calibration.calibration.models import CalibrationGas


SETUP, RECORD, REVIEW, VERIFY = "Setup", "Record levels", "Review", "Verify"
CUSTOM_GAS = "Other (enter manually)"
REFERENCE_GAS = "isobutylene"


@st.cache_data(ttl=600)
def _builtin_gases() -> list[dict]:
    # The package's own response-factor table: every value relative to
    # isobutylene = 1.00 on a 10.6 eV lamp. Cached because it is read from
    # a packaged YAML file and never changes within a session.
    return [
        {
            "name": gas.name,
            "mw": gas.mw,
            "response_factor": gas.response_factor,
            "ie_ev": gas.ie_ev,
            "source": gas.source,
        }
        for gas in CalibrationGas.list_builtin()
    ]


def _session() -> CalibrationSession | None:
    return st.session_state.get("cal_session")


def _goto(step: str) -> None:
    # The step is read at the top of the script, so the current run has
    # already drawn the old step by the time a button handler changes it.
    # Without an explicit rerun the page would sit on the previous step
    # until the operator clicked something else.
    st.session_state["cal_step"] = step
    st.session_state["cal_error"] = None
    st.rerun()


def _fail(message: str) -> None:
    st.session_state["cal_error"] = message
    st.rerun()


def _capture_into(level_index: int, base_url: str) -> None:
    try:
        value = calibration_capture.fetch_latest_voltage_mv(base_url)
    except Exception as exc:
        _fail(f"Could not read from the edge service: {exc}")
    st.session_state["cal_error"] = None
    _session().levels[level_index].readings_mv.append(value)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("Multi-point calibration")
st.caption(
    "Calibrate one PID sensor against a series of known concentrations, so its "
    "millivolts become real ppm."
)

st.session_state.setdefault("cal_step", SETUP)
st.session_state.setdefault("cal_error", None)
st.session_state.setdefault("cal_level_index", 0)
st.session_state.setdefault("edge_base_url", os.environ.get("ALGAESENSE_EDGE_BASE_URL", "http://localhost:8000"))
st.session_state.setdefault("cal_data_dir", os.environ.get("ALGAESENSE_DATA_DIR", "data"))
st.session_state.setdefault("cal_levels_text", ", ".join(f"{v:g}" for v in DEFAULT_LEVELS_PPM))


with st.sidebar:
    st.divider()
    st.text_input("Data directory", key="cal_data_dir")

    # In the sidebar rather than on the setup step, so it is drawn on every
    # run. Streamlit discards a widget's session-state entry as soon as the
    # widget stops being rendered, which for a setup-only field happens the
    # moment the wizard advances -- and anything still holding a reference
    # to it then reads a key that no longer exists.
    levels_text = st.text_input("ppm levels, comma separated", key="cal_levels_text")

    st.caption(f"Step: {st.session_state['cal_step']}")

base_url = st.session_state["edge_base_url"]
data_dir = Path(st.session_state["cal_data_dir"])
step = st.session_state["cal_step"]

if st.session_state["cal_error"]:
    st.error(st.session_state["cal_error"])


# --------------------------------------------------------------------------
# Step 1 -- Setup
# --------------------------------------------------------------------------

if step == SETUP:
    st.subheader("1. What are you calibrating?")

    col_a, col_b = st.columns(2)
    with col_a:
        reactor_id = st.text_input("Reactor ID", key="cal_reactor_id")
        sensor_id = st.text_input("Sensor ID", key="cal_sensor_id")
        experiment_id = st.text_input("Experiment ID", key="cal_experiment_id")

    gases = _builtin_gases()
    by_name = {gas["name"]: gas for gas in gases}

    with col_b:
        options = [*by_name, CUSTOM_GAS]
        # Isobutylene first, not whatever sorts first alphabetically: it is
        # the reference every other response factor is defined against, and
        # the usual contents of a PID calibration cylinder.
        choice = st.selectbox(
            "Calibration gas",
            options,
            index=options.index(REFERENCE_GAS) if REFERENCE_GAS in options else 0,
            key="cal_gas_choice",
        )
        if choice == CUSTOM_GAS:
            compound = st.text_input("Compound name", key="cal_custom_compound")
            mw = st.number_input("Molecular weight (g/mol)", min_value=0.0, step=0.01, key="cal_custom_mw")
            rf = st.number_input(
                "Response factor (isobutylene = 1.00)", min_value=0.0, step=0.01, key="cal_custom_rf"
            )
        else:
            gas = by_name[choice]
            compound, mw, rf = gas["name"], gas["mw"], gas["response_factor"]
            st.metric("Response factor", f"{rf:.2f}")
            st.caption(
                f"MW {mw:g} g/mol · ionization energy {gas['ie_ev']:g} eV · {gas['source']}. "
                "Response factors are relative to isobutylene = 1.00 on a 10.6 eV lamp."
            )

    st.write(f"**Concentrations to record:** {levels_text} ppm")
    st.caption(
        f"Set them in the sidebar. At least {MIN_LEVELS} distinct levels, including clean air at "
        "0 ppm. Two points would define a straight line whatever the sensor actually did between "
        "them; the extra levels are what reveal whether the response bends."
    )

    with st.expander("Built-in response-factor table"):
        st.caption(
            "Shipped with the package. Widely published approximations (Alphasense AAN 305, "
            "RAE Systems TN-106) -- real lamps drift with age, so treat them as a sound default "
            "rather than a certified measurement."
        )
        st.dataframe(
            pd.DataFrame(gases).rename(
                columns={
                    "name": "Compound",
                    "mw": "MW (g/mol)",
                    "response_factor": "RF (isobutylene=1.00)",
                    "ie_ev": "IE (eV)",
                    "source": "Source",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    if st.button("Start calibration", type="primary", key="cal_start"):
        raw_levels = levels_text.strip() or ", ".join(str(v) for v in DEFAULT_LEVELS_PPM)
        try:
            levels_ppm = [float(part) for part in raw_levels.split(",") if part.strip()]
        except ValueError:
            levels_ppm = []

        missing = [
            name
            for name, value in [
                ("Reactor ID", reactor_id),
                ("Sensor ID", sensor_id),
                ("Experiment ID", experiment_id),
                ("Compound name", compound),
            ]
            if not value
        ]
        if missing:
            _fail(f"Please fill in: {', '.join(missing)}.")
        elif not levels_ppm:
            _fail("Could not read the ppm levels -- enter them as numbers separated by commas.")
        elif len(set(levels_ppm)) < MIN_LEVELS:
            _fail(
                f"At least {MIN_LEVELS} distinct concentrations are needed "
                f"(got {len(set(levels_ppm))}). Fewer cannot show whether the response bends."
            )
        elif 0.0 not in levels_ppm:
            _fail("Include 0 ppm (clean air) -- it is what fixes the sensor's own baseline offset.")
        elif not rf or rf <= 0:
            _fail("Response factor must be greater than zero.")
        else:
            st.session_state["cal_session"] = CalibrationSession(
                reactor_id=reactor_id,
                sensor_id=sensor_id,
                calibration_run_id=f"cal_{sensor_id}_{dt.datetime.now():%Y-%m-%dT%H-%M-%S}",
                experiment_id=experiment_id,
                calibration_compound=compound,
                mw_g_mol=float(mw),
                response_factor=float(rf),
                levels=build_session_levels(levels_ppm),
            )
            st.session_state["cal_level_index"] = 0
            _goto(RECORD)


# --------------------------------------------------------------------------
# Step 2 -- Record each level in turn
# --------------------------------------------------------------------------

elif step == RECORD and _session() is not None:
    session = _session()
    index = min(st.session_state["cal_level_index"], len(session.levels) - 1)
    level = session.levels[index]

    st.subheader(f"2. Record {level.ppm:g} ppm  ({index + 1} of {len(session.levels)})")

    if level.ppm == 0:
        st.write(
            "Flow zero-grade clean air over the sensor. Wait for the reading to stop drifting, "
            f"then capture at least {MIN_READINGS_PER_LEVEL} readings."
        )
    else:
        st.write(
            f"Flow the {level.ppm:g} ppm {session.calibration_compound} standard. Wait for the "
            f"reading to plateau, then capture at least {MIN_READINGS_PER_LEVEL} readings."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Capture from sensor", key=f"cal_capture_{index}"):
            _capture_into(index, base_url)
    with col_b:
        manual = st.number_input("Or enter a reading manually (mV)", step=0.1, key=f"cal_manual_{index}")
        if st.button("Add manual reading", key=f"cal_add_{index}"):
            level.readings_mv.append(float(manual))

    if level.readings_mv:
        st.dataframe(
            pd.DataFrame({"#": range(1, len(level.readings_mv) + 1), "mV": level.readings_mv}),
            hide_index=True,
            use_container_width=True,
        )
        if st.button("Clear this level's readings", key=f"cal_clear_{index}"):
            level.readings_mv.clear()

    if level.is_complete:
        is_last = index == len(session.levels) - 1
        label = "Review calibration" if is_last else "Next level"
        if st.button(label, type="primary", key=f"cal_next_{index}"):
            if is_last:
                _goto(REVIEW)
            else:
                st.session_state["cal_level_index"] = index + 1
                st.rerun()
    else:
        st.warning(f"At least {MIN_READINGS_PER_LEVEL} readings are needed before continuing.")

    if index > 0 and st.button("Back to previous level", key=f"cal_back_{index}"):
        st.session_state["cal_level_index"] = index - 1
        st.rerun()


# --------------------------------------------------------------------------
# Step 3 -- Review and save
# --------------------------------------------------------------------------

elif step == REVIEW and _session() is not None:
    session = _session()
    st.subheader("3. Review")

    quality = assess_quality(session)

    st.dataframe(
        pd.DataFrame(
            {
                "ppm": [lq.ppm for lq in quality.levels],
                "mean mV": [round(lq.mean_mv, 2) for lq in quality.levels],
                "scatter mV": [round(lq.std_mv, 2) for lq in quality.levels],
                "readings": [lq.n for lq in quality.levels],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.line_chart(
        pd.DataFrame(
            {"ppm": [lq.ppm for lq in quality.levels], "mV": [lq.mean_mv for lq in quality.levels]}
        ),
        x="ppm",
        y="mV",
    )

    col_a, col_b = st.columns(2)
    col_a.metric("Linear fit R²", f"{quality.r_squared:.4f}")
    col_b.metric("Curvature share", f"{100 * quality.curvature_share:.0f}%")
    st.caption(
        "R² is meaningful here because there are more levels than a straight line has "
        "coefficients -- unlike a two-point calibration, where it sits near 1.0 whatever the "
        "sensor did. Curvature share is how much of what the line misses is a real bend "
        "rather than scatter."
    )

    for warning in quality.warnings:
        st.warning(warning)
    if not quality.warnings:
        st.success(
            "Every level looks settled, the series rises monotonically, and a straight line "
            "describes it well."
        )

    default_method = "polynomial_deg2" if quality.suggests_quadratic else SUPPORTED_FIT_METHODS[0]
    method = st.selectbox(
        "Fit method",
        SUPPORTED_FIT_METHODS,
        index=SUPPORTED_FIT_METHODS.index(default_method),
        key="cal_method",
    )

    if st.button("Fit calibration", type="primary", key="cal_fit"):
        try:
            st.session_state["cal_model"] = fit_calibration(session, method=method)
            st.session_state["cal_error"] = None
        except Exception as exc:
            _fail(f"Fit failed: {exc}")

    model = st.session_state.get("cal_model")
    if model is not None:
        st.write("**Fitted curve**")
        if model.b2_mv_per_ppm2_asgas:
            st.latex(
                rf"\text{{mV}} = {model.b0_mv:.3f} + {model.b1_mv_per_ppm_asgas:.4f}\,\text{{ppm}} "
                rf"{model.b2_mv_per_ppm2_asgas:+.6f}\,\text{{ppm}}^2"
            )
        else:
            st.latex(rf"\text{{mV}} = {model.b0_mv:.3f} + {model.b1_mv_per_ppm_asgas:.4f}\,\text{{ppm}}")
        st.write(
            f"Slope uncertainty ± {model.b1_stderr:.4f} mV/ppm "
            f"({100 * model.b1_stderr / abs(model.b1_mv_per_ppm_asgas):.2f}% of the slope) · "
            f"fit R² {model.r_squared:.4f}"
        )

        if st.button("Save calibration", key="cal_save"):
            saved = None
            try:
                saved = save_calibration(session, model, data_dir)
            except Exception as exc:
                _fail(f"Save failed: {exc}")

            # Deliberately outside the try: _goto raises Streamlit's
            # RerunException, which a broad `except Exception` here would
            # swallow and mis-report as a failed save.
            if saved is not None:
                st.session_state["cal_saved_path"] = str(saved)
                _goto(VERIFY)


# --------------------------------------------------------------------------
# Step 4 -- Verify the calibration produces real concentrations
# --------------------------------------------------------------------------

elif step == VERIFY and _session() is not None:
    session = _session()
    st.subheader("4. Verify")
    st.success(f"Saved as `{session.calibration_run_id}`")
    st.caption(st.session_state.get("cal_saved_path", ""))
    st.info(
        "Paste that id into the Monitoring page's **VOC calibration_run_id** field to switch the "
        "live chart from its placeholder conversion to these real ppm values."
    )

    st.write(
        "Below, live readings from this sensor are converted through the calibration you just "
        "saved. Clean air should read near zero ppm."
    )

    if st.button("Read live VOC in ppm", type="primary", key="cal_verify"):
        try:
            st.session_state["cal_verify_ppm"] = calibration_capture.fetch_recent_ppm(
                base_url, session.sensor_id, session.calibration_run_id, data_dir
            )
            st.session_state["cal_error"] = None
        except Exception as exc:
            _fail(f"Verification read failed: {exc}")

    ppm_values = st.session_state.get("cal_verify_ppm")
    if ppm_values:
        st.metric("Latest", f"{ppm_values[-1]:.3f} ppm")
        st.line_chart(pd.DataFrame({"ppm": ppm_values}))

    if st.button("Calibrate another sensor", key="cal_restart"):
        for key in ("cal_session", "cal_model", "cal_saved_path", "cal_verify_ppm"):
            st.session_state.pop(key, None)
        st.session_state["cal_level_index"] = 0
        _goto(SETUP)


else:
    st.warning("No calibration in progress.")
    if st.button("Start over", key="cal_reset"):
        _goto(SETUP)
