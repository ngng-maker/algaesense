"""Guided zero-and-span calibration for one sensor-reactor pair.

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

from algaesense_agent.dashboard import zero_span_capture
from algaesense_agent.dashboard.zero_span_session import (
    MIN_READINGS_PER_LEVEL,
    SUPPORTED_FIT_METHODS,
    ZeroSpanSession,
    assess_quality,
    fit_zero_span,
    save_zero_span,
)


STEPS = ["Setup", "Zero (clean air)", "Span (known gas)", "Review", "Verify"]


def _session() -> ZeroSpanSession | None:
    return st.session_state.get("zs_session")


def _goto(step: str) -> None:
    # The step is read at the top of the script, so the current run has
    # already drawn the old step by the time a button handler changes it.
    # Without an explicit rerun the page would sit on the previous step
    # until the operator clicked something else.
    st.session_state["zs_step"] = step
    st.session_state["zs_error"] = None
    st.rerun()


def _fail(message: str) -> None:
    # Same reason as _goto: the error banner is rendered near the top, above
    # every button that could set one.
    st.session_state["zs_error"] = message
    st.rerun()


def _capture_into(bucket: str, base_url: str) -> None:
    try:
        value = zero_span_capture.fetch_latest_voltage_mv(base_url)
    except Exception as exc:
        _fail(f"Could not read from the edge service: {exc}")
    st.session_state["zs_error"] = None
    getattr(_session(), bucket).append(value)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

# No st.set_page_config here: app.py owns it for the whole multi-page app,
# and Streamlit only honours the first call.

st.title("Zero & span calibration")
st.caption(
    "Two-point calibration of one PID sensor on one reactor: record clean air, "
    "record a certified span gas, fit the line, verify it."
)

st.session_state.setdefault("zs_step", STEPS[0])
st.session_state.setdefault("zs_error", None)
st.session_state.setdefault("edge_base_url", os.environ.get("ALGAESENSE_EDGE_BASE_URL", "http://localhost:8000"))
st.session_state.setdefault("zs_data_dir", os.environ.get("ALGAESENSE_DATA_DIR", "data"))

with st.sidebar:
    st.header("Connection")
    st.text_input("algaesense-edge URL", key="edge_base_url")
    st.text_input("Data directory", key="zs_data_dir")
    st.divider()
    st.header("Progress")
    for step in STEPS:
        marker = "➡️" if step == st.session_state["zs_step"] else "•"
        st.write(f"{marker} {step}")

base_url = st.session_state["edge_base_url"]
data_dir = Path(st.session_state["zs_data_dir"])
step = st.session_state["zs_step"]

if st.session_state["zs_error"]:
    st.error(st.session_state["zs_error"])


# --------------------------------------------------------------------------
# Step 1 -- Setup
# --------------------------------------------------------------------------

if step == STEPS[0]:
    st.subheader("1. What are you calibrating?")

    col_a, col_b = st.columns(2)
    with col_a:
        reactor_id = st.text_input("Reactor ID", key="zs_reactor_id")
        sensor_id = st.text_input("Sensor ID", key="zs_sensor_id")
        experiment_id = st.text_input("Experiment ID", key="zs_experiment_id")
    with col_b:
        compound = st.text_input("Span gas compound", key="zs_compound")
        span_ppm = st.number_input("Span gas concentration (ppm)", min_value=0.0, step=1.0, key="zs_span_ppm")
        mw = st.number_input("Molecular weight (g/mol)", min_value=0.0, step=0.01, key="zs_mw")
        rf = st.number_input("Response factor", min_value=0.0, step=0.01, key="zs_rf")

    st.info(
        "A two-point calibration assumes the sensor's response between and beyond "
        "these two points is a straight line. It cannot detect curvature -- if this "
        "sensor is known to compress at high concentration, run a multi-point "
        "standard addition instead."
    )

    if st.button("Start calibration", type="primary", key="zs_start"):
        missing = [
            name
            for name, value in [
                ("Reactor ID", reactor_id),
                ("Sensor ID", sensor_id),
                ("Experiment ID", experiment_id),
                ("Span gas compound", compound),
            ]
            if not value
        ]
        if missing:
            _fail(f"Please fill in: {', '.join(missing)}.")
        elif span_ppm <= 0:
            _fail("Span gas concentration must be greater than zero.")
        elif rf <= 0:
            _fail("Response factor must be greater than zero.")
        else:
            st.session_state["zs_session"] = ZeroSpanSession(
                reactor_id=reactor_id,
                sensor_id=sensor_id,
                calibration_run_id=f"zerospan_{sensor_id}_{dt.datetime.now():%Y-%m-%dT%H-%M-%S}",
                experiment_id=experiment_id,
                calibration_compound=compound,
                mw_g_mol=float(mw),
                response_factor=float(rf),
                span_ppm=float(span_ppm),
            )
            _goto(STEPS[1])


# --------------------------------------------------------------------------
# Steps 2 and 3 -- record each level
# --------------------------------------------------------------------------

elif step in (STEPS[1], STEPS[2]) and _session() is not None:
    session = _session()
    is_zero = step == STEPS[1]
    bucket = "zero_readings_mv" if is_zero else "span_readings_mv"
    readings = getattr(session, bucket)

    if is_zero:
        st.subheader("2. Record the zero")
        st.write(
            "Flow zero-grade clean air over the sensor. Wait for the reading to stop "
            "drifting, then capture at least "
            f"{MIN_READINGS_PER_LEVEL} readings."
        )
    else:
        st.subheader("3. Record the span")
        st.write(
            f"Flow the {session.span_ppm:g} ppm {session.calibration_compound} span gas. "
            "Wait for the reading to plateau, then capture at least "
            f"{MIN_READINGS_PER_LEVEL} readings."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Capture from sensor", key=f"zs_capture_{bucket}"):
            _capture_into(bucket, base_url)
    with col_b:
        manual = st.number_input("Or enter a reading manually (mV)", step=0.1, key=f"zs_manual_{bucket}")
        if st.button("Add manual reading", key=f"zs_add_{bucket}"):
            readings.append(float(manual))

    if readings:
        st.write(f"**{len(readings)} reading(s) captured**")
        st.dataframe(
            pd.DataFrame({"#": range(1, len(readings) + 1), "mV": readings}),
            hide_index=True,
            use_container_width=True,
        )
        if st.button("Clear these readings", key=f"zs_clear_{bucket}"):
            readings.clear()

    enough = session.has_enough_zero if is_zero else session.has_enough_span
    if enough:
        if st.button("Continue", type="primary", key=f"zs_next_{bucket}"):
            _goto(STEPS[2] if is_zero else STEPS[3])
    else:
        st.warning(f"At least {MIN_READINGS_PER_LEVEL} readings are needed before continuing.")


# --------------------------------------------------------------------------
# Step 4 -- Review and save
# --------------------------------------------------------------------------

elif step == STEPS[3] and _session() is not None:
    session = _session()
    st.subheader("4. Review")

    quality = assess_quality(session)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Zero", f"{quality.zero_mean_mv:.2f} mV", f"± {quality.zero_std_mv:.2f} scatter")
    col_b.metric("Span", f"{quality.span_mean_mv:.2f} mV", f"± {quality.span_std_mv:.2f} scatter")
    col_c.metric("Separation", f"{quality.separation_mv:.2f} mV")

    for warning in quality.warnings:
        st.warning(warning)
    if not quality.warnings:
        st.success("Both levels look settled and well separated.")

    st.caption(
        "R² is deliberately not shown as a pass criterion here: with only two "
        "concentrations it is near 1.0 by construction and says almost nothing about "
        "sensor quality. The scatter figures above are what actually matter."
    )

    method = st.selectbox("Fit method", SUPPORTED_FIT_METHODS, key="zs_method")

    if st.button("Fit calibration", type="primary", key="zs_fit"):
        try:
            st.session_state["zs_model"] = fit_zero_span(session, method=method)
            st.session_state["zs_error"] = None
        except Exception as exc:
            _fail(f"Fit failed: {exc}")

    model = st.session_state.get("zs_model")
    if model is not None:
        st.write("**Fitted line**")
        st.latex(
            rf"\text{{mV}} = {model.b0_mv:.3f} + {model.b1_mv_per_ppm_asgas:.5f} \times \text{{ppm}}"
        )
        st.write(
            f"Slope uncertainty ± {model.b1_stderr:.5f} mV/ppm "
            f"({100 * model.b1_stderr / abs(model.b1_mv_per_ppm_asgas):.2f}% of the slope)."
        )

        if st.button("Save calibration", key="zs_save"):
            saved = None
            try:
                saved = save_zero_span(session, model, data_dir)
            except Exception as exc:
                _fail(f"Save failed: {exc}")

            # Deliberately outside the try: _goto raises Streamlit's
            # RerunException, which a broad `except Exception` here would
            # swallow and mis-report as a failed save.
            if saved is not None:
                st.session_state["zs_saved_path"] = str(saved)
                _goto(STEPS[4])


# --------------------------------------------------------------------------
# Step 5 -- Verify the calibration produces real concentrations
# --------------------------------------------------------------------------

elif step == STEPS[4] and _session() is not None:
    session = _session()
    st.subheader("5. Verify")
    st.success(f"Saved as `{session.calibration_run_id}`")
    st.caption(st.session_state.get("zs_saved_path", ""))

    st.write(
        "Below, live readings from this sensor are converted through the calibration "
        "you just saved. Clean air should read near zero ppm."
    )

    if st.button("Read live VOC in ppm", type="primary", key="zs_verify"):
        try:
            st.session_state["zs_verify_ppm"] = zero_span_capture.fetch_recent_ppm(
                base_url, session.sensor_id, session.calibration_run_id, data_dir
            )
            st.session_state["zs_error"] = None
        except Exception as exc:
            st.session_state["zs_error"] = f"Verification read failed: {exc}"

    ppm_values = st.session_state.get("zs_verify_ppm")
    if ppm_values:
        st.metric("Latest", f"{ppm_values[-1]:.3f} ppm")
        st.line_chart(pd.DataFrame({"ppm": ppm_values}))

    if st.button("Calibrate another sensor", key="zs_restart"):
        for key in ("zs_session", "zs_model", "zs_saved_path", "zs_verify_ppm"):
            st.session_state.pop(key, None)
        _goto(STEPS[0])


else:
    st.warning("No calibration in progress.")
    if st.button("Start over", key="zs_reset"):
        _goto(STEPS[0])
