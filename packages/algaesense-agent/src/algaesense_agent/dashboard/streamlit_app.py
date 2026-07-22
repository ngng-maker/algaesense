"""Live operator dashboard: VOC + camera biomass readings streamed from a
real algaesense-edge instance, plus a Slack panel to send commands to the
AI agent -- run with `streamlit run streamlit_app.py`, not imported as a
library module.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st
from jaxsr_calibration.camera.calibration import BiomassCameraModel, apply_biomass_calibration, load_biomass_calibration

# NOTE ON COMMENT STYLE IN THIS FILE: Streamlit's "magic commands" feature
# renders any bare top-level string expression as page content (confirmed
# by testing) -- including ones inside a function body once that function
# runs, not just at module level. Only a genuine first-statement docstring
# is exempt. That makes this project's usual "separate triple-quoted
# technical-detail block" convention actively wrong here: every such block
# would show up as literal text on the page. This file uses plain `#`
# comments for that reason, everywhere else in the codebase still uses
# the triple-quoted convention.

from algaesense_agent.dashboard.history_db import (
    list_experiments,
    load_experiment_camera_readings,
    load_experiment_voc_readings,
)

# Two independent, auto-refreshing sections (st.fragment(run_every=...)),
# so the live readings keep updating without re-running the whole page
# (and losing whatever's typed into the Slack message box). Both sections
# degrade gracefully rather than crashing: the readings panel reports a
# clear connection error if algaesense-edge isn't reachable yet, and the
# Slack panel explains what env vars are still needed if Slack isn't wired
# up yet -- neither blocks the other from working.

st.set_page_config(page_title="AlgaeSense", layout="wide")


def _edge_base_url() -> str:
    # Which algaesense-edge instance to poll -- overridable per-session
    # from the sidebar, so switching reactors doesn't need an app restart.
    default = os.environ.get("ALGAESENSE_EDGE_BASE_URL", "http://localhost:8000")
    return st.session_state.get("edge_base_url", default)


def _history_db_path() -> Path:
    # Where the past-experiments SQLite archive lives -- populated by the
    # separate `algaesense-dashboard-sync` CLI (see history_db.py), not by
    # this app itself. Defaults alongside wherever ALGAESENSE_DATA_DIR
    # points, if set.
    default_dir = os.environ.get("ALGAESENSE_DATA_DIR", ".")
    default = str(Path(default_dir) / "dashboard_history.db")
    return Path(os.environ.get("ALGAESENSE_HISTORY_DB_PATH", default))


def _greenness(rgb: list[float]) -> float:
    # Excess Green Index (2*G - R - B) -- same formula as
    # jaxsr_calibration.camera.calibration.greenness_index, duplicated here
    # (not imported) since this is a display-only computation with no need
    # to depend on the calibration package just to plot a number.
    r, g, b = rgb[0], rgb[1], rgb[2]
    return 2 * g - r - b


# VOC_SENSOR_FULL_SCALE_MV/VOC_SENSOR_FULL_SCALE_PPM: a ROUGH PLACEHOLDER
# linear conversion (raw ISB output spans 0-3.3V per algaesense-edge's
# acquisition/voc.py, mapped straight across the sensor's stated 0-5 ppm
# range), used ONLY until a real standard-addition calibration exists for
# this sensor -- same spirit as this project's other documented
# approximations (see CLAUDE.md's PAR-calibration gotcha). Once a real
# calibration_run_id is entered in the sidebar, that's used instead and
# this placeholder is skipped entirely.
_VOC_SENSOR_FULL_SCALE_MV = 3300.0
_VOC_SENSOR_FULL_SCALE_PPM = 5.0


def _voc_ppm_placeholder(voltage_mv: float) -> float:
    return (voltage_mv / _VOC_SENSOR_FULL_SCALE_MV) * _VOC_SENSOR_FULL_SCALE_PPM


def _data_dir() -> Path:
    return Path(os.environ.get("ALGAESENSE_DATA_DIR", "."))


def _load_camera_calibration(calibration_run_id: str) -> BiomassCameraModel | None:
    # Returns None (rather than raising) on any failure -- a missing or
    # unreadable calibration run shouldn't crash the whole dashboard, just
    # fall back to showing raw greenness with no blank-correction.
    try:
        return load_biomass_calibration(calibration_run_id, data_dir=_data_dir() / "derived" / "calibrations" / "camera_zero")
    except Exception:
        return None


def _parse_timestamp(value: str) -> dt.datetime:
    # Every raw timestamp this app sees (live API or the history db) is an
    # ISO-8601 UTC string -- handles both a trailing "Z" and an explicit
    # "+00:00" offset, since either can show up depending on the source.
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _elapsed_seconds_since_start(rows: list[dict]) -> list[float]:
    if not rows:
        return []
    start = _parse_timestamp(rows[0]["timestamp"])
    return [(_parse_timestamp(row["timestamp"]) - start).total_seconds() for row in rows]


def _elapsed_hours_since_start(rows: list[dict]) -> list[float]:
    return [seconds / 3600.0 for seconds in _elapsed_seconds_since_start(rows)]


def _render_experiment_header(voc_rows: list[dict], camera_rows: list[dict]) -> None:
    # Pulls experiment_id/reactor_id/sensor_id/camera_id straight off the
    # first available row -- both the live API and the history db return
    # rows carrying these fields, so this works identically for either
    # source without needing a separate metadata lookup.
    first_row = (voc_rows or camera_rows or [None])[0]
    if first_row is None:
        return

    experiment_id = first_row.get("experiment_id", "?")
    reactor_id = first_row.get("reactor_id", "?")
    sensor_id = voc_rows[0].get("sensor_id", "?") if voc_rows else "?"
    camera_id = camera_rows[0].get("camera_id", "?") if camera_rows else "?"

    st.subheader(f"Experiment: {experiment_id}")
    cols = st.columns(4)
    cols[0].metric("Reactor", reactor_id)
    cols[1].metric("VOC sensor", sensor_id)
    cols[2].metric("Camera", camera_id)
    if voc_rows:
        started_at = voc_rows[0]["timestamp"][:19].replace("T", " ")
        cols[3].metric("Started (UTC)", started_at)


def _render_readings(voc_rows: list[dict], camera_rows: list[dict]) -> None:
    _render_experiment_header(voc_rows, camera_rows)

    voc_col, camera_col = st.columns(2)

    with voc_col:
        st.subheader("VOC (PID sensor)")
        if voc_rows:
            # A real calibration_run_id (sidebar) would give a properly
            # fitted ppm conversion (jaxsr_calibration.calibration.apply.
            # apply_calibration) -- until one exists for this sensor, ppm
            # values here use the rough placeholder linear scaling above,
            # which is clearly labeled as such rather than presented as
            # a real measurement.
            ppm_values = [_voc_ppm_placeholder(row["pid_voltage_mv"]) for row in voc_rows]
            st.metric("Latest reading (ppm, approx.)", f"{ppm_values[-1]:.2f}")
            st.caption(
                "Approximate placeholder conversion (linear across the sensor's stated "
                "0-5 ppm range) -- run a standard-addition calibration via Slack for a real one."
            )
            # x-axis: seconds since this experiment's first VOC reading,
            # not a raw timestamp -- much easier to read at the ~1 Hz
            # sampling rate this sensor actually runs at. Fixed y-axis
            # domain [0, 5] (the sensor's stated range) needs an Altair
            # chart -- st.line_chart auto-scales its axis and has no
            # option to pin a fixed range.
            df = pd.DataFrame(
                {"seconds_since_start": _elapsed_seconds_since_start(voc_rows), "ppm": ppm_values}
            )
            chart = (
                alt.Chart(df)
                .mark_line()
                .encode(
                    x=alt.X("seconds_since_start", title="Seconds since start"),
                    y=alt.Y("ppm", title="ppm (approx.)", scale=alt.Scale(domain=[0, _VOC_SENSOR_FULL_SCALE_PPM])),
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No VOC readings yet.")

    with camera_col:
        st.subheader("Camera (biomass)")
        # Only rows that actually carry a feature vector can be scored --
        # a capture that failed partway through might not.
        scored_rows = [row for row in camera_rows if row.get("image_feature_vector")]
        if scored_rows:
            camera_calibration_run_id = st.session_state.get("camera_calibration_run_id", "").strip()
            calibration_model = _load_camera_calibration(camera_calibration_run_id) if camera_calibration_run_id else None

            greenness_values = [_greenness(row["image_feature_vector"]) for row in scored_rows]
            metric_cols = st.columns(2)
            metric_cols[0].metric("Latest greenness (2G-R-B)", f"{greenness_values[-1]:.1f}")

            df_data = {"hours_since_start": _elapsed_hours_since_start(scored_rows), "raw greenness": greenness_values}
            if calibration_model is not None:
                # apply_biomass_calibration returns a signal RELATIVE to
                # the blank baseline (positive = greener/more biomass than
                # clean medium), NOT an absolute concentration -- this
                # project has no calibration mapping greenness to an
                # absolute unit like g/L (that would need a second
                # reference measurement, e.g. dry weight or OD, which
                # doesn't exist here). See camera/calibration.py's own
                # module docstring.
                biomass_signal = [
                    apply_biomass_calibration(row["image_feature_vector"], calibration_model) for row in scored_rows
                ]
                metric_cols[1].metric("Latest biomass signal (blank-corrected)", f"{biomass_signal[-1]:.1f}")
                df_data["biomass signal (blank-corrected)"] = biomass_signal
            elif camera_calibration_run_id:
                st.caption(f"Could not load camera calibration {camera_calibration_run_id!r} -- showing raw greenness only.")
            else:
                st.caption("Enter a camera calibration_run_id in the sidebar for a blank-corrected biomass signal.")

            # x-axis: hours since this experiment's first camera capture --
            # the camera samples far less often (~hourly) than the VOC
            # sensor, so hours read more naturally here than seconds.
            df = pd.DataFrame(df_data).set_index("hours_since_start")
            st.line_chart(df)
        elif camera_rows:
            st.info("Camera readings present but missing feature vectors.")
        else:
            st.info("No camera readings yet.")


@st.fragment(run_every=2)
def _live_readings() -> None:
    base_url = _edge_base_url()

    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            voc_rows = client.get("/sensors/voc/recent", params={"limit": 300}).json()
            camera_rows = client.get("/sensors/camera/recent", params={"limit": 100}).json()
    except httpx.HTTPError as exc:
        st.error(f"Could not reach algaesense-edge at {base_url}: {exc}")
        return

    _render_readings(voc_rows, camera_rows)


def _past_experiment_readings() -> None:
    db_path = _history_db_path()
    if not db_path.exists():
        st.info(
            f"No history database found at {db_path}. Run `algaesense-dashboard-sync "
            "--data-dir ... --db-path ...` after copying an experiment's raw data onto "
            "this machine, then reload this page."
        )
        return

    experiments = list_experiments(db_path)
    if not experiments:
        st.info(f"{db_path} exists but has no ingested experiments yet.")
        return

    labels = [f"{e['experiment_id']} ({e['voc_row_count']} VOC rows)" for e in experiments]
    selected = st.selectbox("Past experiment", labels, key="past_experiment_selector")
    experiment_id = experiments[labels.index(selected)]["experiment_id"]

    voc_rows = load_experiment_voc_readings(db_path, experiment_id)
    camera_rows = load_experiment_camera_readings(db_path, experiment_id)
    _render_readings(voc_rows, camera_rows)


def _slack_client():
    token = os.environ.get("SLACK_BOT_TOKEN")
    if not token:
        return None

    from slack_sdk import WebClient

    return WebClient(token=token)


@st.fragment(run_every=3)
def _slack_panel() -> None:
    st.subheader("Chat with the AI agent (Slack)")

    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    client = _slack_client()

    if client is None or not channel_id:
        st.info(
            "Slack isn't configured yet. Set the SLACK_BOT_TOKEN and "
            "SLACK_CHANNEL_ID environment variables before starting this "
            "app -- see profile/README.md for how to create the Slack app "
            "and get both values."
        )
        return

    # conversations_history is a real Slack Web API call, not a stand-in --
    # this reads the actual channel Hermes is also listening on, so
    # messages sent here (and the agent's replies) show up the same as if
    # typed directly in Slack.
    try:
        history = client.conversations_history(channel=channel_id, limit=20)
        for message in reversed(history["messages"]):
            sender = message.get("user", message.get("bot_id", "unknown"))
            st.markdown(f"**{sender}**: {message.get('text', '')}")
    except Exception as exc:
        st.error(f"Failed to load Slack history: {exc}")
        return

    message_text = st.text_input("Message to the agent", key="slack_message_input")
    if st.button("Send") and message_text:
        try:
            client.chat_postMessage(channel=channel_id, text=message_text)
            st.success("Sent.")
        except Exception as exc:
            st.error(f"Failed to send: {exc}")


st.title("AlgaeSense")

# Passing both value= and key= to a widget is a known Streamlit
# footgun -- Streamlit's own docs warn the value= can silently
# re-overwrite whatever the user just typed on a later rerun. Setting
# the session-state default ONCE, before creating the widget, then
# passing only key= (no value=) is the documented-safe way to give a
# widget a default without that risk.
if "edge_base_url" not in st.session_state:
    st.session_state["edge_base_url"] = os.environ.get("ALGAESENSE_EDGE_BASE_URL", "http://localhost:8000")
if "camera_calibration_run_id" not in st.session_state:
    st.session_state["camera_calibration_run_id"] = os.environ.get("ALGAESENSE_CAMERA_CALIBRATION_RUN_ID", "")

with st.sidebar:
    st.text_input(
        "algaesense-edge URL",
        key="edge_base_url",
        help="The reactor's Raspberry Pi network API address, e.g. http://192.168.1.42:8000",
    )
    st.text_input(
        "Camera calibration_run_id (optional)",
        key="camera_calibration_run_id",
        help="From a completed camera zero-point calibration (Slack) -- shows a blank-corrected "
        "biomass signal alongside raw greenness when set.",
    )
    st.divider()
    view = st.radio("View", ["Live", "Past experiment"], key="dashboard_view")

if view == "Live":
    _live_readings()
else:
    _past_experiment_readings()

_slack_panel()
