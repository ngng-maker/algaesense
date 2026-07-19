"""Unit tests for algaesense_agent.mcp_calibration.sessions: the shared,
file-backed session storage every calibration wizard builds on.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from algaesense_agent.mcp_calibration.sessions import (
    SessionAlreadyFinishedError,
    SessionNotFoundError,
    append_step,
    create_session,
    load_session,
    mark_finished,
)


def test_create_session_writes_a_file_and_returns_a_readable_id(tmp_path: Path) -> None:
    now = dt.datetime(2026, 7, 17, 9, 0, 0, tzinfo=dt.timezone.utc)

    session = create_session(tmp_path, kind="standard_addition", experiment_id="exp_01", context={"a": 1}, now=now)

    assert session.session_id == "cal_standard_addition_exp_01_2026-07-17T09-00-00"
    assert (tmp_path / f"{session.session_id}.yaml").exists()
    assert session.status == "in_progress"
    assert session.steps == []


def test_load_session_round_trips(tmp_path: Path) -> None:
    created = create_session(tmp_path, kind="camera_zero", experiment_id="exp_01", context={"camera_id": "CAM01"})

    loaded = load_session(tmp_path, created.session_id)

    assert loaded == created


def test_load_session_raises_for_unknown_id(tmp_path: Path) -> None:
    with pytest.raises(SessionNotFoundError):
        load_session(tmp_path, "cal_does_not_exist")


def test_append_step_persists_across_reloads(tmp_path: Path) -> None:
    session = create_session(tmp_path, kind="reference_jar", experiment_id="exp_01", context={"sensors": ["PID01"]})

    append_step(tmp_path, session.session_id, {"sensor_id": "PID01", "pid_voltage_mv": 1.0})
    reloaded = load_session(tmp_path, session.session_id)

    assert reloaded.steps == [{"sensor_id": "PID01", "pid_voltage_mv": 1.0}]


def test_append_step_raises_once_finished(tmp_path: Path) -> None:
    session = create_session(tmp_path, kind="reference_jar", experiment_id="exp_01", context={"sensors": ["PID01"]})
    mark_finished(tmp_path, session.session_id)

    with pytest.raises(SessionAlreadyFinishedError):
        append_step(tmp_path, session.session_id, {"sensor_id": "PID01", "pid_voltage_mv": 1.0})
