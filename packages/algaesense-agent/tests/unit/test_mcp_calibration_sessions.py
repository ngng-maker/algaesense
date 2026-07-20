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


def test_append_step_never_leaves_a_partial_session_file_after_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test, same shape as calibration.apply.persist_calibration's:
    _write used to write directly to the session's final path, so a crash
    mid-write could leave a truncated/corrupt YAML file behind -- which a
    later load_session call (e.g. a resumed Hermes tool call mid-session)
    would then fail to parse, or worse, parse partially."""
    session = create_session(tmp_path, kind="reference_jar", experiment_id="exp_01", context={"sensors": ["PID01"]})
    session_path = tmp_path / f"{session.session_id}.yaml"
    original_contents = session_path.read_text(encoding="utf-8")

    def _crash_mid_write(self: Path, *args, **kwargs):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(Path, "write_text", _crash_mid_write)

    with pytest.raises(OSError, match="simulated crash mid-write"):
        append_step(tmp_path, session.session_id, {"sensor_id": "PID01", "pid_voltage_mv": 1.0})

    # The final file is untouched -- still exactly what create_session wrote,
    # not truncated or partially overwritten.
    assert session_path.read_text(encoding="utf-8") == original_contents
    assert list(tmp_path.glob("*.tmp")) == []
