"""Durable, file-backed storage for an in-progress guided calibration
session -- shared by all three calibration types (standard-addition,
reference-jar, camera zero-point).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


"""
A calibration session can span many minutes (a standard-addition dwell is
300s per spike level by default) across several separate tool calls from
Hermes, so it needs to survive between calls -- an in-memory dict inside
the MCP server process would be lost if that process restarted mid-session.
Persisting each session as its own YAML file (same durability reasoning as
labwiki's raw/ layer) means a session survives a server restart, and the
file itself is a real, inspectable record of exactly what was recorded and
when, not just a transient object.
"""


class SessionNotFoundError(LookupError):
    """Raised when a session_id doesn't correspond to a session file on disk."""


class SessionAlreadyFinishedError(RuntimeError):
    """Raised when trying to record a step (or finish again) on a session
    that's already been finished."""


@dataclass
class CalibrationSession:
    """One in-progress (or completed) guided calibration session."""

    session_id: str
    kind: str
    experiment_id: str
    context: dict
    steps: list[dict] = field(default_factory=list)
    status: str = "in_progress"
    created_at: str = ""


def _session_path(sessions_dir: Path, session_id: str) -> Path:
    return Path(sessions_dir) / f"{session_id}.yaml"


def _write(sessions_dir: Path, session: CalibrationSession) -> None:
    sessions_dir_path = Path(sessions_dir)
    sessions_dir_path.mkdir(parents=True, exist_ok=True)

    """
    `sort_keys=False` preserves the dataclass's own field order in the
    written file, matching this project's existing YAML-writing
    convention elsewhere (persist_calibration, labwiki's raw sources).
    """
    _session_path(sessions_dir_path, session.session_id).write_text(
        yaml.safe_dump(asdict(session), sort_keys=False), encoding="utf-8"
    )


def create_session(
    sessions_dir: Path,
    kind: str,
    experiment_id: str,
    context: dict,
    now: dt.datetime | None = None,
) -> CalibrationSession:
    """Start a new session and write its initial (empty-steps) file."""

    now = now or dt.datetime.now(dt.timezone.utc)

    """
    Session IDs are timestamp-based and filesystem-safe (colons replaced
    with hyphens), the same run_id convention already used by
    jaxsr_calibration.diagnostics.fleet_zero and
    algaesense_edge.service's camera clip filenames.
    """
    session_id = f"cal_{kind}_{experiment_id}_{now.strftime('%Y-%m-%dT%H-%M-%S')}"

    session = CalibrationSession(
        session_id=session_id,
        kind=kind,
        experiment_id=experiment_id,
        context=context,
        created_at=now.isoformat(),
    )
    _write(sessions_dir, session)

    return session


def load_session(sessions_dir: Path, session_id: str) -> CalibrationSession:
    path = _session_path(sessions_dir, session_id)

    if not path.exists():
        raise SessionNotFoundError(f"No calibration session found with id {session_id!r}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    return CalibrationSession(**raw)


def append_step(sessions_dir: Path, session_id: str, step: dict) -> CalibrationSession:
    """Record one more step onto an in-progress session."""

    session = load_session(sessions_dir, session_id)

    if session.status != "in_progress":
        raise SessionAlreadyFinishedError(
            f"session {session_id!r} is already {session.status!r}; cannot record another step"
        )

    session.steps.append(step)
    _write(sessions_dir, session)

    return session


def mark_finished(sessions_dir: Path, session_id: str) -> CalibrationSession:
    session = load_session(sessions_dir, session_id)
    session.status = "finished"
    _write(sessions_dir, session)
    return session
