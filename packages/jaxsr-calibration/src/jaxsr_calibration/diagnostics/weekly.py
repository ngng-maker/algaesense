"""Weekly diagnostic rollup: a simple composed summary over results other
diagnostics already produced, rather than a diagnostic that collects its
own raw data.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from jaxsr_calibration.calibration.config import SensorConfig
from jaxsr_calibration.diagnostics.models import DiagnosticStatus, WeeklyAuditResult
from jaxsr_calibration.diagnostics.swap_pilot import SwapPilotResult
from jaxsr_calibration.processing.config import DiagnosticThresholds


"""
Not given an explicit field list or algorithm in the spec (§23, §27) --
only the prose description of what it should check (sensor-id
variance-share drift, backup currency, lamp cleaning due dates) and the
function signature `run_weekly_audit(output_markdown=None) ->
WeeklyAuditResult`. The design below is intentionally simple: it composes
already-computed SwapPilotResults and SensorConfigs rather than re-deriving
everything from raw historical files, which keeps it decoupled and
testable without needing a real multi-week diagnostic history to exist yet.
"""


"""
Approximation: we don't yet track actual cumulative lamp-on hours (that
needs real acquisition data -- SensorConfig only records
lamp_hours_at_install and the calendar date of install). Using calendar age
since install as a proxy for "due for cleaning" is a simplification, noted
here rather than presented as more precise than it is.
"""
_DEFAULT_LAMP_CLEANING_AGE_DAYS = 180


def run_weekly_audit(
    output_markdown: Path | None = None,
    *,
    swap_pilot_results: list[SwapPilotResult] | None = None,
    sensor_configs: list[SensorConfig] | None = None,
    backup_current: bool | None = None,
    lamp_cleaning_age_days: int = _DEFAULT_LAMP_CLEANING_AGE_DAYS,
    thresholds: DiagnosticThresholds | None = None,
    today: dt.date | None = None,
) -> WeeklyAuditResult:
    """Compose a weekly rollup from already-computed diagnostic results."""

    """
    `swap_pilot_results` should be given oldest-first if more than one is
    provided -- the delta is computed as the most recent result's sensor_id
    variance share minus the previous one's.
    """

    limits = (thresholds or DiagnosticThresholds()).swap_pilot

    """
    `today` is injectable (rather than always using dt.date.today()) so
    tests can pass a fixed date and get fully deterministic results instead
    of depending on whatever day the test happens to run.
    """
    today = today or dt.date.today()

    sensor_variance_share_delta: float | None = None
    latest_sensor_share: float | None = None
    if swap_pilot_results:
        latest_sensor_share = swap_pilot_results[-1].variance_share.get("sensor_id")
        if len(swap_pilot_results) >= 2:
            previous_sensor_share = swap_pilot_results[-2].variance_share.get("sensor_id")
            if latest_sensor_share is not None and previous_sensor_share is not None:
                sensor_variance_share_delta = latest_sensor_share - previous_sensor_share

    sensors_due_for_cleaning: list[str] = []
    for sensor in sensor_configs or []:
        lamp_age_days = (today - sensor.lamp_install_date).days
        if lamp_age_days >= lamp_cleaning_age_days:
            sensors_due_for_cleaning.append(sensor.id)

    summary_status = _classify(
        latest_sensor_share=latest_sensor_share,
        max_sensor_variance_share=limits.max_sensor_variance_share,
        sensors_due_for_cleaning=sensors_due_for_cleaning,
        backup_current=backup_current,
    )

    report_path: Path | None = None
    if output_markdown is not None:
        report_path = _write_markdown_report(
            output_markdown,
            summary_status=summary_status,
            sensor_variance_share_delta=sensor_variance_share_delta,
            latest_sensor_share=latest_sensor_share,
            sensors_due_for_cleaning=sensors_due_for_cleaning,
            backup_current=backup_current,
        )

    return WeeklyAuditResult(
        summary_status=summary_status,
        sensor_variance_share_delta=sensor_variance_share_delta,
        sensors_due_for_cleaning=sensors_due_for_cleaning,
        backup_current=backup_current,
        report_path=report_path,
    )


def _classify(
    *,
    latest_sensor_share: float | None,
    max_sensor_variance_share: float,
    sensors_due_for_cleaning: list[str],
    backup_current: bool | None,
) -> DiagnosticStatus:
    """Roll up the individual signals into one overall status."""

    """
    RED if the sensor-id variance share is over its healthy limit or
    backups are confirmed stale; YELLOW if anything else looks like it
    needs attention (a lamp due for cleaning); GREEN otherwise. A simple,
    worst-case-wins rollup -- not spec-mandated, see module technical block.
    """

    if backup_current is False:
        return "RED"

    if latest_sensor_share is not None and latest_sensor_share > max_sensor_variance_share:
        return "RED"

    if sensors_due_for_cleaning:
        return "YELLOW"

    return "GREEN"


def _write_markdown_report(
    path: Path,
    *,
    summary_status: DiagnosticStatus,
    sensor_variance_share_delta: float | None,
    latest_sensor_share: float | None,
    sensors_due_for_cleaning: list[str],
    backup_current: bool | None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Weekly Diagnostic Audit -- {dt.date.today().isoformat()}",
        "",
        f"**Summary status: {summary_status}**",
        "",
        f"- Latest sensor-id variance share: {_fmt_pct(latest_sensor_share)}",
        f"- Change since previous swap-pilot: {_fmt_pct(sensor_variance_share_delta, signed=True)}",
        f"- Backup current: {backup_current if backup_current is not None else 'unknown'}",
        f"- Sensors due for lamp cleaning: {', '.join(sensors_due_for_cleaning) or 'none'}",
    ]

    """
    `"\\n".join(lines)` glues the list of strings above into one multi-line
    string, with a real newline character inserted between each pair --
    simpler than repeatedly using `+=` to build the string in a loop.
    """
    path.write_text("\n".join(lines), encoding="utf-8")

    return path


def _fmt_pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"

    """
    `{value:+.1%}` / `{value:.1%}`: Python's "%" format spec multiplies by
    100 and appends a "%" sign automatically (so 0.183 becomes "18.3%"),
    and the leading "+" flag forces a plus sign on positive numbers --
    useful here since "the share went up" vs. "went down" is exactly what
    we want a reader to see at a glance for a *delta*, not just a plain
    value.
    """
    return f"{value:+.1%}" if signed else f"{value:.1%}"
