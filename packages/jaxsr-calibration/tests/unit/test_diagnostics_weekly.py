"""Unit tests for jaxsr_calibration.diagnostics.weekly.run_weekly_audit."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from jaxsr_calibration.calibration.config import SensorConfig
from jaxsr_calibration.diagnostics.models import SwapPilotResult
from jaxsr_calibration.diagnostics.weekly import run_weekly_audit

_TODAY = dt.date(2026, 7, 15)


def _sensor(sensor_id: str, install_date: dt.date) -> SensorConfig:
    return SensorConfig(
        id=sensor_id,
        model="PID-AH2",
        serial="X",
        lamp_install_date=install_date,
        lamp_hours_at_install=0,
        calibration_gas="isobutylene",
        factory_sensitivity_mV_per_ppm=20.0,
        associated_rh_sensor="SHT01",
        associated_reactor="R01",
    )


def test_run_weekly_audit_with_no_inputs_is_green() -> None:
    result = run_weekly_audit(today=_TODAY)

    assert result.summary_status == "GREEN"
    assert result.sensor_variance_share_delta is None
    assert result.sensors_due_for_cleaning == []
    assert result.report_path is None


def test_run_weekly_audit_computes_delta_between_last_two_swap_pilots() -> None:
    older = SwapPilotResult(
        variance_share={"sensor_id": 0.10, "reactor_id": 0.10, "residual": 0.80},
        mixedlm_summary="",
    )
    newer = SwapPilotResult(
        variance_share={"sensor_id": 0.18, "reactor_id": 0.10, "residual": 0.72},
        mixedlm_summary="",
    )

    result = run_weekly_audit(swap_pilot_results=[older, newer], today=_TODAY)

    # 0.18 - 0.10 = +0.08 -- variance share got worse by 8 percentage points.
    assert result.sensor_variance_share_delta is not None
    assert result.sensor_variance_share_delta > 0.079
    assert result.sensor_variance_share_delta < 0.081


def test_run_weekly_audit_red_when_sensor_variance_share_too_high() -> None:
    bad = SwapPilotResult(
        variance_share={"sensor_id": 0.45, "reactor_id": 0.10, "residual": 0.45},
        mixedlm_summary="",
    )

    result = run_weekly_audit(swap_pilot_results=[bad], today=_TODAY)

    # Default threshold (DiagnosticThresholds().swap_pilot.max_sensor_variance_share) is 0.30.
    assert result.summary_status == "RED"


def test_run_weekly_audit_red_when_backup_stale() -> None:
    result = run_weekly_audit(backup_current=False, today=_TODAY)

    assert result.summary_status == "RED"


def test_run_weekly_audit_flags_old_lamps_as_yellow() -> None:
    old_sensor = _sensor("PID01", install_date=_TODAY - dt.timedelta(days=200))
    fresh_sensor = _sensor("PID02", install_date=_TODAY - dt.timedelta(days=10))

    result = run_weekly_audit(
        sensor_configs=[old_sensor, fresh_sensor], today=_TODAY, lamp_cleaning_age_days=180
    )

    assert result.sensors_due_for_cleaning == ["PID01"]
    assert result.summary_status == "YELLOW"


def test_run_weekly_audit_writes_markdown_report(tmp_path: Path) -> None:
    out_path = tmp_path / "reports" / "weekly" / "2026-W29.md"

    result = run_weekly_audit(output_markdown=out_path, backup_current=True, today=_TODAY)

    assert result.report_path == out_path
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "Summary status: GREEN" in text
    assert "Backup current: True" in text
