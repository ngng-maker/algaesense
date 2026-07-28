"""The zero-and-span calibration procedure for a single sensor-reactor
pair, as pure logic with no Streamlit in it.

Zero and span is the classic two-point field calibration. You put the
sensor on clean air containing none of the analyte and record what it
reads (the zero), then on a gas of known concentration and record that
(the span). Two points define the line `voltage = b0 + b1 * ppm`, and
inverting that line turns every later voltage into a concentration.

It is deliberately simpler than the multi-point standard addition this
package also supports, and the simplicity costs something real -- see
`ZeroSpanQuality` below for what a two-point fit cannot tell you.

Kept separate from the Streamlit page on purpose: every rule about what
makes a calibration valid lives here, where it can be tested directly,
rather than being tangled up with widget state.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from jaxsr_calibration.calibration.apply import persist_calibration
from jaxsr_calibration.calibration.models import SensitivityModel
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor


MIN_READINGS_PER_LEVEL = 3
"""Below this there is no way to see whether the sensor had actually
settled, and the whole point of the dwell step is that it had."""

SUPPORTED_FIT_METHODS = ("ols", "robust", "wls")
"""`polynomial_deg2` is deliberately absent. A quadratic needs three
coefficients and zero/span supplies only two distinct concentrations, so
the fit would be underdetermined. Curvature requires multi-point standard
addition -- see `ZeroSpanQuality.curvature_untested`."""


@dataclass
class ZeroSpanSession:
    """One in-progress calibration of one sensor on one reactor."""

    reactor_id: str
    sensor_id: str
    calibration_run_id: str
    experiment_id: str

    calibration_compound: str
    mw_g_mol: float
    response_factor: float

    span_ppm: float
    """The certified concentration of the span gas cylinder."""

    zero_readings_mv: list[float] = field(default_factory=list)
    span_readings_mv: list[float] = field(default_factory=list)

    sample_t_c: float = 25.0
    sample_rh_pct: float = 50.0
    lamp_hours: float = 0.0

    def clear_zero(self) -> None:
        self.zero_readings_mv = []

    def clear_span(self) -> None:
        self.span_readings_mv = []

    @property
    def has_enough_zero(self) -> bool:
        return len(self.zero_readings_mv) >= MIN_READINGS_PER_LEVEL

    @property
    def has_enough_span(self) -> bool:
        return len(self.span_readings_mv) >= MIN_READINGS_PER_LEVEL

    @property
    def ready_to_fit(self) -> bool:
        return self.has_enough_zero and self.has_enough_span


@dataclass
class ZeroSpanQuality:
    """What can and cannot be concluded from a two-point calibration.

    R-squared is deliberately NOT the headline here. With only two
    distinct concentrations you are fitting two coefficients through two
    clusters of points, so R-squared comes out near 1.0 almost regardless
    of how good the sensor is -- quoting it as a pass criterion would be
    close to meaningless. What actually indicates a trustworthy two-point
    calibration is whether the sensor was STABLE at each level while you
    recorded it, which is what the two scatter figures below measure.
    """

    zero_mean_mv: float
    zero_std_mv: float
    span_mean_mv: float
    span_std_mv: float
    separation_mv: float
    """How far the span reading sits above the zero reading. A small
    separation means the span gas is barely distinguishable from clean
    air, and the resulting slope will be dominated by noise."""

    warnings: list[str] = field(default_factory=list)

    curvature_untested: bool = True
    """Always true for zero/span: two points cannot reveal whether the
    response between and beyond them is straight. Stated explicitly so it
    is a known limitation rather than a silent assumption."""


ZERO_INSTABILITY_MV = 2.0
SPAN_INSTABILITY_FRACTION = 0.05
MIN_SEPARATION_MV = 10.0


def assess_quality(session: ZeroSpanSession) -> ZeroSpanQuality:
    """Judge the recorded readings before anything is fitted, so a bad
    calibration is caught while the operator is still at the bench."""

    zero = np.asarray(session.zero_readings_mv, dtype=float)
    span = np.asarray(session.span_readings_mv, dtype=float)

    zero_mean, zero_std = float(np.mean(zero)), float(np.std(zero, ddof=1))
    span_mean, span_std = float(np.mean(span)), float(np.std(span, ddof=1))
    separation = span_mean - zero_mean

    warnings: list[str] = []
    if zero_std > ZERO_INSTABILITY_MV:
        warnings.append(
            f"Zero readings are unsettled (scatter {zero_std:.2f} mV). Let the sensor dwell "
            "longer on clean air before recording."
        )
    if span_mean != 0 and span_std > abs(span_mean) * SPAN_INSTABILITY_FRACTION:
        warnings.append(
            f"Span readings are unsettled (scatter {span_std:.2f} mV, "
            f"{100 * span_std / abs(span_mean):.1f}% of the level). Let the span gas stabilise longer."
        )
    if separation < MIN_SEPARATION_MV:
        warnings.append(
            f"Span sits only {separation:.1f} mV above zero. The slope will be dominated by "
            "noise -- use a more concentrated span gas."
        )
    if separation <= 0:
        warnings.append(
            "Span reads at or below zero. Check the span gas is actually flowing and that the "
            "cylinder concentration was entered correctly."
        )

    return ZeroSpanQuality(
        zero_mean_mv=zero_mean,
        zero_std_mv=zero_std,
        span_mean_mv=span_mean,
        span_std_mv=span_std,
        separation_mv=separation,
        warnings=warnings,
    )


def build_readings_frame(session: ZeroSpanSession) -> pl.DataFrame:
    """Shape the two recorded levels into the frame
    `fit_sensitivity_per_sensor` already expects, so the real fitting code
    is reused rather than a second implementation existing here."""

    if not session.ready_to_fit:
        raise ValueError(
            f"build_readings_frame: need at least {MIN_READINGS_PER_LEVEL} readings at both "
            f"zero and span; have {len(session.zero_readings_mv)} and "
            f"{len(session.span_readings_mv)}."
        )

    spike = [0.0] * len(session.zero_readings_mv) + [session.span_ppm] * len(session.span_readings_mv)
    voltage = list(session.zero_readings_mv) + list(session.span_readings_mv)
    n = len(spike)
    base_time = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)

    return pl.DataFrame(
        {
            "timestamp": [base_time + dt.timedelta(seconds=i) for i in range(n)],
            "sensor_id": [session.sensor_id] * n,
            "spike_ppm_asgas": spike,
            "pid_voltage_mv": voltage,
            "sample_t_c": [session.sample_t_c] * n,
            "sample_rh_pct": [session.sample_rh_pct] * n,
            "lamp_hours": [session.lamp_hours] * n,
            "calibration_compound": [session.calibration_compound] * n,
            "mw_g_mol": [session.mw_g_mol] * n,
            "response_factor": [session.response_factor] * n,
            "response_factor_stderr": [None] * n,
            "calibration_source": ["zero-span-ui"] * n,
            "calibration_is_builtin": [False] * n,
        }
    )


def fit_zero_span(session: ZeroSpanSession, method: str = "ols") -> SensitivityModel:
    """Fit the two-point line using the package's real fitting code."""

    if method not in SUPPORTED_FIT_METHODS:
        raise ValueError(
            f"fit_zero_span: method {method!r} is not usable with a two-point calibration; "
            f"choose one of {sorted(SUPPORTED_FIT_METHODS)}. A quadratic fit needs at least "
            "three distinct concentrations -- run a multi-point standard addition instead."
        )

    models = fit_sensitivity_per_sensor(build_readings_frame(session), method=method)
    return models[session.sensor_id]


def save_zero_span(session: ZeroSpanSession, model: SensitivityModel, data_dir: Path) -> Path:
    """Persist the fitted calibration where `apply_calibration` will find
    it, under the same layout every other calibration in this project
    uses."""
    out_dir = Path(data_dir) / "derived" / "calibrations" / "standard_addition"
    return persist_calibration(
        {session.sensor_id: model},
        session.calibration_run_id,
        session.experiment_id,
        out_dir,
    )
