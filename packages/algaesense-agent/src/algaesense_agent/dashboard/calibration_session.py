"""Multi-point gas calibration of a single sensor-reactor pair, as pure
logic with no Streamlit in it.

The procedure: flow a series of known concentrations over the sensor --
clean air first, then each certified level in turn -- and record what it
reads at each. Fitting a curve through those points gives the relationship
between millivolts and concentration, and inverting it turns every later
reading into a real number.

Why several levels rather than just clean air and one span gas: two points
define a straight line no matter what the sensor actually did between
them. Only a third and fourth level can show whether the response is
genuinely straight or bends -- and PID sensors do bend, compressing near
the top of their range. A two-point calibration cannot detect that, and
would quietly report the compressed region as though it were linear.

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


DEFAULT_LEVELS_PPM = [0.0, 1.0, 3.0, 5.0]
"""Clean air plus three certified concentrations spanning this PID's
stated 0-5 ppm working range. Spaced 0/1/3/5 rather than evenly: two
points sit in the lower half, where the reactor actually operates, while
still reaching full scale to pin down the top of the curve."""

MIN_LEVELS = 4
"""Three points are the mathematical minimum for a quadratic; a fourth
leaves a degree of freedom to judge whether that curvature is real or is
just fitting noise."""

MIN_READINGS_PER_LEVEL = 3
"""Below this there is no way to see whether the sensor had actually
settled, and the whole point of the dwell step is that it had."""

SUPPORTED_FIT_METHODS = ("wls", "ols", "robust", "polynomial_deg2")
"""All four are usable once there are at least three distinct levels.
`wls` leads because PID scatter grows with concentration: weighting each
level by how tightly its own replicates clustered stops the noisiest end
of the range dominating the fit."""


@dataclass
class LevelRecording:
    """One known concentration, and the readings taken at it."""

    ppm: float
    readings_mv: list[float] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.readings_mv) >= MIN_READINGS_PER_LEVEL


@dataclass
class CalibrationSession:
    """One in-progress calibration of one sensor on one reactor."""

    reactor_id: str
    sensor_id: str
    calibration_run_id: str
    experiment_id: str

    calibration_compound: str
    mw_g_mol: float
    response_factor: float

    levels: list[LevelRecording] = field(default_factory=list)

    sample_t_c: float = 25.0
    sample_rh_pct: float = 50.0
    lamp_hours: float = 0.0

    @property
    def distinct_level_count(self) -> int:
        return len({level.ppm for level in self.levels})

    @property
    def completed_levels(self) -> list[LevelRecording]:
        return [level for level in self.levels if level.is_complete]

    @property
    def ready_to_fit(self) -> bool:
        return (
            len(self.completed_levels) == len(self.levels)
            and self.distinct_level_count >= MIN_LEVELS
        )


def build_session_levels(levels_ppm: list[float]) -> list[LevelRecording]:
    """Turn a list of planned concentrations into empty recordings.

    Sorted ascending so the operator works upward through the range --
    going downward risks a high concentration's residue lingering in the
    line and inflating the next, lower reading.
    """
    return [LevelRecording(ppm=ppm) for ppm in sorted(set(levels_ppm))]


@dataclass
class LevelQuality:
    ppm: float
    mean_mv: float
    std_mv: float
    n: int


@dataclass
class CalibrationQuality:
    """What the recorded points say about whether this calibration can be
    trusted -- judged before anything is saved, while the operator is
    still at the bench and can redo a level."""

    levels: list[LevelQuality]
    r_squared: float
    """A real measure here, unlike in a two-point calibration where it is
    near 1.0 by construction: with more levels than the line has
    coefficients, the fit can genuinely fail to describe the points."""

    curvature_share: float
    """Of the variance a straight line fails to explain, the share a
    quadratic accounts for.

    Deliberately a share rather than a raw R-squared difference. A decent
    sensor's linear R-squared is already ~0.998, leaving so little
    unexplained that even perfectly capturing real curvature moves the
    absolute number by ~0.002 -- indistinguishable from noise by eye. The
    useful question is what fraction of the line's leftover error is
    structure rather than scatter, and that comes out near 1.0 for a
    genuinely bent response.
    """

    warnings: list[str] = field(default_factory=list)
    suggests_quadratic: bool = False


LEVEL_INSTABILITY_MV = 2.0
LEVEL_INSTABILITY_FRACTION = 0.05
MIN_TOTAL_SPAN_MV = 10.0

CURVATURE_SHARE_THRESHOLD = 0.5
"""Half the line's leftover error being structure rather than scatter. Well
clear of what an extra free parameter explains by chance -- roughly
1/(n-2), a few percent at this many readings."""


def assess_quality(session: CalibrationSession) -> CalibrationQuality:
    """Judge the recorded readings before anything is persisted."""

    levels: list[LevelQuality] = []
    warnings: list[str] = []

    for level in session.levels:
        values = np.asarray(level.readings_mv, dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        levels.append(LevelQuality(ppm=level.ppm, mean_mv=mean, std_mv=std, n=int(values.size)))

        """
        Judged against whichever tolerance is looser: an absolute floor
        near zero ppm, where a percentage of a tiny number is meaningless,
        and a proportional one higher up, where absolute millivolt scatter
        naturally grows with concentration.
        """
        tolerance = max(LEVEL_INSTABILITY_MV, abs(mean) * LEVEL_INSTABILITY_FRACTION)
        if std > tolerance:
            warnings.append(
                f"{level.ppm:g} ppm readings are unsettled (scatter {std:.2f} mV). Let the sensor "
                "dwell longer at that concentration before recording."
            )

    ordered = sorted(levels, key=lambda lq: lq.ppm)

    total_span = ordered[-1].mean_mv - ordered[0].mean_mv
    if total_span < MIN_TOTAL_SPAN_MV:
        warnings.append(
            f"The whole set of levels spans only {total_span:.1f} mV. The fit will be dominated by "
            "noise -- check the gas actually reached the sensor at each level."
        )

    """
    Monotonicity is checked on level means rather than individual
    readings: a higher concentration reading lower than a lower one is
    physically impossible for a PID, so it points at a procedural mistake
    -- a mislabelled cylinder, or a level recorded before the previous gas
    cleared -- rather than at ordinary noise.
    """
    for earlier, later in zip(ordered, ordered[1:]):
        if later.mean_mv <= earlier.mean_mv:
            warnings.append(
                f"{later.ppm:g} ppm reads no higher than {earlier.ppm:g} ppm "
                f"({later.mean_mv:.1f} vs {earlier.mean_mv:.1f} mV). Check the cylinder labels, and "
                "that each gas fully cleared before the next level was recorded."
            )

    r_squared, curvature_share = _linear_fit_quality(session)
    suggests_quadratic = curvature_share > CURVATURE_SHARE_THRESHOLD
    if suggests_quadratic:
        warnings.append(
            f"The response bends: {100 * curvature_share:.0f}% of what a straight line fails to "
            "explain is curvature, not scatter. Use the polynomial_deg2 fit method -- detecting "
            "this is exactly what the extra levels are for."
        )

    return CalibrationQuality(
        levels=ordered,
        r_squared=r_squared,
        curvature_share=curvature_share,
        warnings=warnings,
        suggests_quadratic=suggests_quadratic,
    )


def _linear_fit_quality(session: CalibrationSession) -> tuple[float, float]:
    """R-squared of a straight-line fit, and the share of its unexplained
    variance that a quadratic accounts for."""
    recorded = [level for level in session.levels if level.readings_mv]
    if len(recorded) < 2:
        return 0.0, 0.0

    ppm = np.concatenate([np.full(len(l.readings_mv), l.ppm) for l in recorded])
    mv = np.concatenate([np.asarray(l.readings_mv, dtype=float) for l in recorded])

    total_ss = float(np.sum((mv - mv.mean()) ** 2))
    if total_ss <= 0.0:
        return 0.0, 0.0

    def _r2(degree: int) -> float:
        coefficients = np.polyfit(ppm, mv, degree)
        residual = mv - np.polyval(coefficients, ppm)
        return 1.0 - float(np.sum(residual**2)) / total_ss

    linear_r2 = _r2(1)
    distinct = len({level.ppm for level in recorded})
    if distinct < 3:
        return linear_r2, 0.0

    unexplained = 1.0 - linear_r2
    if unexplained <= 0.0:
        return linear_r2, 0.0

    quadratic_r2 = _r2(2)
    return linear_r2, max(0.0, min(1.0, (quadratic_r2 - linear_r2) / unexplained))


def build_readings_frame(session: CalibrationSession) -> pl.DataFrame:
    """Shape the recorded levels into the frame
    `fit_sensitivity_per_sensor` already expects, so the real fitting code
    is reused rather than a second implementation existing here."""

    if not session.ready_to_fit:
        raise ValueError(
            f"build_readings_frame: need at least {MIN_LEVELS} distinct concentrations with "
            f"{MIN_READINGS_PER_LEVEL} readings each; have {session.distinct_level_count} distinct "
            f"levels, {len(session.completed_levels)} of {len(session.levels)} complete."
        )

    spike: list[float] = []
    voltage: list[float] = []
    for level in session.levels:
        spike.extend([level.ppm] * len(level.readings_mv))
        voltage.extend(level.readings_mv)

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
            "calibration_source": ["dashboard-multipoint"] * n,
            "calibration_is_builtin": [False] * n,
        }
    )


def fit_calibration(session: CalibrationSession, method: str = "wls") -> SensitivityModel:
    """Fit the calibration curve using the package's real fitting code."""

    if method not in SUPPORTED_FIT_METHODS:
        raise ValueError(
            f"fit_calibration: method {method!r} is not supported; choose one of "
            f"{sorted(SUPPORTED_FIT_METHODS)}."
        )

    models = fit_sensitivity_per_sensor(build_readings_frame(session), method=method)
    return models[session.sensor_id]


def save_calibration(session: CalibrationSession, model: SensitivityModel, data_dir: Path) -> Path:
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
