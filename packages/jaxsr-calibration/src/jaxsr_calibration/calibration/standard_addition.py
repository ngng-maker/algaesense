"""Standard-addition calibration: relate sensor voltage to a known injected
concentration of a chosen calibration compound.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats as scipy_stats

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.calibration.models import CalibrationGas, SensitivityModel
from jaxsr_calibration.validation import require_columns, require_implemented_method


"""
`fit_sensitivity_per_sensor` is real, tested regression logic.
`run_standard_addition` is a stub -- see its technical block below for why
(same "no live acquisition backend yet" situation as
jaxsr_calibration.diagnostics).
"""


"""
Not spec-mandated numeric values -- see fleet_zero.py's `_classify_sensor`
for the same PASS/SUSPECT/FAIL-tiering pattern this mirrors. The spec DOES
explicitly say "Acceptance: R^2 >= 0.95 per sensor" (§20/§26), so the PASS
bar itself is spec-derived; only the SUSPECT/FAIL split point below it is
our own reasonable choice.
"""
_R_SQUARED_PASS = 0.95
_R_SQUARED_SUSPECT = 0.85

"""
Columns fit_sensitivity_per_sensor's input `df` is expected to carry,
denormalized per row (mirroring how the persisted calibration schema, spec
§18, already stores calibration_compound/response_factor/mw_g_mol as
row-level columns rather than separate metadata).
"""
_REQUIRED_COLUMNS = {
    "sensor_id",
    "spike_ppm_asgas",
    "pid_voltage_mv",
    "sample_t_c",
    "sample_rh_pct",
    "lamp_hours",
    "calibration_compound",
    "mw_g_mol",
    "response_factor",
}

"""
`method` values this module actually has a fit implementation for --
checked at the earliest reachable point in each function below (before any
live-acquisition raise or per-sensor loop), not merely wherever a fit is
eventually attempted.
"""
_IMPLEMENTED_METHODS = {"ols", "robust"}


def run_standard_addition(
    experiment_id: str,
    calibration_gas: CalibrationGas,
    spike_ppm_list: list[float],
    dwell_seconds: int = 300,
    method: str = "ols",
) -> pl.DataFrame:
    """Drive the interactive spike-and-recover procedure: log a baseline,
    prompt the operator to physically inject the calibration standard at
    each level in `spike_ppm_list`, dwell and record after each."""

    """
    This inherently requires live hardware acquisition AND a human present
    to perform each injection -- there's no meaningful "offline" version of
    this specific function the way there is for e.g. run_fleet_zero (which
    can at least analyze already-collected data). Once algaesense-edge
    exists and this function can drive it, it will return the raw
    per-timestamp readings DataFrame the spec describes; until then, if you
    already have such a DataFrame (e.g. from a test fixture, or produced by
    some other means), call fit_sensitivity_per_sensor(df, method=...)
    directly -- that part is fully implemented.
    """

    """
    Guarded even though this function is currently a stub that always
    raises below regardless of method -- so an invalid method name is
    named correctly the day algaesense_edge exists and this stub is
    replaced with a real implementation, rather than this param being
    silently inert until then.
    """
    require_implemented_method(method, _IMPLEMENTED_METHODS, "run_standard_addition")

    raise LiveAcquisitionNotAvailableError(
        "run_standard_addition needs to drive an interactive, live spike-and-"
        "recover procedure (needs algaesense-edge, a later phase). If you "
        "already have collected spike-and-recover data as a DataFrame, call "
        "fit_sensitivity_per_sensor(df, method=...) directly instead."
    )


def fit_sensitivity_per_sensor(df: pl.DataFrame, method: str = "ols") -> dict[str, SensitivityModel]:
    """Fit `voltage = b0 + b1 * spike_ppm_asgas` per sensor from
    already-collected spike-and-recover data."""

    """
    `df` must contain (see _REQUIRED_COLUMNS above): one row per reading,
    with `spike_ppm_asgas` labeling which injection level that reading
    belongs to (0.0 for baseline/no-injection rows), and the calibration
    compound's identity denormalized onto every row via
    calibration_compound/mw_g_mol/response_factor(_stderr).
    """

    require_implemented_method(method, _IMPLEMENTED_METHODS, "fit_sensitivity_per_sensor")
    require_columns(df, _REQUIRED_COLUMNS, "fit_sensitivity_per_sensor")

    results: dict[str, SensitivityModel] = {}

    for (sensor_id,), sensor_df in df.partition_by("sensor_id", as_dict=True).items():
        spike_ppm = sensor_df["spike_ppm_asgas"].to_numpy()
        voltage = sensor_df["pid_voltage_mv"].to_numpy()

        if len(np.unique(spike_ppm)) < 2:
            raise ValueError(
                f"sensor {sensor_id}: need at least 2 distinct spike_ppm_asgas "
                "levels (including baseline=0) to fit a slope; got "
                f"{sorted(set(spike_ppm.tolist()))}."
            )

        if method == "ols":
            design = sm.add_constant(spike_ppm)
            result = sm.OLS(voltage, design).fit()
            b0, b1 = result.params

            """
            `result.bse` ("basic standard errors") has one entry per
            coefficient in the same order as `result.params` -- index 1 is
            the slope's standard error, matching b1 at index 1.
            """
            b1_stderr = float(result.bse[1])
            r_squared = float(result.rsquared)
        else:
            """
            Theil-Sen: a median-of-pairwise-slopes estimator, far less
            sensitive to a single bad outlier reading than OLS's
            least-squares fit. `scipy.stats.theilslopes` returns a 95%
            confidence interval on the slope by default; deriving a
            stderr-equivalent from its half-width keeps `SensitivityModel`'s
            existing `b1_stderr` field meaningful regardless of which
            method produced it, rather than adding a second, method-specific
            uncertainty field.
            """
            slope, intercept, low_slope, high_slope = scipy_stats.theilslopes(voltage, spike_ppm)
            b0, b1 = intercept, slope
            b1_stderr = float((high_slope - low_slope) / (2 * 1.96))

            predicted = b0 + b1 * spike_ppm
            ss_res = float(np.sum((voltage - predicted) ** 2))
            ss_tot = float(np.sum((voltage - np.mean(voltage)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        gas = _reconstruct_calibration_gas(sensor_df, sensor_id)
        b1_iso_equiv = b1 * gas.response_factor if gas.has_rf else None

        results[sensor_id] = SensitivityModel(
            sensor_id=sensor_id,
            calibration_gas=gas,
            b0_mv=float(b0),
            b1_mv_per_ppm_asgas=float(b1),
            b1_mv_per_ppm_iso_equiv=b1_iso_equiv,
            b1_stderr=b1_stderr,
            r_squared=r_squared,
            fit_method=method,
            mean_sample_t_c=float(sensor_df["sample_t_c"].mean()),
            mean_sample_rh_pct=float(sensor_df["sample_rh_pct"].mean()),
            lamp_hours=float(sensor_df["lamp_hours"].max()),
            status=_classify(r_squared),
        )

    return results


def _classify(r_squared: float) -> str:
    if r_squared >= _R_SQUARED_PASS:
        return "PASS"

    if r_squared >= _R_SQUARED_SUSPECT:
        return "SUSPECT"

    return "FAIL"


def _reconstruct_calibration_gas(sensor_df: pl.DataFrame, sensor_id: str) -> CalibrationGas:
    """Rebuild a CalibrationGas from the denormalized columns on one
    sensor's calibration rows."""

    """
    Validates that exactly one compound was used -- a calibration run
    mixing compounds mid-way would silently corrupt the fit otherwise.
    """
    compounds = sensor_df["calibration_compound"].unique().to_list()
    if len(compounds) != 1:
        raise ValueError(
            f"sensor {sensor_id}: expected exactly one calibration_compound, got {compounds}"
        )
    compound = compounds[0]

    """
    `.drop_nulls()` removes any None/null entries from the column before we
    look at its first value -- response_factor is legitimately all-null
    when the compound's RF is unknown (spec §24 Option 3), and we want
    `has_rf` semantics preserved (None, not some accidental 0.0) rather
    than crashing on a null lookup.
    """
    rf_values = sensor_df["response_factor"].drop_nulls()
    response_factor = float(rf_values[0]) if rf_values.len() > 0 else None

    rf_stderr_col = "response_factor_stderr"
    response_factor_stderr = None
    if rf_stderr_col in sensor_df.columns:
        rf_stderr_values = sensor_df[rf_stderr_col].drop_nulls()
        if rf_stderr_values.len() > 0:
            response_factor_stderr = float(rf_stderr_values[0])

    source = "user"
    if "calibration_source" in sensor_df.columns:
        source_values = sensor_df["calibration_source"].drop_nulls()
        if source_values.len() > 0:
            source = source_values[0]

    is_builtin = False
    if "calibration_is_builtin" in sensor_df.columns:
        is_builtin = bool(sensor_df["calibration_is_builtin"][0])

    return CalibrationGas(
        name=compound,
        mw=float(sensor_df["mw_g_mol"][0]),
        response_factor=response_factor,
        response_factor_stderr=response_factor_stderr,
        source=source,
        is_builtin=is_builtin,
    )
