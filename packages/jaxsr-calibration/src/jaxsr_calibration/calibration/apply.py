"""Saving, loading, and using a fitted sensor calibration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import yaml

from jaxsr_calibration.calibration.models import (
    CalibrationGas,
    CalibrationUnitUnavailableError,
    SensitivityModel,
)

"""
Combines what used to be two files (persist.py, apply.py) -- the full
lifecycle of a fitted SensitivityModel after fit_sensitivity_per_sensor
produces it: save it to disk, load it back, and use it to convert a raw
voltage reading into a concentration.

Two additions beyond the spec's bare positional signature for
apply_calibration, both documented inline where they appear: a required
`data_dir` keyword (apply_calibration has to know *where* to load
`calibration_run_id` from) and an optional `voltage_stderr` keyword (the
spec's own uncertainty-propagation formula needs Var(voltage), which
nothing in the bare signature supplies).

Disk layout, per calibration_run_id:
    {out_dir}/{calibration_run_id}.parquet   -- one row per sensor
    {out_dir}/{calibration_run_id}.yaml      -- full CalibrationGas provenance
"""

ExtrapolationPolicy = Literal["clip", "linear", "nan"]
OutputUnit = Literal["as_calibrated", "isobutylene_equiv", "both"]


def persist_calibration(
    models: dict[str, SensitivityModel],
    calibration_run_id: str,
    experiment_id: str,
    out_dir: Path,
) -> Path:
    """Save every sensor's fitted calibration to disk."""

    """
    Writes one Parquet file (spec §18's schema, one row per sensor) plus a
    YAML sidecar recording the full CalibrationGas.

    All sensors in `models` are assumed to share the same CalibrationGas (one
    compound per calibration run) -- fit_sensitivity_per_sensor already
    enforces "one compound per sensor" internally; this function additionally
    checks "one compound across the whole run" before writing anything.
    """

    if not models:
        raise ValueError("persist_calibration: models is empty, nothing to write")

    gases = {model.calibration_gas.name for model in models.values()}
    if len(gases) != 1:
        raise ValueError(
            f"persist_calibration: all sensors in one calibration run must share "
            f"the same compound, got {sorted(gases)}"
        )
    gas = next(iter(models.values())).calibration_gas

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "calibration_run_id": calibration_run_id,
            "experiment_id": experiment_id,
            "sensor_id": model.sensor_id,
            "calibration_compound": model.calibration_gas.name,
            "response_factor": model.calibration_gas.response_factor,
            "mw_g_mol": model.calibration_gas.mw,
            "mean_sample_t_c": model.mean_sample_t_c,
            "mean_sample_rh_pct": model.mean_sample_rh_pct,
            "lamp_hours": model.lamp_hours,
            "b0_mv": model.b0_mv,
            "b1_mv_per_ppm_asgas": model.b1_mv_per_ppm_asgas,
            "b1_mv_per_ppm_iso_equiv": model.b1_mv_per_ppm_iso_equiv,
            "b1_stderr": model.b1_stderr,
            "r_squared": model.r_squared,
            "status": model.status,
            "fit_method": model.fit_method,
        }
        for model in models.values()
    ]
    table = pl.DataFrame(rows)
    parquet_path = out_dir / f"{calibration_run_id}.parquet"
    table.write_parquet(parquet_path)

    sidecar = {
        "calibration_run_id": calibration_run_id,
        "experiment_id": experiment_id,
        "calibration_gas": {
            "name": gas.name,
            "mw": gas.mw,
            "response_factor": gas.response_factor,
            "response_factor_stderr": gas.response_factor_stderr,
            "ie_ev": gas.ie_ev,
            "source": gas.source,
            "is_builtin": gas.is_builtin,
        },
    }
    yaml_path = out_dir / f"{calibration_run_id}.yaml"

    with yaml_path.open("w", encoding="utf-8") as f:
        """
        `sort_keys=False` preserves the dict's own insertion order in the
        written file (calibration_run_id first, then experiment_id, etc.)
        rather than yaml's default alphabetical re-sort, which reads more
        naturally for a human opening this file directly.
        """
        yaml.safe_dump(sidecar, f, sort_keys=False)

    return parquet_path


def load_calibration(calibration_run_id: str, data_dir: Path) -> dict[str, SensitivityModel]:
    """Load a previously-saved calibration run back from disk."""

    """
    Returns `dict[sensor_id, SensitivityModel]` (spec §18), the reverse of
    persist_calibration.
    """

    parquet_path = data_dir / f"{calibration_run_id}.parquet"
    yaml_path = data_dir / f"{calibration_run_id}.yaml"
    if not parquet_path.exists():
        raise FileNotFoundError(f"No calibration Parquet file at {parquet_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"No calibration YAML sidecar at {yaml_path}")

    table = pl.read_parquet(parquet_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        sidecar = yaml.safe_load(f)

    gas = CalibrationGas(**sidecar["calibration_gas"])

    models: dict[str, SensitivityModel] = {}

    """
    `.iter_rows(named=True)` yields each row as a dict keyed by column name,
    which is more readable here than iterating positionally and having to
    remember which numeric index corresponds to which field.
    """
    for row in table.iter_rows(named=True):
        models[row["sensor_id"]] = SensitivityModel(
            sensor_id=row["sensor_id"],
            calibration_gas=gas,
            b0_mv=row["b0_mv"],
            b1_mv_per_ppm_asgas=row["b1_mv_per_ppm_asgas"],
            b1_mv_per_ppm_iso_equiv=row["b1_mv_per_ppm_iso_equiv"],
            b1_stderr=row["b1_stderr"],
            r_squared=row["r_squared"],
            fit_method=row["fit_method"],
            mean_sample_t_c=row["mean_sample_t_c"],
            mean_sample_rh_pct=row["mean_sample_rh_pct"],
            lamp_hours=row["lamp_hours"],
            status=row["status"],
        )
    return models


def apply_calibration(
    voltage: pl.Series,
    sensor_id: str,
    sample_t_c: pl.Series,
    sample_rh_pct: pl.Series,
    calibration_run_id: str,
    extrapolation_policy: ExtrapolationPolicy = "clip",
    output_unit: OutputUnit = "as_calibrated",
    *,
    data_dir: Path,
    voltage_stderr: float = 0.0,
) -> tuple[pl.Series | pl.DataFrame, pl.Series | pl.DataFrame, str]:
    """Convert a raw voltage reading into a concentration, using a saved calibration."""

    """
    Inverts `ppm = (voltage - b0) / b1` for the given sensor's calibration,
    with uncertainty propagated per the spec's explicit delta-method formula
    (§28):

        Var(ppm_asgas) = Var(voltage) / b1^2 + (voltage / b1^2)^2 * Var(b1)

    `sample_t_c`/`sample_rh_pct` are accepted (matching the spec's own
    signature) but not yet used to adjust the slope -- that only matters once
    a T/RH-aware sensitivity *surface* exists (spec §29's optional
    build_sensitivity_surface, not built in this milestone). A single
    calibration's b0/b1 are treated as constants here.

    `extrapolation_policy` is implemented against the one boundary we can
    always justify without extra stored data -- concentration can't be
    physically negative: "clip" floors negative results at 0, "nan" replaces
    them with NaN, "linear" leaves the raw (possibly negative, e.g. from
    sensor noise near zero concentration) linear-inversion value untouched.
    A fuller implementation clamping to the *calibrated spike range* would
    need that range stored on SensitivityModel, which isn't tracked yet.
    """

    models = load_calibration(calibration_run_id, data_dir)
    if sensor_id not in models:
        raise KeyError(f"No calibration for sensor {sensor_id!r} in run {calibration_run_id!r}")
    model = models[sensor_id]
    gas = model.calibration_gas

    b0, b1 = model.b0_mv, model.b1_mv_per_ppm_asgas
    voltage_np = voltage.to_numpy()

    ppm_asgas = (voltage_np - b0) / b1
    ppm_asgas = _apply_extrapolation_policy(ppm_asgas, extrapolation_policy)

    """
    The spec's formula verbatim: Var(voltage)/b1^2 + (voltage/b1^2)^2 *
    Var(b1). Using raw `voltage_np` (not the already-inverted ppm) in the
    second term's coefficient is what the spec's own formula literally
    specifies, so that's what's implemented here even though a from-scratch
    delta-method derivation would use (voltage - b0)/b1^2 instead -- this is
    a case where we follow the spec's given math exactly rather than
    re-deriving our own.
    """
    var_voltage = voltage_stderr**2
    var_b1 = model.b1_stderr**2
    var_ppm_asgas = var_voltage / (b1**2) + (voltage_np / (b1**2)) ** 2 * var_b1
    stderr_ppm_asgas = np.sqrt(var_ppm_asgas)

    if output_unit == "as_calibrated":
        return (
            pl.Series("ppm_asgas", ppm_asgas),
            pl.Series("ppm_asgas_stderr", stderr_ppm_asgas),
            f"ppm_asgas:{gas.name}",
        )

    if output_unit == "isobutylene_equiv":
        if not gas.has_rf:
            raise CalibrationUnitUnavailableError(
                f"sensor {sensor_id}: calibration compound {gas.name!r} has no known "
                "response factor, so isobutylene-equivalent output isn't available. "
                "Request output_unit='as_calibrated' instead, or supply a "
                "response_factor for this compound."
            )
        ppm_iso, stderr_iso = _to_iso_equiv(ppm_asgas, var_ppm_asgas, gas)
        return (
            pl.Series("ppm_isobutylene_equiv", ppm_iso),
            pl.Series("ppm_isobutylene_equiv_stderr", stderr_iso),
            "ppm_isobutylene_equiv",
        )

    if output_unit == "both":
        if gas.has_rf:
            ppm_iso, stderr_iso = _to_iso_equiv(ppm_asgas, var_ppm_asgas, gas)
        else:
            """
            spec §28: for output_unit="both", an unknown RF means the
            isobutylene-equivalent column is NaN-filled rather than raising
            (unlike explicitly requesting "isobutylene_equiv" alone, which
            does raise) -- "both" is meant to degrade gracefully.
            """
            ppm_iso = np.full_like(ppm_asgas, math.nan)
            stderr_iso = np.full_like(ppm_asgas, math.nan)
        values = pl.DataFrame({"ppm_asgas": ppm_asgas, "ppm_isobutylene_equiv": ppm_iso})
        stderrs = pl.DataFrame(
            {"ppm_asgas_stderr": stderr_ppm_asgas, "ppm_isobutylene_equiv_stderr": stderr_iso}
        )
        return values, stderrs, "ppm_asgas_and_isobutylene_equiv"

    raise ValueError(f"Unknown output_unit: {output_unit!r}")


def _apply_extrapolation_policy(ppm: np.ndarray, policy: ExtrapolationPolicy) -> np.ndarray:
    """Apply the requested rule for handling a physically-impossible negative concentration."""

    if policy == "clip":
        """
        `np.clip(a, a_min, a_max)` caps every element of `a` to be within
        [a_min, a_max]; passing `a_max=None` means "no upper cap", only the
        lower one (0.0, since concentration can't be negative) applies.
        """
        return np.clip(ppm, a_min=0.0, a_max=None)

    if policy == "linear":
        return ppm

    if policy == "nan":
        """
        `np.where(condition, if_true, if_false)` builds a new array,
        elementwise choosing between the two value arrays based on
        `condition` -- here, replacing negative entries with NaN and leaving
        everything else unchanged.
        """
        return np.where(ppm < 0.0, np.nan, ppm)

    raise ValueError(f"Unknown extrapolation_policy: {policy!r}")


def _to_iso_equiv(
    ppm_asgas: np.ndarray, var_ppm_asgas: np.ndarray, gas: CalibrationGas
) -> tuple[np.ndarray, np.ndarray]:
    """Convert calibration-gas-unit ppm into isobutylene-equivalent ppm."""

    """
    `ppm_iso = ppm_asgas * RF`, with uncertainty per spec §28: RF is treated
    as exact (Var(ppm_iso) = Var(ppm_asgas) * RF^2) unless the CalibrationGas
    carries a response_factor_stderr, in which case RF's own uncertainty is
    propagated too via the standard product-of-two-uncertain-quantities
    delta method: Var(X*Y) ~= Y^2*Var(X) + X^2*Var(Y) for independent X, Y.
    """

    rf = gas.response_factor
    ppm_iso = ppm_asgas * rf

    if gas.response_factor_stderr is None:
        var_ppm_iso = var_ppm_asgas * (rf**2)
    else:
        var_rf = gas.response_factor_stderr**2
        var_ppm_iso = (rf**2) * var_ppm_asgas + (ppm_asgas**2) * var_rf

    return ppm_iso, np.sqrt(var_ppm_iso)
