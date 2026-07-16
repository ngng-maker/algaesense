"""Calibration compounds and fitted sensor calibration results."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


"""
Combines what used to be three separate files (models.py, response_factors.py,
errors.py) -- all three were thin (a table lookup, two exception classes,
two dataclasses) and existed only to support this subpackage's calibration
data. Kept together as one deep module: the public surface is "give me a
CalibrationGas or a SensitivityModel", and the response-factor table loading
plus the one calibration-specific exception are internal details of that.
"""


class CalibrationUnitUnavailableError(ValueError):
    """Raised when isobutylene-equivalent ppm is requested but not available for this calibration."""

    """
    Specifically: the calibration's compound has no known response factor
    (spec §28), so there is nothing to convert isobutylene-equivalent units
    from.
    """


def _load_builtin_table() -> dict[str, dict]:
    """Load the shipped table of known compounds and their response factors."""

    """
    `importlib.resources.files(package)` locates a package's own data files
    correctly regardless of *how* the package is installed -- a normal
    site-packages install, an editable `pip install -e .` install (what this
    project uses locally), or even a zipped package -- unlike hardcoding a
    path relative to `__file__`, which can break under some of those
    installation modes.
    """
    data_path = (
        importlib.resources.files("jaxsr_calibration.calibration") / "data" / "response_factors.yaml"
    )

    """
    `.open("r")` on an `importlib.resources` Traversable works like a normal
    file object; `as f` gives us that open handle inside the `with` block,
    closed automatically afterwards.
    """
    with data_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return raw["compounds"]


def load_response_factor_table(overrides_path: Path | None = None) -> dict[str, dict]:
    """Look up the built-in compound table, optionally with project-specific overrides."""

    """
    Returns {compound_name: {"mw", "response_factor", "ie_ev", "source"}}.
    Entries in `overrides_path` (if given and it exists) replace or add to
    the built-in table -- spec §24's "merged over the built-in table at load
    time. Same schema" behavior.
    """
    table = _load_builtin_table()

    if overrides_path is not None and overrides_path.exists():
        with overrides_path.open("r", encoding="utf-8") as f:
            overrides_raw = yaml.safe_load(f) or {}
        overrides = overrides_raw.get("compounds", {})

        """
        `dict.update(other)` overwrites any key present in both with
        `other`'s value, and adds any key only present in `other` -- exactly
        "override where specified, keep built-in elsewhere".
        """
        table.update(overrides)

    return table


@dataclass(frozen=True)
class CalibrationGas:
    """The VOC standard used for a calibration run."""

    """
    Which compound, its molecular weight, and its response factor (RF)
    relative to isobutylene on a PID lamp (spec §24).

    `frozen=True` makes instances immutable (assigning to `gas.mw = 5` after
    construction raises an error) -- appropriate here because a
    CalibrationGas is a record of *what was used* for a specific past
    calibration; nothing should ever mutate it after the fact.
    """

    name: str
    mw: float

    """
    None means "response factor not known" -- not "zero response" or "not
    applicable". Downstream code (isobutylene-equivalent unit conversion)
    must always check `has_rf` before using this.
    """
    response_factor: float | None

    response_factor_stderr: float | None = None
    ie_ev: float | None = None
    source: str = "user"
    is_builtin: bool = False

    @classmethod
    def builtin(cls, name: str, overrides_path: Path | None = None) -> CalibrationGas:
        """Look up a compound by name from the built-in response-factor table."""

        """
        `overrides_path` is an addition beyond the spec's own bare
        `builtin(cls, name)` signature -- an optional parameter defaulting to
        None, so `CalibrationGas.builtin("isoprene")` (the spec's own usage
        example) still works unchanged; passing
        `overrides_path=Path("configs/response_factors_overrides.yaml")` is
        purely additive.
        """
        table = load_response_factor_table(overrides_path)

        """
        Compound names are matched case-insensitively (the built-in table
        itself uses lowercase keys) so a user typing "Isoprene" or
        "ISOPRENE" at an interactive prompt still resolves correctly.
        """
        key = name.lower()

        if key not in table:
            known = ", ".join(sorted(table))
            raise ValueError(
                f"{name!r} is not in the response-factor table (known: {known}). "
                "Use CalibrationGas.custom(...) for an unlisted compound."
            )

        entry = table[key]
        return cls(
            name=key,
            mw=entry["mw"],
            response_factor=entry.get("response_factor"),
            ie_ev=entry.get("ie_ev"),
            source=entry.get("source", "builtin"),
            is_builtin=True,
        )

    @classmethod
    def custom(cls, name: str, mw: float, response_factor: float | None = None, **kw) -> CalibrationGas:
        """Describe a compound that isn't in the built-in table."""

        """
        `response_factor=None` (the default) is valid -- see `has_rf` below
        -- the tool still works, just reports results in compound-specific
        units only (spec §24's "Option 3"). `**kw` forwards any other
        CalibrationGas fields the caller wants to set
        (response_factor_stderr, ie_ev, source) without this method needing
        to name every one of them individually.
        """
        return cls(name=name, mw=mw, response_factor=response_factor, is_builtin=False, **kw)

    @property
    def has_rf(self) -> bool:
        """Whether this compound's response factor is actually known."""
        return self.response_factor is not None


@dataclass
class SensitivityModel:
    """A fitted calibration line for one sensor."""

    """
    The line is `voltage = b0 + b1 * ppm` (spec §26, §34) -- the output of
    fit_sensitivity_per_sensor.
    """

    sensor_id: str
    calibration_gas: CalibrationGas
    b0_mv: float

    """
    Slope in calibration-gas units -- e.g. "mV per ppm of isoprene", not yet
    corrected to an isobutylene-equivalent scale.
    """
    b1_mv_per_ppm_asgas: float

    """
    None iff calibration_gas.has_rf is False -- see fit_sensitivity_per_sensor.
    """
    b1_mv_per_ppm_iso_equiv: float | None

    b1_stderr: float
    r_squared: float
    fit_method: Literal["ols", "robust", "polynomial_deg2"]
    mean_sample_t_c: float
    mean_sample_rh_pct: float
    lamp_hours: float
    status: Literal["PASS", "SUSPECT", "FAIL"]
