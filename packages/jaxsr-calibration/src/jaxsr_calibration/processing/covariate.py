"""Covariate (temperature/humidity) correction: fitting and applying an
ambient-response model.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
import statsmodels.api as sm
import yaml

from jaxsr_calibration.processing.errors import TrainingWindowInsufficientError
from jaxsr_calibration.validation import require_columns, require_implemented_method

"""
Combines what used to be two files (models.py, covariate.py) --
CovariateModel had no logic of its own, only fit_covariate_model
(below) produces it, so it lives next to the function that creates it
rather than in its own near-empty file (the same pattern
camera/calibration.py already uses for BiomassCameraModel).

fit_covariate_model's "robust" and "symbolic" methods are still not
implemented (see that function's docstring) -- only apply_covariate_correction
is new as of this milestone.
"""

"""
`TYPE_CHECKING` is a constant that is always False at runtime but treated as
True by type-checkers (mypy/pyright) and IDEs. Importing jaxsr only inside
this block means the string annotation "jaxsr.SymbolicRegressor | None"
below (on CovariateModel.symbolic_regressor) works whether or not this
import actually runs -- a machine without jaxsr installed (or a different
version) can still import this module without error; only code that
actually *uses* the symbolic method at runtime needs jaxsr present.
"""
if TYPE_CHECKING:
    import jaxsr


@dataclass
class CovariateModel:
    """A fitted model of how a sensor's signal responds to ambient temperature and humidity."""

    """
    Used to *subtract out* that ambient effect later (apply_covariate_correction).

    For method="ols"/"robust" the model is the linear form
    `voltage ~ alpha + beta_rh*RH + gamma_t*T + delta_rh_t*RH*T`, so
    alpha/beta_rh/gamma_t/delta_rh_t/covariance are populated and
    symbolic_regressor is None. For method="symbolic" it's the reverse:
    those four coefficients are None and symbolic_regressor holds a fitted
    jaxsr.SymbolicRegressor instead.
    """

    sensor_id: str
    method: Literal["ols", "robust", "symbolic"]
    alpha: float | None
    beta_rh: float | None
    gamma_t: float | None
    delta_rh_t: float | None

    """
    The 4x4 parameter covariance matrix from the OLS/robust fit -- lets
    later code propagate uncertainty through the correction, not just use
    the point-estimate coefficients. Only meaningful for the linear methods,
    None for the symbolic one.
    """
    covariance: np.ndarray | None

    """
    A forward-reference string annotation (quoted) rather than a direct
    `jaxsr.SymbolicRegressor | None` -- this is what lets the import above
    stay inside `if TYPE_CHECKING:` and never actually execute at runtime
    for callers who only use the ols/robust path.
    """
    symbolic_regressor: "jaxsr.SymbolicRegressor | None"

    training_window: tuple[dt.datetime, dt.datetime]
    r_squared: float


"""
Below this many surviving rows, a 4-parameter regression (alpha, beta_rh,
gamma_t, delta_rh_t) is considered too under-determined to trust, even if
the RH-range check alone would pass. 10 is a generous margin above the
mathematical minimum of 4 -- not a spec-mandated number, just a sanity floor.
"""
_MIN_TRAINING_ROWS = 10

"""
`method` values this module actually has a fit implementation for --
"symbolic" (fitting through jaxsr.SymbolicRegressor + jaxsr.Constraints
instead of plain regression) is still real, planned, and not built yet.
"""
_IMPLEMENTED_METHODS = {"ols", "robust"}

_REQUIRED_COLUMNS = {"sensor_id", "sample_rh_pct", "sample_t_c", "pid_voltage_mv", "timestamp"}


def fit_covariate_model(
    df: pl.DataFrame,
    training_mask: pl.Series,
    method: str = "ols",
    min_rh_range_pct: float = 20.0,
) -> CovariateModel | None:
    """Fit how one sensor's voltage responds to ambient temperature and humidity."""

    """
    Fits `voltage ~ alpha + beta_rh*RH + gamma_t*T + delta_rh_t*(RH*T)` via
    ordinary least squares.

    `df` is expected to already be filtered to one sensor (this function
    doesn't group by sensor_id itself -- callers like run_ambient_baseline do
    that grouping and call this once per sensor). `training_mask` is a
    boolean pl.Series the same length as `df`, selecting which rows count as
    the "training window" (e.g. the first 30 minutes of a run, or -- for
    ambient baseline -- the whole collection window).
    """

    """
    "symbolic" (fitting through jaxsr.SymbolicRegressor + jaxsr.Constraints
    instead of plain regression) is a real, planned feature -- just not
    built yet. Raising NotImplementedError (rather than silently falling
    back to "ols") means a caller who explicitly asked for it finds out
    immediately rather than getting a result they didn't ask for.
    """
    require_implemented_method(method, _IMPLEMENTED_METHODS, "fit_covariate_model")
    require_columns(df, _REQUIRED_COLUMNS, "fit_covariate_model")

    """
    Boolean-index the polars DataFrame down to just the training rows.
    `pl.DataFrame.filter(mask)` keeps only rows where `mask` is True --
    equivalent in spirit to `df[mask]` in pandas, but polars requires the
    explicit `.filter(...)` call rather than supporting `[]` indexing by a
    boolean series directly.
    """
    training_df = df.filter(training_mask)

    if training_df.height == 0:
        """
        No data at all for this sensor in the training window -- there's
        nothing to fit, but this isn't necessarily an error condition (e.g.
        a sensor that was legitimately excluded for this run), so we signal
        "no model" via None rather than raising.
        """
        return None

    sensor_ids = training_df["sensor_id"].unique().to_list()
    if len(sensor_ids) != 1:
        """
        This function fits ONE sensor at a time by design (see docstring); a
        caller accidentally passing multi-sensor data would otherwise
        silently fit a meaningless pooled model across different physical
        sensors, which is a much worse failure mode than an explicit error.
        """
        raise ValueError(
            f"fit_covariate_model expects data for exactly one sensor_id, got {sensor_ids}"
        )
    sensor_id = sensor_ids[0]

    rh = training_df["sample_rh_pct"].to_numpy()
    temp = training_df["sample_t_c"].to_numpy()
    voltage = training_df["pid_voltage_mv"].to_numpy()

    rh_range = float(np.nanmax(rh) - np.nanmin(rh)) if len(rh) else 0.0
    if training_df.height < _MIN_TRAINING_ROWS or rh_range < min_rh_range_pct:
        raise TrainingWindowInsufficientError(
            f"sensor {sensor_id}: training window has {training_df.height} rows "
            f"spanning {rh_range:.1f}% RH (need >= {_MIN_TRAINING_ROWS} rows and "
            f">= {min_rh_range_pct}% RH range to fit a reliable RH coefficient)."
        )

    """
    Build the design matrix for `voltage ~ alpha + beta_rh*RH + gamma_t*T +
    delta_rh_t*(RH*T)`. `np.column_stack` lays several 1-D arrays out side by
    side as columns of a single 2-D array -- here that gives us an
    (n_rows, 3) matrix of [RH, T, RH*T] for every training row.
    """
    interaction = rh * temp
    design = np.column_stack([rh, temp, interaction])

    """
    `sm.add_constant` prepends a column of all-1.0s -- that's what makes the
    fitted "coefficient" on it become the intercept (alpha) rather than
    forcing the fitted line through the origin.
    """
    design = sm.add_constant(design)

    if method == "ols":
        """
        `sm.OLS(y, X).fit()` is statsmodels' ordinary-least-squares
        regression: it solves for the coefficient vector that minimizes
        squared error between `design @ coefficients` and `voltage`, and
        also computes standard errors/covariance/R^2 as a side effect --
        all of which we use below rather than re-deriving them by hand.
        """
        result = sm.OLS(voltage, design, missing="drop").fit()
        alpha, beta_rh, gamma_t, delta_rh_t = result.params
        covariance = np.asarray(result.cov_params())
        r_squared = float(result.rsquared)
    else:
        """
        `sm.RLM` (robust linear model, Huber norm by default) downweights
        the influence of outlier readings rather than letting a single bad
        sample pull the fit the way plain least-squares would. It doesn't
        expose `.rsquared`/`.cov_params()` the same way OLS does, so both
        are computed by hand here from the fitted line, keeping
        `CovariateModel`'s fields meaningful regardless of which method
        produced them.
        """
        result = sm.RLM(voltage, design, missing="drop").fit()
        alpha, beta_rh, gamma_t, delta_rh_t = result.params
        covariance = np.asarray(result.bcov_scaled)
        predicted = design @ result.params
        ss_res = float(np.sum((voltage - predicted) ** 2))
        ss_tot = float(np.sum((voltage - np.mean(voltage)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    training_timestamps = training_df["timestamp"].to_list()

    return CovariateModel(
        sensor_id=sensor_id,
        method=method,
        alpha=float(alpha),
        beta_rh=float(beta_rh),
        gamma_t=float(gamma_t),
        delta_rh_t=float(delta_rh_t),
        covariance=covariance,
        symbolic_regressor=None,
        training_window=(min(training_timestamps), max(training_timestamps)),
        r_squared=r_squared,
    )


def apply_covariate_correction(df: pl.DataFrame, models: dict[str, CovariateModel]) -> pl.DataFrame:
    """Subtract each sensor's predicted ambient-only voltage from its raw reading."""

    """
    For a sensor with a fitted `CovariateModel`, the predicted baseline at a
    given RH/T is `alpha + beta_rh*RH + gamma_t*T + delta_rh_t*(RH*T)` -- the
    voltage the sensor would read from ambient temperature/humidity alone,
    with zero VOC present. Subtracting that prediction from the actual
    reading leaves (ideally) just the VOC-driven signal: in clean air the
    corrected value should sit near 0 regardless of RH/T, since the ambient
    contribution has been removed.

    Sensors with no entry in `models` (e.g. one that failed
    TrainingWindowInsufficientError during ambient baseline) are passed
    through with a null correction rather than dropped -- losing the
    covariate correction for one sensor shouldn't discard its raw data.
    """

    require_columns(df, {"sensor_id", "sample_rh_pct", "sample_t_c", "pid_voltage_mv"}, "apply_covariate_correction")

    corrected_frames = []
    for (sensor_id,), sensor_df in df.partition_by("sensor_id", as_dict=True).items():
        model = models.get(sensor_id)
        if model is None or model.method not in ("ols", "robust"):
            """
            No fitted model for this sensor (or it's a "symbolic" model,
            which this function doesn't yet know how to evaluate -- that's
            deferred until jaxsr.SymbolicRegressor integration is built) --
            pass the raw voltage through unchanged rather than erroring.
            `ols` and `robust` models share the same linear
            alpha/beta_rh/gamma_t/delta_rh_t shape, so both are evaluated
            the same way below -- this predicate previously only matched
            "ols" exactly, which meant a fitted "robust" model would
            silently never actually get applied here.
            """
            corrected = sensor_df["pid_voltage_mv"]
        else:
            rh = sensor_df["sample_rh_pct"]
            temp = sensor_df["sample_t_c"]
            predicted_baseline = (
                model.alpha + model.beta_rh * rh + model.gamma_t * temp + model.delta_rh_t * (rh * temp)
            )
            corrected = sensor_df["pid_voltage_mv"] - predicted_baseline
        corrected_frames.append(
            sensor_df.with_columns(corrected.alias("pid_voltage_mv_covariate_corrected"))
        )

    return pl.concat(corrected_frames)


def persist_covariate_models(
    models: dict[str, CovariateModel],
    ambient_baseline_run_id: str,
    out_dir: Path,
) -> Path:
    """Save every sensor's fitted ambient-covariate model to disk, so a
    later caller (e.g. discover_led_response_dynamics) can apply the same
    correction without needing to re-fit it from a fresh blank/ambient
    recording every time."""

    """
    Only `alpha`/`beta_rh`/`gamma_t`/`delta_rh_t`/`method`/`r_squared`/
    `training_window` are written -- `covariance` (a numpy array) and
    `symbolic_regressor` (a live, unpicklable jaxsr object) aren't
    JSON/YAML-safe, and `apply_covariate_correction` doesn't need either
    of them to actually apply a correction. Mirrors
    `calibration/apply.py`'s persist_calibration/load_calibration split
    and its atomic write-to-temp-then-replace pattern.
    """
    if not models:
        raise ValueError("persist_covariate_models: models is empty, nothing to write")

    unsupported = [m.sensor_id for m in models.values() if m.method not in ("ols", "robust")]
    if unsupported:
        raise ValueError(
            f"persist_covariate_models: sensor(s) {sorted(unsupported)} use a 'symbolic' "
            "model -- only 'ols'/'robust' models (real alpha/beta_rh/gamma_t/delta_rh_t "
            "coefficients, not a live SymbolicRegressor) can be persisted here"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "sensor_id": model.sensor_id,
            "method": model.method,
            "alpha": model.alpha,
            "beta_rh": model.beta_rh,
            "gamma_t": model.gamma_t,
            "delta_rh_t": model.delta_rh_t,
            "r_squared": model.r_squared,
            "training_window_start": model.training_window[0].isoformat(),
            "training_window_end": model.training_window[1].isoformat(),
        }
        for model in models.values()
    ]

    yaml_path = out_dir / f"{ambient_baseline_run_id}.yaml"
    tmp_yaml_path = out_dir / f".{ambient_baseline_run_id}.yaml.tmp"
    tmp_yaml_path.write_text(yaml.safe_dump({"models": rows}, sort_keys=False), encoding="utf-8")
    tmp_yaml_path.replace(yaml_path)
    return yaml_path


def load_covariate_models(ambient_baseline_run_id: str, data_dir: Path) -> dict[str, CovariateModel]:
    """Load a previously-persisted ambient-covariate correction run back
    from disk -- the reverse of persist_covariate_models. `covariance`
    and `symbolic_regressor` come back `None` (never persisted; not
    needed by apply_covariate_correction)."""
    yaml_path = Path(data_dir) / f"{ambient_baseline_run_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No ambient-covariate YAML file at {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        sidecar = yaml.safe_load(f)

    models: dict[str, CovariateModel] = {}
    for row in sidecar["models"]:
        models[row["sensor_id"]] = CovariateModel(
            sensor_id=row["sensor_id"],
            method=row["method"],
            alpha=row["alpha"],
            beta_rh=row["beta_rh"],
            gamma_t=row["gamma_t"],
            delta_rh_t=row["delta_rh_t"],
            covariance=None,
            symbolic_regressor=None,
            training_window=(
                dt.datetime.fromisoformat(row["training_window_start"]),
                dt.datetime.fromisoformat(row["training_window_end"]),
            ),
            r_squared=row["r_squared"],
        )
    return models
