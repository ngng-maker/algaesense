"""Feature extraction and two different bridges into existing JAXSR, which
answer two different questions.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl

from jaxsr_calibration.logging_.schema import ExperimentMeta
from jaxsr_calibration.processing.errors import (
    MixedCalibrationCompoundError,
    TargetContainsNaNError,
)


"""
- `load_features_for_jaxsr` (spec Part X, §34) compares ONE summarized
  outcome per experiment ACROSS MANY experiments run at different
  conditions -- "how does average VOC output depend on light level?" -- by
  feeding `jaxsr.SymbolicRegressor.fit`. It necessarily averages away
  everything that happened *within* one run to get that one summary point.

- `load_timeseries_for_jaxsr` (new; not in the original spec) preserves the
  FULL, un-averaged trajectory from a SINGLE experiment run, for
  `jaxsr.discover_dynamics` -- "what's the actual shape of the rise over
  time, for this one run?" This is what you want when the trend/behavior
  *within* one run is the thing you care about, not a single averaged
  number.

Both expect their input `timeseries` to already carry per-row calibrated
VOC concentration columns (`ppm_asgas`, `ppm_asgas_stderr`) -- produced
upstream by looping
`jaxsr_calibration.calibration.apply.apply_calibration` over each sensor
and attaching the result back onto the timeseries -- plus, for campaigns
with a camera, the `biomass_signal_arb`/`biomass_reading_age_s` columns
`jaxsr_calibration.processing.fusion.fuse_multirate` already adds. Neither
function re-runs calibration or fusion itself.
"""

"""
Bumped from the spec's original 1 to add the biomass columns.
"""
FEATURES_SCHEMA_VERSION = 2


def extract_features(
    timeseries: pl.DataFrame,
    metadata: ExperimentMeta,
    analysis_window: tuple[dt.datetime, dt.datetime],
) -> pl.DataFrame:
    """Collapse a per-timestamp timeseries into one row per (experiment_id,
    reactor_id, sensor_id)."""

    """
    Input must already be fused, calibrated, and covariate-corrected.
    Output matches spec §19's schema, extended with this project's biomass
    columns.
    """

    start, end = analysis_window
    windowed = timeseries.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") <= end))
    if windowed.height == 0:
        raise ValueError(
            f"extract_features: no rows fall within analysis_window {analysis_window}"
        )

    has_biomass = "biomass_signal_arb" in windowed.columns
    has_iso_equiv = "ppm_iso_equiv" in windowed.columns

    rows: list[dict] = []

    """
    Grouping by all three keys at once (rather than nesting three separate
    partition_by calls) gives us exactly the "(experiment, reactor,
    sensor)" triples spec §19 wants, one dict entry per unique combination.
    """
    groups = windowed.partition_by(["experiment_id", "reactor_id", "sensor_id"], as_dict=True)

    for (experiment_id, reactor_id, sensor_id), group in groups.items():
        group = group.sort("timestamp")

        row: dict = {
            "experiment_id": experiment_id,
            "campaign_id": metadata.campaign_id,
            "reactor_id": reactor_id,
            "sensor_id": sensor_id,
            "analysis_window_start": start,
            "analysis_window_end": end,
        }

        """
        Controllable variables: whatever this reactor's condition dict in
        the experiment metadata contains (e.g. par_umol_m2_s,
        reactor_temp_c) -- flattened directly onto the row, per-campaign
        column set, exactly as spec §19 describes ("user-defined per
        campaign").
        """
        row.update(metadata.conditions.get(reactor_id, {}))

        row["mean_sample_t_c"] = float(group["sample_t_c"].mean())
        row["mean_sample_rh_pct"] = float(group["sample_rh_pct"].mean())
        row["lamp_hours"] = float(group["lamp_hours"][-1])

        ppm = group["ppm_asgas"].to_numpy()
        ppm_stderr = group["ppm_asgas_stderr"].to_numpy()
        n = len(ppm)
        row["mean_voc_ppm_asgas"] = float(np.mean(ppm))

        """
        Standard error of a MEAN of n independent, unequally-uncertain
        readings: Var(mean) = mean(sigma_i^2) / n -- the standard
        "propagate then average" delta-method result for averaging
        independent uncertain quantities.
        """
        row["mean_voc_ppm_asgas_stderr"] = float(np.sqrt(np.mean(ppm_stderr**2) / n))
        row["p95_voc_ppm_asgas"] = float(np.percentile(ppm, 95))

        elapsed_hours = np.array(
            [(t - group["timestamp"][0]).total_seconds() / 3600.0 for t in group["timestamp"]]
        )
        if len(np.unique(elapsed_hours)) >= 2:
            row["voc_slope_ppm_asgas_h"] = float(np.polyfit(elapsed_hours, ppm, 1)[0])
        else:
            row["voc_slope_ppm_asgas_h"] = 0.0

        if has_iso_equiv:
            ppm_iso = group["ppm_iso_equiv"].to_numpy()
            if np.any(np.isnan(ppm_iso)):
                row["mean_voc_ppm_iso_equiv"] = math.nan
                row["mean_voc_ppm_iso_equiv_stderr"] = math.nan
            else:
                iso_stderr = group["ppm_iso_equiv_stderr"].to_numpy()
                row["mean_voc_ppm_iso_equiv"] = float(np.mean(ppm_iso))
                row["mean_voc_ppm_iso_equiv_stderr"] = float(np.sqrt(np.mean(iso_stderr**2) / n))
        else:
            row["mean_voc_ppm_iso_equiv"] = math.nan
            row["mean_voc_ppm_iso_equiv_stderr"] = math.nan

        if has_biomass:
            """
            The LAST row in the window has the freshest (lowest-age)
            biomass pairing available -- use it as "the" biomass reading
            for this window, rather than e.g. averaging biomass_signal_arb
            across rows (which would just repeat the same value many times
            between camera captures and not add information).
            """
            row["biomass_signal_arb"] = float(group["biomass_signal_arb"][-1])
            row["biomass_reading_age_s"] = float(group["biomass_reading_age_s"][-1])
        else:
            row["biomass_signal_arb"] = math.nan
            row["biomass_reading_age_s"] = math.nan

        row["calibration_run_id"] = group["calibration_run_id"][0] if "calibration_run_id" in group.columns else None
        row["calibration_compound"] = group["calibration_compound"][0] if "calibration_compound" in group.columns else None
        row["calibration_response_factor"] = (
            float(group["calibration_response_factor"][0])
            if "calibration_response_factor" in group.columns and group["calibration_response_factor"][0] is not None
            else math.nan
        )
        row["features_schema_version"] = FEATURES_SCHEMA_VERSION

        rows.append(row)

    return pl.DataFrame(rows)


"""
Columns that are identifiers, provenance, or targets -- never
auto-selected as model input features by load_features_for_jaxsr below.
"""
_ID_COLUMNS = {"experiment_id", "campaign_id", "reactor_id", "sensor_id"}
_PROVENANCE_COLUMNS = {
    "analysis_window_start",
    "analysis_window_end",
    "calibration_run_id",
    "calibration_compound",
    "calibration_response_factor",
    "features_schema_version",
}
_TARGET_CANDIDATE_COLUMNS = {
    "mean_voc_ppm_asgas",
    "mean_voc_ppm_asgas_stderr",
    "mean_voc_ppm_iso_equiv",
    "mean_voc_ppm_iso_equiv_stderr",
    "p95_voc_ppm_asgas",
    "voc_slope_ppm_asgas_h",
}


def load_features_for_jaxsr(
    features_df: pl.DataFrame,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    include_categorical: bool = True,
    return_stderr: bool = False,
    allow_mixed: bool = False,
):
    """The bridge from this package's derived features into existing
    JAXSR: returns `(X, y, feature_names)`, or `(X, y, y_stderr,
    feature_names)` if `return_stderr=True`, ready for
    `jaxsr.SymbolicRegressor.fit`."""

    if "calibration_compound" in features_df.columns:
        compounds = features_df["calibration_compound"].unique().to_list()
        if len(compounds) > 1 and not allow_mixed:
            raise MixedCalibrationCompoundError(
                f"features_df contains rows calibrated against multiple compounds "
                f"({sorted(c for c in compounds if c is not None)}); pass allow_mixed=True "
                "to combine them anyway (only meaningful via the *_iso_equiv columns, "
                "and only if every compound has a known response factor)."
            )

    if target not in features_df.columns:
        raise ValueError(f"target column {target!r} not found in features_df")

    target_values = features_df[target].to_numpy()
    if np.any(np.isnan(target_values)):
        suggestion = (
            "mean_voc_ppm_asgas"
            if target != "mean_voc_ppm_asgas"
            else "a target with no missing values"
        )
        raise TargetContainsNaNError(
            f"target column {target!r} contains NaN values (likely from an unknown "
            f"response factor, if targeting an *_iso_equiv column). Try target={suggestion!r} instead."
        )

    if feature_columns is None:
        excluded = _ID_COLUMNS | _PROVENANCE_COLUMNS | _TARGET_CANDIDATE_COLUMNS | {target}

        """
        Auto-detect: every numeric column not in one of the excluded sets
        above -- covers both "controllable variables" (e.g. par_umol_m2_s,
        which only exists per-campaign so can't be named in advance) and
        "observed covariates" (mean_sample_t_c, mean_sample_rh_pct,
        lamp_hours, biomass_signal_arb) in one pass.
        """
        feature_columns = [
            name
            for name, dtype in zip(features_df.columns, features_df.dtypes)
            if name not in excluded and dtype.is_numeric()
        ]

    numeric_block = features_df.select(feature_columns).to_numpy()
    feature_names = list(feature_columns)

    if include_categorical:
        """
        `to_dummies` one-hot-encodes each named categorical column into a
        set of 0/1 indicator columns (one per distinct value) -- e.g.
        sensor_id="PID01"/"PID02" becomes two columns
        "sensor_id_PID01"/"sensor_id_PID02" -- the standard way to hand a
        categorical identity to a model that only accepts numeric input.
        """
        categorical_cols = [c for c in ("sensor_id", "reactor_id") if c in features_df.columns]
        if categorical_cols:
            dummies = features_df.select(categorical_cols).to_dummies()
            numeric_block = np.hstack([numeric_block, dummies.to_numpy()])
            feature_names.extend(dummies.columns)

    X = numeric_block.astype(float)
    y = target_values.astype(float)

    if return_stderr:
        stderr_col = f"{target}_stderr"
        if stderr_col in features_df.columns:
            y_stderr = features_df[stderr_col].to_numpy().astype(float)
        else:
            y_stderr = np.full_like(y, math.nan)
        return X, y, y_stderr, feature_names

    return X, y, feature_names


"""
Default state variables for load_timeseries_for_jaxsr -- the things whose
TREND OVER TIME (not just their average) is usually the actual point of
interest for a single run: the VOC concentration, and biomass if a camera
is present. Controllable settings like PAR are deliberately NOT included
here BY DEFAULT -- within a run held at one static setpoint, PAR is
constant, so it has no within-run trend to discover; comparing how the
discovered dynamics change ACROSS experiments run at different settings is
a separate, human (or later, agent) analysis step, not something one
discover_dynamics call does alone.

The exception: a run driven by a time-varying control profile (a
ramp/sinusoid/step light schedule -- see
algaesense_edge.actuators.control_profiles) genuinely does vary PAR within
the run, so it's meaningful there. A caller with that kind of data passes
`state_columns=["ppm_asgas", "reactor_par_umol_m2_s"]` explicitly (see
algaesense_agent.mcp_pipeline.pipeline.discover_led_response_dynamics) --
this function itself needed no change to support that, since
`state_columns` was already a plain caller-supplied parameter.
"""
_DEFAULT_STATE_COLUMNS = ["ppm_asgas", "biomass_signal_arb"]


def load_timeseries_for_jaxsr(
    timeseries: pl.DataFrame,
    state_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare ONE experiment's full, un-averaged trajectory for
    `jaxsr.discover_dynamics(X, t, state_names=...)`."""

    """
    Returns `(X, t, state_names)`:
    - `X`: state variable values over time -- shape `(n_samples,)` if
      there's exactly one state column, `(n_samples, n_states)` if there's
      more than one (matching what `jaxsr.discover_dynamics` itself
      documents accepting).
    - `t`: elapsed seconds since this experiment's first reading (JAXSR
      needs plain numeric time, not timestamps).
    - `state_names`: the column names in `X`'s order.

    Unlike `load_features_for_jaxsr`, this function refuses data spanning
    more than one experiment/reactor/sensor -- mixing multiple runs
    together would produce a nonsensical trajectory (e.g. time jumping
    backwards, or averaging across runs that had different conditions),
    which defeats the entire point of preserving the within-run trend.
    """

    required = {"experiment_id", "reactor_id", "sensor_id", "timestamp"}
    missing = required - set(timeseries.columns)
    if missing:
        raise ValueError(
            f"load_timeseries_for_jaxsr: timeseries is missing required columns: {sorted(missing)}"
        )

    for id_column in ("experiment_id", "reactor_id", "sensor_id"):
        distinct_values = timeseries[id_column].unique().to_list()
        if len(distinct_values) != 1:
            raise ValueError(
                f"load_timeseries_for_jaxsr expects data from exactly one {id_column}, "
                f"got {distinct_values}. This function preserves a single experiment's own "
                "trajectory; to compare summarized outcomes across multiple experiments, "
                "use load_features_for_jaxsr instead."
            )

    sorted_timeseries = timeseries.sort("timestamp")

    if state_columns is None:
        state_columns = [c for c in _DEFAULT_STATE_COLUMNS if c in sorted_timeseries.columns]
        if not state_columns:
            raise ValueError(
                "load_timeseries_for_jaxsr: none of the default state columns "
                f"{_DEFAULT_STATE_COLUMNS} were found; pass state_columns explicitly."
            )

    X = sorted_timeseries.select(state_columns).to_numpy().astype(float)
    if X.shape[1] == 1:
        """
        `jaxsr.discover_dynamics`'s own docstring documents `(n_samples,)`
        as the expected shape for a single state variable (as opposed to
        `(n_samples, n_states)` for several) -- `.ravel()` flattens our
        (n_samples, 1) array down to that 1-D shape.
        """
        X = X.ravel()

    timestamps = sorted_timeseries["timestamp"].to_list()
    t0 = timestamps[0]
    t = np.array([(ts - t0).total_seconds() for ts in timestamps], dtype=float)

    return X, t, list(state_columns)
