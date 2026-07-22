"""Read/compute-only bridge from derived experiment features to JAXSR fits
and next-experiment suggestions.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import jaxsr
import numpy as np
import polars as pl

from jaxsr_calibration.calibration.apply import apply_calibration
from jaxsr_calibration.processing.features import load_features_for_jaxsr, load_timeseries_for_jaxsr

from algaesense_agent.labwiki.wiki import query_labwiki
from algaesense_agent.raw_readers import load_raw_voc_readings


"""
This module is the actual logic behind the `mcp_pipeline` MCP server --
kept separate from server.py (the thin FastMCP tool-registration layer) so
every function here is a plain, directly-testable Python call with no MCP
protocol involved, same split already used throughout jaxsr_calibration
(e.g. calibration/apply.py vs cli.py). Nothing in this module has a side
effect: it only reads already-written Parquet files and returns computed
results.
"""


class CampaignNotFoundError(FileNotFoundError):
    """Raised when a campaign has no derived-features Parquet files yet."""


def load_campaign_features(campaign_id: str, data_dir: Path) -> pl.DataFrame:
    """Load and concatenate every experiment's derived-features file for
    one campaign."""

    """
    Per the derived-feature layout
    (`data/derived/features/{campaign_id}/{experiment_id}.parquet`), one
    file per experiment run under that campaign -- concatenating them is
    exactly "compare summarized outcomes across many experiments", the
    `load_features_for_jaxsr` use case.
    """
    campaign_dir = Path(data_dir) / "derived" / "features" / campaign_id
    paths = sorted(campaign_dir.glob("*.parquet")) if campaign_dir.exists() else []

    if not paths:
        raise CampaignNotFoundError(
            f"No derived-features Parquet files found for campaign {campaign_id!r} "
            f"under {campaign_dir}"
        )

    return pl.concat([pl.read_parquet(p) for p in paths])


@dataclass
class FitResult:
    """A fitted symbolic model plus enough context to run active learning
    against it later."""

    expression: str
    coefficients: list[float]
    selected_features: list[str]
    complexity: int
    metrics: dict
    feature_names: list[str]
    feature_bounds: list[tuple[float, float]]


def default_basis_library(n_features: int) -> jaxsr.BasisLibrary:
    """Build a generic, no-assumptions basis library for a first-pass fit."""

    """
    Constant + linear + degree-2 polynomial terms is a reasonable
    general-purpose starting library for "we don't yet know the functional
    form" -- the same spirit as jaxsr_calibration.processing.config's
    BasisConfig default (polynomial_degree=2), but this is a fresh,
    independent library rather than reused from there: that one is scoped
    to per-experiment covariate correction, this one is campaign-level
    design-of-experiments, a different question entirely.
    """
    return (
        jaxsr.BasisLibrary(n_features=n_features)
        .add_constant()
        .add_linear()
        .add_polynomials(max_degree=2)
    )


def fit_symbolic_model(
    campaign_id: str,
    data_dir: Path,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    max_terms: int = 5,
    include_categorical: bool = True,
) -> FitResult:
    """Fit a symbolic-regression model over one campaign's experiments."""

    features_df = load_campaign_features(campaign_id, data_dir)

    X, y, feature_names = load_features_for_jaxsr(
        features_df,
        target=target,
        feature_columns=feature_columns,
        include_categorical=include_categorical,
    )

    library = default_basis_library(n_features=X.shape[1])
    model = jaxsr.SymbolicRegressor(basis_library=library, max_terms=max_terms)
    model.fit(X, y)

    """
    Bounds are taken from the observed data's own min/max per feature --
    the natural "safe to suggest within" range for the active-learning step
    below, rather than an arbitrarily chosen range that might not match
    what this reactor/sensor combination has ever actually run at.
    """
    feature_bounds = [(float(np.min(X[:, i])), float(np.max(X[:, i]))) for i in range(X.shape[1])]

    return FitResult(
        expression=model.expression_,
        coefficients=[float(c) for c in model.coefficients_],
        selected_features=list(model.selected_features_),
        complexity=int(model.complexity_),
        metrics={k: float(v) for k, v in model.metrics_.items()},
        feature_names=feature_names,
        feature_bounds=feature_bounds,
    )


@dataclass
class SuggestionResult:
    """Next experimental conditions JAXSR's active learner recommends
    trying, plus the fit they were derived from."""

    points: list[dict[str, float]]
    scores: list[float]
    acquisition: str
    fit: FitResult


def suggest_next_experiments(
    campaign_id: str,
    data_dir: Path,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    n_points: int = 3,
    kappa: float = 2.0,
    max_terms: int = 5,
) -> SuggestionResult:
    """Fit a model over one campaign, then suggest the next `n_points`
    experimental conditions to try."""

    """
    `include_categorical=False` here specifically: active learning needs a
    continuous, bounded input space to generate candidate points in --
    a one-hot categorical dummy column (e.g. sensor_id_PID01) has no
    meaningful "in-between value" to suggest, unlike a numeric condition
    like PAR or temperature. This is a separate fit from
    `fit_symbolic_model`'s general-purpose one (which is fine to include
    categoricals in), not a reuse of the same result.
    """
    fit = fit_symbolic_model(
        campaign_id,
        data_dir,
        target=target,
        feature_columns=feature_columns,
        max_terms=max_terms,
        include_categorical=False,
    )

    features_df = load_campaign_features(campaign_id, data_dir)
    X, y, _ = load_features_for_jaxsr(
        features_df, target=target, feature_columns=feature_columns, include_categorical=False
    )

    library = default_basis_library(n_features=X.shape[1])
    model = jaxsr.SymbolicRegressor(basis_library=library, max_terms=max_terms)
    model.fit(X, y)

    learner = jaxsr.ActiveLearner(
        model=model,
        bounds=fit.feature_bounds,
        acquisition=jaxsr.UCB(kappa=kappa),
    )
    result = learner.suggest(n_points=n_points)

    """
    `result.points` is a `(n_points, n_features)` array in the same
    feature order as `fit.feature_names` -- zipping each row against that
    name list turns it into a self-describing dict (e.g. {"par_umol_m2_s":
    250.0, "reactor_temp_c": 30.0}) instead of a bare positional array,
    which is what an MCP tool caller (and a human reading the agent's
    response in Slack) actually needs to act on it.
    """
    points = [
        dict(zip(fit.feature_names, (float(v) for v in row)))
        for row in np.asarray(result.points)
    ]

    return SuggestionResult(
        points=points,
        scores=[float(s) for s in result.scores],
        acquisition=result.acquisition,
        fit=fit,
    )


@dataclass
class LabwikiFinding:
    """One matched page's full content, not just the lines `query_labwiki`
    happened to filter on. A page's most useful content (an operator's
    note, a fit expression) is often on a DIFFERENT line than the
    condition name that made it match -- e.g. a summary page's "Campaign:
    [[camp_01]]" line and its "## Notes" section are both real, useful
    parts of the same page, but only one of them literally contains the
    search term. Returning the whole page, not just the matching lines,
    is what actually makes "relevant past findings" useful here."""

    path: Path
    matching_lines: list[str]
    full_content: str


@dataclass
class LabwikiContext:
    """Every labwiki page that mentioned one topic, surfaced alongside a
    JAXSR suggestion so a human deciding what to try next sees relevant
    prior findings without a separate query_labwiki_topic call."""

    topic: str
    findings: list[LabwikiFinding]


@dataclass
class SuggestionWithContext:
    """A JAXSR active-learning suggestion, plus whatever the labwiki
    already says about the conditions/target involved."""

    suggestion: SuggestionResult
    labwiki_context: list[LabwikiContext]


def suggest_next_experiments_with_context(
    campaign_id: str,
    data_dir: Path,
    wiki_root: Path,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    n_points: int = 3,
    kappa: float = 2.0,
    max_terms: int = 5,
    extra_topics: list[str] | None = None,
) -> SuggestionWithContext:
    """`suggest_next_experiments`, plus a labwiki lookup for each
    condition/target JAXSR's fit actually used -- surfacing relevant past
    findings (operator notes, prior fit expressions, prior active-learning
    proposals) alongside the new quantitative suggestion.

    This does NOT feed labwiki content into JAXSR itself -- `jaxsr`'s
    active learner is a numerical optimizer with no notion of free-text
    knowledge, and nothing about its actual math changes here. What this
    adds is a query_labwiki_topic lookup, run automatically for the
    campaign and every feature/target name the fit used, so a human (or
    the agent, before relaying the suggestion) sees relevant qualitative
    history in the same response instead of needing a separate lookup --
    the same "read the compiled knowledge directly" idea query_labwiki_topic
    already uses on its own, just triggered automatically here rather than
    requiring someone to think to ask for it.
    """
    suggestion = suggest_next_experiments(
        campaign_id,
        data_dir,
        target=target,
        feature_columns=feature_columns,
        n_points=n_points,
        kappa=kappa,
        max_terms=max_terms,
    )

    """
    `dict.fromkeys(...)` dedupes while preserving order -- the campaign_id
    and target are always searched, plus every feature name the fit
    actually selected (the concrete conditions this suggestion is about),
    plus any caller-supplied extras (e.g. an entity ID not captured by
    the fit's own feature names).
    """
    topics = list(dict.fromkeys([campaign_id, target, *suggestion.fit.feature_names, *(extra_topics or [])]))

    labwiki_context = [
        LabwikiContext(topic=topic, findings=findings)
        for topic in topics
        if (findings := _labwiki_findings_for_topic(campaign_id, topic, wiki_root))
    ]

    return SuggestionWithContext(suggestion=suggestion, labwiki_context=labwiki_context)


def _labwiki_findings_for_topic(campaign_id: str, topic: str, wiki_root: Path) -> list[LabwikiFinding]:
    matches = query_labwiki(campaign_id, topic, wiki_root)
    return [
        LabwikiFinding(path=match.path, matching_lines=match.matching_lines, full_content=match.path.read_text(encoding="utf-8"))
        for match in matches
    ]


@dataclass
class DynamicsDiscoveryResult:
    """A discovered ODE per state variable, fit over one experiment's
    real, un-averaged VOC trajectory (see discover_led_response_dynamics).
    """

    state_names: list[str]
    n_samples: int
    equations: dict[str, str]
    metrics: dict[str, dict[str, float]]
    coefficients: dict[str, list[float]]
    selected_features: dict[str, list[str]]


"""
`jaxsr.discover_dynamics`'s own DynamicsResult holds a raw numpy array
(`derivatives`) and live, unpicklable `SymbolicRegressor` objects
(`models`) -- neither JSON-safe, same problem `mcp_diagnostics/server.py`
already solved for `CovariateModel`. `DynamicsDiscoveryResult` above is a
hand-shaped, JSON-safe replacement: `equations`/`metrics` are copied
straight through (already plain), `coefficients`/`selected_features` are
extracted per state from the live `SymbolicRegressor` objects, and the
raw `derivatives` array plus the live models themselves are deliberately
not exposed -- internal detail, not needed by an MCP tool caller.
"""


def discover_led_response_dynamics(
    experiment_id: str,
    reactor_id: str,
    sensor_id: str,
    calibration_run_id: str,
    data_dir: Path,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    max_terms: int = 5,
    derivative_method: str = "finite_difference",
) -> DynamicsDiscoveryResult:
    """Feed one experiment's real, per-second VOC trajectory -- with the
    LED's actually-applied PAR as a second state variable -- into
    `jaxsr.discover_dynamics`, to find how light dynamically drives the
    VOC response, not just its static level."""

    """
    Meant to be run over data collected during a control-profile run (a
    ramp/sinusoid/step light schedule -- see
    algaesense_edge.actuators.control_profiles): a static PAR setpoint
    gives PAR no within-run trend to discover anything from.
    `jaxsr.discover_dynamics`'s default basis library already includes
    polynomial (degree 3) and pairwise-interaction terms across every
    state variable passed in, so PAR x VOC interaction terms are already
    candidates here with no custom BasisLibrary needed.
    """

    """
    The raw `timestamp` column is always tz-aware UTC (VOC_RAW_SCHEMA uses
    `pa.timestamp("ns", tz="UTC")` specifically to rule out naive-timestamp
    ambiguity -- see that schema's own module docstring). A naive
    `since`/`until` here is genuinely ambiguous (UTC? the caller's local
    time? something else?), and guessing UTC on the caller's behalf could
    silently filter the wrong window -- so this raises a clear error
    instead of guessing.
    """
    for name, value in (("since", since), ("until", until)):
        if value is not None and value.tzinfo is None:
            raise ValueError(
                f"discover_led_response_dynamics: {name} must be timezone-aware "
                f"(the raw timestamp column is always tz-aware UTC) -- got a naive "
                f"datetime {value!r}, which is ambiguous about what timezone was meant."
            )

    readings = load_raw_voc_readings(data_dir, experiment_id)
    readings = readings.filter(
        (pl.col("reactor_id") == reactor_id) & (pl.col("sensor_id") == sensor_id)
    )
    if since is not None:
        readings = readings.filter(pl.col("timestamp") >= since)
    if until is not None:
        readings = readings.filter(pl.col("timestamp") <= until)

    if readings.height == 0:
        raise ValueError(
            f"No raw VOC readings found for reactor {reactor_id!r}, sensor {sensor_id!r} "
            f"in experiment {experiment_id!r} within the requested window."
        )

    """
    A clear, specific error here (rather than letting a downstream
    numpy/jaxsr error surface) for the two real ways this column can be
    empty: the experiment predates AcquisitionService recording
    actually-applied PAR into each row, or the LED was simply never
    actuated during this window.
    """
    par_null_count = readings["reactor_par_umol_m2_s"].null_count()
    if par_null_count == readings.height:
        raise ValueError(
            f"reactor_par_umol_m2_s is entirely null for reactor {reactor_id!r} in "
            f"experiment {experiment_id!r} -- either this experiment predates PAR "
            "recording being wired up, or the LED was never actuated during this "
            "window. Nothing to discover a light-response equation from."
        )
    """
    A PARTIALLY-null column (some rows recorded, some not) is a distinct,
    equally real gap: it happens when PAR recording started mid-experiment
    (a service restart, or the fix landing partway through a long-running
    experiment), and is worse than all-null in one way -- it wouldn't be
    caught by the all-null check above, and would silently feed a
    mixed-validity state variable into jaxsr.discover_dynamics, which has
    no way to know some of those PAR values are fabricated nulls rather
    than real recorded data.
    """
    if par_null_count > 0:
        raise ValueError(
            f"reactor_par_umol_m2_s is null for {par_null_count} of {readings.height} rows "
            f"for reactor {reactor_id!r} in experiment {experiment_id!r} -- likely because "
            "PAR recording started partway through this window (a service restart, or this "
            "experiment straddling when PAR recording was first wired up). Narrow the "
            "since/until window to a range where every row has a recorded PAR value."
        )

    """
    `apply_calibration`'s `data_dir` is the exact directory holding
    `{calibration_run_id}.parquet`/`.yaml`, not the top-level data_dir --
    matching the one convention already established for this
    (mcp_calibration/server.py's finish_standard_addition_session).
    """
    calibration_dir = data_dir / "derived" / "calibrations" / "standard_addition"
    ppm, ppm_stderr, _ = apply_calibration(
        readings["pid_voltage_mv"],
        sensor_id,
        readings["sample_t_c"],
        readings["sample_rh_pct"],
        calibration_run_id,
        data_dir=calibration_dir,
    )
    readings = readings.with_columns(ppm.alias("ppm_asgas"), ppm_stderr.alias("ppm_asgas_stderr"))

    X, t, state_names = load_timeseries_for_jaxsr(
        readings, state_columns=["ppm_asgas", "reactor_par_umol_m2_s"]
    )

    result = jaxsr.discover_dynamics(
        X, t, state_names=state_names, max_terms=max_terms, derivative_method=derivative_method
    )

    return DynamicsDiscoveryResult(
        state_names=state_names,
        n_samples=len(t),
        equations=result.equations,
        metrics=result.metrics,
        coefficients={name: [float(c) for c in model.coefficients_] for name, model in result.models.items()},
        selected_features={name: list(model.selected_features_) for name, model in result.models.items()},
    )
