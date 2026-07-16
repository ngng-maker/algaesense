"""Pydantic schemas for configs/preprocessing.yaml and
configs/diagnostic_thresholds.yaml.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


"""
PreprocessingConfig's shape is a literal transcription of the spec's own
`configs/preprocessing.yaml` example (§15) -- every section
(covariate_regression, common_mode, spectral, concentration, features) and
default value matches it. DiagnosticThresholds is not given a literal
example in the spec, so it is a reasonable minimal design, same caveat as
ReactorConfig in calibration/config.py.
"""


"""
`Literal[...]` is a type-hint that restricts a value to one of a fixed,
specific set of strings (not just "any string") -- pydantic enforces this
at validation time, so passing method="OLS" (wrong case) or method="lasso"
(not one of the allowed options) fails immediately with a clear error
instead of silently doing the wrong thing three pipeline stages later.
"""
CovariateMethod = Literal["ols", "robust", "symbolic"]
CommonModeMethod = Literal["median", "trimmed_mean"]
ExtrapolationPolicy = Literal["clip", "linear", "nan"]


class BasisConfig(BaseModel):
    """Only used when covariate_regression.method == "symbolic"."""

    """
    Describes the basis functions jaxsr.BasisLibrary should build (spec
    §15's `symbolic.basis` block).
    """

    polynomial_degree: int = 2

    """
    list[str] of transcendental function names, e.g. ["log", "exp"] --
    matched against whatever jaxsr.BasisLibrary.add_transcendental accepts.
    """
    transcendental: list[str] = Field(default_factory=lambda: ["log", "exp"])


class SymbolicCovariateConfig(BaseModel):
    """The `symbolic:` sub-block under covariate_regression -- only read
    when method="symbolic" is chosen; ignored otherwise."""

    max_terms: int = 4

    basis: BasisConfig = Field(default_factory=BasisConfig)


class CovariateRegressionConfig(BaseModel):
    method: CovariateMethod = "ols"

    """
    A free-form string describing which slice of the run to fit on, e.g.
    "first_30min" -- kept as a string rather than a fixed enum because the
    spec's own example uses a human-readable label like this rather than a
    numeric duration, and other labels ("last_2h_of_run" appears elsewhere)
    follow the same pattern.
    """
    training_window: str = "first_30min"

    min_rh_range_pct: float = 20.0

    """
    `SymbolicCovariateConfig | None = None`: this whole sub-block is
    optional and only meaningful if method="symbolic".
    """
    symbolic: SymbolicCovariateConfig | None = None


class CommonModeConfig(BaseModel):
    method: CommonModeMethod = "median"

    outlier_std_threshold: float = 3.0

    min_healthy_fraction: float = 0.75


class SpectralConfig(BaseModel):
    """configs/preprocessing.yaml's `spectral:` block."""

    """
    Frequencies (in Hz) of KNOWN recurring artifacts to notch-filter out,
    e.g. a pump's PWM cycle -- populated once you've identified them via
    jaxsr_calibration.processing.spectral.lomb_scargle.
    """
    known_artifact_freqs_hz: list[float] = Field(default_factory=list)

    notch_q: float = 30.0

    min_amplitude_to_flag: float = 0.05


class ConcentrationConfig(BaseModel):
    extrapolation_policy: ExtrapolationPolicy = "clip"


class FeaturesConfig(BaseModel):
    analysis_window: str = "last_2h_of_run"


class PreprocessingConfig(BaseModel):
    """configs/preprocessing.yaml, matching spec §15 exactly."""

    preprocessing_schema_version: int = 1

    covariate_regression: CovariateRegressionConfig = Field(
        default_factory=CovariateRegressionConfig
    )

    common_mode: CommonModeConfig = Field(default_factory=CommonModeConfig)

    spectral: SpectralConfig = Field(default_factory=SpectralConfig)

    concentration: ConcentrationConfig = Field(default_factory=ConcentrationConfig)

    features: FeaturesConfig = Field(default_factory=FeaturesConfig)


class FleetZeroThresholds(BaseModel):
    """Pass/fail limits for jaxsr_calibration.diagnostics.run_fleet_zero."""

    max_mean_mv: float = 5.0

    max_std_mv: float = 1.0

    """
    Drift over the clean-air window -- a sensor that's quiet (low std) but
    steadily climbing is still unhealthy, just in a way mean/std alone
    wouldn't catch.
    """
    max_abs_slope_mv_per_min: float = 0.05

    """
    A sensor is "SUSPECT" rather than an outright "FAIL" if it's over
    threshold but not by much; this multiplier defines "by much". E.g. with
    the default 2.0, a mean 1.5x over max_mean_mv is SUSPECT, and one 2.5x
    over is FAIL. Not a spec-mandated number -- see run_fleet_zero's own
    technical block for the full reasoning.
    """
    fail_multiplier: float = 2.0


class AmbientBaselineThresholds(BaseModel):
    min_r_squared: float = 0.6


class SwapPilotThresholds(BaseModel):
    max_sensor_variance_share: float = 0.30

    max_reactor_variance_share: float = 0.30


class DiagnosticThresholds(BaseModel):
    """configs/diagnostic_thresholds.yaml."""

    """
    Not given a literal example in the spec (unlike PreprocessingConfig
    above) -- this groups one threshold sub-model per diagnostic already
    named in spec Part VII, using the pass/fail numbers already mentioned
    in the spec's own prose (e.g. "R^2 below 0.6 means...", "variance
    share... should be < 30%").
    """

    fleet_zero: FleetZeroThresholds = Field(default_factory=FleetZeroThresholds)

    ambient_baseline: AmbientBaselineThresholds = Field(
        default_factory=AmbientBaselineThresholds
    )

    swap_pilot: SwapPilotThresholds = Field(default_factory=SwapPilotThresholds)
