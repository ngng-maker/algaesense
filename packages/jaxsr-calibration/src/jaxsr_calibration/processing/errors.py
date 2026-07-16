"""Custom exceptions raised by jaxsr_calibration.processing (spec §34)."""

from __future__ import annotations


class TrainingWindowInsufficientError(RuntimeError):
    """Raised when the chosen training window doesn't have enough variation
    to fit a reliable covariate model."""

    """
    Raised by fit_covariate_model when the chosen training window doesn't
    contain enough RH variation (spec's `min_rh_range_pct`) to reliably
    estimate a humidity-sensitivity coefficient -- fitting anyway would give
    a number that looks precise but is actually unconstrained by the data.
    """


class CommonModeInsufficientFleetError(RuntimeError):
    """Raised when too few sensors agree to trust a common-mode estimate."""

    """
    Raised by subtract_common_mode when, at some timestamp, fewer than
    `min_healthy_fraction` of the fleet survive outlier exclusion -- there
    aren't enough agreeing sensors left to trust a "common mode" estimate.
    """


class TargetContainsNaNError(ValueError):
    """Raised when the requested JAXSR target column has missing values."""

    """
    Raised by load_features_for_jaxsr when the requested target column
    contains NaN values with no usable fallback (e.g. requesting
    mean_voc_ppm_iso_equiv when some rows' calibration had no known
    response factor).
    """


class MixedCalibrationCompoundError(ValueError):
    """Raised when features from more than one calibration compound are
    combined without explicit permission."""

    """
    Raised by load_features_for_jaxsr when the input features_df contains
    rows calibrated against more than one distinct compound and
    `allow_mixed=True` wasn't passed -- comparing raw ppm_asgas values
    across different compounds is not meaningful without RF correction.
    """
