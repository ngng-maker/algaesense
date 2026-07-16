"""Signal processing: covariate regression, common-mode subtraction, spectral
filtering, dual-rate VOC/camera fusion, and the bridge into JAXSR (spec
§12's public API list, extended with this project's fuse_multirate)."""

from jaxsr_calibration.processing.common_mode import subtract_common_mode
from jaxsr_calibration.processing.config import DiagnosticThresholds, PreprocessingConfig
from jaxsr_calibration.processing.covariate import (
    CovariateModel,
    apply_covariate_correction,
    fit_covariate_model,
)
from jaxsr_calibration.processing.features import (
    extract_features,
    load_features_for_jaxsr,
    load_timeseries_for_jaxsr,
)
from jaxsr_calibration.processing.fusion import fuse_multirate
from jaxsr_calibration.processing.spectral import lomb_scargle, notch_filter_known_artifacts

__all__ = [
    "fit_covariate_model",
    "apply_covariate_correction",
    "subtract_common_mode",
    "lomb_scargle",
    "notch_filter_known_artifacts",
    "fuse_multirate",
    "extract_features",
    "load_features_for_jaxsr",
    "load_timeseries_for_jaxsr",
    "CovariateModel",
    "PreprocessingConfig",
    "DiagnosticThresholds",
]
