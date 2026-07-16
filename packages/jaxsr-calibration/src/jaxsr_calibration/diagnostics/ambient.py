"""Ambient baseline diagnostic: characterize each sensor's response to room
temperature/humidity by fitting a CovariateModel over an overnight
ambient-air collection window (spec §21).
"""

from __future__ import annotations

import polars as pl

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.diagnostics.models import AmbientBaselineResult
from jaxsr_calibration.processing.covariate import fit_covariate_model
from jaxsr_calibration.processing.errors import TrainingWindowInsufficientError


def run_ambient_baseline(
    duration_h: int,
    method: str = "ols",
    *,
    readings: pl.DataFrame | None = None,
) -> AmbientBaselineResult:
    """Fit one CovariateModel per sensor from an ambient-air collection
    window."""

    """
    Same "no live acquisition yet" caveat as run_fleet_zero: `duration_h`
    documents the intended collection length, but this function analyzes
    already-collected data passed via `readings` rather than acquiring it
    itself (see jaxsr_calibration.errors for why).

    Any single sensor whose data doesn't have enough RH variation to fit
    reliably (fit_covariate_model raising TrainingWindowInsufficientError)
    is skipped rather than failing the whole call -- one
    under-characterized sensor shouldn't prevent reporting results for the
    rest of the fleet.
    """

    if readings is None:
        raise LiveAcquisitionNotAvailableError(
            "run_ambient_baseline has no live-acquisition backend yet; pass "
            "readings=<a DataFrame of already-collected ambient-air data> instead."
        )

    covariate_models = {}
    r_squared_per_sensor: dict[str, float] = {}

    for (sensor_id,), sensor_df in readings.partition_by("sensor_id", as_dict=True).items():
        """
        The whole window counts as "training data" for ambient baseline
        (unlike per-experiment covariate correction, which trains on only
        a sub-window of a run -- see fit_covariate_model's technical
        block).
        """
        mask = pl.Series([True] * sensor_df.height)

        try:
            model = fit_covariate_model(sensor_df, mask, method=method)
        except TrainingWindowInsufficientError:
            """
            Skip this sensor; see technical block above for why we don't
            propagate.
            """
            continue

        if model is None:
            continue

        covariate_models[sensor_id] = model
        r_squared_per_sensor[sensor_id] = model.r_squared

    return AmbientBaselineResult(
        covariate_models=covariate_models,
        r_squared_per_sensor=r_squared_per_sensor,
    )
