"""Unit tests for jaxsr_calibration.processing.covariate.persist_covariate_models
/ load_covariate_models."""

from __future__ import annotations

import datetime as dt

import pytest

from jaxsr_calibration.processing.covariate import (
    CovariateModel,
    load_covariate_models,
    persist_covariate_models,
)


def _make_model(sensor_id: str = "PID01", method: str = "ols") -> CovariateModel:
    return CovariateModel(
        sensor_id=sensor_id,
        method=method,
        alpha=10.0,
        beta_rh=0.2,
        gamma_t=0.5,
        delta_rh_t=0.01,
        covariance=None,
        symbolic_regressor=None,
        training_window=(
            dt.datetime(2026, 7, 22, 5, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 7, 22, 6, 0, 0, tzinfo=dt.timezone.utc),
        ),
        r_squared=0.98,
    )


def test_persist_then_load_recovers_the_same_coefficients(tmp_path) -> None:
    models = {"PID01": _make_model("PID01"), "PID02": _make_model("PID02", method="robust")}
    persist_covariate_models(models, "ambient_run_01", tmp_path)

    loaded = load_covariate_models("ambient_run_01", tmp_path)

    assert set(loaded) == {"PID01", "PID02"}
    for sensor_id, model in models.items():
        recovered = loaded[sensor_id]
        assert recovered.sensor_id == model.sensor_id
        assert recovered.method == model.method
        assert recovered.alpha == pytest.approx(model.alpha)
        assert recovered.beta_rh == pytest.approx(model.beta_rh)
        assert recovered.gamma_t == pytest.approx(model.gamma_t)
        assert recovered.delta_rh_t == pytest.approx(model.delta_rh_t)
        assert recovered.r_squared == pytest.approx(model.r_squared)
        assert recovered.training_window == model.training_window
        # Deliberately not persisted -- not needed by apply_covariate_correction.
        assert recovered.covariance is None
        assert recovered.symbolic_regressor is None


def test_persist_covariate_models_rejects_empty_dict(tmp_path) -> None:
    with pytest.raises(ValueError, match="nothing to write"):
        persist_covariate_models({}, "ambient_run_01", tmp_path)


def test_persist_covariate_models_rejects_symbolic_method(tmp_path) -> None:
    model = _make_model(method="symbolic")
    with pytest.raises(ValueError, match="symbolic"):
        persist_covariate_models({"PID01": model}, "ambient_run_01", tmp_path)


def test_load_covariate_models_raises_clearly_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No ambient-covariate YAML file"):
        load_covariate_models("does_not_exist", tmp_path)


def test_persisted_model_applies_the_same_correction_as_the_original(tmp_path) -> None:
    """The real point of persisting this: a loaded model must actually
    behave the same as the original when fed into
    apply_covariate_correction, not just round-trip its own fields."""
    import polars as pl

    from jaxsr_calibration.processing.covariate import apply_covariate_correction

    model = _make_model("PID01")
    persist_covariate_models({"PID01": model}, "ambient_run_01", tmp_path)
    loaded = load_covariate_models("ambient_run_01", tmp_path)

    df = pl.DataFrame(
        {
            "sensor_id": ["PID01", "PID01"],
            "sample_rh_pct": [50.0, 60.0],
            "sample_t_c": [30.0, 32.0],
            "pid_voltage_mv": [100.0, 105.0],
        }
    )

    original_corrected = apply_covariate_correction(df, {"PID01": model})
    loaded_corrected = apply_covariate_correction(df, loaded)

    assert loaded_corrected["pid_voltage_mv_covariate_corrected"].to_list() == pytest.approx(
        original_corrected["pid_voltage_mv_covariate_corrected"].to_list()
    )
