"""Unit tests for jaxsr_calibration.processing.features (extract_features and
load_features_for_jaxsr), tested with a hand-built already-calibrated
timeseries so each function's own logic is isolated from the rest of the
pipeline (which is instead exercised together in the Milestone 4 end-to-end
DoD test)."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.logging_.schema import ExperimentMeta
from jaxsr_calibration.processing.errors import MixedCalibrationCompoundError, TargetContainsNaNError
from jaxsr_calibration.processing.features import (
    extract_features,
    load_features_for_jaxsr,
    load_timeseries_for_jaxsr,
)

_START = dt.datetime(2026, 7, 22, 8, 0, 0, tzinfo=dt.timezone.utc)


def _meta(**conditions_per_reactor: dict) -> ExperimentMeta:
    return ExperimentMeta(
        experiment_id="exp_test",
        started_at=_START,
        operator="tester",
        campaign_id="campaign_test",
        conditions=conditions_per_reactor,
        sensor_assignment={"PID01": "R01"},
        calibration_run="cal_test",
    )


def _timeseries(n: int = 100, with_biomass: bool = True, with_iso: bool = True) -> pl.DataFrame:
    timestamps = [_START + dt.timedelta(seconds=i) for i in range(n)]
    # A clean, steadily rising ppm signal so voc_slope_ppm_asgas_h comes out
    # reliably positive and p95 sits meaningfully above the mean.
    ppm = np.linspace(1.0, 2.0, n)
    data = {
        "timestamp": timestamps,
        "experiment_id": ["exp_test"] * n,
        "reactor_id": ["R01"] * n,
        "sensor_id": ["PID01"] * n,
        "sample_t_c": [32.0] * n,
        "sample_rh_pct": [55.0] * n,
        "lamp_hours": [12.0] * n,
        "ppm_asgas": ppm,
        "ppm_asgas_stderr": [0.05] * n,
        "calibration_run_id": ["cal_test"] * n,
        "calibration_compound": ["isoprene"] * n,
        "calibration_response_factor": [0.63] * n,
    }
    if with_iso:
        data["ppm_iso_equiv"] = ppm * 0.63
        data["ppm_iso_equiv_stderr"] = [0.05 * 0.63] * n
    if with_biomass:
        data["biomass_signal_arb"] = np.linspace(0.0, 5.0, n)
        data["biomass_reading_age_s"] = [30.0] * n
    return pl.DataFrame(data)


def test_extract_features_one_row_per_experiment_reactor_sensor() -> None:
    ts = _timeseries()
    meta = _meta(R01={"par_umol_m2_s": 200.0, "reactor_temp_c": 32.0})

    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))

    assert features.height == 1
    row = features.row(0, named=True)
    assert row["experiment_id"] == "exp_test"
    assert row["reactor_id"] == "R01"
    assert row["sensor_id"] == "PID01"
    assert row["campaign_id"] == "campaign_test"
    assert row["par_umol_m2_s"] == pytest.approx(200.0)
    assert row["reactor_temp_c"] == pytest.approx(32.0)
    assert row["features_schema_version"] == 2


def test_extract_features_computes_mean_p95_and_slope() -> None:
    ts = _timeseries(n=100)
    meta = _meta(R01={"par_umol_m2_s": 200.0})

    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))
    row = features.row(0, named=True)

    # ppm ramps linearly from 1.0 to 2.0 over 99 seconds -> mean ~= 1.5.
    assert row["mean_voc_ppm_asgas"] == pytest.approx(1.5, abs=0.02)
    assert row["p95_voc_ppm_asgas"] > row["mean_voc_ppm_asgas"]
    # Slope should be positive (ppm rising) -- 1.0 ppm over 99s = 1.0/(99/3600) ppm/h.
    assert row["voc_slope_ppm_asgas_h"] > 0


def test_extract_features_includes_iso_equiv_when_available() -> None:
    ts = _timeseries(with_iso=True)
    meta = _meta(R01={})

    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))
    row = features.row(0, named=True)

    assert not math.isnan(row["mean_voc_ppm_iso_equiv"])
    assert row["mean_voc_ppm_iso_equiv"] == pytest.approx(row["mean_voc_ppm_asgas"] * 0.63, abs=0.02)


def test_extract_features_nan_fills_iso_equiv_when_unavailable() -> None:
    ts = _timeseries(with_iso=False)
    meta = _meta(R01={})

    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))
    row = features.row(0, named=True)

    assert math.isnan(row["mean_voc_ppm_iso_equiv"])


def test_extract_features_uses_freshest_biomass_reading() -> None:
    ts = _timeseries(with_biomass=True, n=100)
    meta = _meta(R01={})

    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))
    row = features.row(0, named=True)

    # biomass_signal_arb ramps 0.0 -> 5.0; the LAST row (freshest) should be used.
    assert row["biomass_signal_arb"] == pytest.approx(5.0, abs=0.1)


def test_extract_features_raises_on_empty_window() -> None:
    ts = _timeseries()
    meta = _meta(R01={})
    far_future = _START + dt.timedelta(days=365)

    with pytest.raises(ValueError, match="no rows fall within"):
        extract_features(ts, meta, (far_future, far_future + dt.timedelta(seconds=10)))


def test_load_features_for_jaxsr_returns_x_y_feature_names() -> None:
    ts = _timeseries()
    meta = _meta(R01={"par_umol_m2_s": 200.0, "reactor_temp_c": 32.0})
    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))

    X, y, feature_names = load_features_for_jaxsr(features, target="mean_voc_ppm_asgas")

    assert X.shape[0] == 1
    assert y.shape == (1,)
    assert "par_umol_m2_s" in feature_names
    assert "mean_sample_t_c" in feature_names
    # Provenance/id/target columns must never leak into the feature set.
    assert "experiment_id" not in feature_names
    assert "mean_voc_ppm_asgas" not in feature_names


def test_load_features_for_jaxsr_one_hot_encodes_categorical_when_requested() -> None:
    ts1 = _timeseries()
    ts2 = _timeseries()
    ts2 = ts2.with_columns(pl.lit("PID02").alias("sensor_id"))
    combined = pl.concat([ts1, ts2])
    meta = _meta(R01={"par_umol_m2_s": 200.0})
    features = extract_features(combined, meta, (_START, _START + dt.timedelta(seconds=99)))

    X_with_cat, _, names_with_cat = load_features_for_jaxsr(features, include_categorical=True)
    X_without_cat, _, names_without_cat = load_features_for_jaxsr(features, include_categorical=False)

    assert X_with_cat.shape[1] > X_without_cat.shape[1]
    assert any(name.startswith("sensor_id_") for name in names_with_cat)
    assert not any(name.startswith("sensor_id_") for name in names_without_cat)


def test_load_features_for_jaxsr_raises_on_mixed_compounds_by_default() -> None:
    ts1 = _timeseries()
    ts2 = _timeseries().with_columns(
        pl.lit("PID02").alias("sensor_id"), pl.lit("acetone").alias("calibration_compound")
    )
    combined = pl.concat([ts1, ts2])
    meta = _meta(R01={"par_umol_m2_s": 200.0})
    features = extract_features(combined, meta, (_START, _START + dt.timedelta(seconds=99)))

    with pytest.raises(MixedCalibrationCompoundError):
        load_features_for_jaxsr(features)

    # allow_mixed=True should let it through without raising.
    X, y, names = load_features_for_jaxsr(features, allow_mixed=True)
    assert X.shape[0] == 2


def test_load_features_for_jaxsr_raises_target_contains_nan() -> None:
    ts = _timeseries(with_iso=False)  # mean_voc_ppm_iso_equiv will be NaN
    meta = _meta(R01={"par_umol_m2_s": 200.0})
    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))

    with pytest.raises(TargetContainsNaNError, match="mean_voc_ppm_asgas"):
        load_features_for_jaxsr(features, target="mean_voc_ppm_iso_equiv")


def test_load_features_for_jaxsr_return_stderr() -> None:
    ts = _timeseries()
    meta = _meta(R01={"par_umol_m2_s": 200.0})
    features = extract_features(ts, meta, (_START, _START + dt.timedelta(seconds=99)))

    X, y, y_stderr, feature_names = load_features_for_jaxsr(
        features, target="mean_voc_ppm_asgas", return_stderr=True
    )

    assert y_stderr.shape == y.shape
    assert not np.isnan(y_stderr).any()


def test_load_timeseries_for_jaxsr_preserves_full_trajectory_not_an_average() -> None:
    n = 100
    ts = _timeseries(n=n)  # ppm ramps 1.0 -> 2.0 across n rows, per _timeseries()

    X, t, state_names = load_timeseries_for_jaxsr(ts)

    # Every one of the n rows should survive -- nothing averaged away, unlike
    # extract_features which would collapse this same data to 1 row.
    assert len(t) == n
    assert X.shape[0] == n
    assert "ppm_asgas" in state_names
    assert "biomass_signal_arb" in state_names  # _timeseries() includes biomass by default


def test_load_timeseries_for_jaxsr_time_starts_at_zero() -> None:
    ts = _timeseries(n=50)

    _, t, _ = load_timeseries_for_jaxsr(ts)

    assert t[0] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(49.0)  # 50 rows, 1 second apart, per _timeseries()


def test_load_timeseries_for_jaxsr_single_state_column_is_1d() -> None:
    ts = _timeseries(n=50, with_biomass=False)

    X, _, state_names = load_timeseries_for_jaxsr(ts, state_columns=["ppm_asgas"])

    # jaxsr.discover_dynamics documents (n_samples,) -- not (n_samples, 1) --
    # as the expected shape for a single state variable.
    assert X.ndim == 1
    assert state_names == ["ppm_asgas"]


def test_load_timeseries_for_jaxsr_multiple_state_columns_is_2d() -> None:
    ts = _timeseries(n=50, with_biomass=True)

    X, _, state_names = load_timeseries_for_jaxsr(
        ts, state_columns=["ppm_asgas", "biomass_signal_arb"]
    )

    assert X.shape == (50, 2)
    assert state_names == ["ppm_asgas", "biomass_signal_arb"]


def test_load_timeseries_for_jaxsr_rejects_multiple_experiments() -> None:
    ts1 = _timeseries(n=20)
    ts2 = _timeseries(n=20).with_columns(pl.lit("exp_other").alias("experiment_id"))
    combined = pl.concat([ts1, ts2])

    with pytest.raises(ValueError, match="exactly one experiment_id"):
        load_timeseries_for_jaxsr(combined)


def test_load_timeseries_for_jaxsr_rejects_multiple_sensors() -> None:
    ts1 = _timeseries(n=20)
    ts2 = _timeseries(n=20).with_columns(pl.lit("PID02").alias("sensor_id"))
    combined = pl.concat([ts1, ts2])

    with pytest.raises(ValueError, match="exactly one sensor_id"):
        load_timeseries_for_jaxsr(combined)
