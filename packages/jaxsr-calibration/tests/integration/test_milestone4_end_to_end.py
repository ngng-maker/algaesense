"""Milestone 4 Definition-of-Done: on synthetic dual-rate data (VOC @1 Hz,
camera @1/hr), the full pipeline -- calibrate, apply calibration, covariate-
correct, common-mode-subtract, fuse VOC+camera, extract features, bridge to
JAXSR -- produces a valid fused feature table, and a REAL upstream
`jaxsr.SymbolicRegressor.fit` call runs cleanly against it.

This is an integration test (spec Part XIV's "integration" tier): it
exercises many modules together rather than one function in isolation, and
it's slower than a typical unit test (fitting several sensors' calibrations
plus a real symbolic regression search) -- both reasons it lives under
tests/integration/ rather than tests/unit/.
"""

from __future__ import annotations

import datetime as dt

import jaxsr
import numpy as np
import polars as pl
import pytest

from jaxsr_calibration.calibration.apply import apply_calibration
from jaxsr_calibration.calibration.apply import persist_calibration
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from jaxsr_calibration.logging_.schema import ExperimentMeta
from jaxsr_calibration.processing.features import (
    extract_features,
    load_features_for_jaxsr,
    load_timeseries_for_jaxsr,
)
from jaxsr_calibration.processing.fusion import fuse_multirate
from tests.fixtures.synthetic_readings import make_dual_rate_experiment, make_standard_addition_readings

_START = dt.datetime(2026, 7, 22, 8, 0, 0, tzinfo=dt.timezone.utc)


def _build_experiment(experiment_id: str, par_umol_m2_s: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame, ExperimentMeta]:
    """One synthetic experiment: higher PAR -> a faster-rising true VOC
    signal AND a faster-growing biomass signal, so the fitted model has real
    PAR-vs-target structure to find (not just noise)."""
    voc_df, camera_df = make_dual_rate_experiment(
        experiment_id=experiment_id,
        reactor_id="R01",
        sensor_id="PID01",
        duration_h=0.1,  # 360 s -- short, to keep this test fast
        camera_interval_h=1.0,
        voc_baseline_mv=2.0,
        voc_signal_ppm_asgas_over_time=0.001 * par_umol_m2_s,
        voc_b1_mv_per_ppm=4.0,
        voc_noise_std=0.05,
        biomass_values=[par_umol_m2_s / 50.0],
        seed=seed,
    )
    meta = ExperimentMeta(
        experiment_id=experiment_id,
        started_at=_START,
        operator="tester",
        campaign_id="campaign_e2e_test",
        conditions={"R01": {"par_umol_m2_s": par_umol_m2_s}},
        sensor_assignment={"PID01": "R01"},
        calibration_run="cal_e2e_test",
    )
    return voc_df, camera_df, meta


def test_milestone4_full_pipeline_feeds_a_real_jaxsr_symbolic_regressor(tmp_path) -> None:
    # 1. Calibrate the (single, shared-across-the-campaign) sensor once.
    cal_df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 2.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.02}},
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        calibration_compound="isoprene",
        response_factor=0.63,
        seed=100,
    )
    models = fit_sensitivity_per_sensor(cal_df)
    assert models["PID01"].status == "PASS"
    data_dir = tmp_path / "calibrations"
    persist_calibration(models, "cal_e2e_test", "campaign_e2e_test", data_dir)

    # 2. Build several synthetic experiments across a range of PAR values.
    all_features = []
    for i, par in enumerate([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]):
        voc_df, camera_df, meta = _build_experiment(f"exp_{i:02d}", par_umol_m2_s=par, seed=200 + i)

        # 3. Apply the shared calibration to this experiment's raw voltage.
        ppm, ppm_stderr, _ = apply_calibration(
            voc_df["pid_voltage_mv"], "PID01", voc_df["sample_t_c"], voc_df["sample_rh_pct"],
            "cal_e2e_test", data_dir=data_dir,
        )
        voc_df = voc_df.with_columns(
            ppm.alias("ppm_asgas"),
            ppm_stderr.alias("ppm_asgas_stderr"),
            pl.lit("cal_e2e_test").alias("calibration_run_id"),
            pl.lit("isoprene").alias("calibration_compound"),
            pl.lit(0.63).alias("calibration_response_factor"),
        )

        # 4. Fuse the fast VOC stream with the slow camera stream.
        fused = fuse_multirate(voc_df, camera_df)

        # 5. Collapse into one feature row for this experiment/reactor/sensor.
        features = extract_features(fused, meta, (voc_df["timestamp"][0], voc_df["timestamp"][-1]))
        all_features.append(features)

    features_df = pl.concat(all_features)
    assert features_df.height == 8  # one row per synthetic experiment

    # 6. Persist and reload as Parquet -- per the spec's own artifact layout
    # (data/derived/features/{campaign_id}/{experiment_id}.parquet), the
    # fused feature table is meant to be a real on-disk artifact, not just an
    # in-memory DataFrame. Round-tripping through Parquet here confirms the
    # table is actually valid Parquet, not just valid in memory.
    features_path = tmp_path / "features" / "campaign_e2e_test.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(features_path)
    features_df = pl.read_parquet(features_path)
    assert features_df.height == 8

    # 7. Bridge into JAXSR's expected (X, y, feature_names) shape.
    X, y, feature_names = load_features_for_jaxsr(
        features_df, target="mean_voc_ppm_asgas", include_categorical=False
    )
    assert X.shape[0] == 8
    assert "par_umol_m2_s" in feature_names
    assert "biomass_signal_arb" in feature_names

    # 7. The actual point of this DoD: hand X, y to REAL, unmodified upstream
    # jaxsr.SymbolicRegressor and confirm it fits without error. A small
    # basis library and max_terms keep this fast -- this test is checking
    # the *pipe connects*, not evaluating model quality.
    basis = (
        jaxsr.BasisLibrary(n_features=X.shape[1], feature_names=feature_names)
        .add_constant()
        .add_linear()
    )
    model = jaxsr.SymbolicRegressor(basis_library=basis, max_terms=3)
    fitted = model.fit(np.asarray(X), np.asarray(y))

    assert fitted is model  # spec's own example relies on .fit() returning self
    # A fitted model should be able to predict on the same X without error.
    predictions = fitted.predict(np.asarray(X))
    assert predictions.shape == y.shape


def test_milestone4_single_experiment_trajectory_feeds_real_jaxsr_dynamics_discovery(tmp_path) -> None:
    """The second half of the Milestone 4 correction: prove that ONE
    experiment's full, un-averaged trajectory (not a single summarized row)
    can be handed to real `jaxsr.discover_dynamics` and recover the true
    rate of change -- this is the "see the exact trend within one run" path,
    complementary to the cross-experiment SymbolicRegressor path above.
    """
    cal_df = make_standard_addition_readings(
        {"PID01": {"b0_mv": 2.0, "b1_mv_per_ppm": 4.0, "noise_std": 0.02}},
        spike_ppm_list=[0.0, 1.0, 5.0, 20.0],
        calibration_compound="isoprene",
        response_factor=0.63,
        seed=300,
    )
    models = fit_sensitivity_per_sensor(cal_df)
    data_dir = tmp_path / "calibrations"
    persist_calibration(models, "cal_dynamics_test", "campaign_e2e_test", data_dir)

    # One experiment, held at a fixed PAR, with a clean known rate of rise
    # (0.02 ppm/s) so we can check discover_dynamics actually recovers it --
    # not just that the call doesn't crash.
    true_rate_ppm_per_s = 0.02
    voc_df, camera_df = make_dual_rate_experiment(
        experiment_id="exp_dynamics_test",
        reactor_id="R01",
        sensor_id="PID01",
        duration_h=0.1,
        camera_interval_h=1.0,
        voc_baseline_mv=2.0,
        voc_signal_ppm_asgas_over_time=true_rate_ppm_per_s,
        voc_b1_mv_per_ppm=4.0,
        voc_noise_std=0.02,
        seed=301,
    )

    ppm, ppm_stderr, _ = apply_calibration(
        voc_df["pid_voltage_mv"], "PID01", voc_df["sample_t_c"], voc_df["sample_rh_pct"],
        "cal_dynamics_test", data_dir=data_dir,
    )
    voc_df = voc_df.with_columns(ppm.alias("ppm_asgas"), ppm_stderr.alias("ppm_asgas_stderr"))
    fused = fuse_multirate(voc_df, camera_df)

    # The point of this whole correction: pull out the FULL trajectory
    # (hundreds of rows), not one averaged summary row.
    X, t, state_names = load_timeseries_for_jaxsr(fused, state_columns=["ppm_asgas"])
    assert len(t) == voc_df.height  # every single reading preserved, nothing averaged away

    # Hand it to REAL, unmodified upstream jaxsr.discover_dynamics.
    result = jaxsr.discover_dynamics(X, t, state_names=state_names, max_terms=2)

    assert "ppm_asgas" in result.equations
    # `coefficients_` on the fitted per-state SymbolicRegressor is the actual
    # discovered rate of change -- this is the real assertion: not just "the
    # call didn't crash" but "it recovered the true 0.02 ppm/s rate this
    # synthetic experiment was built with, from the trajectory alone".
    discovered_rate = result.models["ppm_asgas"].coefficients_[0]
    assert discovered_rate == pytest.approx(true_rate_ppm_per_s, abs=0.002)
