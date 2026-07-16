"""Unit tests for jaxsr_calibration.processing.fusion.fuse_multirate."""

from __future__ import annotations

import pytest

from jaxsr_calibration.processing.fusion import fuse_multirate
from tests.fixtures.synthetic_readings import make_dual_rate_experiment


def test_fuse_multirate_attaches_most_recent_camera_reading() -> None:
    voc_df, camera_df = make_dual_rate_experiment(
        duration_h=2.0, camera_interval_h=1.0, biomass_values=[10.0, 20.0, 30.0]
    )

    fused = fuse_multirate(voc_df, camera_df)

    assert fused.height == voc_df.height
    assert "biomass_signal_arb" in fused.columns
    assert "biomass_reading_age_s" in fused.columns

    # A VOC row right at the experiment start (t=0) should pair with the
    # first camera capture (biomass=10.0, taken at the same instant, age=0).
    first_row = fused.row(0, named=True)
    assert first_row["biomass_signal_arb"] == pytest.approx(10.0)
    assert first_row["biomass_reading_age_s"] == pytest.approx(0.0)

    # A VOC row taken 30 minutes after the second camera capture (which
    # landed at exactly the 1-hour mark) should still be paired with that
    # second capture (biomass=20.0), not the first or third.
    row_at_90min = fused.row(90 * 60, named=True)  # row index == elapsed seconds here
    assert row_at_90min["biomass_signal_arb"] == pytest.approx(20.0)
    assert row_at_90min["biomass_reading_age_s"] == pytest.approx(30 * 60, abs=1)


def test_fuse_multirate_age_resets_after_each_new_capture() -> None:
    voc_df, camera_df = make_dual_rate_experiment(
        duration_h=2.0, camera_interval_h=1.0, biomass_values=[10.0, 20.0, 30.0]
    )

    fused = fuse_multirate(voc_df, camera_df)

    ages = fused["biomass_reading_age_s"].to_numpy()
    # Age should climb steadily within each hourly window (roughly +1 per
    # VOC row, since VOC samples once per second) and then drop back near 0
    # right after each new camera capture -- i.e. it's NOT monotonically
    # increasing across the whole experiment.
    assert ages[0] == pytest.approx(0.0)
    assert ages[3599] == pytest.approx(3599.0, abs=1)  # just before the 2nd capture
    assert ages[3600] == pytest.approx(0.0, abs=1)  # right at the 2nd capture


def test_fuse_multirate_keeps_experiments_and_reactors_separate() -> None:
    voc_a, camera_a = make_dual_rate_experiment(
        experiment_id="exp_A", reactor_id="R01", duration_h=1.0, camera_interval_h=1.0,
        biomass_values=[100.0, 100.0],
    )
    voc_b, camera_b = make_dual_rate_experiment(
        experiment_id="exp_B", reactor_id="R02", duration_h=1.0, camera_interval_h=1.0,
        biomass_values=[999.0, 999.0],
    )
    import polars as pl

    voc_combined = pl.concat([voc_a, voc_b])
    camera_combined = pl.concat([camera_a, camera_b])

    fused = fuse_multirate(voc_combined, camera_combined)

    exp_a_biomass = fused.filter(pl.col("experiment_id") == "exp_A")["biomass_signal_arb"].unique().to_list()
    exp_b_biomass = fused.filter(pl.col("experiment_id") == "exp_B")["biomass_signal_arb"].unique().to_list()
    # exp_A's VOC rows must only ever pair with exp_A's own camera readings
    # (100.0), never exp_B's (999.0), even though both experiments' data was
    # combined into one input table.
    assert exp_a_biomass == [100.0]
    assert exp_b_biomass == [999.0]
