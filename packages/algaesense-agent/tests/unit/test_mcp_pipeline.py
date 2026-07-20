"""Unit tests for algaesense_agent.mcp_pipeline: pure-Python pipeline logic
plus a check that the MCP tool wrappers call through to it correctly.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from algaesense_edge.acquisition.writer import PartitionedParquetWriter
from jaxsr_calibration.calibration.apply import persist_calibration
from jaxsr_calibration.calibration.standard_addition import fit_sensitivity_per_sensor
from jaxsr_calibration.logging_.schema import VOC_RAW_SCHEMA

from algaesense_agent.mcp_pipeline.pipeline import (
    CampaignNotFoundError,
    discover_led_response_dynamics,
    fit_symbolic_model,
    load_campaign_features,
    suggest_next_experiments,
)


def _write_synthetic_campaign(data_dir: Path, campaign_id: str, n_experiments: int = 8) -> None:
    """Write one derived-features Parquet file per synthetic experiment,
    with a known linear relationship between PAR and VOC output."""

    """
    `mean_voc_ppm_asgas = 2.0 * par_umol_m2_s + 5.0`, no noise -- a
    trivially recoverable relationship, so a successful fit here is a real
    signal that `fit_symbolic_model`/`suggest_next_experiments` wired
    `load_features_for_jaxsr` -> `jaxsr.SymbolicRegressor` ->
    `jaxsr.ActiveLearner` together correctly, not that the model happened
    to get lucky on noisy data.
    """
    campaign_dir = data_dir / "derived" / "features" / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    par_values = np.linspace(100.0, 500.0, n_experiments)

    for i, par in enumerate(par_values):
        experiment_id = f"exp_{i:02d}"
        row = {
            "experiment_id": experiment_id,
            "campaign_id": campaign_id,
            "reactor_id": "R01",
            "sensor_id": "PID01",
            "par_umol_m2_s": float(par),
            "mean_sample_t_c": 30.0,
            "mean_sample_rh_pct": 60.0,
            "mean_voc_ppm_asgas": float(2.0 * par + 5.0),
        }
        pl.DataFrame([row]).write_parquet(campaign_dir / f"{experiment_id}.parquet")


def test_load_campaign_features_concatenates_every_experiment(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01", n_experiments=5)

    features_df = load_campaign_features("camp_01", tmp_path)

    assert features_df.height == 5
    assert set(features_df["experiment_id"].to_list()) == {f"exp_0{i}" for i in range(5)}


def test_load_campaign_features_raises_for_unknown_campaign(tmp_path: Path) -> None:
    with pytest.raises(CampaignNotFoundError):
        load_campaign_features("does_not_exist", tmp_path)


def test_fit_symbolic_model_recovers_known_linear_relationship(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")

    """
    `include_categorical=False`: this synthetic fixture only has one
    sensor_id/reactor_id value, so the one-hot dummy columns
    `load_features_for_jaxsr` would otherwise add are just noise for what
    this test is actually checking (the par_umol_m2_s -> VOC
    relationship).
    """
    result = fit_symbolic_model(
        "camp_01",
        data_dir=tmp_path,
        target="mean_voc_ppm_asgas",
        feature_columns=["par_umol_m2_s"],
        include_categorical=False,
    )

    assert result.feature_names == ["par_umol_m2_s"]
    # A clean linear relationship should fit essentially perfectly.
    assert result.metrics["r2"] > 0.99


def test_suggest_next_experiments_returns_points_within_observed_bounds(tmp_path: Path) -> None:
    _write_synthetic_campaign(tmp_path, "camp_01")

    result = suggest_next_experiments(
        "camp_01",
        data_dir=tmp_path,
        target="mean_voc_ppm_asgas",
        feature_columns=["par_umol_m2_s"],
        n_points=3,
    )

    assert len(result.points) == 3
    lo, hi = result.fit.feature_bounds[0]
    for point in result.points:
        assert lo <= point["par_umol_m2_s"] <= hi


async def test_mcp_server_fit_tool_matches_direct_pipeline_call(tmp_path: Path, monkeypatch) -> None:
    """The FastMCP tool wrapper should return the same result as calling
    the pipeline function directly -- confirms the server wiring (argument
    passing, dataclass -> dict conversion) doesn't silently change behavior."""

    _write_synthetic_campaign(tmp_path, "camp_01")
    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path))

    """
    Imported after setting the env var, and via importlib.reload if
    already imported by an earlier test in this session, so `_data_dir()`
    picks up this test's `tmp_path` rather than whatever value an earlier
    import might have already read.
    """
    import importlib

    from algaesense_agent.mcp_pipeline import server as server_module

    importlib.reload(server_module)

    direct_result = fit_symbolic_model(
        "camp_01", data_dir=tmp_path, target="mean_voc_ppm_asgas", feature_columns=["par_umol_m2_s"]
    )

    tool_result = await server_module.mcp.call_tool(
        "fit_campaign_model",
        {
            "campaign_id": "camp_01",
            "target": "mean_voc_ppm_asgas",
            "feature_columns": ["par_umol_m2_s"],
        },
    )

    """
    `call_tool` returns a list of MCP content blocks, not the raw Python
    return value -- a dict-returning tool comes back as a single
    `TextContent` block whose `.text` is the JSON-serialized dict, so we
    parse that back into a dict rather than asserting on wire-format
    objects directly.
    """
    import json

    structured = json.loads(tool_result[0].text)

    assert structured["expression"] == direct_result.expression
    assert structured["feature_names"] == direct_result.feature_names


"""
Tests for discover_led_response_dynamics: feeding one experiment's real,
per-second VOC trajectory -- with the LED's actually-applied PAR as a
second state variable -- into jaxsr.discover_dynamics.
"""

_DYN_START = dt.datetime(2026, 7, 25, 8, 0, 0, tzinfo=dt.timezone.utc)

# Known calibration: voltage = 2.0 + 4.0*ppm_asgas (no noise, so
# apply_calibration recovers ppm exactly, isolating this test to whether
# the dynamics-discovery wiring itself works, not calibration noise).
_CAL_B0_MV = 2.0
_CAL_B1_MV_PER_PPM = 4.0

# Known ground truth: d(ppm)/dt = _TRUE_RATE * par(t), where
# par(t) = _PAR0 + _PAR_SLOPE * t (a ramp, same shape control_profiles.py's
# "ramp" produces) -- so ppm(t) is the closed-form integral of that, and
# reactor_par_umol_m2_s should show up in the discovered ppm_asgas equation.
_TRUE_RATE = 0.01
_PAR0 = 50.0
_PAR_SLOPE = 1.0
_PPM0 = 1.0


def _persist_known_calibration(data_dir: Path, calibration_run_id: str, sensor_id: str = "PID01") -> None:
    """Persist a real, exactly-recoverable calibration run (2 spike levels,
    no noise) via the real fit_sensitivity_per_sensor + persist_calibration
    -- not a hand-built SensitivityModel."""
    cal_rows = [
        {
            "sensor_id": sensor_id,
            "spike_ppm_asgas": spike,
            "pid_voltage_mv": _CAL_B0_MV + _CAL_B1_MV_PER_PPM * spike,
            "sample_t_c": 25.0,
            "sample_rh_pct": 50.0,
            "lamp_hours": 10.0,
            "calibration_compound": "isoprene",
            "mw_g_mol": 68.12,
            "response_factor": 0.63,
        }
        for spike in (0.0, 0.0, 10.0, 10.0)
    ]
    models = fit_sensitivity_per_sensor(pl.DataFrame(cal_rows))
    calibration_dir = data_dir / "derived" / "calibrations" / "standard_addition"
    persist_calibration(models, calibration_run_id, "exp_calibration_source", calibration_dir)


def _write_dynamics_experiment(
    data_dir: Path,
    experiment_id: str,
    reactor_id: str = "R01",
    sensor_id: str = "PID01",
    n: int = 200,
    all_null_par: bool = False,
    n_par_nulls: int = 0,
) -> None:
    """Write n one-second raw VOC rows for one reactor/sensor, with a known
    PAR ramp and a VOC trajectory whose true rate of change is a known
    linear function of that PAR. `n_par_nulls` nulls out just that many
    leading rows' PAR (rather than every row, like `all_null_par`) --
    simulating PAR recording starting partway through the window."""
    writer = PartitionedParquetWriter(
        base_dir=data_dir / "raw",
        experiment_id=experiment_id,
        partition_key="sensor_id",
        partition_value=sensor_id,
        schema=VOC_RAW_SCHEMA,
    )
    for i in range(n):
        t = float(i)
        par = _PAR0 + _PAR_SLOPE * t
        ppm = _PPM0 + _TRUE_RATE * (_PAR0 * t + _PAR_SLOPE * t * t / 2.0)
        voltage = _CAL_B0_MV + _CAL_B1_MV_PER_PPM * ppm
        writer.write_row(
            {
                "timestamp": _DYN_START + dt.timedelta(seconds=i),
                "experiment_id": experiment_id,
                "sensor_id": sensor_id,
                "reactor_id": reactor_id,
                "pid_voltage_mv": voltage,
                "sample_t_c": 25.0,
                "sample_rh_pct": 50.0,
                "sample_flow_sccm": None,
                "pump_pwm": None,
                "lamp_hours": 10.0,
                "reactor_par_umol_m2_s": None if (all_null_par or i < n_par_nulls) else par,
                "reactor_temp_c": None,
                "reactor_od": None,
                "reactor_ph": None,
                "light_state": "on",
                "room_t_c": None,
                "room_rh_pct": None,
                "acquisition_status": "OK",
            }
        )
    writer.close()


def test_discover_led_response_dynamics_recovers_par_as_a_selected_feature(tmp_path: Path) -> None:
    _write_dynamics_experiment(tmp_path, "exp_dynamics_test")
    _persist_known_calibration(tmp_path, "cal_dynamics_test")

    result = discover_led_response_dynamics(
        "exp_dynamics_test", "R01", "PID01", "cal_dynamics_test", data_dir=tmp_path, max_terms=3
    )

    assert result.n_samples == 200
    assert "ppm_asgas" in result.equations
    # The real check: PAR shows up as a term jaxsr actually selected for
    # the VOC equation, not just "the call didn't crash".
    assert "reactor_par_umol_m2_s" in result.selected_features["ppm_asgas"]


def test_discover_led_response_dynamics_raises_for_all_null_par(tmp_path: Path) -> None:
    _write_dynamics_experiment(tmp_path, "exp_no_par_history", all_null_par=True)
    _persist_known_calibration(tmp_path, "cal_dynamics_test")

    with pytest.raises(ValueError, match="entirely null"):
        discover_led_response_dynamics(
            "exp_no_par_history", "R01", "PID01", "cal_dynamics_test", data_dir=tmp_path
        )


def test_discover_led_response_dynamics_raises_for_partially_null_par(tmp_path: Path) -> None:
    """Regression test for a real gap: the old all-null-only guard would
    have missed this case entirely -- a PARTIALLY-null PAR column (PAR
    recording started partway through the window, e.g. a service restart
    mid-experiment) would have silently fed a mixed-validity state
    variable into jaxsr.discover_dynamics instead of raising a clear
    error."""
    _write_dynamics_experiment(tmp_path, "exp_partial_par_history", n_par_nulls=10)
    _persist_known_calibration(tmp_path, "cal_dynamics_test")

    with pytest.raises(ValueError, match="10 of 200"):
        discover_led_response_dynamics(
            "exp_partial_par_history", "R01", "PID01", "cal_dynamics_test", data_dir=tmp_path
        )


def test_discover_led_response_dynamics_raises_for_naive_since_or_until(tmp_path: Path) -> None:
    """Regression test for a real gap: `since`/`until` used to be filtered
    against the raw tz-aware UTC `timestamp` column with no normalization
    or validation -- a naive datetime is genuinely ambiguous about which
    timezone was meant, so this must raise a clear error rather than
    silently filtering against the wrong window (or erroring deep inside
    polars with a confusing message)."""
    _write_dynamics_experiment(tmp_path, "exp_dynamics_test")
    _persist_known_calibration(tmp_path, "cal_dynamics_test")
    naive_since = dt.datetime(2026, 7, 25, 8, 0, 0)  # no tzinfo

    with pytest.raises(ValueError, match="timezone-aware"):
        discover_led_response_dynamics(
            "exp_dynamics_test", "R01", "PID01", "cal_dynamics_test", data_dir=tmp_path, since=naive_since
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        discover_led_response_dynamics(
            "exp_dynamics_test", "R01", "PID01", "cal_dynamics_test", data_dir=tmp_path, until=naive_since
        )


async def test_mcp_server_dynamics_tool_matches_direct_pipeline_call(tmp_path: Path, monkeypatch) -> None:
    _write_dynamics_experiment(tmp_path, "exp_dynamics_test")
    _persist_known_calibration(tmp_path, "cal_dynamics_test")
    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path))

    import importlib

    from algaesense_agent.mcp_pipeline import server as server_module

    importlib.reload(server_module)

    direct_result = discover_led_response_dynamics(
        "exp_dynamics_test", "R01", "PID01", "cal_dynamics_test", data_dir=tmp_path, max_terms=3
    )

    tool_result = await server_module.mcp.call_tool(
        "discover_led_response_dynamics",
        {
            "experiment_id": "exp_dynamics_test",
            "reactor_id": "R01",
            "sensor_id": "PID01",
            "calibration_run_id": "cal_dynamics_test",
            "max_terms": 3,
        },
    )

    import json

    structured = json.loads(tool_result[0].text)

    assert structured["equations"] == direct_result.equations
    assert structured["selected_features"] == direct_result.selected_features
