"""Lightweight synthetic data generators shared across this package's tests
(diagnostics, calibration, and processing/fusion).

This is deliberately NOT the fuller `synthetic_pid_stream` fixture the spec
describes in §36 (with drift, RH/T profiles, notch-filterable artifacts, and
several named test scenarios, all in service of end-to-end preprocessing
regression tests) -- these generators are simpler, purpose-built ones for
exercising individual functions.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl


def make_fleet_readings(
    sensor_specs: dict[str, dict[str, float]],
    n_samples: int = 300,
    sample_period_s: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a multi-sensor synthetic clean-air readings table.

    `sensor_specs` maps sensor_id -> {"mean_mv", "std_mv", "slope_mv_per_min"}
    describing the *true* generating parameters for that sensor -- tests then
    assert that run_fleet_zero's classification matches what those true
    parameters should produce (e.g. a sensor built with mean_mv=50 when the
    PASS threshold is 5 should come back "FAIL").
    """
    # `np.random.default_rng(seed)` -- see test_covariate.py's synthetic
    # generator for why we use this reproducible RNG style rather than
    # numpy's older global-state random functions.
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 15, 8, 0, 0, tzinfo=dt.timezone.utc)

    frames: list[pl.DataFrame] = []
    for sensor_id, spec in sensor_specs.items():
        mean_mv = spec.get("mean_mv", 0.0)
        std_mv = spec.get("std_mv", 0.2)
        slope_mv_per_min = spec.get("slope_mv_per_min", 0.0)

        minutes = np.arange(n_samples) * (sample_period_s / 60.0)
        noise = rng.normal(0.0, std_mv, size=n_samples)
        voltage = mean_mv + slope_mv_per_min * minutes + noise
        timestamps = [
            base_time + dt.timedelta(seconds=i * sample_period_s) for i in range(n_samples)
        ]

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": [sensor_id] * n_samples,
                    "pid_voltage_mv": voltage,
                }
            )
        )

    # `pl.concat` stacks the per-sensor frames into one combined table, the
    # same shape run_fleet_zero expects to receive as `readings`.
    return pl.concat(frames)


def make_ambient_readings(
    sensor_specs: dict[str, dict[str, float]],
    n_samples: int = 200,
    rh_range: tuple[float, float] = (20.0, 80.0),
    t_range: tuple[float, float] = (28.0, 34.0),
    seed: int = 0,
) -> pl.DataFrame:
    """Build a multi-sensor synthetic ambient-air table with a known
    RH/T -> voltage relationship per sensor, for testing
    run_ambient_baseline / fit_covariate_model.

    `sensor_specs` maps sensor_id -> {"alpha", "beta_rh", "gamma_t",
    "delta_rh_t", "noise_std"} -- the true linear-model coefficients used to
    generate that sensor's synthetic voltage, so a test can assert the fitted
    CovariateModel recovers values close to them.
    """
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 15, 20, 0, 0, tzinfo=dt.timezone.utc)

    frames: list[pl.DataFrame] = []
    for sensor_id, spec in sensor_specs.items():
        alpha = spec.get("alpha", 10.0)
        beta_rh = spec.get("beta_rh", 0.2)
        gamma_t = spec.get("gamma_t", 0.5)
        delta_rh_t = spec.get("delta_rh_t", 0.0)
        noise_std = spec.get("noise_std", 0.05)

        rh = rng.uniform(rh_range[0], rh_range[1], size=n_samples)
        temp = rng.uniform(t_range[0], t_range[1], size=n_samples)
        noise = rng.normal(0.0, noise_std, size=n_samples)
        voltage = alpha + beta_rh * rh + gamma_t * temp + delta_rh_t * (rh * temp) + noise

        timestamps = [base_time + dt.timedelta(seconds=i) for i in range(n_samples)]
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": [sensor_id] * n_samples,
                    "sample_rh_pct": rh,
                    "sample_t_c": temp,
                    "pid_voltage_mv": voltage,
                }
            )
        )

    return pl.concat(frames)


def make_swap_pilot_readings(
    sensor_ids: list[str],
    reactor_ids: list[str],
    sensor_effect_std: float,
    reactor_effect_std: float,
    residual_std: float,
    n_blocks: int | None = None,
    obs_per_block: int = 20,
    baseline_mv: float = 50.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a synthetic Latin-square swap-pilot table with KNOWN true
    sensor/reactor/residual variance contributions, for testing
    run_swap_pilot's variance decomposition.

    Each sensor gets one fixed random "effect" (a constant offset drawn from
    N(0, sensor_effect_std^2)) and likewise each reactor -- exactly the
    generating process a crossed-random-effects model assumes. `n_blocks`
    defaults to `len(reactor_ids)`, giving each sensor a turn with every
    reactor exactly once (a true Latin square) if `len(sensor_ids) ==
    len(reactor_ids)`.
    """
    rng = np.random.default_rng(seed)
    n_blocks = n_blocks or len(reactor_ids)

    sensor_effects = {s: rng.normal(0.0, sensor_effect_std) for s in sensor_ids}
    reactor_effects = {r: rng.normal(0.0, reactor_effect_std) for r in reactor_ids}

    rows: list[dict] = []
    base_time = dt.datetime(2026, 7, 20, 8, 0, 0, tzinfo=dt.timezone.utc)
    t = 0
    for block in range(n_blocks):
        for i, sensor_id in enumerate(sensor_ids):
            # Rotate which reactor this sensor is paired with, one position
            # per block -- the "Latin square" rotation the spec describes.
            reactor_id = reactor_ids[(i + block) % len(reactor_ids)]
            for _ in range(obs_per_block):
                voltage = (
                    baseline_mv
                    + sensor_effects[sensor_id]
                    + reactor_effects[reactor_id]
                    + rng.normal(0.0, residual_std)
                )
                rows.append(
                    {
                        "timestamp": base_time + dt.timedelta(seconds=t),
                        "sensor_id": sensor_id,
                        "reactor_id": reactor_id,
                        "pid_voltage_mv": voltage,
                    }
                )
                t += 1

    return pl.DataFrame(rows)


def make_standard_addition_readings(
    sensor_specs: dict[str, dict[str, float]],
    spike_ppm_list: list[float],
    calibration_compound: str = "isoprene",
    mw_g_mol: float = 68.12,
    response_factor: float | None = 0.63,
    n_per_level: int = 10,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a synthetic spike-and-recover table for testing
    fit_sensitivity_per_sensor.

    `sensor_specs` maps sensor_id -> {"b0_mv", "b1_mv_per_ppm", "noise_std"}
    -- the true generating line for that sensor, so a test can assert the
    fitted SensitivityModel recovers values close to them. `spike_ppm_list`
    should include 0.0 for the baseline (no-injection) level, matching how a
    real run always starts with a baseline dwell before any injection.
    """
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 15, 7, 0, 0, tzinfo=dt.timezone.utc)

    rows: list[dict] = []
    t = 0
    for sensor_id, spec in sensor_specs.items():
        b0 = spec.get("b0_mv", 0.0)
        b1 = spec.get("b1_mv_per_ppm", 5.0)
        noise_std = spec.get("noise_std", 0.2)
        for spike_ppm in spike_ppm_list:
            for _ in range(n_per_level):
                voltage = b0 + b1 * spike_ppm + rng.normal(0.0, noise_std)
                rows.append(
                    {
                        "timestamp": base_time + dt.timedelta(seconds=t),
                        "sensor_id": sensor_id,
                        "spike_ppm_asgas": spike_ppm,
                        "pid_voltage_mv": voltage,
                        "sample_t_c": 32.0 + rng.normal(0.0, 0.1),
                        "sample_rh_pct": 55.0 + rng.normal(0.0, 0.5),
                        "lamp_hours": 12.0,
                        "calibration_compound": calibration_compound,
                        "mw_g_mol": mw_g_mol,
                        "response_factor": response_factor,
                        "response_factor_stderr": None,
                        "calibration_source": "test-fixture",
                        "calibration_is_builtin": True,
                    }
                )
                t += 1

    return pl.DataFrame(rows)


def make_common_mode_readings(
    sensor_ids: list[str],
    n_samples: int = 60,
    common_signal_amplitude: float = 5.0,
    individual_noise_std: float = 0.2,
    outlier_sensor_ids: list[str] | None = None,
    outlier_offset: float = 50.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Build a synthetic multi-sensor table sharing one common-mode signal
    (a slow sine wave, standing in for something like a shared HVAC cycle)
    plus small independent per-sensor noise, for testing subtract_common_mode.

    `outlier_sensor_ids` (if given) get a large constant offset added on top
    -- simulating a malfunctioning sensor that should be excluded from the
    common-mode estimate at every timestamp.
    """
    rng = np.random.default_rng(seed)
    outlier_sensor_ids = outlier_sensor_ids or []
    base_time = dt.datetime(2026, 7, 21, 9, 0, 0, tzinfo=dt.timezone.utc)

    # One shared "common mode" value per timestamp -- every sensor sees the
    # same value here before its own noise/offset is added.
    common_signal = common_signal_amplitude * np.sin(np.linspace(0, 2 * np.pi, n_samples))

    rows: list[dict] = []
    for sensor_id in sensor_ids:
        offset = outlier_offset if sensor_id in outlier_sensor_ids else 0.0
        noise = rng.normal(0.0, individual_noise_std, size=n_samples)
        values = common_signal + offset + noise
        for i in range(n_samples):
            rows.append(
                {
                    "timestamp": base_time + dt.timedelta(seconds=i),
                    "sensor_id": sensor_id,
                    "pid_voltage_mv": values[i],
                }
            )

    return pl.DataFrame(rows)


def make_dual_rate_experiment(
    experiment_id: str = "exp_dual_rate_test",
    reactor_id: str = "R01",
    sensor_id: str = "PID01",
    camera_id: str = "CAM01",
    duration_h: float = 2.0,
    camera_interval_h: float = 1.0,
    voc_baseline_mv: float = 2.0,
    voc_signal_ppm_asgas_over_time: float = 0.02,
    voc_b1_mv_per_ppm: float = 4.0,
    voc_noise_std: float = 0.05,
    biomass_values: list[float] | None = None,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a matched pair of (VOC timeseries @ ~1 Hz, camera timeseries @
    ~hourly) for one experiment/reactor -- the two different sampling rates
    the user's original request centers on. Used to test fuse_multirate and,
    end to end, the whole Milestone 4 pipeline.

    The VOC signal is built as a slowly ramping "true" ppm_asgas
    (`voc_signal_ppm_asgas_over_time` ppm per second) converted to voltage
    via a known `b0 + b1*ppm` line, so a test can both fit a calibration
    against it and sanity-check the shape of the resulting feature table.
    `biomass_values` (one per camera capture) defaults to a simple increasing
    sequence if not given, standing in for a culture's biomass slowly
    increasing over the experiment.
    """
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 22, 8, 0, 0, tzinfo=dt.timezone.utc)

    n_voc_samples = int(duration_h * 3600)
    voc_timestamps = [base_time + dt.timedelta(seconds=i) for i in range(n_voc_samples)]
    true_ppm = voc_signal_ppm_asgas_over_time * np.arange(n_voc_samples)
    voltage = voc_baseline_mv + voc_b1_mv_per_ppm * true_ppm + rng.normal(
        0.0, voc_noise_std, size=n_voc_samples
    )

    voc_df = pl.DataFrame(
        {
            "timestamp": voc_timestamps,
            "experiment_id": [experiment_id] * n_voc_samples,
            "sensor_id": [sensor_id] * n_voc_samples,
            "reactor_id": [reactor_id] * n_voc_samples,
            "pid_voltage_mv": voltage,
            "sample_t_c": 32.0 + rng.normal(0.0, 0.1, size=n_voc_samples),
            "sample_rh_pct": 55.0 + rng.normal(0.0, 0.5, size=n_voc_samples),
            "lamp_hours": [12.0] * n_voc_samples,
            "light_state": ["on"] * n_voc_samples,
            "acquisition_status": ["OK"] * n_voc_samples,
        }
    )

    n_camera_samples = int(duration_h / camera_interval_h) + 1
    camera_timestamps = [
        base_time + dt.timedelta(hours=i * camera_interval_h) for i in range(n_camera_samples)
    ]
    if biomass_values is None:
        # A simple increasing sequence (0, 1, 2, ...) -- standing in for
        # biomass slowly increasing over the course of an experiment.
        biomass_values = [float(i) for i in range(n_camera_samples)]

    camera_df = pl.DataFrame(
        {
            "timestamp": camera_timestamps,
            "experiment_id": [experiment_id] * n_camera_samples,
            "reactor_id": [reactor_id] * n_camera_samples,
            "camera_id": [camera_id] * n_camera_samples,
            "biomass_signal_arb": biomass_values,
        }
    )

    return voc_df, camera_df
