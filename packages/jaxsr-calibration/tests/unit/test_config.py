"""Unit tests for the Pydantic config models: SensorConfig, ReactorConfig,
RotationSchedule (calibration/config.py), PreprocessingConfig,
DiagnosticThresholds (processing/config.py), and CameraConfig (camera/config.py).

Each test builds a plain Python dict shaped like what `yaml.safe_load` would
hand back from an actual YAML file, then constructs the model from it -- this
tests the *validation* logic without needing a real file on disk.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jaxsr_calibration.calibration.config import ReactorConfig, RotationSchedule, SensorConfig
from jaxsr_calibration.camera.config import CameraConfig
from jaxsr_calibration.processing.config import DiagnosticThresholds, PreprocessingConfig


def test_sensor_config_accepts_spec_example() -> None:
    # This dict is a direct transcription of the worked example in spec §12 --
    # if this test ever fails after an edit to SensorConfig, it means that
    # edit broke compatibility with the spec's own documented example.
    raw = {
        "id": "PID01",
        "model": "PID-AH2",
        "serial": "XXXXXXXX",
        "lamp_install_date": "2026-01-15",
        "lamp_hours_at_install": 0,
        "calibration_gas": "isobutylene",
        # Deliberately using the mixed-case YAML key here (not the snake_case
        # Python attribute name) to prove the `Field(alias=...)` +
        # `populate_by_name` combination in SensorConfig works.
        "factory_sensitivity_mV_per_ppm": 20.0,
        "associated_rh_sensor": "SHT01",
        "associated_reactor": "R01",
    }
    sensor = SensorConfig(**raw)
    # `.lamp_install_date` should have been parsed into a real `date` object,
    # not left as the string "2026-01-15" -- checking `.year` proves that.
    assert sensor.lamp_install_date.year == 2026
    assert sensor.factory_sensitivity_mv_per_ppm == 20.0


def test_sensor_config_also_accepts_snake_case_key() -> None:
    # Because `populate_by_name = True`, the plain Python attribute spelling
    # should work too, not just the YAML alias -- useful if this model is ever
    # constructed directly from Python code rather than parsed YAML.
    raw = {
        "id": "PID02",
        "model": "PID-AH2",
        "serial": "YYYYYYYY",
        "lamp_install_date": "2026-02-01",
        "lamp_hours_at_install": 10,
        "calibration_gas": "isoprene",
        "factory_sensitivity_mv_per_ppm": 18.5,
        "associated_rh_sensor": "SHT02",
        "associated_reactor": "R02",
    }
    sensor = SensorConfig(**raw)
    assert sensor.factory_sensitivity_mv_per_ppm == 18.5


def test_sensor_config_rejects_missing_required_field() -> None:
    # Deliberately omitting "model" (a required field) -- pydantic should
    # refuse to construct the object at all rather than silently defaulting
    # it to None or an empty string.
    raw = {
        "id": "PID03",
        "serial": "ZZZZZZZZ",
        "lamp_install_date": "2026-01-01",
        "lamp_hours_at_install": 0,
        "calibration_gas": "isobutylene",
        "factory_sensitivity_mV_per_ppm": 20.0,
        "associated_rh_sensor": "SHT03",
        "associated_reactor": "R03",
    }
    # `pytest.raises(...)` is a context manager: the test only passes if the
    # code inside the `with` block raises exactly the given exception type.
    with pytest.raises(ValidationError):
        SensorConfig(**raw)


def test_reactor_config_defaults() -> None:
    # Only "id" and "model" are required; the safety-bound fields should fall
    # back to their declared defaults when not specified in YAML.
    reactor = ReactorConfig(id="R01", model="pioreactor_20mL")
    assert reactor.max_par_umol_m2_s == 15000.0
    assert reactor.min_reactor_temp_c < reactor.max_reactor_temp_c


def test_rotation_schedule_assignments() -> None:
    schedule = RotationSchedule(
        period_id="2026-W29",
        assignments={"PID01": "R01", "PID02": "R02"},
    )
    assert schedule.assignments["PID01"] == "R01"


def test_preprocessing_config_matches_spec_example() -> None:
    # Transcribed from the spec's own configs/preprocessing.yaml example (§15).
    raw = {
        "preprocessing_schema_version": 1,
        "covariate_regression": {
            "method": "ols",
            "training_window": "first_30min",
            "min_rh_range_pct": 20,
            "symbolic": {
                "max_terms": 4,
                "basis": {"polynomial_degree": 2, "transcendental": ["log", "exp"]},
            },
        },
        "common_mode": {
            "method": "median",
            "outlier_std_threshold": 3.0,
            "min_healthy_fraction": 0.75,
        },
        "spectral": {
            "known_artifact_freqs_hz": [0.00028, 0.00056],
            "notch_q": 30.0,
            "min_amplitude_to_flag": 0.05,
        },
        "concentration": {"extrapolation_policy": "clip"},
        "features": {"analysis_window": "last_2h_of_run"},
    }
    config = PreprocessingConfig(**raw)
    assert config.covariate_regression.method == "ols"
    assert config.covariate_regression.symbolic is not None
    assert config.covariate_regression.symbolic.basis.polynomial_degree == 2
    assert config.spectral.known_artifact_freqs_hz == [0.00028, 0.00056]


def test_preprocessing_config_rejects_unknown_method() -> None:
    # "lasso" is not one of the Literal["ols", "robust", "symbolic"] options,
    # so this should fail validation rather than silently accepting a typo.
    with pytest.raises(ValidationError):
        PreprocessingConfig(covariate_regression={"method": "lasso"})


def test_preprocessing_config_all_defaults() -> None:
    # Every field has a default, so an empty config should still construct
    # successfully -- this matters because a new user's first preprocessing.yaml
    # might be nearly empty before they've tuned anything.
    config = PreprocessingConfig()
    assert config.preprocessing_schema_version == 1


def test_diagnostic_thresholds_defaults() -> None:
    thresholds = DiagnosticThresholds()
    assert thresholds.swap_pilot.max_sensor_variance_share == 0.30


def test_camera_config_defaults() -> None:
    camera = CameraConfig(id="CAM01", associated_reactor="R01")
    # 60 minutes = hourly, matching the requirement that the camera samples
    # far less often than the ~1 Hz VOC sensor.
    assert camera.capture_interval_min == 60
    # Each hourly capture records a short clip (not a single still photo) --
    # 10s at 10fps by default.
    assert camera.capture_duration_s == 10.0
    assert camera.frame_rate_fps == 10.0
    assert camera.blank_calibration_run_id is None
