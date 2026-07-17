"""Phase 1b Definition-of-Done, per the project plan: "the service streams
both sensor types at their respective rates to local Parquet, and rejects
an out-of-range LED command over the network while accepting an in-range
one."

Runs against REAL hardware (Alphasense PID + ADS1115, OV5647 camera, WS2811
LED strip) -- proving the whole chain (acquisition -> writer -> Parquet ->
API state -> live network endpoint) actually connects on the real rig, the
same role Milestone 4's end-to-end test played for jaxsr-calibration. Marked
`@pytest.mark.hardware`: run this for real, on the Pi, not on a dev machine
(there is no mocked-hardware version of this test anymore -- see
acquisition/voc.py, acquisition/camera.py, actuators/actuators.py for why).
No T/RH sensor is wired up yet, so this test runs with `trh_reader=None`,
per VOC_RAW_SCHEMA's nullable sample_t_c/sample_rh_pct fields.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from fastapi.testclient import TestClient
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.acquisition.camera import create_hardware_camera_capture
from algaesense_edge.acquisition.voc import create_hardware_voc_reader
from algaesense_edge.actuators.actuators import LEDActuator, create_hardware_led
from algaesense_edge.api.app import create_app
from algaesense_edge.api.state import AppState
from algaesense_edge.service import AcquisitionService

_START = dt.datetime(2026, 7, 25, 8, 0, 0, tzinfo=dt.timezone.utc)

"""
Adjust to the real strip's actual pixel count before running this test
for real.
"""
_LED_NUM_PIXELS = 30


@pytest.mark.hardware
def test_phase1b_streams_both_sensor_types_and_gates_led_commands(tmp_path) -> None:
    state = AppState()
    reactor = ReactorConfig(id="R01", model="pioreactor_20mL", max_par_umol_m2_s=500.0)
    state.led_actuators["R01"] = LEDActuator(
        hardware=create_hardware_led(gpio_pin=18, num_pixels=_LED_NUM_PIXELS, pixel_order="BRG"),
        reactor_config=reactor,
        par_per_full_duty_umol_m2_s=1000.0,
    )

    service = AcquisitionService(
        experiment_id="exp_phase1b_test",
        reactor_id="R01",
        sensor_id="PID01",
        camera_id="CAM01",
        voc_reader=create_hardware_voc_reader(),
        trh_reader=None,
        camera_capture=create_hardware_camera_capture(),
        camera_clip_dir=tmp_path / "clips",
        raw_data_dir=tmp_path / "raw",
        state=state,
        camera_capture_duration_s=0.5,  # short, to keep this test fast
        camera_frame_rate_fps=5.0,
    )

    # --- "streams both sensor types at their respective rates" ---
    # VOC: once per second, several ticks (fast stream).
    for i in range(5):
        service.run_voc_tick(_START + dt.timedelta(seconds=i))
    # Camera: once per hour, only one tick in this same window (slow stream) --
    # this ratio (many VOC ticks per one camera tick) IS "different sampling
    # frequencies", exercised here the same way it happens for real.
    service.run_camera_tick(_START)
    service.close()

    # Real Parquet files landed in the exact layout jaxsr-calibration expects.
    voc_path = tmp_path / "raw" / "experiments" / "exp_phase1b_test" / "sensor_id=PID01" / "hour=2026-07-25T08.parquet"
    camera_path = tmp_path / "raw" / "experiments" / "exp_phase1b_test" / "camera_id=CAM01" / "hour=2026-07-25T08.parquet"
    assert voc_path.exists()
    assert camera_path.exists()
    assert pl.read_parquet(voc_path).height == 5
    assert pl.read_parquet(camera_path).height == 1

    # The API's live buffer reflects the same readings, independent of the
    # Parquet files -- this is what a "brain machine" would actually poll.
    assert len(state.recent_voc_readings()) == 5
    assert len(state.recent_camera_readings()) == 1

    # --- "rejects an out-of-range LED command... while accepting an in-range one" ---
    client = TestClient(create_app(state))

    in_range = client.post("/actuators/led/R01", json={"par_umol_m2_s": 300.0})
    assert in_range.status_code == 200
    assert in_range.json()["applied_par_umol_m2_s"] == 300.0

    out_of_range = client.post("/actuators/led/R01", json={"par_umol_m2_s": 5000.0})
    assert out_of_range.status_code == 422
    assert "exceeds reactor" in out_of_range.json()["detail"]
