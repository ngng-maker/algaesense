"""Ties acquisition (sensor readers, camera capture, Parquet writers) and
the API's live-reading buffer together into one thing a caller can drive
one "tick" at a time.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from jaxsr_calibration.logging_.schema import CAMERA_RAW_SCHEMA, VOC_RAW_SCHEMA

from algaesense_edge.acquisition.camera import CameraCapture, process_clip
from algaesense_edge.acquisition.voc import TRHSensorReader, VOCSensorReader
from algaesense_edge.acquisition.writer import PartitionedParquetWriter
from algaesense_edge.actuators.actuators import UnsafeSetpointError
from algaesense_edge.actuators.control_profiles import evaluate_control_profile
from algaesense_edge.api.state import AppState


"""
Deliberately exposes `run_voc_tick`/`run_camera_tick` as individually
callable methods rather than only an internal infinite loop -- that's
what makes this testable (call a tick a known number of times, inspect
exactly what happened) instead of only being verifiable by actually
running it forever on real hardware. `cli.py`'s `start` command is the
thin wrapper that calls these on a real schedule (once/second, once/hour)
forever.
"""


@dataclass
class AcquisitionService:
    """One instance per (experiment, reactor, sensor, camera): turns
    hardware readings into raw Parquet rows and live API state, tick by
    tick."""

    experiment_id: str

    reactor_id: str

    sensor_id: str

    camera_id: str

    voc_reader: VOCSensorReader

    camera_capture: CameraCapture

    camera_clip_dir: Path

    raw_data_dir: Path

    state: AppState

    """
    Optional: no T/RH sensor is wired up on this rig yet. When absent,
    `run_voc_tick` records null `sample_t_c`/`sample_rh_pct` -- fields
    VOC_RAW_SCHEMA already allows to be null for exactly this reason (not
    every rig has every ancillary sensor).
    """
    trh_reader: TRHSensorReader | None = None

    camera_capture_duration_s: float = 10.0

    camera_frame_rate_fps: float = 10.0

    lamp_hours: float = 0.0

    light_state: str = "on"

    def __post_init__(self) -> None:
        self.voc_writer = PartitionedParquetWriter(
            base_dir=self.raw_data_dir,
            experiment_id=self.experiment_id,
            partition_key="sensor_id",
            partition_value=self.sensor_id,
            schema=VOC_RAW_SCHEMA,
        )
        self.camera_writer = PartitionedParquetWriter(
            base_dir=self.raw_data_dir,
            experiment_id=self.experiment_id,
            partition_key="camera_id",
            partition_value=self.camera_id,
            schema=CAMERA_RAW_SCHEMA,
        )
        self.camera_clip_dir.mkdir(parents=True, exist_ok=True)

    def run_voc_tick(self, timestamp: dt.datetime) -> dict:
        """Read the VOC + T/RH sensors once, write the row to Parquet, and
        make it available to the API's recent-readings buffer."""

        """
        `reactor_par_umol_m2_s` reads AppState's generic "last
        successfully applied setpoint" cache (see
        AppState.last_applied_setpoint, updated by the manual
        `/actuators/led/{reactor_id}` endpoint and by
        `tick_control_profiles`) rather than real hardware -- avoids
        adding a hardware touch to this ~1Hz tick just to record what the
        LED is doing. Two known, deliberately-accepted limitations: (1)
        doesn't survive a mid-experiment service restart, since the cache
        starts empty even if the physical LED is still lit from before
        the restart; (2) a one-tick (~1s) lag, since this tick runs BEFORE
        `tick_control_profiles` in cli.py's loop, so this row reflects the
        PREVIOUS tick's applied value, not the one about to be set this
        tick -- negligible next to the PID sensor's own physical response
        time, but worth knowing if debugging an apparent off-by-one-second
        light/VOC correlation.
        """
        row = {
            "timestamp": timestamp,
            "experiment_id": self.experiment_id,
            "sensor_id": self.sensor_id,
            "reactor_id": self.reactor_id,
            "pid_voltage_mv": self.voc_reader.read_voltage_mv(),
            "sample_t_c": self.trh_reader.read_temperature_c() if self.trh_reader is not None else None,
            "sample_rh_pct": self.trh_reader.read_humidity_pct() if self.trh_reader is not None else None,
            "sample_flow_sccm": None,
            "pump_pwm": None,
            "lamp_hours": self.lamp_hours,
            "reactor_par_umol_m2_s": self.state.last_applied_setpoint.get((self.reactor_id, "led")),
            "reactor_temp_c": None,
            "reactor_od": None,
            "reactor_ph": None,
            "light_state": self.light_state,
            "room_t_c": None,
            "room_rh_pct": None,
            "acquisition_status": "OK",
        }
        self.voc_writer.write_row(row)
        self.state.record_voc_reading(row)
        return row

    def run_camera_tick(self, timestamp: dt.datetime) -> dict:
        """Record and process one camera clip, write the row to Parquet,
        and make it available to the API's recent-readings buffer."""

        """
        Colons aren't valid in Windows filenames, hence the replace --
        same reasoning as jaxsr_calibration.diagnostics.fleet_zero's
        run_id.
        """
        clip_name = f"{self.camera_id}_{timestamp.isoformat().replace(':', '-')}.mp4"
        clip_path = self.camera_clip_dir / clip_name

        self.camera_capture.record_clip(
            duration_s=self.camera_capture_duration_s,
            frame_rate_fps=self.camera_frame_rate_fps,
            out_path=clip_path,
        )
        features = process_clip(clip_path)

        row = {
            "timestamp": timestamp,
            "experiment_id": self.experiment_id,
            "reactor_id": self.reactor_id,
            "camera_id": self.camera_id,
            "video_path": str(clip_path),
            "capture_duration_s": self.camera_capture_duration_s,
            "frame_rate_fps": self.camera_frame_rate_fps,
            "frame_count": features.frame_count,
            "image_feature_vector": features.rgb,
            "exposure_us": None,
            "gain": None,
            "light_state": self.light_state,
            "acquisition_status": "OK",
        }
        self.camera_writer.write_row(row)
        self.state.record_camera_reading(row)
        return row

    def tick_control_profiles(self, now: dt.datetime) -> dict[tuple[str, str], str]:
        """Evaluate every (reactor, actuator_kind)'s currently-running
        control profile (if any) and apply the resulting setpoint.
        Returns, per `(reactor_id, actuator_kind)` that had an active
        profile, one of "applied" or "rejected" -- callers (cli.py's loop)
        can print/act on a rejection, but nothing here raises."""

        """
        Re-evaluating and re-applying on every single tick (not just once
        at profile start) is what makes this safe: `apply_setpoint()`
        re-runs its own bounds-check every time (`LEDActuator.set_par()`
        underneath, for the one actuator kind that exists today), so a
        profile can never "outrun" the actuator's safety limits even
        though the profile's own math is trusted, unreviewed-per-run data
        (see control_profiles.py's module docstring). Driven generically
        through `ControlProfileActuator` (see actuators.py) so this loop
        doesn't need to know whether it's actually driving an LED or a
        future heater/stirrer.
        """
        results: dict[tuple[str, str], str] = {}
        for (reactor_id, actuator_kind), active_profile in list(self.state.active_control_profiles.items()):
            actuator = self.state.control_actuators.get((reactor_id, actuator_kind))
            if actuator is None:
                continue

            elapsed_s = (now - active_profile.started_at).total_seconds()
            target_value = evaluate_control_profile(active_profile.profile, elapsed_s)

            try:
                applied = actuator.apply_setpoint(target_value)
            except UnsafeSetpointError:
                """
                A profile that ever asks for an out-of-bounds value is
                stopped outright rather than clamped and continued -- a
                profile whose math produces an unsafe value is a bad
                profile, not a one-off blip to silently paper over.
                """
                actuator.turn_off()
                self.state.last_applied_setpoint[(reactor_id, actuator_kind)] = 0.0
                self.state.stop_control_profile(reactor_id, actuator_kind)
                results[(reactor_id, actuator_kind)] = "rejected"
            else:
                self.state.last_applied_setpoint[(reactor_id, actuator_kind)] = applied
                results[(reactor_id, actuator_kind)] = "applied"

        return results

    def close(self) -> None:
        """Flush any buffered-but-unwritten rows -- call when acquisition stops."""
        self.voc_writer.close()
        self.camera_writer.close()
