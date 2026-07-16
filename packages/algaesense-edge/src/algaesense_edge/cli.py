"""Command-line entry point for algaesense-edge: `algaesense-edge start`
runs acquisition (on a background thread) and the network API (in the
foreground) together, using either real hardware or mocks.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path

import click
import uvicorn
from jaxsr_calibration.calibration.config import ReactorConfig

from algaesense_edge.acquisition.camera import MockCameraCapture, create_hardware_camera_capture
from algaesense_edge.acquisition.i2c import scan_i2c
from algaesense_edge.acquisition.voc import (
    MockTRHSensorReader,
    MockVOCSensorReader,
    create_hardware_trh_reader,
    create_hardware_voc_reader,
)
from algaesense_edge.actuators.actuators import LEDActuator, MockLEDHardware, create_hardware_led
from algaesense_edge.api.app import create_app
from algaesense_edge.api.state import AppState
from algaesense_edge.service import AcquisitionService


@click.group()
def cli() -> None:
    """algaesense-edge: Raspberry Pi sensor acquisition + actuator control."""


@cli.command()
@click.option("--experiment", required=True, help="Experiment ID, e.g. exp_2026-07-25_batch03.")
@click.option("--reactor", required=True, help="Reactor ID this Pi instance is responsible for.")
@click.option("--sensor", required=True, help="VOC sensor ID.")
@click.option("--camera", required=True, help="Camera ID.")
@click.option("--max-par", type=float, required=True, help="This reactor's safety-max PAR, umol/m^2/s.")
@click.option("--par-per-full-duty", type=float, required=True, help="Measured PAR at 100% LED duty cycle (see hardware setup docs).")
@click.option("--led-gpio-pin", type=int, default=17, help="GPIO pin the LED's PWM control line is wired to.")
@click.option("--raw-data-dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--camera-clip-dir", type=click.Path(path_type=Path), default=Path("data/clips"))
@click.option("--camera-interval-min", type=float, default=60.0, help="Minutes between camera captures.")
@click.option("--camera-duration-s", type=float, default=10.0, help="Seconds each camera clip records for.")
@click.option("--camera-fps", type=float, default=10.0, help="Camera recording frame rate.")
@click.option("--host", default="0.0.0.0", help="Network API bind address.")
@click.option("--port", type=int, default=8000, help="Network API port.")
@click.option(
    "--mock-hardware",
    is_flag=True,
    default=False,
    help="Use mock sensors/camera/LED instead of real hardware -- for testing this command's wiring off a Pi.",
)
def start(
    experiment: str,
    reactor: str,
    sensor: str,
    camera: str,
    max_par: float,
    par_per_full_duty: float,
    led_gpio_pin: int,
    raw_data_dir: Path,
    camera_clip_dir: Path,
    camera_interval_min: float,
    camera_duration_s: float,
    camera_fps: float,
    host: str,
    port: int,
    mock_hardware: bool,
) -> None:
    """Start sensor acquisition and the network API together."""

    if mock_hardware:
        voc_reader = MockVOCSensorReader()
        trh_reader = MockTRHSensorReader()
        camera_capture = MockCameraCapture()
        led_hardware = MockLEDHardware()
    else:
        voc_reader = create_hardware_voc_reader()
        trh_reader = create_hardware_trh_reader()
        camera_capture = create_hardware_camera_capture()
        led_hardware = create_hardware_led(gpio_pin=led_gpio_pin)

    state = AppState()
    reactor_config = ReactorConfig(id=reactor, model="pioreactor_20mL", max_par_umol_m2_s=max_par)
    state.led_actuators[reactor] = LEDActuator(
        hardware=led_hardware,
        reactor_config=reactor_config,
        par_per_full_duty_umol_m2_s=par_per_full_duty,
    )

    service = AcquisitionService(
        experiment_id=experiment,
        reactor_id=reactor,
        sensor_id=sensor,
        camera_id=camera,
        voc_reader=voc_reader,
        trh_reader=trh_reader,
        camera_capture=camera_capture,
        camera_clip_dir=camera_clip_dir,
        raw_data_dir=raw_data_dir,
        state=state,
        camera_capture_duration_s=camera_duration_s,
        camera_frame_rate_fps=camera_fps,
    )

    stop_event = threading.Event()

    def acquisition_loop() -> None:
        """A plain background thread running two independent schedules
        (VOC ~every second, camera every camera_interval_min) -- simple
        and sufficient for this project's sampling rates; a real
        scheduler library would be overkill for "do X every second, do Y
        every hour"."""

        next_camera_tick = time.monotonic()
        camera_interval_s = camera_interval_min * 60.0
        while not stop_event.is_set():
            service.run_voc_tick(dt.datetime.now(dt.timezone.utc))
            if time.monotonic() >= next_camera_tick:
                service.run_camera_tick(dt.datetime.now(dt.timezone.utc))
                next_camera_tick = time.monotonic() + camera_interval_s
            """
            ~1 Hz VOC sampling.
            """
            stop_event.wait(1.0)
        service.close()

    acquisition_thread = threading.Thread(target=acquisition_loop, daemon=True)
    acquisition_thread.start()

    try:
        """
        Runs in the foreground (blocks) until interrupted (Ctrl-C) --
        acquisition keeps running on its own thread the whole time.
        """
        uvicorn.run(create_app(state), host=host, port=port)
    finally:
        stop_event.set()
        acquisition_thread.join(timeout=5.0)


@cli.command("scan-i2c")
@click.option("--bus", "bus_number", type=int, default=1, help="I2C bus number, e.g. 1 for /dev/i2c-1.")
def scan_i2c_command(bus_number: int) -> None:
    """Scan the I2C bus and report which addresses have a device on them."""

    try:
        result = scan_i2c(bus_number=bus_number)
    except (ImportError, RuntimeError) as exc:
        """
        Both scan_i2c's "no hardware extra installed" (ImportError) and
        "no I2C bus found" (RuntimeError) cases already carry a clear,
        specific message -- forward it through click's clean-exit
        mechanism rather than letting a raw traceback reach the terminal.
        """
        raise click.ClickException(str(exc)) from exc

    if not result:
        click.echo("No I2C devices responded.")
    for address, status in result.items():
        click.echo(f"  {address}: {status}")


if __name__ == "__main__":
    cli()
