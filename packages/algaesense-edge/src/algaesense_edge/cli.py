"""Command-line entry point for algaesense-edge: `algaesense-edge start`
runs acquisition (on a background thread) and the network API (in the
foreground) together, against real hardware.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path

import click
import uvicorn
from jaxsr_calibration.calibration.config import ReactorConfig
from jaxsr_calibration.storage import get_storage_backend

from algaesense_edge.acquisition.camera import create_hardware_camera_capture
from algaesense_edge.acquisition.i2c import scan_i2c
from algaesense_edge.acquisition.voc import create_hardware_trh_reader, create_hardware_voc_reader
from algaesense_edge.actuators.actuators import LEDActuator, create_hardware_led
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
@click.option("--led-gpio-pin", type=int, default=18, help="GPIO pin (BCM numbering) the LED strip's data line is wired to.")
@click.option("--led-num-pixels", type=int, required=True, help="Number of pixels on the WS2811 LED strip.")
@click.option("--led-pixel-order", default="BRG", help="Strip's color channel order, e.g. BRG for this project's ALITOVE strip.")
@click.option(
    "--voc-i2c-address",
    type=lambda s: int(s, 0),
    default="0x48",
    help="ADS1115 I2C address the VOC sensor's ADC responds at.",
)
@click.option(
    "--trh-i2c-address",
    type=lambda s: int(s, 0),
    default=None,
    help="BME280 I2C address, if a temperature/humidity sensor is wired up. Omit to run without one.",
)
@click.option("--raw-data-dir", type=click.Path(path_type=Path), default=Path("data/raw"))
@click.option("--camera-clip-dir", type=click.Path(path_type=Path), default=Path("data/clips"))
@click.option("--camera-interval-min", type=float, default=60.0, help="Minutes between camera captures.")
@click.option("--camera-duration-s", type=float, default=10.0, help="Seconds each camera clip records for.")
@click.option("--camera-fps", type=float, default=10.0, help="Camera recording frame rate.")
@click.option("--host", default="0.0.0.0", help="Network API bind address.")
@click.option("--port", type=int, default=8000, help="Network API port.")
@click.option(
    "--storage-backend",
    type=click.Choice(["none", "local", "firebase", "sftp"]),
    default="none",
    envvar="ALGAESENSE_STORAGE_BACKEND",
    help="Where completed hours' Parquet files go once written. 'none' (default) keeps "
    "everything on this machine's disk, same as before this option existed. See "
    "docs/remote_storage_setup.md.",
)
@click.option(
    "--storage-local-root",
    envvar="ALGAESENSE_STORAGE_LOCAL_ROOT",
    default=None,
    help="Required if --storage-backend=local: a directory (external drive, NAS mount, "
    "any other machine's disk) to mirror raw files into instead of keeping them here.",
)
@click.option(
    "--storage-firebase-credentials",
    envvar="ALGAESENSE_STORAGE_FIREBASE_CREDENTIALS",
    default=None,
    help="Required if --storage-backend=firebase: path to a Firebase service-account JSON key file.",
)
@click.option(
    "--storage-firebase-bucket",
    envvar="ALGAESENSE_STORAGE_FIREBASE_BUCKET",
    default=None,
    help="Required if --storage-backend=firebase: the Firebase Storage bucket name.",
)
@click.option(
    "--storage-sftp-host",
    envvar="ALGAESENSE_STORAGE_SFTP_HOST",
    default=None,
    help="Required if --storage-backend=sftp: hostname/IP of the machine to push files onto "
    "(e.g. your laptop's Tailscale address), which must already be running an SSH server.",
)
@click.option(
    "--storage-sftp-port",
    type=int,
    envvar="ALGAESENSE_STORAGE_SFTP_PORT",
    default=22,
    help="Required if --storage-backend=sftp: SSH port on the destination machine.",
)
@click.option(
    "--storage-sftp-username",
    envvar="ALGAESENSE_STORAGE_SFTP_USERNAME",
    default=None,
    help="Required if --storage-backend=sftp: username to authenticate as on the destination machine.",
)
@click.option(
    "--storage-sftp-private-key",
    envvar="ALGAESENSE_STORAGE_SFTP_PRIVATE_KEY",
    default=None,
    help="Required if --storage-backend=sftp: path to this Pi's private key, "
    "authorized on the destination machine's SSH server.",
)
@click.option(
    "--storage-sftp-remote-root",
    envvar="ALGAESENSE_STORAGE_SFTP_REMOTE_ROOT",
    default=None,
    help="Required if --storage-backend=sftp: directory on the destination machine to write "
    "raw files into (created automatically if it doesn't exist).",
)
def start(
    experiment: str,
    reactor: str,
    sensor: str,
    camera: str,
    max_par: float,
    par_per_full_duty: float,
    led_gpio_pin: int,
    led_num_pixels: int,
    led_pixel_order: str,
    voc_i2c_address: int,
    trh_i2c_address: int | None,
    raw_data_dir: Path,
    camera_clip_dir: Path,
    camera_interval_min: float,
    camera_duration_s: float,
    camera_fps: float,
    host: str,
    port: int,
    storage_backend: str,
    storage_local_root: str | None,
    storage_firebase_credentials: str | None,
    storage_firebase_bucket: str | None,
    storage_sftp_host: str | None,
    storage_sftp_port: int,
    storage_sftp_username: str | None,
    storage_sftp_private_key: str | None,
    storage_sftp_remote_root: str | None,
) -> None:
    """Start sensor acquisition and the network API together, against real hardware."""

    """
    `--raw-data-dir` always stays this machine's *working* buffer -- a
    partial hour's rows always get written here first (writers need
    somewhere to build up a table before it's complete). What
    `--storage-backend` controls is only what happens to a file the
    moment its hour completes: 'none' leaves it here for good (the
    original behavior); 'local'/'firebase'/'sftp' upload it elsewhere
    and delete this copy right away, so raw-data-dir never accumulates
    more than the current, still-being-written hour for each
    sensor/camera.
    """
    remote_storage_backend = get_storage_backend(
        {
            "backend": storage_backend,
            "local_root_dir": storage_local_root,
            "firebase_credentials_path": storage_firebase_credentials,
            "firebase_bucket_name": storage_firebase_bucket,
            "sftp_host": storage_sftp_host,
            "sftp_port": storage_sftp_port,
            "sftp_username": storage_sftp_username,
            "sftp_private_key_path": storage_sftp_private_key,
            "sftp_remote_root_dir": storage_sftp_remote_root,
        }
    )

    voc_reader = create_hardware_voc_reader(i2c_address=voc_i2c_address)

    """
    `trh_reader` stays `None` when `--trh-i2c-address` is omitted -- no
    temperature/humidity sensor is required to run; VOC_RAW_SCHEMA's
    sample_t_c/sample_rh_pct fields are nullable for exactly this case.
    """
    trh_reader = create_hardware_trh_reader(i2c_address=trh_i2c_address) if trh_i2c_address is not None else None

    camera_capture = create_hardware_camera_capture()
    led_hardware = create_hardware_led(
        gpio_pin=led_gpio_pin, num_pixels=led_num_pixels, pixel_order=led_pixel_order
    )

    state = AppState(experiment_id=experiment, raw_data_dir=raw_data_dir)
    reactor_config = ReactorConfig(id=reactor, model="pioreactor_20mL", max_par_umol_m2_s=max_par)
    led_actuator = LEDActuator(
        hardware=led_hardware,
        reactor_config=reactor_config,
        par_per_full_duty_umol_m2_s=par_per_full_duty,
    )
    state.led_actuators[reactor] = led_actuator

    """
    Dual-registered: `led_actuators` (above) backs the manual single-setpoint
    path; `control_actuators` (same object, generic key) is what
    `AcquisitionService.tick_control_profiles` drives -- see AppState's
    own docstring for why these are kept separate.
    """
    state.control_actuators[(reactor, "led")] = led_actuator

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
        remote_storage_backend=remote_storage_backend,
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
            Runs every tick (same ~1 Hz cadence as VOC sampling), not just
            once when a profile starts -- see AcquisitionService.tick_control_profiles.
            """
            profile_results = service.tick_control_profiles(dt.datetime.now(dt.timezone.utc))
            for (reactor_id, actuator_kind), outcome in profile_results.items():
                if outcome == "rejected":
                    click.echo(
                        f"{actuator_kind!r} control profile for reactor {reactor_id!r} rejected an "
                        "unsafe setpoint; profile stopped and actuator turned off."
                    )

            """
            ~1 Hz VOC sampling.
            """
            stop_event.wait(1.0)
        click.echo("acquisition_loop: stop_event set, flushing writers...")
        service.close()
        click.echo("acquisition_loop: writers flushed, thread exiting.")

    acquisition_thread = threading.Thread(target=acquisition_loop, daemon=True)
    acquisition_thread.start()

    try:
        """
        Runs in the foreground (blocks) until interrupted (Ctrl-C) --
        acquisition keeps running on its own thread the whole time.
        """
        uvicorn.run(create_app(state), host=host, port=port)
    finally:
        click.echo("uvicorn.run() returned, signaling acquisition thread to stop...")
        stop_event.set()
        acquisition_thread.join(timeout=5.0)
        click.echo(f"acquisition_thread.join() returned, still alive={acquisition_thread.is_alive()}")


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
