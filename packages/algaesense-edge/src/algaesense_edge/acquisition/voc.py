"""VOC PID sensor (and optional companion T/RH sensor) reading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


"""
Confirmed real hardware (2026-07-16): an Alphasense PID sensor with its
ISB (Individual Sensor Board) for signal conditioning, outputting an
analog 0-3.3V signal -- read via an ADS1115 ADC (Adafruit STEMMA QT
variant) over I2C at address 0x48 (its default when ADDR is tied to
GND/floating, confirmed via i2cdetect), single-ended on channel 0 (A0).

A companion T/RH sensor is not yet acquired -- `TRHSensorReader` stays
optional everywhere it's used (service.py, cli.py); a row with no T/RH
reading simply has `sample_t_c`/`sample_rh_pct` as null, which
VOC_RAW_SCHEMA already supports (see jaxsr_calibration.logging_.schema --
those fields are nullable precisely because not every rig has every
ancillary sensor). `Bme280TRHSensorReader` below is a real, ready-to-use
driver for when a BME280 is wired up later, not a placeholder.

Honesty note: these real-hardware code paths cannot be run or verified on
a non-Pi dev machine (`board`/`busio`, from Adafruit Blinka, only work on
actual supported single-board-computer hardware) -- they're built against
each library's well-known, standard, widely-documented usage pattern, and
against this project's own confirmed wiring, but are untested on this dev
machine by nature, same situation as jaxsr_calibration's own scan_i2c.
They're marked `@pytest.mark.hardware` and skipped by default here; only
run for real on the Pi itself.
"""


class VOCSensorReader(Protocol):
    """Anything that can report the PID sensor's current raw voltage."""

    def read_voltage_mv(self) -> float:
        """Return the sensor's current output voltage, in millivolts."""
        ...


class TRHSensorReader(Protocol):
    """Anything that can report the companion T/RH sensor's current reading."""

    def read_temperature_c(self) -> float: ...
    def read_humidity_pct(self) -> float: ...


@dataclass
class Ads1115VOCSensorReader:
    """Real `VOCSensorReader` backed by an ADS1115 ADC over I2C, via the
    `adafruit-circuitpython-ads1x15` library."""

    """
    Part of this package's `[hardware]` extra -- NOT hand-rolled I2C
    register writes.

    The Alphasense PID's ISB output connects to the ADS1115's channel 0
    (A0), single-ended (the ISB's 0V signal return goes to GND, not a
    second ADS1115 input) -- `channel` stays configurable in case a future
    rig wires it elsewhere, but 0 is this project's actual, tested wiring.
    `i2c_address` defaults to 0x48, this ADS1115 board's confirmed address
    (ADDR pin tied to GND).
    """

    i2c_address: int = 0x48

    channel: int = 0

    """
    Deliberately UNANNOTATED (unlike i2c_address/channel above) -- an
    unannotated attribute on a @dataclass-decorated class is just a
    normal class attribute, not a dataclass field, so it does NOT appear
    in the generated __init__ or __repr__ (we don't want callers passing a
    pre-made connection object in, and don't want it dumped in a repr).
    `_connect()` sets a real per-instance value the first time it runs.
    """
    _analog_in = None

    def _connect(self):
        """Open the actual I2C/ADC connection, the first time it's needed."""

        """
        Imported lazily (inside the method, not at module load time) so
        importing this module doesn't require these packages to be
        installed -- only actually connecting to hardware does. Mirrors
        jaxsr_calibration.diagnostics.i2c.scan_i2c's same lazy-import
        pattern, for the same reason: these are Pi-only dependencies.
        """
        try:
            import board
            import busio
            from adafruit_ads1x15.analog_in import AnalogIn
            from adafruit_ads1x15.ads1115 import ADS1115
        except ImportError as exc:
            raise ImportError(
                "Ads1115VOCSensorReader requires the 'hardware' extra "
                "(adafruit-circuitpython-ads1x15). Install with "
                "`pip install algaesense-edge[hardware]` on a Raspberry Pi."
            ) from exc

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS1115(i2c, address=self.i2c_address)

        """
        `AnalogIn(ads, self.channel)` takes the channel directly as a
        plain integer (0-3) -- confirmed working against real hardware.
        The library's `P0`-style module constants are just aliases for
        these same integers; passing the int directly avoids an
        unnecessary `getattr` lookup for something that was never
        anything but a plain channel number.
        """
        self._analog_in = AnalogIn(ads, self.channel)

    def read_voltage_mv(self) -> float:
        if self._analog_in is None:
            self._connect()

        """
        `.voltage` on an AnalogIn reads the channel and returns volts;
        this project's raw schema stores millivolts (VOC_RAW_SCHEMA's
        `pid_voltage_mv`), hence the *1000 conversion at this boundary.
        """
        return self._analog_in.voltage * 1000.0


@dataclass
class Bme280TRHSensorReader:
    """Real `TRHSensorReader` backed by a BME280 over I2C, via the
    `adafruit-circuitpython-bme280` library. Not yet wired up on this
    rig -- ready to use the moment one is."""

    """
    0x76 is BME280's default address when its SDO pin is tied to ground
    (the more common wiring); some breakout boards default to 0x77
    instead -- check the specific board's datasheet if 0x76 doesn't
    respond, once this sensor is actually connected.
    """

    i2c_address: int = 0x76

    """
    Same "deliberately unannotated" reasoning as Ads1115VOCSensorReader's
    _analog_in above.
    """
    _sensor = None

    def _connect(self):
        try:
            import board
            import busio
            from adafruit_bme280 import basic as adafruit_bme280
        except ImportError as exc:
            raise ImportError(
                "Bme280TRHSensorReader requires the 'hardware' extra "
                "(adafruit-circuitpython-bme280). Install with "
                "`pip install algaesense-edge[hardware]` on a Raspberry Pi."
            ) from exc

        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_bme280.Adafruit_BME280_I2C(i2c, address=self.i2c_address)

    def read_temperature_c(self) -> float:
        if self._sensor is None:
            self._connect()
        return float(self._sensor.temperature)

    def read_humidity_pct(self) -> float:
        if self._sensor is None:
            self._connect()
        return float(self._sensor.humidity)


def create_hardware_voc_reader(i2c_address: int = 0x48, channel: int = 0) -> VOCSensorReader:
    """Construct the real ADS1115-backed VOC sensor reader (see
    `Ads1115VOCSensorReader`)."""
    return Ads1115VOCSensorReader(i2c_address=i2c_address, channel=channel)


def create_hardware_trh_reader(i2c_address: int = 0x76) -> TRHSensorReader:
    """Construct the real BME280-backed T/RH sensor reader (see
    `Bme280TRHSensorReader`). Only call this once a BME280 is actually
    wired up -- until then, run without a T/RH reader at all (see
    service.py/cli.py's `trh_reader: TRHSensorReader | None`)."""
    return Bme280TRHSensorReader(i2c_address=i2c_address)
