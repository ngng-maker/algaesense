"""VOC PID sensor (and its companion T/RH sensor) reading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


"""
The exact ADC chip and T/RH sensor model were left as open ⚠️ hardware
decisions in the original spec ("ADS1115 or MCP3428"; "SHT35 or BME280")
and never resolved. Since this package needs to be genuinely installable
and usable on a real Pi right away, this module commits to concrete,
well-documented defaults rather than waiting indefinitely: an ADS1115 ADC
and a BME280 T/RH sensor, both extremely common on Raspberry Pi projects,
each with a stable, actively-maintained Adafruit CircuitPython driver.

Honesty note: these real-hardware code paths cannot be run or verified on
a non-Pi dev machine (`board`/`busio`, from Adafruit Blinka, only work on
actual supported single-board-computer hardware) -- they're built against
each library's well-known, standard, widely-documented usage pattern, but
are untested here by nature, same situation as jaxsr_calibration's own
scan_i2c. They're marked `@pytest.mark.hardware` and skipped by default;
the mock readers below remain the fully-tested path for everything else.
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
class MockVOCSensorReader:
    """A `VOCSensorReader` that always returns a fixed (or externally
    updated) voltage."""

    """
    Used for local development and tests on any machine, with no real
    hardware involved.
    """

    voltage_mv: float = 0.0

    def read_voltage_mv(self) -> float:
        return self.voltage_mv


@dataclass
class MockTRHSensorReader:
    """A `TRHSensorReader` counterpart to MockVOCSensorReader."""

    temperature_c: float = 25.0

    humidity_pct: float = 50.0

    def read_temperature_c(self) -> float:
        return self.temperature_c

    def read_humidity_pct(self) -> float:
        return self.humidity_pct


@dataclass
class Ads1115VOCSensorReader:
    """Real `VOCSensorReader` backed by an ADS1115 ADC over I2C, via the
    `adafruit-circuitpython-ads1x15` library."""

    """
    Part of this package's `[hardware]` extra -- NOT hand-rolled I2C
    register writes.

    The PID sensor's analog output connects to one of the ADS1115's four
    input channels (`channel`, 0-3); `i2c_address` defaults to the chip's
    standard address (0x48 when its ADDR pin is tied to ground, the most
    common wiring).
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
            import adafruit_ads1x15.ads1115 as ads1115_module
            import board
            import busio
            from adafruit_ads1x15.analog_in import AnalogIn
        except ImportError as exc:
            raise ImportError(
                "Ads1115VOCSensorReader requires the 'hardware' extra "
                "(adafruit-circuitpython-ads1x15). Install with "
                "`pip install algaesense-edge[hardware]` on a Raspberry Pi."
            ) from exc

        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ads1115_module.ADS1115(i2c, address=self.i2c_address)

        """
        ADS1115's four input channels are addressed via constants P0-P3;
        `getattr(ads1115_module, f"P{n}")` looks up the right one from
        `self.channel` without needing a 4-branch if/elif chain.
        """
        channel_pin = getattr(ads1115_module, f"P{self.channel}")
        self._analog_in = AnalogIn(ads, channel_pin)

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
    `adafruit-circuitpython-bme280` library."""

    """
    0x76 is BME280's default address when its SDO pin is tied to ground
    (the more common wiring); some breakout boards default to 0x77
    instead -- check the specific board's datasheet if 0x76 doesn't
    respond.
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
    `Bme280TRHSensorReader`)."""
    return Bme280TRHSensorReader(i2c_address=i2c_address)
