"""Unit tests for algaesense_edge.acquisition.voc.

create_hardware_voc_reader/create_hardware_trh_reader now return REAL
Ads1115VOCSensorReader/Bme280TRHSensorReader instances (not a permanent
NotImplementedError) -- constructing them is real, tested code; only the
first actual hardware READ (which lazily connects) needs real Pi hardware
and isn't runnable here, so that's what's checked below: it fails with a
clear ImportError (since adafruit-circuitpython-ads1x15/-bme280 aren't
installed on this dev machine), not a confusing crash.
"""

from __future__ import annotations

import pytest

from algaesense_edge.acquisition.voc import (
    Ads1115VOCSensorReader,
    Bme280TRHSensorReader,
    MockTRHSensorReader,
    MockVOCSensorReader,
    create_hardware_trh_reader,
    create_hardware_voc_reader,
)


def test_mock_voc_sensor_reader_returns_fixed_voltage() -> None:
    reader = MockVOCSensorReader(voltage_mv=12.5)

    assert reader.read_voltage_mv() == 12.5


def test_mock_voc_sensor_reader_value_can_be_changed_between_reads() -> None:
    # Since this is a real dataclass (not a frozen one), a test can simulate
    # a sensor's reading changing over time by just reassigning the attribute
    # between calls -- no special "set_value" method needed.
    reader = MockVOCSensorReader(voltage_mv=0.0)
    assert reader.read_voltage_mv() == 0.0

    reader.voltage_mv = 5.0
    assert reader.read_voltage_mv() == 5.0


def test_mock_trh_sensor_reader_returns_fixed_readings() -> None:
    reader = MockTRHSensorReader(temperature_c=31.5, humidity_pct=58.0)

    assert reader.read_temperature_c() == 31.5
    assert reader.read_humidity_pct() == 58.0


def test_create_hardware_voc_reader_returns_a_real_ads1115_reader() -> None:
    reader = create_hardware_voc_reader(i2c_address=0x49, channel=2)

    assert isinstance(reader, Ads1115VOCSensorReader)
    assert reader.i2c_address == 0x49
    assert reader.channel == 2


def test_create_hardware_trh_reader_returns_a_real_bme280_reader() -> None:
    reader = create_hardware_trh_reader(i2c_address=0x77)

    assert isinstance(reader, Bme280TRHSensorReader)
    assert reader.i2c_address == 0x77


def test_ads1115_reader_read_fails_clearly_without_hardware_extra_installed() -> None:
    # adafruit-circuitpython-ads1x15 isn't installed on this dev machine (it's
    # part of the Pi-only 'hardware' extra) -- the first actual read attempt
    # should fail with a clear, actionable ImportError, not a confusing
    # traceback deep inside some other module.
    reader = Ads1115VOCSensorReader()
    with pytest.raises(ImportError, match="hardware"):
        reader.read_voltage_mv()


def test_bme280_reader_read_fails_clearly_without_hardware_extra_installed() -> None:
    reader = Bme280TRHSensorReader()
    with pytest.raises(ImportError, match="hardware"):
        reader.read_temperature_c()
