"""Unit tests for algaesense_edge.acquisition.voc.

Construction and clear-failure-without-hardware-extra behavior are
regular tests (no hardware needed for either). Actually reading a live
voltage/temperature/humidity needs the real ADS1115/BME280 wired up per
this module's confirmed pinout, so those are `@pytest.mark.hardware` --
run for real on the Pi (`pytest -m hardware`), not here.
"""

from __future__ import annotations

import pytest

from algaesense_edge.acquisition.voc import (
    Ads1115VOCSensorReader,
    Bme280TRHSensorReader,
    create_hardware_trh_reader,
    create_hardware_voc_reader,
)
from tests.conftest import hardware_extra_importable


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
    """
    adafruit-circuitpython-ads1x15 (and the Blinka `board`/`busio`
    modules it needs) aren't installed on this dev machine -- the first
    actual read attempt should fail with a clear, actionable ImportError,
    not a confusing traceback deep inside some other module.
    """
    if hardware_extra_importable("board", "busio"):
        pytest.skip("board/busio are installed in this environment (e.g. on the Pi) -- "
                     "this test only verifies the ImportError path when they're absent.")
    reader = Ads1115VOCSensorReader()
    with pytest.raises(ImportError, match="hardware"):
        reader.read_voltage_mv()


def test_bme280_reader_read_fails_clearly_without_hardware_extra_installed() -> None:
    if hardware_extra_importable("board", "busio"):
        pytest.skip("board/busio are installed in this environment (e.g. on the Pi) -- "
                     "this test only verifies the ImportError path when they're absent.")
    reader = Bme280TRHSensorReader()
    with pytest.raises(ImportError, match="hardware"):
        reader.read_temperature_c()


@pytest.mark.hardware
def test_ads1115_reader_reads_a_real_voltage_from_hardware() -> None:
    """Run only on the Pi, with the ADS1115 and Alphasense PID sensor
    wired per this module's confirmed pinout (0x48, channel 0)."""
    reader = create_hardware_voc_reader()
    voltage_mv = reader.read_voltage_mv()
    assert 0.0 <= voltage_mv <= 3300.0


@pytest.mark.hardware
def test_bme280_reader_reads_real_temperature_and_humidity_from_hardware() -> None:
    """Run only on the Pi, once a BME280 is actually wired up -- not yet
    acquired as of 2026-07-16, so this will fail until then."""
    reader = create_hardware_trh_reader()
    assert -40.0 <= reader.read_temperature_c() <= 85.0
    assert 0.0 <= reader.read_humidity_pct() <= 100.0
