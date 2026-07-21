"""Unit tests for algaesense_edge.acquisition.i2c.scan_i2c.

Only the "no hardware extra installed" error path is a real unit test --
everything else needs a physical I2C bus and is marked
@pytest.mark.hardware, skipped by default.
"""

from __future__ import annotations

import pytest

from algaesense_edge.acquisition.i2c import scan_i2c
from tests.conftest import hardware_extra_importable


def test_scan_i2c_raises_clear_error_without_hardware_extra() -> None:
    # This dev/CI machine does not have smbus2 installed (it's a Linux-only,
    # optional 'hardware' extra -- see pyproject.toml) -- scan_i2c should
    # fail with a clear, actionable ImportError rather than a confusing
    # traceback deep inside some other module.
    if hardware_extra_importable("smbus2"):
        pytest.skip("smbus2 is installed in this environment (e.g. on the Pi) -- "
                     "this test only verifies the ImportError path when it's absent.")
    with pytest.raises(ImportError, match="hardware"):
        scan_i2c()


@pytest.mark.hardware
def test_scan_i2c_on_real_bus_returns_status_dict() -> None:
    # Skipped by default via pyproject.toml's `addopts = "-m 'not hardware'"`.
    # Run with `pytest -m hardware` on a real Raspberry Pi (with smbus2
    # installed and I2C enabled) to actually exercise this.
    result = scan_i2c()
    assert isinstance(result, dict)
    for status in result.values():
        assert status in ("OK", "TIMEOUT", "ERROR")
