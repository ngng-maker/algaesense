"""I2C bus scan: which addresses on the bus have a device that responds."""

from __future__ import annotations

from typing import Literal

I2CStatus = Literal["OK", "TIMEOUT", "ERROR"]


"""
Genuinely hardware-bound -- there is no meaningful way to unit-test the
actual scanning logic without a real I2C bus (a Raspberry Pi with sensors
wired up), so the scanning behavior itself is only covered by an opt-in
@pytest.mark.hardware test. What IS tested in plain unit tests is the clear,
helpful error this raises on any machine that doesn't have the 'hardware'
extra installed -- which is every machine except a properly set-up
Raspberry Pi.

This used to live in jaxsr_calibration (the hardware-agnostic "brain"
package) -- moved here because scanning a physical bus is real hardware I/O,
which belongs with the rest of this package's hardware-touching code
(acquisition/voc.py, actuators/), not inside an analysis package that's
otherwise designed to work from already-collected data alone.
"""


"""
Addresses 0x00-0x02 and 0x78-0x7F are reserved by the I2C specification for
special bus purposes (general call, high-speed mode, etc.) -- real sensors
are never assigned addresses in those ranges, so we don't scan them.
"""
_SCANNABLE_ADDRESS_RANGE = range(0x03, 0x78)


def scan_i2c(bus_number: int = 1) -> dict[str, I2CStatus]:
    """Scan the I2C bus and report which addresses have a device on them."""

    """
    Scans addresses 0x03-0x77 on the given bus. Addresses with no device at
    all are simply omitted from the result (an empty bus position isn't an
    error -- most of the 117 scannable addresses are expected to be empty on
    any real rig).

    Known limitation: smbus2's synchronous read calls don't reliably
    distinguish "the device timed out" from "a generic bus I/O error" on
    typical Linux I2C-dev hardware, so in practice this implementation
    reports most failures as "ERROR" rather than "TIMEOUT" -- TIMEOUT is
    kept in the return type for hardware/drivers where the distinction IS
    available.
    """

    try:
        """
        Imported lazily (inside the function, not at module load time) so
        that importing this module at all doesn't require smbus2 to be
        installed -- only *calling* scan_i2c does. This matters because
        smbus2 uses Linux-only system calls and won't even install on most
        non-Linux development machines.
        """
        import smbus2
    except ImportError as exc:
        raise ImportError(
            "scan_i2c requires the 'hardware' extra (smbus2). Install with "
            "`pip install algaesense-edge[hardware]` on a Raspberry Pi with "
            "I2C enabled."
        ) from exc

    try:
        bus = smbus2.SMBus(bus_number)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"No I2C bus device found at /dev/i2c-{bus_number}. scan_i2c must "
            "run on a machine with real I2C hardware and the bus enabled."
        ) from exc

    results: dict[str, I2CStatus] = {}

    try:
        for address in _SCANNABLE_ADDRESS_RANGE:
            try:
                """
                A zero-length "quick write" is the conventional way I2C
                scanners probe for a device's presence: if any device is
                listening at this address, it acknowledges the write at the
                hardware protocol level even though we're not sending it any
                actual data.
                """
                bus.write_quick(address)
            except OSError:
                """
                No device acknowledged at this address -- expected for the
                overwhelming majority of addresses, not reported as an error.
                """
                continue
            else:
                results[hex(address)] = "OK"
    finally:
        """
        Always release the bus handle, even if something above raised --
        this mirrors the guarantee a `with` block would give, written
        explicitly here because SMBus's own context-manager support varies
        across smbus2 versions.
        """
        bus.close()

    return results
