"""Shared exceptions used by more than one jaxsr_calibration subpackage."""

from __future__ import annotations


"""
Previously duplicated identically in calibration/errors.py, camera/errors.py,
and diagnostics/errors.py. Centralized here instead, since importing one
shared marker exception doesn't create any meaningful coupling between those
subpackages -- it just avoids three copies of the same class drifting out of
sync with each other.
"""


class LiveAcquisitionNotAvailableError(RuntimeError):
    """Raised when a function needs live sensor/camera hardware that isn't available yet."""

    """
    Several functions across calibration, camera, and diagnostics are split
    into a real, tested "analysis" half (operating on already-collected data)
    and a "drive live hardware" half that doesn't exist yet -- that's
    algaesense_edge, the Raspberry-Pi-side package. This is what those
    live-hardware halves raise. Pass already-collected data via the
    function's `readings=` (or equivalent) keyword argument instead, or call
    this function once algaesense_edge is available.
    """
