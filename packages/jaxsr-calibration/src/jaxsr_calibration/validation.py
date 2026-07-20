"""Shared input-validation helpers used across calibration/diagnostics/processing."""

from __future__ import annotations

import polars as pl


"""
Two distinct exception types on purpose, matching what the codebase already
does at individual call sites before this module existed: a `ValueError` for
"this DataFrame doesn't have the shape this function needs" (a caller data
problem), and a `NotImplementedError` for "this function knows about this
method name but hasn't implemented it yet" (a planned-but-not-built
capability). Conflating the two into one exception type would make a caller's
`except` clause unable to tell "you gave me bad data" apart from "you asked
for something that doesn't exist yet".
"""


def require_columns(df: pl.DataFrame, required: set[str] | frozenset[str], fn_name: str) -> None:
    """Raise a clear ValueError if `df` is missing any of `required` columns."""

    """
    Several functions across this package used to check this inline (e.g.
    `standard_addition.fit_sensitivity_per_sensor`'s own `_REQUIRED_COLUMNS`
    check) while others didn't check at all, surfacing a raw polars
    `ColumnNotFoundError`/`SchemaError` from wherever the first bad column
    access happened -- not a message naming which function was called or
    which column was actually missing. Centralizing here means every
    `readings=`/`df=` entry point gets the same caller-legible message for
    free.
    """
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fn_name}: df is missing required columns: {sorted(missing)}")


def require_implemented_method(method: str, implemented: set[str] | frozenset[str], fn_name: str) -> None:
    """Raise a clear NotImplementedError if `method` isn't one of `implemented`."""

    """
    Called at the EARLIEST reachable point in a function -- before any
    per-sensor loop, partition_by, or upstream I/O -- not merely wherever a
    fit happens to be attempted. A guard placed inside a loop can be
    silently skipped entirely if the loop never executes (e.g. an empty/
    zero-sensor `readings` frame), letting an invalid method pass through
    unnoticed rather than being rejected.
    """
    if method not in implemented:
        raise NotImplementedError(
            f"{fn_name}(method={method!r}) is not implemented yet; "
            f"only {sorted(implemented)} are available so far."
        )
