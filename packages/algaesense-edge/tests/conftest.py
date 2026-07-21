"""Shared pytest fixtures/helpers for algaesense-edge's test suite."""

from __future__ import annotations

import importlib.util


def hardware_extra_importable(*module_names: str) -> bool:
    """True if every one of `module_names` is importable in this environment."""

    """
    The "fails clearly without the hardware extra installed" tests (e.g.
    test_led.py's test_neopixel_hardware_fails_clearly_without_hardware_extra_installed)
    only make sense on a machine where these libraries genuinely aren't
    installed -- their whole point is confirming a clear ImportError, not a
    confusing traceback, when someone forgot `pip install algaesense-edge[hardware]`.
    On the Raspberry Pi, where the hardware extra IS (correctly) installed
    for real use, that precondition doesn't hold, and the same call instead
    hits a real hardware/permission error (no root, no device wired up,
    etc.) -- a different, unrelated failure that isn't a regression. This
    helper lets those tests self-skip when their precondition is already
    false, rather than permanently failing on the one machine that's
    actually supposed to have the extra installed.
    """
    return all(importlib.util.find_spec(name) is not None for name in module_names)
