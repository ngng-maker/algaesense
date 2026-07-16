"""Top-level package for jaxsr-calibration."""

"""
This package is a standalone import root (`import jaxsr_calibration`), NOT
a merge into the upstream `jaxsr` package's namespace. Upstream `jaxsr` has
no plugin/entry-point mechanism for external packages to inject themselves
into `jaxsr.*`, so instead this package simply *depends on* `jaxsr` as a
normal library and imports it directly wherever it needs
`SymbolicRegressor`, `Constraints`, etc. (see jaxsr_calibration.camera and
.processing for where that happens once those modules exist).

`__version__` is a convention (not a Python language requirement) that
lets other code/tools introspect the installed version at runtime, e.g.
`import jaxsr_calibration; print(jaxsr_calibration.__version__)`.
"""

__version__ = "0.1.0"
