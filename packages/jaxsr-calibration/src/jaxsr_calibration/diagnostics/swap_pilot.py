"""Sensor-swap Latin-square pilot: separate "this reading is weird because
of the sensor" from "this reading is weird because of the reactor" by
rotating sensors across reactors and fitting a mixed-effects variance
decomposition (spec §22).
"""

from __future__ import annotations

import polars as pl
import statsmodels.formula.api as smf

from jaxsr_calibration.errors import LiveAcquisitionNotAvailableError
from jaxsr_calibration.diagnostics.models import SwapPilotResult


def run_swap_pilot(
    n_blocks: int = 4,
    block_hours: int = 4,
    *,
    readings: pl.DataFrame | None = None,
) -> SwapPilotResult:
    """Fit a crossed-random-effects model partitioning voltage variance
    into a sensor_id share, a reactor_id share, and an unexplained
    residual share."""

    """
    `n_blocks`/`block_hours` document the intended physical rotation
    protocol (spec §22: N blocks of `block_hours` each, sensors rotated
    one reactor-position between blocks) -- same "no live acquisition
    backend yet" situation as the other diagnostics in this milestone, so
    they are accepted for signature compatibility but not used directly;
    the actual rotation structure just needs to already be reflected in
    `readings` (each sensor_id paired with more than one distinct
    reactor_id across rows).

    Statistical approach: statsmodels' MixedLM supports *crossed* random
    effects (sensor and reactor are crossed, not nested -- every sensor
    visits every reactor, not "sensors within reactors") via its
    `vc_formula` ("variance components formula") argument, fit under one
    trivial top-level `groups` (every row in the same group), which is the
    standard statsmodels recipe for this. See this module's tests for a
    worked numeric example.
    """

    if readings is None:
        raise LiveAcquisitionNotAvailableError(
            "run_swap_pilot has no live-acquisition backend yet; pass "
            "readings=<a DataFrame spanning the swap-pilot rotation> instead."
        )

    """
    statsmodels' formula API (`smf.mixedlm`) works on pandas DataFrames,
    not polars ones, so we convert here at the boundary -- everywhere else
    in this package works in polars.
    """
    pdf = readings.select(["sensor_id", "reactor_id", "pid_voltage_mv"]).to_pandas()

    """
    MixedLM always requires a top-level `groups` argument (which rows
    belong to the same "cluster"). We don't want any real top-level
    grouping here -- sensor_id and reactor_id themselves are the structure
    we care about -- so every row gets the same constant group label,
    making the top-level grouping a no-op and letting `vc_formula` alone
    carry the sensor/reactor crossed-random-effects structure.
    """
    pdf["_all_one_group"] = "all"

    """
    `vc_formula` maps a name we choose ("sensor_id", "reactor_id") to a
    patsy formula describing a random-effects design matrix. `"0 +
    C(sensor_id)"` means "one indicator column per distinct sensor_id
    value, no shared intercept" -- i.e. "let each sensor have its own
    random deviation from the overall mean". Same idea for reactor_id.
    """
    model = smf.mixedlm(
        "pid_voltage_mv ~ 1",
        pdf,
        groups=pdf["_all_one_group"],
        vc_formula={
            "sensor_id": "0 + C(sensor_id)",
            "reactor_id": "0 + C(reactor_id)",
        },
    )

    """
    When a true variance component is at or near zero (e.g. a very
    healthy sensor fleet with almost no sensor-to-sensor variation), the
    optimizer is fitting right at the boundary of what's mathematically
    allowed (variance can't be negative) -- statsmodels' default
    optimizer sometimes struggles to converge cleanly in exactly that
    case. Passing a list of optimizers makes `.fit()` retry with
    alternatives (lbfgs/cg/powell, all standard general-purpose
    optimizers) instead of giving up after the first one struggles.
    """
    result = model.fit(method=["lbfgs", "cg", "powell"])

    """
    `model.exog_vc.names` gives the vc_formula component names in the
    same order as `result.vcomp`'s values -- zipping them together is how
    we recover "which variance number belongs to which effect".
    """
    variances = dict(zip(model.exog_vc.names, result.vcomp))
    residual_variance = float(result.scale)
    total_variance = sum(variances.values()) + residual_variance

    variance_share = {name: float(v) / total_variance for name, v in variances.items()}
    variance_share["residual"] = residual_variance / total_variance

    """
    `result.summary()` returns a statsmodels Summary object; str(...)
    renders it as the same human-readable text block you'd see printed in
    a notebook, which is what spec §22's "mixedlm_summary" is for -- an
    operator reading the raw fit output, not just the three numbers.
    """
    return SwapPilotResult(
        variance_share=variance_share,
        mixedlm_summary=str(result.summary()),
    )
