"""MCP server exposing jaxsr-calibration's sensor-health diagnostics as
tools Hermes can call. Read-only, same trust level as mcp_pipeline -- no
propose/apply split needed, nothing here writes to a live experiment or
touches hardware.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from algaesense_agent.mcp_diagnostics.diagnostics import (
    ambient_baseline_check,
    fleet_zero_check,
    swap_pilot_check,
    weekly_audit_check,
)

mcp = FastMCP("algaesense-diagnostics")


def _data_dir() -> Path:
    return Path(os.environ.get("ALGAESENSE_DATA_DIR", "data"))


@mcp.tool()
def run_fleet_zero_check(experiment_id: str, duration_min: int = 60) -> dict:
    """Check every sensor reads ~0 with low noise on clean air, from an
    already-collected clean-air experiment run."""
    result = fleet_zero_check(_data_dir(), experiment_id, duration_min)
    return asdict(result)


@mcp.tool()
def run_ambient_baseline_check(experiment_id: str, duration_h: int = 12, method: str = "ols") -> dict:
    """Characterize each sensor's response to room temperature/humidity,
    from an already-collected ambient-air experiment run."""
    result = ambient_baseline_check(_data_dir(), experiment_id, duration_h, method)

    """
    `CovariateModel.covariance` is a numpy array and `.symbolic_regressor`
    can be a live, unpicklable jaxsr.SymbolicRegressor -- neither survives
    plain `asdict()` + JSON, so each model gets converted by hand rather
    than trusting the generic dataclass conversion here.
    """
    covariate_models = {
        sensor_id: {
            "sensor_id": model.sensor_id,
            "method": model.method,
            "alpha": model.alpha,
            "beta_rh": model.beta_rh,
            "gamma_t": model.gamma_t,
            "delta_rh_t": model.delta_rh_t,
            "covariance": model.covariance.tolist() if model.covariance is not None else None,
            "has_symbolic_regressor": model.symbolic_regressor is not None,
        }
        for sensor_id, model in result.covariate_models.items()
    }
    return {"covariate_models": covariate_models, "r_squared_per_sensor": result.r_squared_per_sensor}


@mcp.tool()
def run_swap_pilot_check(experiment_id: str, n_blocks: int = 4) -> dict:
    """Latin-square sensor/reactor swap to separate sensor vs. reactor
    effects, from an already-collected rotation experiment run."""
    result = swap_pilot_check(_data_dir(), experiment_id, n_blocks)
    return asdict(result)


@mcp.tool()
def run_weekly_audit_check(
    swap_pilot_variance_shares: list[dict[str, float]] | None = None,
    sensors_yaml_path: str | None = None,
    backup_current: bool | None = None,
    lamp_cleaning_age_days: int = 180,
) -> dict:
    """Compose the weekly diagnostic rollup from swap-pilot history
    (oldest first, e.g. from prior run_swap_pilot_check calls) and the
    sensor fleet's config file."""
    result = weekly_audit_check(
        swap_pilot_variance_shares=swap_pilot_variance_shares,
        sensors_yaml_path=Path(sensors_yaml_path) if sensors_yaml_path is not None else None,
        backup_current=backup_current,
        lamp_cleaning_age_days=lamp_cleaning_age_days,
    )
    payload = asdict(result)
    payload["report_path"] = str(result.report_path) if result.report_path is not None else None
    return payload


def main() -> None:
    """Entry point for the `algaesense-mcp-diagnostics` console script."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
