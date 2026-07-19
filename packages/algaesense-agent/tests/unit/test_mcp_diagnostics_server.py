"""Unit tests for the mcp_diagnostics FastMCP server's tool wrappers --
specifically that the custom CovariateModel serialization (numpy array,
possibly-live SymbolicRegressor) actually survives a real MCP tool call,
not just the plain Python function underneath it.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
from pathlib import Path

import numpy as np
from algaesense_edge.acquisition.writer import PartitionedParquetWriter
from jaxsr_calibration.logging_.schema import VOC_RAW_SCHEMA

_START = dt.datetime(2026, 7, 17, 8, 0, 0, tzinfo=dt.timezone.utc)


def _payload(raw_result):
    """See test_labwiki_server.py's `_tool_payload` for why both shapes
    need handling."""
    if isinstance(raw_result, tuple):
        return raw_result[1]["result"]
    return json.loads(raw_result[0].text)


async def test_ambient_baseline_tool_serializes_the_covariate_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALGAESENSE_DATA_DIR", str(tmp_path))

    from algaesense_agent.mcp_diagnostics import server as server_module

    importlib.reload(server_module)

    rng = np.random.default_rng(3)
    writer = PartitionedParquetWriter(
        base_dir=tmp_path / "raw", experiment_id="exp_ambient", partition_key="sensor_id", partition_value="PID01", schema=VOC_RAW_SCHEMA
    )
    for i, rh in enumerate(np.linspace(30.0, 70.0, 20)):
        writer.write_row(
            {
                "timestamp": _START + dt.timedelta(minutes=i),
                "experiment_id": "exp_ambient",
                "sensor_id": "PID01",
                "reactor_id": "R01",
                "pid_voltage_mv": 2.0 + 0.05 * rh,
                "sample_t_c": float(rng.uniform(20.0, 30.0)),
                "sample_rh_pct": float(rh),
                "sample_flow_sccm": None,
                "pump_pwm": None,
                "lamp_hours": 10.0,
                "reactor_par_umol_m2_s": None,
                "reactor_temp_c": None,
                "reactor_od": None,
                "reactor_ph": None,
                "light_state": "on",
                "room_t_c": None,
                "room_rh_pct": None,
                "acquisition_status": "OK",
            }
        )
    writer.close()

    result = await server_module.mcp.call_tool("run_ambient_baseline_check", {"experiment_id": "exp_ambient"})
    payload = _payload(result)

    model = payload["covariate_models"]["PID01"]
    assert model["method"] == "ols"
    assert isinstance(model["covariance"], list)
    assert len(model["covariance"]) == 4  # 4x4 covariance matrix, JSON-safe as nested lists
    assert model["has_symbolic_regressor"] is False
