"""MCP server proxying algaesense-edge's actuator API, split into
no-side-effect propose calls and the single side-effecting apply call.
"""

from algaesense_agent.mcp_actuators.actuators import (
    ActuatorNotImplementedError,
    ActuatorProposal,
    apply_led_setpoint,
    propose_led_setpoint,
    propose_stirring_setpoint,
    propose_temperature_setpoint,
)
from algaesense_agent.mcp_actuators.edge_client import (
    EdgeClient,
    SetpointRejectedError,
    UnknownReactorError,
)

__all__ = [
    "propose_led_setpoint",
    "apply_led_setpoint",
    "propose_temperature_setpoint",
    "propose_stirring_setpoint",
    "ActuatorProposal",
    "ActuatorNotImplementedError",
    "EdgeClient",
    "UnknownReactorError",
    "SetpointRejectedError",
]
