"""Starting a fresh experiment run: restarts algaesense-edge's systemd
service on the Pi over SSH, the same connection this project already
uses for everything else (see docs/remote_storage_setup.md).

This is a real, hardware-adjacent action -- it interrupts whatever
experiment is currently running -- so it follows the same propose/apply
split as LED changes (see actuators.py's module docstring): describing
it has no side effect, only `apply_new_experiment_run` actually touches
the Pi.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from algaesense_agent.mcp_actuators.actuators import apply_led_profile
from algaesense_agent.mcp_actuators.edge_client import EdgeClient
from algaesense_agent.pi_ssh import connect_to_pi


@dataclass
class ExperimentRestartProposal:
    """A described, not-yet-triggered restart."""

    reactor_id: str
    led_profile: dict | None
    note: str


def propose_new_experiment_run(reactor_id: str, led_profile: dict | None = None) -> ExperimentRestartProposal:
    """Describe starting a fresh experiment run, without doing it. If
    `led_profile` is given (same shape as propose_led_profile_change's --
    a dict with a `shape` field plus that shape's own parameters, e.g.
    `{"shape": "constant", "par_umol_m2_s": 100.0}` for a plain static
    setpoint), it's applied right after the fresh run starts, rather than
    the LED being left off until a separate command."""
    note = (
        f"Proposing to start a new experiment run for reactor {reactor_id!r}: this restarts "
        "algaesense-edge on the Pi, which STOPS whatever experiment is currently running and "
        "starts a fresh one with a new experiment_id (the Pi's systemd unit generates one from "
        "the current date/time on every restart)."
    )
    if led_profile:
        note += f" Once the fresh run is up, it will also start this LED profile: {led_profile}."
    else:
        note += " The LED will stay off until a separate setpoint/profile is applied."
    note += " Not yet applied -- requires explicit confirmation before calling apply_new_experiment_run."

    return ExperimentRestartProposal(reactor_id=reactor_id, led_profile=led_profile, note=note)


def restart_edge_service(
    host: str,
    username: str,
    private_key_path=None,
    password: str | None = None,
    port: int = 22,
    service_name: str = "algaesense-edge",
) -> str:
    """SSH in and run `sudo systemctl restart {service_name}` non-
    interactively. Returns combined stdout+stderr from the command.

    Requires a one-time Pi-side setup: a sudoers rule letting this SSH
    user run exactly this systemctl command without a password prompt
    (there's no terminal to type one into over a scripted SSH exec) --
    see docs/remote_storage_setup.md for the narrowly-scoped sudoers
    line, not a blanket passwordless-sudo grant.
    """
    client = connect_to_pi(host=host, username=username, private_key_path=private_key_path, password=password, port=port)
    try:
        _stdin, stdout, stderr = client.exec_command(f"sudo systemctl restart {service_name}")
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode() + stderr.read().decode()
        if exit_status != 0:
            raise RuntimeError(f"systemctl restart {service_name!r} failed (exit {exit_status}): {output}")
        return output
    finally:
        client.close()


async def wait_for_edge_healthy(edge: EdgeClient, timeout_s: float = 30.0, poll_interval_s: float = 1.0) -> None:
    """Poll the edge service's `/health` endpoint until it responds or
    `timeout_s` elapses. A fresh restart takes a few real seconds
    (systemd tearing down the old process, the new one re-initializing
    hardware) before the network API is listening again -- applying an
    LED profile before that would just fail with a connection error."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_exc: Exception | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            await edge.health()
            return
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(poll_interval_s)
    raise TimeoutError(f"algaesense-edge did not become healthy within {timeout_s}s") from last_exc


async def apply_new_experiment_run(
    reactor_id: str,
    host: str,
    username: str,
    private_key_path=None,
    password: str | None = None,
    port: int = 22,
    dashboard_url: str | None = None,
    led_profile: dict | None = None,
    edge: EdgeClient | None = None,
) -> dict:
    """Actually restart algaesense-edge on the Pi, starting a fresh
    experiment run, and (if `led_profile` is given) apply it once the
    fresh run is back up. Only call after the user has explicitly
    confirmed the corresponding propose_new_experiment_run result.

    `edge` is required if `led_profile` is given -- the caller (the MCP
    server) builds it via the same `_build_edge_client()` every other
    LED tool uses, so tests can substitute a real in-process edge app
    the same way they already do for those tools.
    """

    """
    `restart_edge_service` is blocking (paramiko), run off the event
    loop via `asyncio.to_thread` so this async MCP tool doesn't block
    Hermes's whole event loop for however long the SSH round-trip takes.
    """
    await asyncio.to_thread(
        restart_edge_service,
        host=host,
        username=username,
        private_key_path=private_key_path,
        password=password,
        port=port,
    )

    result = {
        "reactor_id": reactor_id,
        "status": "restarted",
        "note": "algaesense-edge restarted on the Pi -- a fresh experiment_id is now running.",
    }

    if led_profile:
        if edge is None:
            raise ValueError("apply_new_experiment_run: led_profile was given but no edge client was provided")
        await wait_for_edge_healthy(edge)
        profile_result = await apply_led_profile(edge, reactor_id, led_profile)
        result["led_profile_result"] = profile_result
        result["note"] += f" LED profile started: {led_profile}."

    if dashboard_url:
        result["dashboard_url"] = dashboard_url
        result["note"] += f" Watch it live at {dashboard_url} (Live view)."
    return result
