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

from dataclasses import dataclass

from algaesense_agent.pi_ssh import connect_to_pi


@dataclass
class ExperimentRestartProposal:
    """A described, not-yet-triggered restart."""

    reactor_id: str
    note: str


def propose_new_experiment_run(reactor_id: str) -> ExperimentRestartProposal:
    """Describe starting a fresh experiment run, without doing it."""
    return ExperimentRestartProposal(
        reactor_id=reactor_id,
        note=(
            f"Proposing to start a new experiment run for reactor {reactor_id!r}: this restarts "
            "algaesense-edge on the Pi, which STOPS whatever experiment is currently running and "
            "starts a fresh one with a new experiment_id (the Pi's systemd unit generates one from "
            "the current date/time on every restart). Not yet applied -- requires explicit "
            "confirmation before calling apply_new_experiment_run."
        ),
    )


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


async def apply_new_experiment_run(
    reactor_id: str,
    host: str,
    username: str,
    private_key_path=None,
    password: str | None = None,
    port: int = 22,
    dashboard_url: str | None = None,
) -> dict:
    """Actually restart algaesense-edge on the Pi, starting a fresh
    experiment run. Only call after the user has explicitly confirmed
    the corresponding propose_new_experiment_run result."""

    """
    `restart_edge_service` is blocking (paramiko), run off the event
    loop via `asyncio.to_thread` so this async MCP tool doesn't block
    Hermes's whole event loop for however long the SSH round-trip takes.
    """
    import asyncio

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
    if dashboard_url:
        result["dashboard_url"] = dashboard_url
        result["note"] += f" Watch it live at {dashboard_url} (Live view)."
    return result
