"""Shared SSH-connection helper for everything in this package that talks
to the Pi over the SSH server it already runs -- `dashboard.pi_sync`
(pulling raw files) and `mcp_actuators.experiment_run` (restarting the
edge service) both need the identical "connect with a key or a
password" logic, so it lives here once rather than duplicated in both.
"""

from __future__ import annotations

from pathlib import Path


def connect_to_pi(
    host: str,
    username: str,
    private_key_path: Path | None = None,
    password: str | None = None,
    port: int = 22,
):
    """Returns a connected `paramiko.SSHClient`. Caller is responsible for
    closing it. Pass exactly one of `private_key_path` (recommended --
    works unattended, e.g. from a scheduled task) or `password` (works
    immediately with whatever you already log into the Pi with)."""
    if not private_key_path and not password:
        raise ValueError("connect_to_pi needs either private_key_path or password")

    try:
        import paramiko
    except ImportError as exc:
        raise ImportError(
            "connect_to_pi requires the 'sftp' extra (paramiko) -- "
            "install with `pip install jaxsr-calibration[sftp]`."
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if private_key_path:
        client.connect(hostname=host, port=port, username=username, key_filename=str(private_key_path))
    else:
        client.connect(hostname=host, port=port, username=username, password=password)
    return client
