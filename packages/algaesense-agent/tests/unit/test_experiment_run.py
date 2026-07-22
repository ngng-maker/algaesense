"""Tests for algaesense_agent.mcp_actuators.experiment_run.

Runs a real, local SSH server (paramiko acting as both client and
server over a loopback socket) that handles `exec_command` requests --
same "real, in-process, not faked" approach as this project's other
SSH-based tests (test_pi_sync.py, jaxsr-calibration's test_sftp_backend.py).
"""

from __future__ import annotations

import socket
import threading

import pytest

paramiko = pytest.importorskip("paramiko", reason="paramiko (the 'sftp' extra) isn't installed in this environment")

from algaesense_agent.mcp_actuators.experiment_run import (
    apply_new_experiment_run,
    propose_new_experiment_run,
    restart_edge_service,
)


class _ServerInterface(paramiko.ServerInterface):
    """Accepts the test's known password, and responds to `exec_command`
    by writing back canned output and an exit status this test controls
    -- a real (if scripted) command-execution channel, not a mocked
    SSHClient."""

    def __init__(self, password: str, exit_status: int, response_text: str) -> None:
        self._password = password
        self._exit_status = exit_status
        self._response_text = response_text
        self.received_commands: list[str] = []

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_password(self, username, password):
        return paramiko.AUTH_SUCCESSFUL if password == self._password else paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "password"

    def check_channel_exec_request(self, channel, command):
        self.received_commands.append(command.decode() if isinstance(command, bytes) else command)
        channel.send(self._response_text.encode())
        channel.send_exit_status(self._exit_status)
        threading.Timer(0.05, channel.close).start()
        return True


def _run_one_connection(sock, host_key, server_interface: _ServerInterface) -> None:
    transport = paramiko.Transport(sock)
    transport.add_server_key(host_key)
    transport.start_server(server=server_interface)
    channel = transport.accept(20)
    if channel is not None:
        while transport.is_active():
            if transport.accept(1) is None and not transport.is_active():
                break


@pytest.fixture
def ssh_server():
    """Simulates the Pi's own already-running SSH server."""
    host_key = paramiko.RSAKey.generate(2048)
    password = "test-password-123"

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    server_interface = _ServerInterface(password=password, exit_status=0, response_text="restart ok\n")

    def _accept_loop():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=_run_one_connection, args=(conn, host_key, server_interface), daemon=True).start()

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()

    yield {"host": "127.0.0.1", "port": port, "username": "tester", "password": password, "server_interface": server_interface}

    listener.close()


def test_restart_edge_service_sends_the_expected_command_and_returns_output(ssh_server) -> None:
    output = restart_edge_service(
        host=ssh_server["host"], port=ssh_server["port"], username=ssh_server["username"], password=ssh_server["password"]
    )

    assert "restart ok" in output
    assert ssh_server["server_interface"].received_commands == ["sudo systemctl restart algaesense-edge"]


def test_restart_edge_service_uses_the_given_service_name(ssh_server) -> None:
    restart_edge_service(
        host=ssh_server["host"],
        port=ssh_server["port"],
        username=ssh_server["username"],
        password=ssh_server["password"],
        service_name="some-other-service",
    )

    assert ssh_server["server_interface"].received_commands == ["sudo systemctl restart some-other-service"]


def test_restart_edge_service_raises_clearly_on_nonzero_exit_status(ssh_server) -> None:
    ssh_server["server_interface"]._exit_status = 1
    ssh_server["server_interface"]._response_text = "permission denied\n"

    with pytest.raises(RuntimeError, match="permission denied"):
        restart_edge_service(
            host=ssh_server["host"], port=ssh_server["port"], username=ssh_server["username"], password=ssh_server["password"]
        )


def test_propose_new_experiment_run_has_no_side_effect_and_mentions_reactor() -> None:
    proposal = propose_new_experiment_run("R01")

    assert proposal.reactor_id == "R01"
    assert "R01" in proposal.note
    assert "restart" in proposal.note.lower()


@pytest.mark.asyncio
async def test_apply_new_experiment_run_restarts_and_reports_dashboard_url(ssh_server) -> None:
    result = await apply_new_experiment_run(
        "R01",
        host=ssh_server["host"],
        port=ssh_server["port"],
        username=ssh_server["username"],
        password=ssh_server["password"],
        dashboard_url="http://example-dashboard:8501",
    )

    assert result["reactor_id"] == "R01"
    assert result["status"] == "restarted"
    assert result["dashboard_url"] == "http://example-dashboard:8501"
    assert "http://example-dashboard:8501" in result["note"]
    assert ssh_server["server_interface"].received_commands == ["sudo systemctl restart algaesense-edge"]


@pytest.mark.asyncio
async def test_apply_new_experiment_run_without_dashboard_url_omits_it(ssh_server) -> None:
    result = await apply_new_experiment_run(
        "R01", host=ssh_server["host"], port=ssh_server["port"], username=ssh_server["username"], password=ssh_server["password"]
    )

    assert "dashboard_url" not in result
