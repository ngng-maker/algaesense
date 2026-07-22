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

import http.server

from algaesense_agent.mcp_actuators.edge_client import EdgeClient
from algaesense_agent.mcp_actuators.experiment_run import (
    apply_new_experiment_run,
    ensure_dashboard_running,
    propose_new_experiment_run,
    restart_edge_service,
    wait_for_edge_healthy,
)
from tests.fixtures.real_edge_app import build_real_edge_app, edge_transport


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


@pytest.mark.asyncio
async def test_apply_new_experiment_run_with_led_profile_starts_it_on_the_real_edge_app(ssh_server) -> None:
    """Combines the real local SSH server (standing in for the Pi) with
    a real, in-process algaesense-edge FastAPI app (standing in for the
    edge service that comes back up after the restart) -- both real
    implementations, nothing mocked."""
    app, _state = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))
    profile = {"shape": "constant", "par_umol_m2_s": 100.0}

    try:
        result = await apply_new_experiment_run(
            "R01",
            host=ssh_server["host"],
            port=ssh_server["port"],
            username=ssh_server["username"],
            password=ssh_server["password"],
            led_profile=profile,
            edge=edge,
        )
    finally:
        await edge.close()

    assert result["status"] == "restarted"
    assert result["led_profile_result"]["profile"] == profile
    assert "LED profile started" in result["note"]


@pytest.mark.asyncio
async def test_apply_new_experiment_run_with_led_profile_but_no_edge_client_raises_clearly(ssh_server) -> None:
    with pytest.raises(ValueError, match="no edge client was provided"):
        await apply_new_experiment_run(
            "R01",
            host=ssh_server["host"],
            port=ssh_server["port"],
            username=ssh_server["username"],
            password=ssh_server["password"],
            led_profile={"shape": "constant", "par_umol_m2_s": 100.0},
        )


@pytest.mark.asyncio
async def test_wait_for_edge_healthy_returns_once_the_real_app_responds() -> None:
    app, _state = build_real_edge_app(reactor_id="R01")
    edge = EdgeClient(base_url="http://fake-edge", transport=edge_transport(app))

    try:
        await wait_for_edge_healthy(edge, timeout_s=5.0, poll_interval_s=0.1)
    finally:
        await edge.close()


@pytest.mark.asyncio
async def test_wait_for_edge_healthy_times_out_clearly_when_edge_never_responds() -> None:
    """A real EdgeClient pointed at a host that will never answer --
    genuinely exercises the timeout path, not a mocked failure."""
    edge = EdgeClient(base_url="http://127.0.0.1:1", timeout=0.2)

    try:
        with pytest.raises(TimeoutError, match="did not become healthy"):
            await wait_for_edge_healthy(edge, timeout_s=1.0, poll_interval_s=0.1)
    finally:
        await edge.close()


def test_propose_new_experiment_run_with_led_profile_mentions_it() -> None:
    profile = {"shape": "constant", "par_umol_m2_s": 100.0}

    proposal = propose_new_experiment_run("R01", led_profile=profile)

    assert proposal.led_profile == profile
    assert "LED profile" in proposal.note


def test_propose_new_experiment_run_without_led_profile_says_led_stays_off() -> None:
    proposal = propose_new_experiment_run("R01")

    assert proposal.led_profile is None
    assert "LED will stay off" in proposal.note


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    """A real, minimal local HTTP server standing in for a running
    Streamlit instance -- just enough to answer `/_stcore/health` the
    way the real thing does, over a genuine socket."""

    def do_GET(self):
        if self.path == "/_stcore/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # keep test output quiet


@pytest.fixture
def real_local_http_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _HealthHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.mark.asyncio
async def test_ensure_dashboard_running_returns_true_when_already_up(real_local_http_server) -> None:
    already_running = await ensure_dashboard_running(f"http://localhost:{real_local_http_server}")

    assert already_running is True


@pytest.mark.asyncio
async def test_ensure_dashboard_running_launches_streamlit_when_not_up(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_args = {}

    def _fake_popen(args, **kwargs):
        captured_args["args"] = args
        return None

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    already_running = await ensure_dashboard_running("http://localhost:59999", health_timeout_s=0.3)

    assert already_running is False
    assert "streamlit" in captured_args["args"]
    assert "run" in captured_args["args"]
    assert "59999" in captured_args["args"]


@pytest.mark.asyncio
async def test_ensure_dashboard_running_skips_non_local_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Can't launch a process on a remote machine from here -- must not
    even try (and must not falsely claim it's running either)."""

    def _fake_popen(*args, **kwargs):
        raise AssertionError("should never attempt to launch a process for a non-local URL")

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    already_running = await ensure_dashboard_running("http://100.64.1.2:8501")

    assert already_running is False
