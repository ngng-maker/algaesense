"""Tests for algaesense_agent.dashboard.pi_sync.pull_and_delete_from_pi.

Runs a real, local SFTP server (paramiko acting as both client and
server over a loopback socket, backed by a real temp directory) rather
than mocking the SSH/SFTP protocol -- same "real, in-process, not
faked" approach this project already uses elsewhere.
"""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

import pytest

paramiko = pytest.importorskip("paramiko", reason="paramiko (the 'sftp' extra) isn't installed in this environment")

from algaesense_agent.dashboard.pi_sync import pull_and_delete_from_pi


class _RealLocalSftpServer(paramiko.SFTPServerInterface):
    def __init__(self, server, root_dir: str, *largs, **kwargs) -> None:
        super().__init__(server, *largs, **kwargs)
        self.root_dir = root_dir

    def _real_path(self, path: str) -> str:
        return os.path.join(self.root_dir, path.lstrip("/"))

    def list_folder(self, path):
        real_path = self._real_path(path)
        try:
            entries = []
            for name in os.listdir(real_path):
                attr = paramiko.SFTPAttributes.from_stat(os.stat(os.path.join(real_path, name)))
                attr.filename = name
                entries.append(attr)
            return entries
        except OSError:
            return paramiko.SFTP_NO_SUCH_FILE

    def stat(self, path):
        try:
            return paramiko.SFTPAttributes.from_stat(os.stat(self._real_path(path)))
        except OSError:
            return paramiko.SFTP_NO_SUCH_FILE

    lstat = stat

    def open(self, path, flags, attr):
        real_path = self._real_path(path)
        mode = "rb" if (flags & os.O_WRONLY) == 0 else "wb"
        try:
            handle = open(real_path, mode)
        except OSError:
            return paramiko.SFTP_NO_SUCH_FILE
        return _FileHandle(handle)

    def remove(self, path):
        try:
            os.remove(self._real_path(path))
            return paramiko.SFTP_OK
        except OSError:
            return paramiko.SFTP_NO_SUCH_FILE

    def mkdir(self, path, attr):
        try:
            os.mkdir(self._real_path(path))
            return paramiko.SFTP_OK
        except OSError:
            return paramiko.SFTP_FAILURE


class _FileHandle(paramiko.SFTPHandle):
    def __init__(self, fileobj) -> None:
        super().__init__()
        self._fileobj = fileobj

    def read(self, offset, length):
        self._fileobj.seek(offset)
        return self._fileobj.read(length)

    def write(self, offset, data):
        self._fileobj.seek(offset)
        self._fileobj.write(data)
        return paramiko.SFTP_OK

    def close(self):
        self._fileobj.close()
        return paramiko.SFTP_OK

    def stat(self):
        return paramiko.SFTPAttributes.from_stat(os.fstat(self._fileobj.fileno()))


class _ServerInterface(paramiko.ServerInterface):
    def __init__(self, authorized_key) -> None:
        self._authorized_key = authorized_key

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_publickey(self, username, key):
        return paramiko.AUTH_SUCCESSFUL if key.get_base64() == self._authorized_key.get_base64() else paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "publickey"


def _run_one_connection(sock, host_key, client_public_key, root_dir: str) -> None:
    transport = paramiko.Transport(sock)
    transport.add_server_key(host_key)
    transport.set_subsystem_handler("sftp", paramiko.SFTPServer, sftp_si=_RealLocalSftpServer, root_dir=root_dir)
    transport.start_server(server=_ServerInterface(client_public_key))
    channel = transport.accept(20)
    if channel is not None:
        while transport.is_active():
            channel = transport.accept(1)
            if channel is None and not transport.is_active():
                break


@pytest.fixture
def sftp_server(tmp_path: Path):
    """Simulates the Pi's own already-running SSH server (a real,
    local, paramiko-based SFTP server backed by a real temp directory
    standing in as the Pi's `data/raw`)."""
    root_dir = tmp_path / "pi_root"
    root_dir.mkdir()

    host_key = paramiko.RSAKey.generate(2048)
    client_key = paramiko.RSAKey.generate(2048)
    client_key_path = tmp_path / "client_key"
    client_key.write_private_key_file(str(client_key_path))

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def _accept_loop():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(
                target=_run_one_connection, args=(conn, host_key, client_key, str(root_dir)), daemon=True
            ).start()

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()

    yield {"host": "127.0.0.1", "port": port, "username": "tester", "private_key_path": client_key_path, "root_dir": root_dir}

    listener.close()


def test_pull_and_delete_from_pi_copies_files_and_removes_them_remotely(sftp_server, tmp_path: Path) -> None:
    pi_raw_dir = sftp_server["root_dir"] / "data" / "raw"
    pi_raw_dir.mkdir(parents=True)
    exp_dir = pi_raw_dir / "experiments" / "exp_01" / "sensor_id=PID01"
    exp_dir.mkdir(parents=True)
    (exp_dir / "hour=2026-08-01T09.parquet").write_bytes(b"real parquet bytes")

    local_data_dir = tmp_path / "laptop_data"
    pulled = pull_and_delete_from_pi(
        host=sftp_server["host"],
        port=sftp_server["port"],
        username=sftp_server["username"],
        private_key_path=sftp_server["private_key_path"],
        remote_raw_dir="/data/raw",
        local_data_dir=local_data_dir,
    )

    assert pulled == 1

    local_file = local_data_dir / "raw" / "experiments" / "exp_01" / "sensor_id=PID01" / "hour=2026-08-01T09.parquet"
    assert local_file.read_bytes() == b"real parquet bytes"

    # Deleted from the Pi's side once copied.
    assert not (exp_dir / "hour=2026-08-01T09.parquet").exists()


def test_pull_and_delete_from_pi_with_nothing_to_pull_is_a_safe_no_op(sftp_server, tmp_path: Path) -> None:
    pulled = pull_and_delete_from_pi(
        host=sftp_server["host"],
        port=sftp_server["port"],
        username=sftp_server["username"],
        private_key_path=sftp_server["private_key_path"],
        remote_raw_dir="/data/raw",
        local_data_dir=tmp_path / "laptop_data",
    )

    assert pulled == 0
