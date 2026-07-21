"""Tests for jaxsr_calibration.storage.sftp_backend.SftpStorageBackend.

Runs a real, local SFTP server (paramiko's own SSHClient talking to a
real paramiko-based SFTPServer over a real loopback socket, backed by a
real local directory) rather than mocking the SFTP protocol -- same
"real, in-process, not faked" approach this project already uses
elsewhere (e.g. httpx.ASGITransport against the real FastAPI app).
"""

from __future__ import annotations

import importlib.util
import os
import socket
import threading
from pathlib import Path

import pytest

paramiko = pytest.importorskip("paramiko", reason="paramiko (the 'sftp' extra) isn't installed in this environment")


class _RealLocalSftpServer(paramiko.SFTPServerInterface):
    """Maps every SFTP operation straight onto a real local directory --
    a genuine (if minimal) SFTP server implementation, not a stand-in
    that fakes responses."""

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
        except FileExistsError:
            return paramiko.SFTP_FAILURE
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


def _run_one_connection(sock: socket.socket, host_key, client_public_key, root_dir: str) -> None:
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
    """Starts one real local SFTP server (a genuine paramiko SSH/SFTP
    server on a loopback socket) backed by a real temp directory, and
    tears it down after the test."""
    root_dir = tmp_path / "sftp_root"
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


def test_sftp_backend_round_trips_a_file(sftp_server) -> None:
    from jaxsr_calibration.storage.sftp_backend import SftpStorageBackend

    backend = SftpStorageBackend(
        host=sftp_server["host"],
        port=sftp_server["port"],
        username=sftp_server["username"],
        private_key_path=sftp_server["private_key_path"],
        remote_root_dir="/data",
    )

    local_file = sftp_server["root_dir"].parent / "local.parquet"
    local_file.write_bytes(b"real parquet-shaped bytes")

    backend.upload_file(local_file, "experiments/exp_01/sensor_id=PID01/hour=2026-08-01T09.parquet")

    uploaded_path = sftp_server["root_dir"] / "data" / "experiments" / "exp_01" / "sensor_id=PID01" / "hour=2026-08-01T09.parquet"
    assert uploaded_path.exists()
    assert uploaded_path.read_bytes() == b"real parquet-shaped bytes"

    downloaded = sftp_server["root_dir"].parent / "downloaded.parquet"
    backend.download_file("experiments/exp_01/sensor_id=PID01/hour=2026-08-01T09.parquet", downloaded)
    assert downloaded.read_bytes() == b"real parquet-shaped bytes"


def test_sftp_backend_download_missing_key_raises_file_not_found(sftp_server) -> None:
    from jaxsr_calibration.storage.sftp_backend import SftpStorageBackend

    backend = SftpStorageBackend(
        host=sftp_server["host"],
        port=sftp_server["port"],
        username=sftp_server["username"],
        private_key_path=sftp_server["private_key_path"],
        remote_root_dir="/data",
    )

    with pytest.raises(FileNotFoundError):
        backend.download_file("experiments/nope/sensor_id=PID01/hour=2026-08-01T09.parquet", sftp_server["root_dir"].parent / "out.parquet")


def test_sftp_backend_list_keys_returns_only_matching_prefix(sftp_server) -> None:
    from jaxsr_calibration.storage.sftp_backend import SftpStorageBackend

    backend = SftpStorageBackend(
        host=sftp_server["host"],
        port=sftp_server["port"],
        username=sftp_server["username"],
        private_key_path=sftp_server["private_key_path"],
        remote_root_dir="/data",
    )

    local_file = sftp_server["root_dir"].parent / "local.parquet"
    local_file.write_bytes(b"x")
    backend.upload_file(local_file, "experiments/exp_01/sensor_id=PID01/hour=2026-08-01T08.parquet")
    backend.upload_file(local_file, "experiments/exp_01/sensor_id=PID01/hour=2026-08-01T09.parquet")
    backend.upload_file(local_file, "experiments/exp_02/sensor_id=PID01/hour=2026-08-01T08.parquet")

    keys = backend.list_keys("experiments/exp_01/")

    assert keys == [
        "experiments/exp_01/sensor_id=PID01/hour=2026-08-01T08.parquet",
        "experiments/exp_01/sensor_id=PID01/hour=2026-08-01T09.parquet",
    ]


def test_sftp_backend_fails_clearly_without_sftp_extra_installed() -> None:
    if importlib.util.find_spec("paramiko") is not None:
        pytest.skip("paramiko is installed in this environment; nothing to test here.")

    from jaxsr_calibration.storage.sftp_backend import SftpStorageBackend

    with pytest.raises(ImportError, match="sftp"):
        SftpStorageBackend(host="x", username="x", private_key_path=Path("x"), remote_root_dir="/x")
