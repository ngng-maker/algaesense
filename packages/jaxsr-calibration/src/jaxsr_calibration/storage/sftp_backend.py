"""A `RemoteStorageBackend` that pushes files over SFTP (SSH file
transfer) to another machine -- e.g. the Pi automatically pushing
completed hourly files straight onto the operator's laptop, with no
cloud account and no shared network filesystem (SMB/NFS) to configure.
Only needs an SSH server already running on the destination machine and
a key pair the source machine can authenticate with -- both of which
this project's Pi/laptop setup already has in some form (see
docs/remote_storage_setup.md).
"""

from __future__ import annotations

import stat
from pathlib import Path
from posixpath import dirname as posix_dirname


class SftpStorageBackend:
    """`host`/`port`/`username`/`private_key_path` describe how to reach
    the destination machine's SSH server. `remote_root_dir` is a
    directory on that machine (created if it doesn't exist) that
    `remote_key`s are resolved relative to -- e.g. `remote_root_dir` =
    "C:/algaesense-data" and `remote_key` =
    "experiments/exp_01/sensor_id=PID01/hour=....parquet" lands at
    "C:/algaesense-data/experiments/exp_01/.../hour=....parquet"."""

    def __init__(
        self,
        host: str,
        username: str,
        private_key_path: Path,
        remote_root_dir: str,
        port: int = 22,
    ) -> None:
        try:
            import paramiko
        except ImportError as exc:
            raise ImportError(
                "SftpStorageBackend requires the 'sftp' extra (paramiko). "
                "Install with `pip install jaxsr-calibration[sftp]`."
            ) from exc

        self._paramiko = paramiko
        self.host = host
        self.port = port
        self.username = username
        self.private_key_path = Path(private_key_path)
        self.remote_root_dir = remote_root_dir.rstrip("/")

    def _connect(self):
        """A fresh SFTP connection per call rather than one held open for
        the writer's whole lifetime -- uploads happen at most once an
        hour, so the (small) reconnect cost is a fair trade for never
        having to detect and recover from a connection that's gone
        stale between uploads."""
        client = self._paramiko.SSHClient()
        client.set_missing_host_key_policy(self._paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.username,
            key_filename=str(self.private_key_path),
        )
        return client

    def _remote_path(self, remote_key: str) -> str:
        return f"{self.remote_root_dir}/{remote_key}"

    def _mkdir_p(self, sftp, remote_dir: str) -> None:
        """SFTP has no `mkdir -p` -- build up each path segment,
        tolerating "already exists" so re-uploading into the same
        experiment's directory on a later hour is a no-op here."""
        parts = remote_dir.strip("/").split("/")
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else f"/{part}"
            try:
                sftp.mkdir(current)
            except OSError:
                pass

    def upload_file(self, local_path: Path, remote_key: str) -> None:
        remote_path = self._remote_path(remote_key)
        client = self._connect()
        try:
            sftp = client.open_sftp()
            self._mkdir_p(sftp, posix_dirname(remote_path))
            sftp.put(str(local_path), remote_path)
        finally:
            client.close()

    def download_file(self, remote_key: str, local_path: Path) -> None:
        remote_path = self._remote_path(remote_key)
        client = self._connect()
        try:
            sftp = client.open_sftp()
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                sftp.get(remote_path, str(local_path))
            except OSError as exc:
                """
                paramiko raises a plain `IOError`/`OSError` for a missing
                remote file, not always already a `FileNotFoundError`
                subtype depending on version -- normalize to
                `FileNotFoundError` so every backend's `download_file`
                raises the exact same thing for "nothing there yet"
                (see `PartitionedParquetWriter.flush`'s hydrate-from-remote
                logic, which specifically catches `FileNotFoundError`).
                """
                raise FileNotFoundError(f"No object at remote key {remote_key!r} on {self.host!r}") from exc
        finally:
            client.close()

    def list_keys(self, prefix: str) -> list[str]:
        prefix_dir = self._remote_path(prefix).rstrip("/")
        client = self._connect()
        try:
            sftp = client.open_sftp()
            keys: list[str] = []
            self._walk(sftp, prefix_dir, keys)
            return sorted(keys)
        finally:
            client.close()

    def _walk(self, sftp, remote_dir: str, keys: list[str]) -> None:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except OSError:
            return  # directory doesn't exist yet -- nothing uploaded under this prefix.

        prefix_root = self.remote_root_dir.rstrip("/") + "/"
        for entry in entries:
            full_path = f"{remote_dir}/{entry.filename}"
            if stat.S_ISDIR(entry.st_mode):
                self._walk(sftp, full_path, keys)
            else:
                keys.append(full_path[len(prefix_root):] if full_path.startswith(prefix_root) else full_path)
