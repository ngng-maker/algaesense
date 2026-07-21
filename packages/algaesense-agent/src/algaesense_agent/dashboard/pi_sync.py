"""Pulls raw experiment files from the Pi over the SSH connection this
project already relies on for everything else, and deletes each file
from the Pi right after it's copied -- so the Pi's SD card never has to
hold more than whatever's been written since the last sync, with no new
SSH server to set up anywhere (the Pi's own sshd, already running, is
all this needs).

This is the reverse of jaxsr_calibration.storage.sftp_backend.SftpStorageBackend
(which pushes from the Pi onto another machine, needing an SSH server on
the *destination*) -- here the laptop is the SSH client, connecting to
the Pi's already-running server, so nothing new needs to be enabled on
either machine beyond an SSH key the Pi's sshd already trusts.
"""

from __future__ import annotations

import stat
from pathlib import Path


def pull_and_delete_from_pi(
    host: str,
    username: str,
    private_key_path: Path,
    remote_raw_dir: str,
    local_data_dir: Path,
    port: int = 22,
) -> int:
    """Copies every file under `remote_raw_dir` (the Pi's `data/raw`)
    into `local_data_dir/raw`, preserving the exact relative layout
    `raw_readers.py` already expects, and deletes each file from the Pi
    immediately after it's confirmed copied. Returns how many files were
    pulled."""
    try:
        import paramiko
    except ImportError as exc:
        raise ImportError(
            "pull_and_delete_from_pi requires the 'sftp' extra (paramiko) -- "
            "install with `pip install jaxsr-calibration[sftp]`."
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, port=port, username=username, key_filename=str(private_key_path))

    try:
        sftp = client.open_sftp()
        remote_files: list[str] = []
        _walk_remote(sftp, remote_raw_dir.rstrip("/"), remote_files)

        local_raw_dir = Path(local_data_dir) / "raw"
        for remote_path in remote_files:
            relative = remote_path[len(remote_raw_dir.rstrip("/")) + 1 :]
            local_path = local_raw_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            """
            Download to a temp file, then rename into place -- same
            atomicity reasoning as PartitionedParquetWriter.flush: if
            the connection drops mid-transfer, we don't want a
            truncated file sitting at the final path, and we
            specifically must not delete the Pi's copy of anything that
            didn't land completely.
            """
            tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
            sftp.get(remote_path, str(tmp_path))
            tmp_path.replace(local_path)

            sftp.remove(remote_path)

        return len(remote_files)
    finally:
        client.close()


def _walk_remote(sftp, remote_dir: str, files: list[str]) -> None:
    try:
        entries = sftp.listdir_attr(remote_dir)
    except OSError:
        return

    for entry in entries:
        full_path = f"{remote_dir}/{entry.filename}"
        if stat.S_ISDIR(entry.st_mode):
            _walk_remote(sftp, full_path, files)
        elif not entry.filename.endswith(".tmp"):
            files.append(full_path)
