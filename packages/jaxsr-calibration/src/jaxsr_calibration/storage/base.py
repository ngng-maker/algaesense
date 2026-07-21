"""The interface every remote-storage backend implements."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class RemoteStorageBackend(Protocol):
    """Deliberately three methods, all file-level -- a new backend (S3,
    Cloudflare R2, a different NAS mount, whatever a future user actually
    has) only ever needs to implement upload/download/list of one file at
    a time. Anything about *which* files belong to an experiment (the
    `raw/experiments/{experiment_id}/...` layout) is decided by the
    caller, not the backend -- `remote_key` is always that same relative
    path, so a backend is just a dumb key-value file store underneath.
    """

    def upload_file(self, local_path: Path, remote_key: str) -> None:
        """Upload the file at `local_path`, stored under `remote_key`
        (e.g. "experiments/exp_01/sensor_id=PID01/hour=2026-07-25T08.parquet").
        Overwrites whatever was previously stored at that key."""
        ...

    def download_file(self, remote_key: str, local_path: Path) -> None:
        """Download the file stored at `remote_key` to `local_path`,
        creating parent directories as needed. Raises `FileNotFoundError`
        if nothing exists at that key."""
        ...

    def list_keys(self, prefix: str) -> list[str]:
        """List every remote key starting with `prefix`. An empty list
        means nothing has been uploaded under that prefix (not an
        error) -- callers use this to check whether e.g. a specific
        hour's file already exists remotely."""
        ...
