"""Picks a `RemoteStorageBackend` from a plain dict of config values (kept
backend-agnostic and independent of any one package's CLI/env-var
conventions -- `algaesense-edge`'s cli.py and `algaesense-agent`'s sync
CLI each build that dict from their own `--flag`/`ALGAESENSE_*` env var
conventions and hand it to `get_storage_backend`, rather than this
module reaching into `os.environ` itself).
"""

from __future__ import annotations

from pathlib import Path

from jaxsr_calibration.storage.base import RemoteStorageBackend
from jaxsr_calibration.storage.local_backend import LocalDiskBackend


def get_storage_backend(config: dict) -> RemoteStorageBackend | None:
    """`config["backend"]` selects which backend to build:

    - missing, `None`, or `"none"` -- no remote storage configured,
      returns `None`. This is the default: every existing call site that
      doesn't pass remote-storage config keeps behaving exactly as
      before (local Parquet files only, nothing uploaded/deleted).
    - `"local"` -- `LocalDiskBackend(root_dir=config["local_root_dir"])`.
      This is the "future users' own local device" option: point
      `local_root_dir` at an external drive or a NAS mount.
    - `"firebase"` -- `FirebaseStorageBackend(credentials_path=...,
      bucket_name=...)`. Imported lazily so `firebase-admin` is only
      ever required on a machine that actually configures this backend.

    Adding a new backend later means adding one more `elif` branch here
    plus a new module next to `local_backend.py`/`firebase_backend.py` --
    nothing else in the codebase needs to change, since every caller
    only ever talks to the returned object through the
    `RemoteStorageBackend` interface.
    """
    backend = config.get("backend")
    if backend is None or backend == "none":
        return None

    if backend == "local":
        return LocalDiskBackend(root_dir=Path(config["local_root_dir"]))

    if backend == "firebase":
        from jaxsr_calibration.storage.firebase_backend import FirebaseStorageBackend

        return FirebaseStorageBackend(
            credentials_path=Path(config["firebase_credentials_path"]),
            bucket_name=config["firebase_bucket_name"],
        )

    raise ValueError(
        f"Unknown storage backend {backend!r}. Expected 'none', 'local', or 'firebase' "
        "(see docs/remote_storage_setup.md for how to add a new one)."
    )
