"""Pluggable remote-storage backends for raw experiment data.

Exists so a Pi with limited SD-card space (or an operator's laptop that
doesn't want gigabytes of Parquet files piling up locally) can offload
completed hourly files to somewhere else -- a cloud bucket, a NAS mount,
any other machine's disk -- without any other part of the codebase caring
which one is actually in use.
"""

from __future__ import annotations

from jaxsr_calibration.storage.base import RemoteStorageBackend
from jaxsr_calibration.storage.factory import get_storage_backend
from jaxsr_calibration.storage.local_backend import LocalDiskBackend

__all__ = ["RemoteStorageBackend", "LocalDiskBackend", "get_storage_backend"]

"""
`FirebaseStorageBackend`/`SftpStorageBackend` are deliberately NOT
imported here -- they each lazily import their own real dependency
(firebase-admin / paramiko) only when actually constructed, so this
package's core import stays free of both. Import them directly from
`jaxsr_calibration.storage.firebase_backend`/`sftp_backend` if needed.
"""
