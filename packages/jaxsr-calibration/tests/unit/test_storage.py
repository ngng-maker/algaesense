"""Tests for jaxsr_calibration.storage: the pluggable remote-storage
backends (LocalDiskBackend, FirebaseStorageBackend, get_storage_backend).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from jaxsr_calibration.storage import LocalDiskBackend, get_storage_backend
from jaxsr_calibration.storage.base import RemoteStorageBackend


def test_local_disk_backend_round_trips_a_file(tmp_path: Path) -> None:
    remote_root = tmp_path / "remote"
    local_file = tmp_path / "local" / "hour=2026-07-25T08.parquet"
    local_file.parent.mkdir(parents=True)
    local_file.write_bytes(b"fake parquet bytes")

    backend = LocalDiskBackend(root_dir=remote_root)
    backend.upload_file(local_file, "experiments/exp_01/sensor_id=PID01/hour=2026-07-25T08.parquet")

    downloaded = tmp_path / "downloaded" / "hour=2026-07-25T08.parquet"
    backend.download_file("experiments/exp_01/sensor_id=PID01/hour=2026-07-25T08.parquet", downloaded)

    assert downloaded.read_bytes() == b"fake parquet bytes"


def test_local_disk_backend_download_missing_key_raises_file_not_found(tmp_path: Path) -> None:
    backend = LocalDiskBackend(root_dir=tmp_path / "remote")

    with pytest.raises(FileNotFoundError):
        backend.download_file("experiments/nope/sensor_id=PID01/hour=2026-07-25T08.parquet", tmp_path / "out.parquet")


def test_local_disk_backend_list_keys_returns_only_matching_prefix(tmp_path: Path) -> None:
    backend = LocalDiskBackend(root_dir=tmp_path / "remote")
    src = tmp_path / "src.parquet"
    src.write_bytes(b"x")
    backend.upload_file(src, "experiments/exp_01/sensor_id=PID01/hour=2026-07-25T08.parquet")
    backend.upload_file(src, "experiments/exp_01/sensor_id=PID01/hour=2026-07-25T09.parquet")
    backend.upload_file(src, "experiments/exp_02/sensor_id=PID01/hour=2026-07-25T08.parquet")

    keys = backend.list_keys("experiments/exp_01/")

    assert keys == [
        "experiments/exp_01/sensor_id=PID01/hour=2026-07-25T08.parquet",
        "experiments/exp_01/sensor_id=PID01/hour=2026-07-25T09.parquet",
    ]


def test_local_disk_backend_list_keys_empty_prefix_returns_empty_list(tmp_path: Path) -> None:
    backend = LocalDiskBackend(root_dir=tmp_path / "remote")

    assert backend.list_keys("experiments/nothing-here/") == []


def test_local_disk_backend_satisfies_the_protocol(tmp_path: Path) -> None:
    backend = LocalDiskBackend(root_dir=tmp_path)
    assert isinstance(backend, RemoteStorageBackend)


def test_get_storage_backend_returns_none_when_unconfigured() -> None:
    assert get_storage_backend({}) is None
    assert get_storage_backend({"backend": None}) is None
    assert get_storage_backend({"backend": "none"}) is None


def test_get_storage_backend_builds_local_backend(tmp_path: Path) -> None:
    backend = get_storage_backend({"backend": "local", "local_root_dir": str(tmp_path)})

    assert isinstance(backend, LocalDiskBackend)
    assert backend.root_dir == tmp_path


def test_get_storage_backend_rejects_unknown_backend_name() -> None:
    with pytest.raises(ValueError, match="Unknown storage backend"):
        get_storage_backend({"backend": "dropbox"})


def test_firebase_backend_fails_clearly_without_cloud_extra_installed() -> None:
    """Mirrors algaesense-edge's hardware_extra_importable pattern: this
    check only makes sense on a machine that genuinely lacks
    firebase-admin -- skip on one (like this dev environment) where it's
    already installed for real use."""
    if importlib.util.find_spec("firebase_admin") is not None:
        pytest.skip("firebase-admin is installed in this environment; nothing to test here.")

    from jaxsr_calibration.storage.firebase_backend import FirebaseStorageBackend

    with pytest.raises(ImportError, match="cloud"):
        FirebaseStorageBackend(credentials_path=Path("nonexistent.json"), bucket_name="fake-bucket")


def test_firebase_backend_rejects_a_missing_credentials_file() -> None:
    """firebase-admin's own `credentials.Certificate` reads the key file
    immediately -- a real, non-mocked check that a bad path is rejected
    clearly, without needing a live Firebase project."""
    if importlib.util.find_spec("firebase_admin") is None:
        pytest.skip("firebase-admin isn't installed in this environment.")

    from jaxsr_calibration.storage.firebase_backend import FirebaseStorageBackend

    with pytest.raises(Exception):
        FirebaseStorageBackend(credentials_path=Path("definitely-does-not-exist.json"), bucket_name="fake-bucket")
