"""A `RemoteStorageBackend` backed by Firebase Storage (a Google Cloud
Storage bucket, managed through a Firebase project). This is the backend
this project uses today; it is not special-cased anywhere else in the
codebase -- everything that touches storage goes through the
`RemoteStorageBackend` interface, so swapping this out for a different
provider later (or for `LocalDiskBackend` pointed at a NAS) needs no
change outside this one file plus `factory.py`.
"""

from __future__ import annotations

from pathlib import Path


class FirebaseStorageBackend:
    """`credentials_path` is a Firebase service-account JSON key file
    (downloaded from the Firebase console, never committed to this
    repo -- see docs/remote_storage_setup.md). `bucket_name` is the
    project's storage bucket, e.g. "my-project.appspot.com"."""

    def __init__(self, credentials_path: Path, bucket_name: str) -> None:
        try:
            import firebase_admin
            from firebase_admin import credentials, storage
        except ImportError as exc:
            raise ImportError(
                "FirebaseStorageBackend requires the 'cloud' extra "
                "(firebase-admin). Install with "
                "`pip install jaxsr-calibration[cloud]`."
            ) from exc

        """
        `firebase_admin.initialize_app` is a one-per-process global --
        calling it twice with different credentials raises. Reusing
        whatever app is already initialized (rather than erroring) lets
        more than one `FirebaseStorageBackend` instance exist in the
        same process (e.g. one per test) without fighting over global
        state.
        """
        try:
            app = firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(str(credentials_path))
            app = firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

        self._bucket = storage.bucket(app=app)

    def upload_file(self, local_path: Path, remote_key: str) -> None:
        blob = self._bucket.blob(remote_key)
        blob.upload_from_filename(str(local_path))

    def download_file(self, remote_key: str, local_path: Path) -> None:
        blob = self._bucket.blob(remote_key)
        if not blob.exists():
            raise FileNotFoundError(f"No object at remote key {remote_key!r} in bucket {self._bucket.name!r}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    def list_keys(self, prefix: str) -> list[str]:
        return [blob.name for blob in self._bucket.list_blobs(prefix=prefix)]
