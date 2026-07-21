"""A `RemoteStorageBackend` that just copies files to another local
directory -- this is the "future users' own local device" option: point
`root_dir` at an external drive, a NAS mount, or any other machine's
disk mounted over the network, and it behaves exactly like a cloud
backend from every caller's point of view. Also what this project's own
tests use to exercise the upload/download/merge logic for real, without
needing a live cloud account.
"""

from __future__ import annotations

import shutil
from pathlib import Path


class LocalDiskBackend:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def upload_file(self, local_path: Path, remote_key: str) -> None:
        dest = self.root_dir / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)

    def download_file(self, remote_key: str, local_path: Path) -> None:
        src = self.root_dir / remote_key
        if not src.exists():
            raise FileNotFoundError(f"No object at remote key {remote_key!r} under {self.root_dir}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)

    def list_keys(self, prefix: str) -> list[str]:
        prefix_dir = self.root_dir / prefix
        if not prefix_dir.exists():
            return []
        return sorted(
            str(p.relative_to(self.root_dir).as_posix())
            for p in prefix_dir.rglob("*")
            if p.is_file()
        )
