"""A small SQLite archive of past experiments' raw readings, so the
Streamlit dashboard can let an operator browse a finished experiment the
same way it shows the live one -- without needing the raw Parquet files
(and the algaesense-edge instance that wrote them) present at query time.

Raw files can be ingested either from a local copy of `data/` (e.g. after
a manual `scp` from the Pi) or synced directly from a configured
`jaxsr_calibration.storage.RemoteStorageBackend` (Firebase, a local/NAS
mount, or any future backend) -- see `sync_and_ingest_experiment`/
`sync_and_ingest_all_experiments` and `main()`'s `--storage-backend` flag.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from jaxsr_calibration.storage import RemoteStorageBackend

from algaesense_agent.raw_readers import (
    NoRawReadingsFoundError,
    list_raw_experiment_ids,
    load_raw_camera_readings,
    load_raw_voc_readings,
)


"""
This is deliberately a separate, additional store, not a replacement for
the raw Parquet files algaesense-edge writes -- those stay the single
source of truth (already relied on by jaxsr-calibration's whole analysis
pipeline, mcp_diagnostics, mcp_pipeline). This database exists purely so
the dashboard has something fast and queryable to list/browse *past*
experiments from, once their raw Parquet files have been synced onto
this machine (see `ingest_experiment`'s own docstring) -- it is a cache
built from that source of truth, not a new one.

`sqlite3` (Python's standard library, no new dependency) is a deliberate
choice over introducing a client-server database for this: the whole
point is a single local file an operator's laptop can hold, with no
separate database service to run or keep alive alongside everything
else in this project.
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS voc_readings (
    experiment_id TEXT NOT NULL,
    reactor_id TEXT NOT NULL,
    sensor_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    pid_voltage_mv REAL NOT NULL,
    reactor_par_umol_m2_s REAL
);
CREATE INDEX IF NOT EXISTS idx_voc_experiment ON voc_readings(experiment_id);

CREATE TABLE IF NOT EXISTS camera_readings (
    experiment_id TEXT NOT NULL,
    reactor_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    image_feature_vector TEXT
);
CREATE INDEX IF NOT EXISTS idx_camera_experiment ON camera_readings(experiment_id);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def ingest_experiment(db_path: Path, data_dir: Path, experiment_id: str) -> dict:
    """Read one experiment's raw VOC/camera Parquet files from `data_dir`
    and (re-)load them into the SQLite archive at `db_path`."""

    """
    Deletes any existing rows for this experiment_id before inserting,
    so re-running this (e.g. after copying a newer snapshot of the same
    still-running experiment's files from the Pi) replaces stale rows
    rather than duplicating them. An experiment missing one of the two
    raw streams entirely (e.g. camera never captured before the
    experiment ended) is not an error -- that stream is just left empty
    for this experiment_id.
    """
    voc_rows: list[tuple] = []
    try:
        voc_df = load_raw_voc_readings(data_dir, experiment_id)
        voc_rows = [
            (experiment_id, row["reactor_id"], row["sensor_id"], row["timestamp"].isoformat(), row["pid_voltage_mv"], row["reactor_par_umol_m2_s"])
            for row in voc_df.sort("timestamp").to_dicts()
        ]
    except NoRawReadingsFoundError:
        pass

    camera_rows: list[tuple] = []
    try:
        camera_df = load_raw_camera_readings(data_dir, experiment_id)
        camera_rows = [
            (
                experiment_id,
                row["reactor_id"],
                row["camera_id"],
                row["timestamp"].isoformat(),
                json.dumps(row["image_feature_vector"]) if row["image_feature_vector"] is not None else None,
            )
            for row in camera_df.sort("timestamp").to_dicts()
        ]
    except NoRawReadingsFoundError:
        pass

    conn = _connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM voc_readings WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM camera_readings WHERE experiment_id = ?", (experiment_id,))
            conn.executemany(
                "INSERT INTO voc_readings VALUES (?, ?, ?, ?, ?, ?)", voc_rows
            )
            conn.executemany(
                "INSERT INTO camera_readings VALUES (?, ?, ?, ?, ?)", camera_rows
            )
    finally:
        conn.close()

    return {"experiment_id": experiment_id, "voc_rows": len(voc_rows), "camera_rows": len(camera_rows)}


def ingest_all_experiments(db_path: Path, data_dir: Path) -> list[dict]:
    """Ingest every experiment currently found under `data_dir` -- the
    bulk version of `ingest_experiment`, for the `algaesense-dashboard-sync`
    CLI command."""
    return [ingest_experiment(db_path, data_dir, experiment_id) for experiment_id in list_raw_experiment_ids(data_dir)]


def list_remote_experiment_ids(backend: RemoteStorageBackend) -> list[str]:
    """List every experiment_id present in the configured remote storage
    backend -- the remote equivalent of `raw_readers.list_raw_experiment_ids`,
    used when this machine is syncing from the cloud/NAS backend instead
    of a local `scp`'d copy."""
    keys = backend.list_keys("experiments/")
    """
    Every key looks like "experiments/{experiment_id}/{partition}=.../
    hour=....parquet" (see PartitionedParquetWriter._remote_key) -- the
    second path segment is always the experiment_id.
    """
    return sorted({key.split("/")[1] for key in keys if len(key.split("/")) > 1})


def download_experiment_from_remote(backend: RemoteStorageBackend, data_dir: Path, experiment_id: str) -> int:
    """Download every raw file the remote backend has for one experiment
    into `data_dir`, preserving the exact relative layout
    `raw_readers.py`'s loaders already expect (`raw/experiments/
    {experiment_id}/...`) -- so the ordinary `ingest_experiment` call
    right after this works unchanged, whether its files came from a local
    `scp` copy or from here. Returns how many files were downloaded."""
    keys = backend.list_keys(f"experiments/{experiment_id}/")
    for key in keys:
        backend.download_file(key, Path(data_dir) / "raw" / key)
    return len(keys)


def sync_and_ingest_experiment(db_path: Path, data_dir: Path, backend: RemoteStorageBackend, experiment_id: str) -> dict:
    """The remote-storage equivalent of running `scp` then
    `ingest_experiment` by hand -- downloads one experiment's files from
    the configured backend, then ingests them the usual way."""
    download_experiment_from_remote(backend, data_dir, experiment_id)
    return ingest_experiment(db_path, data_dir, experiment_id)


def sync_and_ingest_all_experiments(db_path: Path, data_dir: Path, backend: RemoteStorageBackend) -> list[dict]:
    """Bulk version of `sync_and_ingest_experiment`, for the
    `algaesense-dashboard-sync` CLI's `--storage-backend` mode."""
    return [
        sync_and_ingest_experiment(db_path, data_dir, backend, experiment_id)
        for experiment_id in list_remote_experiment_ids(backend)
    ]


def list_experiments(db_path: Path) -> list[dict]:
    """List every experiment currently in the archive, most recently
    started first -- what the dashboard's sidebar picker reads from."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                experiment_id,
                MIN(reactor_id) AS reactor_id,
                MIN(sensor_id) AS sensor_id,
                MIN(timestamp) AS first_timestamp,
                MAX(timestamp) AS last_timestamp,
                COUNT(*) AS voc_row_count
            FROM voc_readings
            GROUP BY experiment_id
            ORDER BY first_timestamp DESC
            """
        ).fetchall()
        columns = ["experiment_id", "reactor_id", "sensor_id", "first_timestamp", "last_timestamp", "voc_row_count"]
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


def load_experiment_voc_readings(db_path: Path, experiment_id: str) -> list[dict]:
    """Same row shape as the live edge API's `/sensors/voc/recent`
    response (including `experiment_id`/`reactor_id`/`sensor_id`), so the
    dashboard can render both through the same code and read experiment
    metadata off either source the same way."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT experiment_id, reactor_id, sensor_id, timestamp, pid_voltage_mv, reactor_par_umol_m2_s "
            "FROM voc_readings WHERE experiment_id = ? ORDER BY timestamp",
            (experiment_id,),
        ).fetchall()
        return [
            {
                "experiment_id": experiment_id,
                "reactor_id": reactor_id,
                "sensor_id": sensor_id,
                "timestamp": timestamp,
                "pid_voltage_mv": voltage,
                "reactor_par_umol_m2_s": par,
            }
            for experiment_id, reactor_id, sensor_id, timestamp, voltage, par in rows
        ]
    finally:
        conn.close()


def load_experiment_camera_readings(db_path: Path, experiment_id: str) -> list[dict]:
    """Same row shape as the live edge API's `/sensors/camera/recent`
    response, same reasoning as `load_experiment_voc_readings`."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT experiment_id, reactor_id, camera_id, timestamp, image_feature_vector "
            "FROM camera_readings WHERE experiment_id = ? ORDER BY timestamp",
            (experiment_id,),
        ).fetchall()
        return [
            {
                "experiment_id": experiment_id,
                "reactor_id": reactor_id,
                "camera_id": camera_id,
                "timestamp": timestamp,
                "image_feature_vector": json.loads(vector) if vector is not None else None,
            }
            for experiment_id, reactor_id, camera_id, timestamp, vector in rows
        ]
    finally:
        conn.close()


def main() -> None:
    """Entry point for the `algaesense-dashboard-sync` console script --
    ingests every experiment found under a data directory into the
    history database. With `--storage-backend`, pulls the raw files from
    the configured remote backend first, replacing the manual `scp`
    step (see docs/remote_storage_setup.md)."""
    import argparse

    from jaxsr_calibration.storage import get_storage_backend

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Root data directory (containing raw/experiments/...)")
    parser.add_argument("--db-path", required=True, type=Path, help="SQLite file to write into")
    parser.add_argument("--experiment-id", default=None, help="Sync/ingest only this experiment (default: every experiment found)")
    parser.add_argument(
        "--storage-backend",
        choices=["none", "local", "firebase", "sftp"],
        default="none",
        help="Pull raw files from this remote backend into --data-dir before ingesting, "
        "instead of assuming --data-dir already has them (e.g. via a manual scp copy). "
        "Not needed at all if the Pi already pushes straight onto this machine via its "
        "own --storage-backend=sftp -- files just show up locally in that case.",
    )
    parser.add_argument("--storage-local-root", default=None, help="Required if --storage-backend=local")
    parser.add_argument("--storage-firebase-credentials", default=None, help="Required if --storage-backend=firebase")
    parser.add_argument("--storage-firebase-bucket", default=None, help="Required if --storage-backend=firebase")
    parser.add_argument("--storage-sftp-host", default=None, help="Required if --storage-backend=sftp")
    parser.add_argument("--storage-sftp-port", type=int, default=22, help="Only used if --storage-backend=sftp")
    parser.add_argument("--storage-sftp-username", default=None, help="Required if --storage-backend=sftp")
    parser.add_argument("--storage-sftp-private-key", default=None, help="Required if --storage-backend=sftp")
    parser.add_argument("--storage-sftp-remote-root", default=None, help="Required if --storage-backend=sftp")
    parser.add_argument(
        "--pull-from-pi",
        action="store_true",
        help="Before ingesting, pull every raw file from the Pi over SSH (reusing whatever "
        "SSH server/key you already use to log into it) and delete it from the Pi once "
        "copied -- an alternative to --storage-backend that needs no new SSH server anywhere, "
        "since the laptop connects to the Pi's already-running one. See docs/remote_storage_setup.md.",
    )
    parser.add_argument("--pi-host", default=None, help="Required if --pull-from-pi")
    parser.add_argument("--pi-port", type=int, default=22, help="Only used if --pull-from-pi")
    parser.add_argument("--pi-username", default=None, help="Required if --pull-from-pi")
    parser.add_argument(
        "--pi-private-key",
        default=None,
        help="Required if --pull-from-pi and not using --pi-password: path to the private key you SSH into the Pi with.",
    )
    parser.add_argument(
        "--pi-password",
        default=None,
        help="Required if --pull-from-pi and not using --pi-private-key: the password you SSH into the Pi with "
        "(fine to try immediately; a key is recommended once you want this to run unattended/scheduled).",
    )
    parser.add_argument(
        "--pi-remote-raw-dir",
        default=None,
        help="Required if --pull-from-pi: path to data/raw on the Pi, e.g. /home/pi/algaesense/algaesense/data/raw",
    )
    args = parser.parse_args()

    if args.pull_from_pi:
        from algaesense_agent.dashboard.pi_sync import pull_and_delete_from_pi

        pulled = pull_and_delete_from_pi(
            host=args.pi_host,
            port=args.pi_port,
            username=args.pi_username,
            private_key_path=args.pi_private_key,
            password=args.pi_password,
            remote_raw_dir=args.pi_remote_raw_dir,
            local_data_dir=args.data_dir,
        )
        print(f"Pulled {pulled} file(s) from the Pi.")

    backend = get_storage_backend(
        {
            "backend": args.storage_backend,
            "local_root_dir": args.storage_local_root,
            "firebase_credentials_path": args.storage_firebase_credentials,
            "firebase_bucket_name": args.storage_firebase_bucket,
            "sftp_host": args.storage_sftp_host,
            "sftp_port": args.storage_sftp_port,
            "sftp_username": args.storage_sftp_username,
            "sftp_private_key_path": args.storage_sftp_private_key,
            "sftp_remote_root_dir": args.storage_sftp_remote_root,
        }
    )

    if backend is not None:
        if args.experiment_id:
            results = [sync_and_ingest_experiment(args.db_path, args.data_dir, backend, args.experiment_id)]
        else:
            results = sync_and_ingest_all_experiments(args.db_path, args.data_dir, backend)
    elif args.experiment_id:
        results = [ingest_experiment(args.db_path, args.data_dir, args.experiment_id)]
    else:
        results = ingest_all_experiments(args.db_path, args.data_dir)

    for result in results:
        print(f"{result['experiment_id']}: {result['voc_rows']} VOC rows, {result['camera_rows']} camera rows")


if __name__ == "__main__":
    main()
