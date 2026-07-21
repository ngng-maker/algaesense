"""A small SQLite archive of past experiments' raw readings, so the
Streamlit dashboard can let an operator browse a finished experiment the
same way it shows the live one -- without needing the raw Parquet files
(and the algaesense-edge instance that wrote them) present at query time.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
    history database."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Root data directory (containing raw/experiments/...)")
    parser.add_argument("--db-path", required=True, type=Path, help="SQLite file to write into")
    parser.add_argument("--experiment-id", default=None, help="Ingest only this experiment (default: every experiment found)")
    args = parser.parse_args()

    if args.experiment_id:
        results = [ingest_experiment(args.db_path, args.data_dir, args.experiment_id)]
    else:
        results = ingest_all_experiments(args.db_path, args.data_dir)

    for result in results:
        print(f"{result['experiment_id']}: {result['voc_rows']} VOC rows, {result['camera_rows']} camera rows")


if __name__ == "__main__":
    main()
