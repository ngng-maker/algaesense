"""Buffers acquired rows in memory and flushes them to partitioned Parquet
files matching the exact layout jaxsr_calibration's raw schemas already
assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


"""
Layout: `{base_dir}/experiments/{experiment_id}/{partition_key}={value}/hour=YYYY-MM-DDTHH.parquet`.
"""


@dataclass
class PartitionedParquetWriter:
    """One writer per (experiment, sensor) or (experiment, camera)."""

    """
    `partition_key`/`partition_value` are e.g. `("sensor_id", "PID01")` or
    `("camera_id", "CAM01")`.

    Buffers rows in memory and only actually writes to disk when the hour
    changes (matching the `hour=YYYY-MM-DDTHH.parquet` partition scheme) or
    when `flush()`/`close()` is called explicitly -- writing one Parquet
    file per row would be both slow and produce an unreasonable number of
    tiny files; buffering an hour's worth at a time matches how the raw
    schema's own partitioning is meant to be used.
    """

    base_dir: Path

    experiment_id: str

    partition_key: str

    partition_value: str

    schema: pa.Schema

    _buffer: list[dict] = field(default_factory=list)

    _current_hour: str | None = field(default=None, init=False)

    def write_row(self, row: dict) -> None:
        """Add one row (a plain dict matching `self.schema`'s field names)
        to the buffer, flushing the previous hour's buffer first if this
        row belongs to a new hour."""

        """
        `strftime("%Y-%m-%dT%H")` turns a timestamp into e.g.
        "2026-07-15T09" -- exactly the hour-partition label the raw schema
        path convention uses.
        """
        row_hour = row["timestamp"].strftime("%Y-%m-%dT%H")
        if self._current_hour is not None and row_hour != self._current_hour:
            self.flush()
        self._current_hour = row_hour
        self._buffer.append(row)

    def flush(self) -> None:
        """Write whatever's currently buffered to its hour's Parquet file,
        then clear the buffer. Safe to call with nothing buffered
        (no-op)."""

        if not self._buffer:
            return

        partition_dir = (
            self.base_dir / "experiments" / self.experiment_id
            / f"{self.partition_key}={self.partition_value}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        out_path = partition_dir / f"hour={self._current_hour}.parquet"

        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        if out_path.exists():
            """
            The process may have been restarted mid-hour (e.g. after a
            crash or a planned restart) -- rather than overwrite whatever
            was already flushed for this hour, read it back and append the
            new rows on top, so no already-written data is lost.
            """
            existing = pq.read_table(out_path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, out_path)

        self._buffer = []

    def close(self) -> None:
        """Flush any remaining buffered rows -- call this when acquisition
        stops, so the last partial hour isn't silently dropped."""
        self.flush()
