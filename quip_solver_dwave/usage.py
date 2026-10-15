"""Durable ledger of QPU access time.

The v0.3 budget lived only in process memory, so a restart re-seeded it and
the miner could spend the same allotment twice. This module is the record of
truth instead: every microsecond the QPU bills is written to SQLite before the
pacer is asked whether to keep mining, so the answer survives a crash, a
container restart and a redeploy.

Usage accumulates into hourly buckets rather than one row per job. A busy day
is ~150k jobs but only 24 rows, the buckets align with the day boundary a
period reset lands on, and the pacer's "spent this period" query stays a sum
over a few hundred rows.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600

# WAL keeps the reader (the pacer) from blocking on the writer, and survives a
# process kill intact. synchronous=NORMAL loses at most the last transactions
# to a host power cut, never to the container restart that actually happens.
_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qpu_usage_hourly (
    hour_start_s   INTEGER PRIMARY KEY,
    jobs           INTEGER NOT NULL,
    access_time_us INTEGER NOT NULL
)
"""


def hour_floor(ts: float) -> int:
    """Floor a unix timestamp to the start of its UTC hour."""
    return int(ts) // SECONDS_PER_HOUR * SECONDS_PER_HOUR


class UsageLedger:
    """Append-only record of billed QPU access time, bucketed by UTC hour."""

    def __init__(self, path: str):
        self._path = path
        if path != ":memory:":
            parent = Path(path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        # Job workers record from pool threads while the session loop reads;
        # one connection guarded by a lock is simpler than a pool and the write
        # rate (a few per second) never makes the lock a bottleneck.
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        for pragma in _PRAGMAS:
            self._db.execute(pragma)
        self._db.execute(_SCHEMA)
        self._db.commit()

    def record(
        self,
        access_time_us: float,
        *,
        now: Optional[float] = None,
    ) -> None:
        """Add one completed job's billed access time to its hour bucket."""
        bucket = hour_floor(now if now is not None else time.time())
        billed = int(access_time_us)
        with self._lock:
            self._db.execute(
                "INSERT INTO qpu_usage_hourly (hour_start_s, jobs, access_time_us) "
                "VALUES (?, 1, ?) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "jobs = jobs + 1, "
                "access_time_us = access_time_us + excluded.access_time_us",
                (bucket, billed),
            )
            self._db.commit()

    def spent_us_since(self, start_ts: float) -> float:
        """Total billed access time in buckets at or after ``start_ts``.

        A period always begins on an hour boundary, so bucket-level granularity
        loses nothing: no bucket ever straddles the reset.
        """
        bucket = hour_floor(start_ts)
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(access_time_us), 0) FROM qpu_usage_hourly "
                "WHERE hour_start_s >= ?",
                (bucket,),
            ).fetchone()
        return float(row[0])

    def jobs_since(self, start_ts: float) -> int:
        """Jobs recorded in buckets at or after ``start_ts``."""
        bucket = hour_floor(start_ts)
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(jobs), 0) FROM qpu_usage_hourly "
                "WHERE hour_start_s >= ?",
                (bucket,),
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()
