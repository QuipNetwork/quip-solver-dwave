"""Durable history behind the round strategy.

Three records, all in the usage database next to ``qpu_usage_hourly``:

* ``qpu_throughput_hourly``: operational sums per UTC hour (completions, busy
  time, round trips, D-Wave service time, access time, concurrency). Every
  throughput metric derives from these at query time, so the estimator can
  change without a migration.
* ``miner_rounds``: one row per qblock round the miner saw, with what the
  strategy predicted at the boundary and what then happened.
* ``energy_margin_daily``: histogram of each job's best energy relative to
  the round's target, in whole energy units. Margins pool across problem
  instances and difficulty changes; raw energies do not.

Sums per hour rather than a row per job, for the same reason as ``usage.py``:
a busy day is up to ~150k jobs and the estimator only ever wants aggregates.
"""

from __future__ import annotations

import functools
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

from quip_solver_dwave.usage import SECONDS_PER_HOUR, hour_floor

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400
MILLI_PER_UNIT = 1000

SOURCE_LIVE = "live"
SOURCE_ATTEMPTS = "attempts"

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS qpu_throughput_hourly (
    hour_start_s      INTEGER PRIMARY KEY,
    source            TEXT    NOT NULL,
    busy_ms           INTEGER NOT NULL DEFAULT 0,
    jobs              INTEGER NOT NULL DEFAULT 0,
    wasted_jobs       INTEGER NOT NULL DEFAULT 0,
    rtt_ms_sum        INTEGER NOT NULL DEFAULT 0,
    rtt_ms_sumsq      INTEGER NOT NULL DEFAULT 0,
    rtt_ms_max        INTEGER NOT NULL DEFAULT 0,
    sapi_ms_sum       INTEGER NOT NULL DEFAULT 0,
    sapi_jobs         INTEGER NOT NULL DEFAULT 0,
    access_us_sum     INTEGER NOT NULL DEFAULT 0,
    inflight_sum      INTEGER NOT NULL DEFAULT 0,
    reads             INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS miner_rounds (
    start_ts_s        INTEGER PRIMARY KEY,
    generation        INTEGER NOT NULL,
    source            TEXT    NOT NULL,
    end_ts_s          INTEGER,
    target_milli      INTEGER,
    joined            INTEGER NOT NULL DEFAULT 0,
    reason            TEXT    NOT NULL DEFAULT '',
    expected_jobs     REAL,
    jobs              INTEGER NOT NULL DEFAULT 0,
    reads             INTEGER NOT NULL DEFAULT 0,
    hits              INTEGER NOT NULL DEFAULT 0,
    hits_coord        INTEGER NOT NULL DEFAULT 0,
    best_energy_milli INTEGER,
    access_us         INTEGER NOT NULL DEFAULT 0,
    won               INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS energy_margin_daily (
    day_start_s       INTEGER NOT NULL,
    margin_unit       INTEGER NOT NULL,
    jobs              INTEGER NOT NULL,
    PRIMARY KEY (day_start_s, margin_unit)
);
CREATE TABLE IF NOT EXISTS seeded_dirs (
    dir_name          TEXT PRIMARY KEY
);
"""


def day_floor(ts: float) -> int:
    """Floor a unix timestamp to the start of its UTC day."""
    return int(ts) // SECONDS_PER_DAY * SECONDS_PER_DAY


def margin_unit(best_energy_milli: int, target_milli: int) -> int:
    """Whole energy units between a job's best energy and the round target.

    Zero or negative means the job cleared the target. Ceiling division on
    the shortfall keeps "one milli over" in bin 1 and "one milli under" in
    bin 0, so ``bin <= 0`` is exactly "cleared".
    """
    return -((target_milli - best_energy_milli) // MILLI_PER_UNIT)


def split_by_hour(start: float, end: float) -> List[Tuple[int, int]]:
    """Split ``[start, end)`` into ``(hour_start_s, milliseconds)`` pieces."""
    out: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        hour = hour_floor(cursor)
        piece_end = min(end, float(hour + SECONDS_PER_HOUR))
        out.append((hour, int(round((piece_end - cursor) * 1000))))
        cursor = piece_end
    return out


class BusyClock:
    """Wall time with at least one job in flight, attributed to UTC hours.

    Busy time is emitted at every completion rather than when the last job
    finishes, so an hour row is never more than one job behind while the
    miner runs continuously.
    """

    def __init__(self) -> None:
        self._inflight = 0
        self._since: Optional[float] = None

    @property
    def inflight(self) -> int:
        return self._inflight

    def start(self, now: float) -> None:
        if self._inflight == 0:
            self._since = now
        self._inflight += 1

    def stop(self, now: float) -> List[Tuple[int, int]]:
        """Close the busy time since the last emission; return its pieces."""
        if self._inflight == 0 or self._since is None:
            return []
        self._inflight -= 1
        pieces = split_by_hour(self._since, now)
        self._since = now if self._inflight > 0 else None
        return pieces


@dataclass(frozen=True)
class JobSample:
    """What one completed job contributes to the history."""

    completed_at: float
    generation: int
    rtt_ms: int
    access_us: int
    inflight_at_submit: int
    reads: int
    best_energy_milli: Optional[int]
    target_milli: Optional[int]
    hits: int
    sapi_ms: Optional[int] = None


@dataclass(frozen=True)
class AttemptRoundSummary:
    """One round as the coordinator's attempts file describes it."""

    generation: int
    first_ts_s: int
    last_ts_s: int
    jobs: int
    hits_coord: int
    won: bool
    best_energy_milli: Optional[int]
    threshold_milli: Optional[int]
    access_us: int
    # (day_start_s, margin_unit) -> jobs, for the histogram seed.
    margins: Dict[Tuple[int, int], int]


# How far before the first attempt a live round may have opened. A Cancel
# precedes its first result by the pipeline's fill time, never by hours; a
# later coordinator run reusing the same generation number is outside this.
_ROUND_MATCH_WINDOW_S = 2 * SECONDS_PER_HOUR


class HistoryStore:
    """SQLite record of throughput, rounds and margins. One lock, one file."""

    def __init__(self, path: str):
        self._path = path
        if path != ":memory:":
            parent = Path(path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)
        # Worker threads write, the seed thread writes, the refresher reads:
        # one connection behind a lock, as the usage ledger does. Reentrant
        # so a batch held by one thread can still call the store's own
        # methods, which each take the lock themselves.
        self._lock = threading.RLock()
        self._batch_depth = 0
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            self._db.execute(pragma)
        self._db.executescript(_SCHEMA)
        # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so
        # columns earlier builds created outlive the code that wrote them:
        # seeded_dirs.lines_seen, whose NOT NULL would reject every
        # mark_seeded, and miner_rounds.p_win, the prediction of a win model
        # that no longer exists.
        for table, column in (("seeded_dirs", "lines_seen"), ("miner_rounds", "p_win")):
            columns = {
                row["name"] for row in self._db.execute(f"PRAGMA table_info({table})")
            }
            if column in columns:
                self._db.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        self._db.commit()

    def _commit(self) -> None:
        """Commit, unless a batch is open: it commits once, at its own end."""
        if self._batch_depth == 0:
            self._db.commit()

    @contextmanager
    def batch(self) -> Generator[None, None, None]:
        """Group several writes into one transaction.

        Reentrant: a method called from inside an open batch still takes
        the lock and calls ``_commit()``, which is a no-op until the
        outermost batch exits. Commits once, at the end, if the body ran
        clean; rolls back everything the batch wrote if it raised.
        """
        with self._lock:
            self._batch_depth += 1
            try:
                yield
            except Exception:
                if self._batch_depth == 1:
                    self._db.rollback()
                raise
            else:
                if self._batch_depth == 1:
                    self._db.commit()
            finally:
                self._batch_depth -= 1

    # -- hourly ---------------------------------------------------------

    def add_busy(self, pieces: Sequence[Tuple[int, int]]) -> None:
        if not pieces:
            return
        with self._lock:
            self._db.executemany(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, busy_ms) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "source = excluded.source, busy_ms = busy_ms + excluded.busy_ms",
                [(hour, SOURCE_LIVE, ms) for hour, ms in pieces],
            )
            self._commit()

    def record_job(self, sample: JobSample, round_start_ts: Optional[int]) -> None:
        """Fold one completed job into its hour, its round and the histogram."""
        hour = hour_floor(sample.completed_at)
        sapi_ms = sample.sapi_ms if sample.sapi_ms is not None else 0
        sapi_jobs = 1 if sample.sapi_ms is not None else 0
        with self._lock:
            self._db.execute(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, jobs, "
                "rtt_ms_sum, rtt_ms_sumsq, rtt_ms_max, sapi_ms_sum, sapi_jobs, "
                "access_us_sum, inflight_sum, reads) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "source = excluded.source, "
                "jobs = jobs + 1, "
                "rtt_ms_sum = rtt_ms_sum + excluded.rtt_ms_sum, "
                "rtt_ms_sumsq = rtt_ms_sumsq + excluded.rtt_ms_sumsq, "
                "rtt_ms_max = MAX(rtt_ms_max, excluded.rtt_ms_max), "
                "sapi_ms_sum = sapi_ms_sum + excluded.sapi_ms_sum, "
                "sapi_jobs = sapi_jobs + excluded.sapi_jobs, "
                "access_us_sum = access_us_sum + excluded.access_us_sum, "
                "inflight_sum = inflight_sum + excluded.inflight_sum, "
                "reads = reads + excluded.reads",
                (
                    hour,
                    SOURCE_LIVE,
                    sample.rtt_ms,
                    sample.rtt_ms * sample.rtt_ms,
                    sample.rtt_ms,
                    sapi_ms,
                    sapi_jobs,
                    sample.access_us,
                    sample.inflight_at_submit,
                    sample.reads,
                ),
            )
            if sample.best_energy_milli is not None and sample.target_milli is not None:
                self._db.execute(
                    "INSERT INTO energy_margin_daily (day_start_s, margin_unit, jobs) "
                    "VALUES (?, ?, 1) "
                    "ON CONFLICT(day_start_s, margin_unit) DO UPDATE SET jobs = jobs + 1",
                    (
                        day_floor(sample.completed_at),
                        margin_unit(sample.best_energy_milli, sample.target_milli),
                    ),
                )
            if round_start_ts is not None:
                self._db.execute(
                    "UPDATE miner_rounds SET "
                    "jobs = jobs + 1, reads = reads + ?, hits = hits + ?, "
                    "access_us = access_us + ?, "
                    "best_energy_milli = CASE WHEN ? IS NULL THEN best_energy_milli "
                    "WHEN best_energy_milli IS NULL OR ? < best_energy_milli THEN ? "
                    "ELSE best_energy_milli END "
                    "WHERE start_ts_s = ?",
                    (
                        sample.reads,
                        sample.hits,
                        sample.access_us,
                        sample.best_energy_milli,
                        sample.best_energy_milli,
                        sample.best_energy_milli,
                        round_start_ts,
                    ),
                )
            self._commit()

    def record_wasted(self, now: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, wasted_jobs) "
                "VALUES (?, ?, 1) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "source = excluded.source, wasted_jobs = wasted_jobs + 1",
                (hour_floor(now), SOURCE_LIVE),
            )
            self._commit()

    def hourly_rows(self, since_ts: float) -> List[sqlite3.Row]:
        with self._lock:
            return self._db.execute(
                "SELECT * FROM qpu_throughput_hourly WHERE hour_start_s >= ? "
                "ORDER BY hour_start_s",
                (hour_floor(since_ts),),
            ).fetchall()

    def margin_counts(self, since_ts: float) -> Dict[int, int]:
        with self._lock:
            rows = self._db.execute(
                "SELECT margin_unit, SUM(jobs) AS jobs FROM energy_margin_daily "
                "WHERE day_start_s >= ? GROUP BY margin_unit",
                (day_floor(since_ts),),
            ).fetchall()
        return {int(r["margin_unit"]): int(r["jobs"]) for r in rows}

    # -- rounds ---------------------------------------------------------

    def open_round(
        self,
        start_ts: float,
        generation: int,
        *,
        joined: bool,
        reason: str,
        expected_jobs: Optional[float],
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO miner_rounds (start_ts_s, generation, source, "
                "joined, reason, expected_jobs) VALUES (?, ?, ?, ?, ?, ?)",
                (int(start_ts), generation, SOURCE_LIVE, int(joined), reason, expected_jobs),
            )
            self._commit()

    def close_round(self, start_ts: float, end_ts: float) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE miner_rounds SET end_ts_s = ? WHERE start_ts_s = ? AND end_ts_s IS NULL",
                (int(end_ts), int(start_ts)),
            )
            self._commit()

    def set_round_target(self, start_ts: float, target_milli: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE miner_rounds SET target_milli = ? WHERE start_ts_s = ?",
                (target_milli, int(start_ts)),
            )
            self._commit()

    def find_live_round(self, generation: int, near_ts: float) -> Optional[int]:
        """The live round of ``generation`` that opened shortly before ``near_ts``."""
        with self._lock:
            row = self._db.execute(
                "SELECT start_ts_s FROM miner_rounds WHERE generation = ? AND source = ? "
                "AND start_ts_s BETWEEN ? AND ? ORDER BY start_ts_s DESC LIMIT 1",
                (generation, SOURCE_LIVE, int(near_ts) - _ROUND_MATCH_WINDOW_S, int(near_ts)),
            ).fetchone()
        return int(row["start_ts_s"]) if row is not None else None

    def apply_attempt_round(
        self, summary: AttemptRoundSummary, *, insert_missing: bool
    ) -> str:
        """Fold an attempts-file round in: outcomes onto a live row, or a new row.

        Returns ``"updated"``, ``"inserted"`` or ``"skipped"``.
        """
        live = self.find_live_round(summary.generation, summary.first_ts_s)
        with self._lock:
            if live is not None:
                self._db.execute(
                    "UPDATE miner_rounds SET hits_coord = ?, won = MAX(won, ?) "
                    "WHERE start_ts_s = ?",
                    (summary.hits_coord, int(summary.won), live),
                )
                self._commit()
                return "updated"
            if not insert_missing:
                return "skipped"
            self._db.execute(
                "INSERT OR IGNORE INTO miner_rounds (start_ts_s, generation, source, "
                "end_ts_s, target_milli, joined, reason, jobs, hits, hits_coord, "
                "best_energy_milli, access_us, won) "
                "VALUES (?, ?, ?, ?, ?, 1, 'attempts', ?, ?, ?, ?, ?, ?)",
                (
                    summary.first_ts_s,
                    summary.generation,
                    SOURCE_ATTEMPTS,
                    summary.last_ts_s,
                    summary.threshold_milli,
                    summary.jobs,
                    # hits counts reads at or below the target; the attempts
                    # file has no per-read count, only hits_coord (attempts
                    # the coordinator accepted). Left at its default of 0.
                    0,
                    summary.hits_coord,
                    summary.best_energy_milli,
                    summary.access_us,
                    int(summary.won),
                ),
            )
            self._commit()
            return "inserted"

    def rounds(self, since_ts: float, limit: Optional[int] = None) -> List[sqlite3.Row]:
        sql = "SELECT * FROM miner_rounds WHERE start_ts_s >= ? ORDER BY start_ts_s DESC"
        params: Tuple[object, ...] = (int(since_ts),)
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(since_ts), int(limit))
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    # -- seeding --------------------------------------------------------

    def seed_margin(self, day_start_s: int, margin: int, jobs: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO energy_margin_daily (day_start_s, margin_unit, jobs) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(day_start_s, margin_unit) DO UPDATE SET jobs = jobs + excluded.jobs",
                (day_start_s, margin, jobs),
            )
            self._commit()

    def seed_hourly(self, hour_start_s: int, *, jobs: int, busy_ms: int, access_us: int) -> None:
        """An approximate hour from the attempts file.

        Adds to an existing row seeded by an earlier directory in the same
        hour; leaves a live row alone. Qblock rounds run about ten minutes,
        so several directories routinely share one hour.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO qpu_throughput_hourly (hour_start_s, source, jobs, "
                "busy_ms, access_us_sum) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(hour_start_s) DO UPDATE SET "
                "jobs = jobs + excluded.jobs, "
                "busy_ms = busy_ms + excluded.busy_ms, "
                "access_us_sum = access_us_sum + excluded.access_us_sum "
                "WHERE qpu_throughput_hourly.source = 'attempts'",
                (hour_start_s, SOURCE_ATTEMPTS, jobs, busy_ms, access_us),
            )
            self._commit()

    def is_seeded(self, dir_name: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM seeded_dirs WHERE dir_name = ?", (dir_name,)
            ).fetchone()
        return row is not None

    def has_live_hours(self, start_hour_s: int, end_hour_s: int) -> bool:
        """Whether a live-source hourly row falls in ``[start_hour_s, end_hour_s]``.

        Used to decide whether the miner was running during an attempts
        directory's time span even when it never opened a round for one of
        the generations in it (see ``attempts.seed_from_attempts``).
        """
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM qpu_throughput_hourly WHERE source = ? "
                "AND hour_start_s BETWEEN ? AND ? LIMIT 1",
                (SOURCE_LIVE, start_hour_s, end_hour_s),
            ).fetchone()
        return row is not None

    def mark_seeded(self, dir_name: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO seeded_dirs (dir_name) VALUES (?)",
                (dir_name,),
            )
            self._commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


# How long a recorder stays quiet after logging a failure. Every job would
# otherwise repeat the same warning at pipeline speed.
_WARN_INTERVAL_S = 60.0


class HistoryRecorder:
    """The session loop's view of the history: never raises, never blocks it.

    Every write from the session thread and the job workers is queued to
    this recorder's one worker thread. Job workers call in after billing,
    and the session thread calls in at Cancel and SetTarget; neither waits
    on SQLite, and the single thread keeps writes in order, so a round is
    opened before its jobs are folded in. A failure costs a row of history
    and nothing else. The busy clock and the generation-to-round map live
    here because the store has no notion of "the current round".

    The seed and pickup threads (``attempts.seed_from_attempts``,
    ``attempts.pickup_outcomes``) are the exception: they write through
    ``self.store`` directly, on their own threads, serialized by the
    store's own lock rather than this recorder's queue.
    """

    def __init__(self, store: HistoryStore):
        self.store = store
        self._lock = threading.Lock()
        self._clock = BusyClock()
        self._last_cancel_generation = 0
        self._open_start: Optional[int] = None
        # Job generation -> round start, kept for the last few rounds so a
        # result that lands after its boundary still finds its row.
        self._round_starts: Dict[int, int] = {}
        self._warned_at = 0.0
        self._closed = False
        self._io = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dwave-history")

    @classmethod
    def open(cls, path: str) -> Optional["HistoryRecorder"]:
        try:
            return cls(HistoryStore(path))
        except Exception as exc:  # noqa: BLE001 - history is optional, mining is not
            logger.warning("history disabled: cannot open %s: %s", path, exc)
            return None

    def _submit(self, what: str, fn) -> None:
        if self._closed:
            return
        try:
            self._io.submit(self._run, what, fn)
        except RuntimeError:
            # The executor is shut down: the session is ending.
            pass

    def _run(self, what: str, fn) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - see class docstring
            now = time.monotonic()
            if now - self._warned_at >= _WARN_INTERVAL_S:
                self._warned_at = now
                logger.warning("history: %s failed: %s", what, exc)

    def flush(self) -> None:
        """Wait for every queued write. Tests and shutdown use it."""
        if self._closed:
            return
        self._io.submit(lambda: None).result()

    def job_started(self, now: float) -> None:
        with self._lock:
            self._clock.start(now)

    def job_finished(self, now: float) -> None:
        with self._lock:
            pieces = self._clock.stop(now)
        if pieces:
            self._submit("busy time", functools.partial(self.store.add_busy, pieces))

    def record_job(self, sample: JobSample) -> None:
        with self._lock:
            start = self._round_starts.get(sample.generation)
        self._submit("job sample", functools.partial(self.store.record_job, sample, start))

    def record_wasted(self, now: float) -> None:
        self._submit("wasted job", functools.partial(self.store.record_wasted, now))

    def round_boundary(
        self,
        cancel_generation: int,
        now: float,
        *,
        joined: bool,
        reason: str,
        expected_jobs: Optional[float],
    ) -> bool:
        """Close the open round and open the next. False when not a fresh boundary.

        ``Cancel(max_generation=N)`` names the generation the coordinator has
        just abandoned. The jobs of the round it opens carry ``N + 1``, which
        is also the generation the coordinator writes to attempts.jsonl, so
        the row is keyed by that and not by the watermark.

        ``start_ts_s`` is the round's primary key, at whole-second resolution.
        Two boundaries landing in the same wall-clock second (routine in a
        fast test, possible in production too) would otherwise collide, and
        ``open_round``'s ``INSERT OR REPLACE`` would silently drop the first
        round's row. Bumping past the previous start keeps every round its
        own row without needing sub-second precision in the schema.
        """
        generation = cancel_generation + 1
        with self._lock:
            if cancel_generation <= self._last_cancel_generation:
                return False
            self._last_cancel_generation = cancel_generation
            previous = self._open_start
            start = int(now)
            if previous is not None and start <= previous:
                start = previous + 1
            self._open_start = start
            self._round_starts[generation] = start
            for old in [g for g in self._round_starts if g < generation - 4]:
                del self._round_starts[old]
            # Submitted while the lock is still held: a record_job for this
            # generation cannot queue its UPDATE ahead of this INSERT.
            if previous is not None:
                # A bumped start can land after `now`; closing at `now`
                # would then close the round before it opened.
                self._submit(
                    "round close",
                    functools.partial(self.store.close_round, previous, max(now, previous)),
                )
            self._submit(
                "round open",
                functools.partial(
                    self.store.open_round,
                    start,
                    generation,
                    joined=joined,
                    reason=reason,
                    expected_jobs=expected_jobs,
                ),
            )
        return True

    def round_target(self, target_milli: int) -> None:
        with self._lock:
            start = self._open_start
        if start is not None:
            self._submit(
                "round target",
                functools.partial(self.store.set_round_target, start, target_milli),
            )

    def close(self) -> None:
        """Drain the queue and close the store. Safe to call twice."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._io.shutdown(wait=True)
        try:
            self.store.close()
        except Exception as exc:  # noqa: BLE001 - nothing left to protect
            logger.warning("history: close failed: %s", exc)
