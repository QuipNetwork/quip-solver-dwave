"""Day-of-month by hour-of-day estimates from the hourly history.

Two cycles drive D-Wave's shared queue: the working day, and the monthly
contract every account runs on. Each UTC hour keys a cell by (day of the
month normalized to 28 bins, hour of the day), 672 cells in all. A cell on
its own would take months to fill, so the estimate is the multiplicative
main-effects model the call-center literature settles on for intraday-by-
day arrival rates (Weinberg, Brown and Stroud 2007; Ibrahim and L'Ecuyer
2013): the global rate times an hour-of-day factor times a day-of-month
factor, 52 numbers that fill from a few weeks of rows. A cell with its own
evidence then shrinks toward that prediction, so an interaction such as
"month end is slow only in working hours" shows once the cell has seen
enough jobs and is invisible before that.

Every number derives from the operational sums in ``qpu_throughput_hourly``
at query time. Throughput is completions over busy time, mean round trip is
round-trip sum over completions, and the D-Wave queue wait is server
service time minus access time.
"""

from __future__ import annotations

import calendar
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Protocol

from quip_solver_dwave.history import HistoryStore

logger = logging.getLogger(__name__)


class Row(Protocol):
    """What a cell needs from a row: field lookup by name.

    A plain ``dict`` (tests) and ``sqlite3.Row`` (``HistoryStore.hourly_rows``)
    both satisfy this; a ``dict`` is a ``typing.Mapping``, but ``sqlite3.Row``
    is not, which is why this is a narrower ``Protocol`` instead.
    """

    def __getitem__(self, key: str, /) -> Any: ...


HOURS_PER_DAY = 24
# Months run 28 to 31 days. Squeezing every month onto the shortest one
# means every bin fills every month; a longer month pairs a few adjacent
# days into one bin instead of leaving bins empty.
DAY_BINS = 28
SLOTS = DAY_BINS * HOURS_PER_DAY
SECONDS_PER_DAY = 86_400

# Each cell sees one hour per month, so the window spans several months
# and a row three months old still carries half its weight.
DEFAULT_WINDOW_DAYS = 183
DEFAULT_HALF_LIFE_DAYS = 91.0
# Pseudo-evidence, in jobs, behind the prediction a cell shrinks toward and
# behind each factor's pull away from 1. About one round's worth: fewer
# jobs than that and the prediction leads.
DEFAULT_PSEUDO_JOBS = 200.0


def day_bin_of(day: int, days_in_month: int) -> int:
    """Calendar day 1..N onto bins 0..27, so every month fills every bin."""
    return (day - 1) * DAY_BINS // days_in_month


def slot_of(ts: float) -> int:
    """(normalized day of the month, UTC hour of the day) as one index."""
    t = datetime.fromtimestamp(int(ts), timezone.utc)
    days_in_month = calendar.monthrange(t.year, t.month)[1]
    return day_bin_of(t.day, days_in_month) * HOURS_PER_DAY + t.hour


def day_bin(slot: int) -> int:
    return slot // HOURS_PER_DAY


def hour_of(slot: int) -> int:
    return slot % HOURS_PER_DAY


def slot_label(slot: int) -> str:
    """``d05 13h``: the normalized day (1..28) and the UTC hour."""
    return f"d{day_bin(slot) + 1:02d} {hour_of(slot):02d}h"


def recency_weight(age_s: float, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    return 0.5 ** (age_s / SECONDS_PER_DAY / half_life_days)


@dataclass(frozen=True)
class SlotStats:
    """Shrunk estimates for one cell.

    ``None`` means no evidence at any level. ``evidence`` is the weighted
    job count observed in this cell itself, before shrinkage.
    """

    jobs_per_s: Optional[float]
    rtt_s: Optional[float]
    queue_s: Optional[float]
    evidence: float


@dataclass(frozen=True)
class Profile:
    """The 672 cell estimates and the two throughput factors behind them.

    A factor is the margin's throughput relative to the global rate, shrunk
    toward 1, so ``1.0`` reads as "no evidence either way".
    """

    slots: List[SlotStats]
    hour_factors: List[float]
    day_factors: List[float]


class _Acc:
    """Weighted operational sums for one cell, one margin, or the whole window."""

    __slots__ = ("jobs", "live_jobs", "busy_s", "rtt_s", "sapi_s", "sapi_jobs", "access_s")

    def __init__(self) -> None:
        self.jobs = 0.0
        # Seeded rows carry completions but no round trips; only live rows
        # may serve as the denominator for the round-trip mean.
        self.live_jobs = 0.0
        self.busy_s = 0.0
        self.rtt_s = 0.0
        self.sapi_s = 0.0
        self.sapi_jobs = 0.0
        self.access_s = 0.0

    def add(self, row: Row, w: float) -> None:
        jobs = float(row["jobs"])
        self.jobs += w * jobs
        if row["source"] == "live":
            self.live_jobs += w * jobs
        self.busy_s += w * float(row["busy_ms"]) / 1000.0
        self.rtt_s += w * float(row["rtt_ms_sum"]) / 1000.0
        self.sapi_s += w * float(row["sapi_ms_sum"]) / 1000.0
        self.sapi_jobs += w * float(row["sapi_jobs"])
        self.access_s += w * float(row["access_us_sum"]) / 1_000_000.0

    def rate(self) -> Optional[float]:
        return self.jobs / self.busy_s if self.busy_s > 0 else None

    def rtt(self) -> Optional[float]:
        return self.rtt_s / self.live_jobs if self.live_jobs > 0 else None

    def queue(self) -> Optional[float]:
        if self.sapi_jobs <= 0 or self.jobs <= 0:
            return None
        return max(0.0, self.sapi_s / self.sapi_jobs - self.access_s / self.jobs)

    def evidence_for(self, measure: str) -> float:
        """The job count backing one measure: seeded rows carry no round
        trip and no SAPI timing, so ``rtt`` and ``queue`` have less evidence
        than ``rate`` does whenever seeded jobs are in the mix."""
        if measure == "rtt":
            return self.live_jobs
        if measure == "queue":
            return self.sapi_jobs
        return self.jobs


_MEASURES = ("rate", "rtt", "queue")


def _shrink(
    value: Optional[float], n: float, prior: Optional[float], pseudo: float
) -> Optional[float]:
    if prior is None:
        return value
    if value is None:
        return prior
    return (n * value + pseudo * prior) / (n + pseudo)


def _factor(margin: _Acc, measure: str, overall: Optional[float], pseudo: float) -> float:
    """One margin's multiplier on the global estimate, shrunk toward 1."""
    value = getattr(margin, measure)()
    if value is None or not overall:
        return 1.0
    n = margin.evidence_for(measure)
    return (n * (value / overall) + pseudo) / (n + pseudo)


def slot_stats(
    rows: Iterable[Row],
    now: float,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    pseudo_jobs: float = DEFAULT_PSEUDO_JOBS,
) -> Profile:
    """672 cell estimates from hourly rows, recency-weighted and shrunk."""
    cells = [_Acc() for _ in range(SLOTS)]
    by_hour = [_Acc() for _ in range(HOURS_PER_DAY)]
    by_day = [_Acc() for _ in range(DAY_BINS)]
    everything = _Acc()
    horizon = now - window_days * SECONDS_PER_DAY
    for row in rows:
        hour = int(row["hour_start_s"])
        if hour < horizon or hour > now:
            continue
        w = recency_weight(now - hour, half_life_days)
        slot = slot_of(hour)
        cells[slot].add(row, w)
        by_hour[hour_of(slot)].add(row, w)
        by_day[day_bin(slot)].add(row, w)
        everything.add(row, w)

    # One pair of factor lists per measure; the rate pair is the one the
    # profile reports.
    overall = {m: getattr(everything, m)() for m in _MEASURES}
    hour_factors = {
        m: [_factor(acc, m, overall[m], pseudo_jobs) for acc in by_hour] for m in _MEASURES
    }
    day_factors = {
        m: [_factor(acc, m, overall[m], pseudo_jobs) for acc in by_day] for m in _MEASURES
    }

    slots: List[SlotStats] = []
    for slot in range(SLOTS):
        mine = cells[slot]
        estimates = []
        for m in _MEASURES:
            base = overall[m]
            prediction = (
                base * hour_factors[m][hour_of(slot)] * day_factors[m][day_bin(slot)]
                if base is not None
                else None
            )
            estimates.append(
                _shrink(getattr(mine, m)(), mine.evidence_for(m), prediction, pseudo_jobs)
            )
        slots.append(
            SlotStats(
                jobs_per_s=estimates[0],
                rtt_s=estimates[1],
                queue_s=estimates[2],
                evidence=mine.jobs,
            )
        )
    return Profile(slots=slots, hour_factors=hour_factors["rate"], day_factors=day_factors["rate"])


# --- snapshot ----------------------------------------------------------

# The chain's convergence target for a round, used until a live round has
# closed. Live rounds are the real measure and take over immediately.
DEFAULT_ROUND_LENGTH_S = 600.0
# Measured on Advantage2_system1 at production size (num_reads=48).
DEFAULT_ACCESS_S_PER_JOB = 0.046
ROUND_LENGTH_SAMPLE = 100


@dataclass(frozen=True)
class Snapshot:
    """Everything the boundary decision reads. Built off the session thread."""

    built_at: float
    profile: Profile
    round_length_s: float
    access_s_per_job: float

    @property
    def slots(self) -> List[SlotStats]:
        return self.profile.slots


def _median(values: List[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def build_snapshot(
    store: HistoryStore,
    now: float,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    pseudo_jobs: float = DEFAULT_PSEUDO_JOBS,
) -> Snapshot:
    since = now - window_days * SECONDS_PER_DAY
    rows = store.hourly_rows(since)
    profile = slot_stats(
        rows, now, window_days=window_days, half_life_days=half_life_days, pseudo_jobs=pseudo_jobs
    )

    jobs = sum(int(r["jobs"]) for r in rows)
    access_us = sum(int(r["access_us_sum"]) for r in rows)
    access_s = access_us / jobs / 1_000_000.0 if jobs > 0 and access_us > 0 else DEFAULT_ACCESS_S_PER_JOB

    lengths = [
        float(r["end_ts_s"] - r["start_ts_s"])
        for r in store.rounds(since_ts=since)
        if r["source"] == "live" and r["end_ts_s"] is not None and r["end_ts_s"] > r["start_ts_s"]
    ][:ROUND_LENGTH_SAMPLE]
    round_length = _median(lengths) if lengths else DEFAULT_ROUND_LENGTH_S

    return Snapshot(
        built_at=now,
        profile=profile,
        round_length_s=round_length,
        access_s_per_job=access_s,
    )


class SnapshotRefresher:
    """Rebuilds the snapshot on its own thread; the boundary reads the latest.

    A rebuild is a few aggregate queries over a few thousand rows, but it is
    IO on the mounted volume, and the boundary decision runs on the session
    thread under the dispatch lock. The decision therefore reads memory only
    and this thread does the reading.
    """

    def __init__(self, store: HistoryStore, *, interval_s: float = 300.0):
        self._store = store
        self._interval = interval_s
        self._lock = threading.Lock()
        self._latest: Optional[Snapshot] = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="dwave-history-snapshot", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def latest(self) -> Optional[Snapshot]:
        with self._lock:
            return self._latest

    def refresh_now(self) -> Optional[Snapshot]:
        """Rebuild synchronously. Returns the snapshot now current."""
        try:
            snap = build_snapshot(self._store, time.time())
        except Exception as exc:  # noqa: BLE001 - keep serving the last good one
            logger.warning("history: snapshot refresh failed: %s", exc)
            return self.latest()
        with self._lock:
            self._latest = snap
        return snap

    def request_refresh(self) -> None:
        """Wake the thread early, after a seed or an outcome pickup."""
        self._wake.set()

    def stop(self) -> None:
        """Signal the thread to stop and wait briefly for it to notice.

        A refresh in flight reads the store; joining here with a bounded
        timeout keeps that read from racing the store's close during
        shutdown, without risking a hang if the thread was never started.
        """
        self._stopping.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stopping.is_set():
            self.refresh_now()
            self._wake.wait(self._interval)
            self._wake.clear()
