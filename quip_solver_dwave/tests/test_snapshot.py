"""The snapshot: the profile plus the round length and access time the decision needs."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quip_solver_dwave.history import HistoryStore, JobSample
from quip_solver_dwave.profile import (
    DEFAULT_ACCESS_S_PER_JOB,
    DEFAULT_ROUND_LENGTH_S,
    HOURS_PER_DAY,
    SnapshotRefresher,
    build_snapshot,
)

MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()
NOW = MONDAY + 6 * 86_400


def _round(store, start, generation, *, jobs, length=600):
    store.open_round(start, generation, joined=True, reason="budget", expected_jobs=None)
    store.close_round(start, start + length)
    for i in range(jobs):
        store.record_job(
            JobSample(
                completed_at=start + 1 + i,
                generation=generation,
                rtt_ms=3000,
                access_us=46_000,
                inflight_at_submit=1,
                reads=48,
                best_energy_milli=-14_540_000,
                target_milli=-14_554_000,
                hits=0,
            ),
            round_start_ts=int(start),
        )


def test_no_history_yields_the_defaults_and_no_evidence():
    snap = build_snapshot(HistoryStore(":memory:"), NOW)
    assert snap.round_length_s == DEFAULT_ROUND_LENGTH_S
    assert snap.access_s_per_job == DEFAULT_ACCESS_S_PER_JOB
    assert all(s.jobs_per_s is None for s in snap.slots)
    assert snap.profile.hour_factors == [1.0] * HOURS_PER_DAY


def test_round_length_is_the_median_of_closed_live_rounds():
    store = HistoryStore(":memory:")
    for i, length in enumerate((500, 900, 600)):
        _round(store, MONDAY + i * 3600, 10 + i, jobs=1, length=length)
    store.open_round(MONDAY + 5 * 3600, 20, joined=True, reason="budget", expected_jobs=None)  # still open
    assert build_snapshot(store, NOW).round_length_s == 600


def test_access_time_per_job_comes_from_the_hourly_sums():
    store = HistoryStore(":memory:")
    _round(store, MONDAY, 2, jobs=10)
    assert build_snapshot(store, NOW).access_s_per_job == pytest.approx(0.046)


def test_the_refresher_serves_the_latest_snapshot_without_a_thread():
    store = HistoryStore(":memory:")
    refresher = SnapshotRefresher(store, interval_s=3600.0)
    assert refresher.latest() is None
    snap = refresher.refresh_now()
    assert snap is not None and refresher.latest() is snap
    refresher.stop()


def test_a_failed_refresh_keeps_the_previous_snapshot(caplog):
    store = HistoryStore(":memory:")
    refresher = SnapshotRefresher(store, interval_s=3600.0)
    first = refresher.refresh_now()
    store.close()  # every query now raises
    assert refresher.refresh_now() is first
    assert any("snapshot" in r.getMessage() for r in caplog.records)
