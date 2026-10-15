"""Rounds: one row per qblock boundary, jobs attributed by generation."""

from __future__ import annotations

import logging

from quip_solver_dwave.history import (
    AttemptRoundSummary,
    HistoryRecorder,
    HistoryStore,
    JobSample,
)

T0 = 1788796800


def _sample(generation: int, at: float, best=-14_540_000, hits=0) -> JobSample:
    return JobSample(
        completed_at=at,
        generation=generation,
        rtt_ms=3000,
        access_us=46_000,
        inflight_at_submit=1,
        reads=48,
        best_energy_milli=best,
        target_milli=-14_554_000,
        hits=hits,
    )


def _summary(generation: int, first: int, **kw) -> AttemptRoundSummary:
    base = dict(
        generation=generation,
        first_ts_s=first,
        last_ts_s=first + 120,
        jobs=3,
        hits_coord=1,
        won=False,
        best_energy_milli=-14_556_000,
        threshold_milli=-14_554_000,
        access_us=138_000,
        margins={},
    )
    base.update(kw)
    # The overridable-dict builder loses per-field types; the fields
    # themselves are checked wherever a test asserts on them.
    return AttemptRoundSummary(**base)  # type: ignore[arg-type]


def test_a_boundary_closes_the_previous_round_and_opens_the_next():
    store = HistoryStore(":memory:")
    store.open_round(T0, 5, joined=True, reason="budget", expected_jobs=None)
    store.close_round(T0, T0 + 600)
    store.open_round(T0 + 600, 6, joined=False, reason="budget-sat-out", expected_jobs=0.0)
    rows = store.rounds(since_ts=0, limit=10)
    assert [(r["generation"], r["end_ts_s"], r["joined"]) for r in rows] == [
        (6, None, 0),
        (5, T0 + 600, 1),
    ]
    assert rows[0]["expected_jobs"] == 0.0


def test_jobs_attribute_to_their_generation_s_round():
    # Cancel(5) abandons generation 5 and opens the round whose jobs carry 6,
    # which is the generation the coordinator writes to attempts.jsonl.
    rec = HistoryRecorder(HistoryStore(":memory:"))
    rec.round_boundary(5, T0, joined=True, reason="budget", expected_jobs=None)
    rec.round_boundary(6, T0 + 600, joined=True, reason="budget", expected_jobs=None)
    # A generation-6 result landing after Cancel(6) still belongs to round 6.
    rec.record_job(_sample(6, T0 + 601, best=-14_560_000, hits=2))
    rec.record_job(_sample(7, T0 + 602))
    rec.flush()
    rows = {r["generation"]: r for r in rec.store.rounds(since_ts=0, limit=10)}
    assert set(rows) == {6, 7}
    assert rows[6]["jobs"] == 1 and rows[6]["hits"] == 2
    assert rows[6]["best_energy_milli"] == -14_560_000
    assert rows[6]["end_ts_s"] == T0 + 600 and rows[7]["end_ts_s"] is None
    assert rows[7]["jobs"] == 1 and rows[7]["best_energy_milli"] == -14_540_000


def test_a_repeated_watermark_is_not_a_new_round():
    rec = HistoryRecorder(HistoryStore(":memory:"))
    assert rec.round_boundary(5, T0, joined=True, reason="budget", expected_jobs=None)
    assert not rec.round_boundary(5, T0 + 1, joined=True, reason="budget", expected_jobs=None)
    assert not rec.round_boundary(4, T0 + 2, joined=True, reason="budget", expected_jobs=None)
    rec.flush()
    assert len(rec.store.rounds(since_ts=0, limit=10)) == 1


def test_two_boundaries_in_the_same_second_open_two_rounds():
    # A round can close and the next open within the same wall-clock second
    # in a fast-moving test harness (and, in principle, in production too).
    # start_ts_s is the primary key: two rounds sharing one would collide and
    # "INSERT OR REPLACE" would silently drop the first round's row.
    rec = HistoryRecorder(HistoryStore(":memory:"))
    rec.round_boundary(5, T0, joined=True, reason="budget", expected_jobs=None)
    rec.round_boundary(6, T0, joined=True, reason="budget", expected_jobs=None)
    rec.flush()
    rows = {r["generation"]: r for r in rec.store.rounds(since_ts=0, limit=10)}
    assert set(rows) == {6, 7}
    starts = {r["start_ts_s"] for r in rows.values()}
    assert len(starts) == 2
    assert rows[6]["end_ts_s"] is not None and rows[7]["end_ts_s"] is None


def test_three_boundaries_in_the_same_second_never_close_a_round_before_it_opens():
    # A third boundary in the same wall-clock second bumps the start twice,
    # so the middle round's own close (at the raw wall time, not the bumped
    # start) must not land before that bumped start.
    rec = HistoryRecorder(HistoryStore(":memory:"))
    rec.round_boundary(5, T0, joined=True, reason="budget", expected_jobs=None)
    rec.round_boundary(6, T0, joined=True, reason="budget", expected_jobs=None)
    rec.round_boundary(7, T0, joined=True, reason="budget", expected_jobs=None)
    rec.flush()
    rows = {r["generation"]: r for r in rec.store.rounds(since_ts=0, limit=10)}
    assert set(rows) == {6, 7, 8}
    starts = {r["start_ts_s"] for r in rows.values()}
    assert len(starts) == 3
    for r in rows.values():
        if r["end_ts_s"] is not None:
            assert r["end_ts_s"] >= r["start_ts_s"]


def test_set_target_fills_the_open_round():
    rec = HistoryRecorder(HistoryStore(":memory:"))
    rec.round_boundary(5, T0, joined=True, reason="budget", expected_jobs=None)
    rec.round_target(-14_554_000)
    rec.flush()
    assert rec.store.rounds(since_ts=0, limit=1)[0]["target_milli"] == -14_554_000


def test_busy_time_flows_from_the_recorder_s_clock():
    rec = HistoryRecorder(HistoryStore(":memory:"))
    rec.job_started(T0 + 1.0)
    rec.job_finished(T0 + 4.0)
    rec.flush()
    assert rec.store.hourly_rows(T0)[0]["busy_ms"] == 3000


def test_apply_attempt_round_updates_a_live_row_and_inserts_a_missing_one():
    store = HistoryStore(":memory:")
    store.open_round(T0, 5, joined=True, reason="budget", expected_jobs=None)
    assert store.apply_attempt_round(_summary(5, T0 + 30, won=True), insert_missing=True) == "updated"
    assert store.apply_attempt_round(_summary(9, T0 + 9000), insert_missing=True) == "inserted"
    assert store.apply_attempt_round(_summary(10, T0 + 12000), insert_missing=False) == "skipped"
    rows = {r["generation"]: r for r in store.rounds(since_ts=0, limit=10)}
    assert rows[5]["won"] == 1 and rows[5]["hits_coord"] == 1 and rows[5]["source"] == "live"
    assert rows[9]["source"] == "attempts" and rows[9]["jobs"] == 3 and rows[9]["joined"] == 1
    # hits counts reads at or below the target; the attempts file has no
    # per-read count, only hits_coord (attempts the coordinator accepted).
    assert rows[9]["hits"] == 0 and rows[9]["hits_coord"] == 1
    assert 10 not in rows


def test_a_live_round_is_only_matched_near_the_attempts_time():
    store = HistoryStore(":memory:")
    store.open_round(T0, 5, joined=True, reason="budget", expected_jobs=None)
    assert store.find_live_round(5, T0 + 30) == T0
    assert store.find_live_round(5, T0 + 3 * 3600) is None  # a later run's generation 5
    assert store.find_live_round(6, T0 + 30) is None


def test_seed_bookkeeping_is_per_directory():
    store = HistoryStore(":memory:")
    assert not store.is_seeded("1258")
    store.mark_seeded("1258")
    assert store.is_seeded("1258")


def test_seeded_hourly_rows_never_overwrite_live_ones():
    store = HistoryStore(":memory:")
    store.seed_hourly(T0, jobs=40, busy_ms=36_000, access_us=1_840_000)
    store.record_job(_sample(5, T0 + 3600 + 5), round_start_ts=None)
    store.seed_hourly(T0 + 3600, jobs=99, busy_ms=1, access_us=1)
    rows = {r["hour_start_s"]: r for r in store.hourly_rows(T0)}
    assert rows[T0]["source"] == "attempts" and rows[T0]["jobs"] == 40
    # The live row is unchanged: the seed is silently dropped, not merged.
    assert rows[T0 + 3600]["source"] == "live" and rows[T0 + 3600]["jobs"] == 1


def test_seed_hourly_adds_every_directory_in_the_hour_instead_of_only_the_first():
    # Qblock rounds run about ten minutes, so several attempts directories
    # share one hour; each must add to the row, not lose to INSERT OR IGNORE.
    store = HistoryStore(":memory:")
    store.seed_hourly(T0, jobs=10, busy_ms=9000, access_us=460_000)
    store.seed_hourly(T0, jobs=10, busy_ms=9000, access_us=460_000)
    store.seed_hourly(T0, jobs=10, busy_ms=9000, access_us=460_000)
    (row,) = store.hourly_rows(T0)
    assert row["source"] == "attempts"
    assert (row["jobs"], row["busy_ms"], row["access_us_sum"]) == (30, 27_000, 1_380_000)


def test_a_recorder_write_failure_is_logged_and_swallowed(caplog):
    class Broken(HistoryStore):
        def record_job(self, sample, round_start_ts):
            raise RuntimeError("disk full")

    rec = HistoryRecorder(Broken(":memory:"))
    with caplog.at_level(logging.WARNING):
        rec.record_job(_sample(5, T0))
        rec.record_job(_sample(5, T0 + 1))
        rec.flush()
    warnings = [r for r in caplog.records if "history" in r.getMessage()]
    assert len(warnings) == 1  # rate-limited: the second failure is silent


def test_close_is_idempotent_and_drains_queued_writes():
    rec = HistoryRecorder(HistoryStore(":memory:"))
    rec.round_boundary(1, T0, joined=True, reason="budget", expected_jobs=None)
    rec.close()
    rec.close()
    rec.record_wasted(T0)  # after close: dropped, never raises
