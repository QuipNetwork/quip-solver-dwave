"""Seeding history from the coordinator's attempts files.

Fixture directories are real rounds cut from qpu-1 on 2026-09-11: 1335 (no
win) and 813 (one win). ``pending`` is the coordinator's name for attempts
made before the chain assigned a qblock id; it is not a round.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from quip_solver_dwave.attempts import (
    Attempt,
    default_attempts_dir,
    hourly_from_attempts,
    parse_attempt,
    pickup_outcomes,
    read_attempts,
    seed_from_attempts,
    summarise_rounds,
)
from quip_solver_dwave.history import HistoryStore, JobSample

FIXTURES = Path(__file__).parent / "fixtures" / "attempts"
NOW = 1789178400  # after the newest fixture round
DAY_813 = 1788825600
HOUR_813 = 1788890400


def _win_line() -> str:
    return (FIXTURES / "813" / "attempts.jsonl").read_text().splitlines()[3]


def _attempt(ts_s: float, generation: int = 1) -> Attempt:
    return Attempt(
        ts_ms=int(ts_s * 1000),
        generation=generation,
        miner_type="QPU-DWAVE",
        miner_id="qpu-0",
        raw_best_energy_milli=-14_400_000,
        threshold_milli=-14_500_000,
        accepted=False,
        won=False,
        device_access_time_us=46_000,
    )


def test_parse_attempt_reads_the_coordinator_fields():
    assert parse_attempt(_win_line()) == Attempt(
        ts_ms=1788893124504,
        generation=382,
        miner_type="QPU-DWAVE",
        miner_id="qpu-0",
        raw_best_energy_milli=-14_550_000,
        threshold_milli=-14_546_432,
        accepted=True,
        won=True,
        device_access_time_us=46_055,
    )


def test_malformed_and_incomplete_lines_are_skipped():
    assert parse_attempt("not json") is None
    assert parse_attempt("[]") is None
    assert parse_attempt('{"ts_ms": 1}') is None
    assert parse_attempt("") is None


def test_only_qpu_attempts_count():
    att = parse_attempt(_win_line())
    assert att is not None and att.is_qpu
    assert not dataclasses.replace(att, miner_type="CPU").is_qpu
    assert not dataclasses.replace(att, miner_type="").is_qpu


def test_attempts_group_into_rounds_by_generation():
    attempts, lines = read_attempts(FIXTURES / "813" / "attempts.jsonl")
    assert lines == 6
    (r,) = summarise_rounds(attempts)
    assert (r.generation, r.first_ts_s, r.last_ts_s) == (382, 1788893117, 1788893124)
    assert (r.jobs, r.hits_coord, r.won) == (6, 1, True)
    assert r.best_energy_milli == -14_550_000
    assert r.threshold_milli == -14_546_432
    assert r.access_us == 276_359
    assert r.margins == {
        (DAY_813, -3): 1,
        (DAY_813, 173): 1,
        (DAY_813, 181): 1,
        (DAY_813, 237): 1,
        (DAY_813, 243): 1,
        (DAY_813, 299): 1,
    }


def test_two_generations_in_one_file_are_two_rounds():
    attempts, _ = read_attempts(FIXTURES / "813" / "attempts.jsonl")
    later = [dataclasses.replace(a, generation=383, ts_ms=a.ts_ms + 600_000) for a in attempts]
    rounds = summarise_rounds(attempts + later)
    assert [r.generation for r in rounds] == [382, 383]
    assert rounds[1].first_ts_s == rounds[0].first_ts_s + 600


def test_busy_time_from_attempts_ends_a_segment_at_a_long_gap():
    base = HOUR_813
    rows = hourly_from_attempts(
        [_attempt(base + 1), _attempt(base + 3), _attempt(base + 100), _attempt(base + 3599), _attempt(base + 3601)],
        before_hour=base + 7200,
    )
    # 1->3 counts (2 s). 3->100 and 100->3599 are parks. 3599->3601 straddles
    # the hour: one second lands in each.
    assert rows[base] == (4, 3000, 4 * 46_000)
    assert rows[base + 3600] == (1, 1000, 46_000)


def test_hours_from_the_current_hour_on_are_left_to_the_live_recorder():
    base = HOUR_813
    assert hourly_from_attempts([_attempt(base + 1), _attempt(base + 2)], before_hour=base) == {}


def test_seed_inserts_only_complete_directories():
    store = HistoryStore(":memory:")
    report = seed_from_attempts(store, str(FIXTURES), now=NOW)
    assert (report.dirs_seeded, report.rounds_inserted, report.rounds_updated) == (1, 1, 0)
    # 1335 is the newest numeric directory: the live round, not history yet.
    # ``pending`` is not a round at all.
    rows = store.rounds(since_ts=0, limit=10)
    assert [r["generation"] for r in rows] == [382]
    row = rows[0]
    assert row["source"] == "attempts" and row["joined"] == 1
    # hits is left at its default: the attempts file has no per-read count.
    assert (row["jobs"], row["hits"], row["hits_coord"], row["won"]) == (6, 0, 1, 1)
    assert row["target_milli"] == -14_546_432 and row["end_ts_s"] == 1788893124
    assert store.margin_counts(0) == {-3: 1, 173: 1, 181: 1, 237: 1, 243: 1, 299: 1}
    (hour,) = store.hourly_rows(0)
    assert hour["hour_start_s"] == HOUR_813 and hour["source"] == "attempts"
    assert hour["jobs"] == 6 and hour["access_us_sum"] == 276_359
    assert 6000 <= hour["busy_ms"] <= 8000  # six completions over ~7 s
    assert store.is_seeded("813") and not store.is_seeded("1335")


def test_seed_is_idempotent():
    store = HistoryStore(":memory:")
    seed_from_attempts(store, str(FIXTURES), now=NOW)
    again = seed_from_attempts(store, str(FIXTURES), now=NOW)
    assert (again.dirs_seeded, again.rounds_inserted, again.rounds_updated) == (0, 0, 0)
    assert len(store.rounds(since_ts=0, limit=10)) == 1
    assert store.margin_counts(0)[-3] == 1
    assert store.hourly_rows(0)[0]["jobs"] == 6


def test_seed_skips_a_directory_the_live_recorder_covered():
    store = HistoryStore(":memory:")
    store.open_round(1788893110, 382, joined=True, reason="budget", expected_jobs=None)
    report = seed_from_attempts(store, str(FIXTURES), now=NOW)
    assert (report.rounds_inserted, report.rounds_updated) == (0, 1)
    (row,) = store.rounds(since_ts=0, limit=10)
    assert row["source"] == "live" and row["won"] == 1 and row["hits_coord"] == 1
    # The live recorder already wrote this round's margins and hours.
    assert store.margin_counts(0) == {}
    assert store.hourly_rows(0) == []


def test_pickup_outcomes_marks_a_live_round_won():
    store = HistoryStore(":memory:")
    store.open_round(1789178290, 58, joined=True, reason="budget", expected_jobs=None)
    store.open_round(1788893110, 382, joined=True, reason="budget", expected_jobs=None)
    assert pickup_outcomes(store, str(FIXTURES)) == 2
    rows = {r["generation"]: r for r in store.rounds(since_ts=0, limit=10)}
    assert rows[382]["won"] == 1 and rows[382]["hits_coord"] == 1
    assert rows[58]["won"] == 0 and rows[58]["hits_coord"] == 0
    # Outcomes only: nothing is inserted for a round the miner never saw.
    assert len(rows) == 2


def test_default_attempts_dir_sits_beside_the_usage_db():
    assert default_attempts_dir("/data/qpu-usage.db") == "/data/attempts"
    assert default_attempts_dir("/srv/node/data/qpu-usage.db") == "/srv/node/data/attempts"


def test_a_stopped_seed_leaves_the_unreached_directory_unmarked(tmp_path):
    # Seeding over a thousand attempt directories can take tens of seconds; a
    # shutdown mid-seed must not leave a directory with its margins written
    # but not marked seeded (they would double-count on the next start).
    root = tmp_path / "attempts"
    for name, generation in (("100", 10), ("200", 20), ("300", 30)):
        d = root / name
        d.mkdir(parents=True)
        line = json.dumps(
            {
                "ts_ms": 1_000_000 + generation * 1000,
                "generation": generation,
                "miner_type": "QPU-DWAVE",
                "raw_best_energy_milli": -100,
                "threshold_milli": -50,
                "accepted": True,
                "device_access_time_us": 1000,
            }
        )
        (d / "attempts.jsonl").write_text(line + "\n")

    store = HistoryStore(":memory:")
    calls = 0

    def stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1  # let the first directory (100) finish, stop before 200

    report = seed_from_attempts(store, str(root), now=2_000_000, stop=stop)

    assert report.dirs_seeded == 1
    assert store.is_seeded("100")
    assert not store.is_seeded("200")  # never reached: unmarked
    assert not store.is_seeded("300")  # the live round, excluded either way
    rows = store.rounds(since_ts=0, limit=10)
    assert [r["generation"] for r in rows] == [10]
    # 200's margins and hours were never written: a jobs count of 2 here
    # would mean 200 was seeded despite the stop, and would double on retry.
    assert sum(store.margin_counts(0).values()) == 1
    assert sum(r["jobs"] for r in store.hourly_rows(0)) == 1


def _attempt_line(miner_id: str, generation: int = 10, ts_ms: int = 1_000_000) -> str:
    return json.dumps(
        {
            "ts_ms": ts_ms,
            "generation": generation,
            "miner_id": miner_id,
            "miner_type": "QPU-DWAVE",
            "raw_best_energy_milli": -100,
            "threshold_milli": -50,
            "accepted": True,
            "device_access_time_us": 1000,
        }
    )


def test_read_attempts_filters_lines_to_the_requested_miner_id(tmp_path):
    path = tmp_path / "attempts.jsonl"
    path.write_text(_attempt_line("qpu-0") + "\n" + _attempt_line("qpu-1") + "\n")

    attempts, lines = read_attempts(path, miner_id="qpu-1")
    assert lines == 2  # every line still counts toward the file's line total
    assert [a.miner_id for a in attempts] == ["qpu-1"]

    attempts_all, _ = read_attempts(path)
    assert [a.miner_id for a in attempts_all] == ["qpu-0", "qpu-1"]  # None: today's behaviour


def test_seeding_a_directory_shared_by_two_miners_counts_only_this_one(tmp_path):
    # The coordinator writes one attempts file per qblock for every miner it
    # sees, so a node running a second QPU miner would otherwise absorb the
    # other miner's jobs, access time and margins into this one's history.
    attempts_dir = tmp_path / "attempts"
    d = attempts_dir / "100"
    d.mkdir(parents=True)
    (d / "attempts.jsonl").write_text(_attempt_line("qpu-0") + "\n" + _attempt_line("qpu-1") + "\n")
    (attempts_dir / "200").mkdir(parents=True)  # the round in progress

    filtered = HistoryStore(":memory:")
    report = seed_from_attempts(filtered, str(attempts_dir), now=2_000_000, miner_id="qpu-0")
    assert report.dirs_seeded == 1
    (row,) = [r for r in filtered.rounds(since_ts=0, limit=10) if r["generation"] == 10]
    assert row["jobs"] == 1

    unfiltered = HistoryStore(":memory:")
    seed_from_attempts(unfiltered, str(attempts_dir), now=2_000_000)  # None: today's behaviour
    (row_both,) = [r for r in unfiltered.rounds(since_ts=0, limit=10) if r["generation"] == 10]
    assert row_both["jobs"] == 2


def test_a_missing_directory_seeds_nothing(tmp_path):
    report = seed_from_attempts(HistoryStore(":memory:"), str(tmp_path / "nope"), now=NOW)
    assert (report.dirs_seeded, report.rounds_inserted) == (0, 0)
    assert pickup_outcomes(HistoryStore(":memory:"), str(tmp_path / "nope")) == 0


def test_a_seed_that_fails_partway_through_a_directory_leaves_nothing_committed(monkeypatch):
    # mark_seeded is the last write of a directory's seed. If it raises, the
    # margins and hourly rows written just before it must not survive either,
    # or the directory would double-count its margins on the next start.
    store = HistoryStore(":memory:")

    def boom(self, dir_name):
        raise RuntimeError("disk full")

    monkeypatch.setattr(HistoryStore, "mark_seeded", boom)
    with pytest.raises(RuntimeError):
        seed_from_attempts(store, str(FIXTURES), now=NOW)
    assert not store.is_seeded("813")
    assert store.margin_counts(0) == {}
    assert store.hourly_rows(0) == []


def test_seeding_a_directory_the_miner_was_live_for_skips_only_margins_and_hours(tmp_path):
    # A miner that starts mid-qblock records live jobs for the generation in
    # progress but has no live round row for it (the Cancel that would have
    # opened one arrived before the recorder existed). Seeding that
    # directory later must not add its margins a second time, but a round
    # row for a generation the live recorder never opened is not double
    # counting.
    store = HistoryStore(":memory:")
    generation = 50
    base = HOUR_813
    for i in range(5):
        store.record_job(
            JobSample(
                completed_at=base + i,
                generation=generation,
                rtt_ms=3000,
                access_us=46_000,
                inflight_at_submit=1,
                reads=1,
                best_energy_milli=-100,
                target_milli=-50,
                hits=0,
            ),
            round_start_ts=None,
        )
    assert sum(store.margin_counts(0).values()) == 5

    attempts_dir = tmp_path / "attempts"
    d = attempts_dir / "500"
    d.mkdir(parents=True)
    lines = [
        json.dumps(
            {
                "ts_ms": int((base + i) * 1000),
                "generation": generation,
                "miner_type": "QPU-DWAVE",
                "raw_best_energy_milli": -100,
                "threshold_milli": -50,
                "accepted": True,
                "device_access_time_us": 1000,
            }
        )
        for i in range(5)
    ]
    (d / "attempts.jsonl").write_text("\n".join(lines) + "\n")
    (attempts_dir / "600").mkdir(parents=True)  # the round in progress

    report = seed_from_attempts(store, str(attempts_dir), now=base + 10_000)

    assert report.dirs_seeded == 1
    assert sum(store.margin_counts(0).values()) == 5  # not doubled
    rows = [r for r in store.rounds(since_ts=0, limit=10) if r["generation"] == generation]
    assert len(rows) == 1 and rows[0]["source"] == "attempts"
