"""Easy rounds are joined at once: mid-round, after a restart, in any slot."""

from __future__ import annotations

import logging
import threading
import time
from typing import List, cast

from quip_solver_core import miner_pb2

from quip_solver_dwave import session_loop
from quip_solver_dwave.budget import BudgetConfig, BudgetPacer
from quip_solver_dwave.history import HistoryStore, JobSample
from quip_solver_dwave.ocean import OceanSampler
from quip_solver_dwave.profile import SnapshotRefresher
from quip_solver_dwave.session_loop import ParticipationGate, _seed_reach
from quip_solver_dwave.strategy import (
    REACH_MIN_JOBS,
    REASON_EASY_TARGET,
    RoundStrategy,
    StrategyConfig,
    TargetReach,
    TargetReachTracker,
    decide_round,
)
from quip_solver_dwave.tests.test_history_wiring import QuickSampler, _configure, _job, _wait_until
from quip_solver_dwave.usage import UsageLedger

TARGET_MILLI = -4_900_000
EASY_BEST_MILLI = -5_000_000


def _easy_tracker(now: float, jobs: int = REACH_MIN_JOBS) -> TargetReachTracker:
    tracker = TargetReachTracker()
    tracker.set_target(TARGET_MILLI)
    for _ in range(jobs):
        tracker.observe(now, EASY_BEST_MILLI)
    return tracker


def _gate(tracker: TargetReachTracker, budget_s: float = 3000.0):
    pacer = BudgetPacer(BudgetConfig(budget_seconds=budget_s, reset_day=9), UsageLedger(":memory:"))
    refresher = SnapshotRefresher(HistoryStore(":memory:"), interval_s=3600.0)
    gate = ParticipationGate(pacer, RoundStrategy(StrategyConfig(), refresher, reach=tracker))
    return gate, pacer, refresher


def test_the_easy_check_needs_no_throughput_history():
    reach = TargetReach(jobs=REACH_MIN_JOBS, reached=REACH_MIN_JOBS, target_milli=TARGET_MILLI)
    d = decide_round(
        now=time.time(),
        headroom_us=1e9,
        accrual_us_per_s=0.0,
        period_end=time.time() + 86_400,
        snapshot=None,
        config=StrategyConfig(),
        explore_draw=0.5,
        reach=reach,
    )
    assert d.join and d.reason == REASON_EASY_TARGET


def test_an_easy_round_is_joined_mid_round_once():
    now = time.time()
    gate, _, refresher = _gate(_easy_tracker(now))
    joined = gate.on_easy_round(now)
    assert joined is not None and joined.allowed and joined.changed
    assert joined.round is not None and joined.round.reason == REASON_EASY_TARGET
    assert gate.participating
    # Already mining: a later target changes nothing.
    assert gate.on_easy_round(now) is None
    # The boundary that follows keeps mining without a second grant.
    boundary = gate.on_qblock_boundary(5, now)
    assert boundary is not None and boundary.allowed and not boundary.changed
    refresher.stop()


def test_a_hard_round_waits_for_the_boundary():
    now = time.time()
    tracker = _easy_tracker(now, jobs=REACH_MIN_JOBS - 1)
    gate, _, refresher = _gate(tracker)
    assert gate.on_easy_round(now) is None
    assert not gate.participating
    refresher.stop()


def test_the_budget_still_decides_first():
    now = time.time()
    gate, pacer, refresher = _gate(_easy_tracker(now))
    pacer.record_access_time(10_000 * 1_000_000, now)  # far past the line
    assert gate.on_easy_round(now) is None
    refresher.stop()


def test_a_gate_without_a_strategy_never_joins_mid_round():
    pacer = BudgetPacer(BudgetConfig(budget_seconds=3000.0, reset_day=9), UsageLedger(":memory:"))
    assert ParticipationGate(pacer).on_easy_round(time.time()) is None


def _record_round(store: HistoryStore, start: float, generation: int, best: int, target: int) -> None:
    store.open_round(start, generation, joined=True, reason="explore", expected_jobs=1.0)
    store.record_job(
        JobSample(
            completed_at=start + 5,
            generation=generation,
            rtt_ms=1000,
            access_us=46_000,
            inflight_at_submit=1,
            reads=48,
            best_energy_milli=best,
            target_milli=target,
            hits=1,
        ),
        int(start),
    )
    store.set_round_target(start, target)
    store.close_round(start, start + 8)


def test_a_restart_seeds_the_evidence_from_recent_rounds():
    now = time.time()
    store = HistoryStore(":memory:")
    _record_round(store, now - 7_200, 1, -4_000_000, -4_000_000)  # older than the window
    _record_round(store, now - 600, 2, EASY_BEST_MILLI, -4_950_000)
    _record_round(store, now - 300, 3, EASY_BEST_MILLI, TARGET_MILLI)
    store.open_round(now - 100, 4, joined=False, reason="slow-slot", expected_jobs=1.0)  # no jobs
    tracker = TargetReachTracker()
    _seed_reach(tracker, store, now)
    # Two mined rounds in the window; the newest round's target stands in.
    assert tracker.summary(now) == TargetReach(jobs=2, reached=2, target_milli=TARGET_MILLI)


def test_the_live_target_wins_over_the_seeded_one():
    now = time.time()
    tracker = TargetReachTracker()
    tracker.set_target(-5_100_000)
    tracker.seed([(now - 10, EASY_BEST_MILLI, TARGET_MILLI)])
    assert tracker.summary(now) == TargetReach(jobs=1, reached=0, target_milli=-5_100_000)


def _run_without_boundary(monkeypatch, sampler: QuickSampler, usage_db: str, attempts_dir: str) -> List:
    """A session that starts mid-round: no Cancel before the first job."""
    sent: List = []

    def responses():
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=2))
        yield _configure(usage_db)
        yield miner_pb2.CoordMsg(
            topology=miner_pb2.Topology(
                nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
            )
        )
        yield miner_pb2.CoordMsg(
            set_target=miner_pb2.SetTarget(max_energy_milli=0, min_solutions=1)
        )
        yield _job(generation=1)
        _wait_until(lambda: any(m.WhichOneof("msg") == "result" for m in sent))
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            def drain():
                for msg in request_iter:
                    sent.append(msg)

            threading.Thread(target=drain, daemon=True).start()
            return responses()

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel())
    monkeypatch.setattr(session_loop.grpc, "channel_ready_future", lambda ch: _Ready())
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")
    session_loop.run_session(
        "unix:///tmp/nope.sock",
        "qpu-test",
        cast(OceanSampler, sampler),
        attempts_dir=attempts_dir,
    )
    return sent


def test_an_easy_restart_mines_the_round_in_progress(monkeypatch, tmp_path, caplog):
    usage_db = str(tmp_path / "usage.db")
    store = HistoryStore(usage_db)
    now = time.time()
    for i in range(REACH_MIN_JOBS):
        _record_round(store, now - 600 + 60 * i, i + 1, EASY_BEST_MILLI, TARGET_MILLI)
    store.close()
    with caplog.at_level(logging.INFO):
        sent = _run_without_boundary(monkeypatch, QuickSampler(), usage_db, str(tmp_path / "attempts"))
    messages = [r.getMessage() for r in caplog.records]
    assert not any("[QPU] waiting:" in m for m in messages)
    assert any("[QPU] easy rounds: 4 of 4 recent jobs reached the target -4900" in m for m in messages)
    assert any("join (easy-target)" in m and m.endswith("(mid-round)") for m in messages)
    assert any(m.WhichOneof("msg") == "job_request" for m in sent)
    assert any(m.WhichOneof("msg") == "result" for m in sent)
