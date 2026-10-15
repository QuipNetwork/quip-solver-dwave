"""The gate asks the strategy only after the budget says yes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List

from quip_solver_dwave.budget import BudgetConfig, BudgetPacer, ParticipationDecision
from quip_solver_dwave.history import HistoryStore
from quip_solver_dwave.profile import SnapshotRefresher
from quip_solver_dwave.session_loop import ParticipationGate
from quip_solver_dwave.strategy import (
    REASON_SLOW_SLOT,
    REASON_FAST_SLOT,
    REASON_NO_DATA,
    RoundDecision,
    RoundStrategy,
    StrategyConfig,
)
from quip_solver_dwave.tests.test_history_wiring import QuickSampler, _run
from quip_solver_dwave.usage import UsageLedger


def ts(y, mo, d, h=0):
    return datetime(y, mo, d, h, tzinfo=timezone.utc).timestamp()


MID_PERIOD = ts(2026, 9, 19)


def _verdict(join: bool, reason: str) -> RoundDecision:
    return RoundDecision(
        join=join, reason=reason, slot=13, expected_jobs=165.0, jobs_per_s=1.1,
        bar_jobs=300.0, rounds_funded=120.0, rounds_remaining=1300.0,
    )


class FakeStrategy:
    def __init__(self, verdicts: List[RoundDecision]):
        self.verdicts = list(verdicts)
        self.calls: List[ParticipationDecision] = []

    def decide(self, now: float, budget: ParticipationDecision) -> RoundDecision:
        self.calls.append(budget)
        return self.verdicts.pop(0)


def _gate(strategy=None, budget_s=3000.0):
    pacer = BudgetPacer(BudgetConfig(budget_seconds=budget_s, reset_day=9), UsageLedger(":memory:"))
    return ParticipationGate(pacer, strategy), pacer


def test_the_budget_decision_reports_its_accrual_rate():
    _, pacer = _gate()
    d = pacer.decide(MID_PERIOD)
    assert d.accrual_us_per_s == 3000.0 * 1_000_000 / (d.period_end - d.period_start)


def test_the_strategy_is_consulted_only_when_the_budget_allows():
    strategy = FakeStrategy([_verdict(True, REASON_FAST_SLOT)])
    gate, pacer = _gate(strategy)
    pacer.record_access_time(1500 * 1_000_000, MID_PERIOD)  # past the line
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None and not result.allowed and result.round is None
    assert strategy.calls == []

    gate2, _ = _gate(FakeStrategy([_verdict(True, REASON_FAST_SLOT)]))
    result = gate2.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None and result.allowed and result.changed
    assert result.round is not None and result.round.reason == REASON_FAST_SLOT


def test_a_skip_verdict_withholds_credits():
    strategy = FakeStrategy([_verdict(False, REASON_SLOW_SLOT)])
    gate, _ = _gate(strategy)
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert not result.allowed and not result.changed and not gate.participating
    assert result.round is not None and result.round.reason == REASON_SLOW_SLOT
    assert strategy.calls[0].headroom_us > 0


def test_a_skip_after_participating_parks_credits():
    gate, _ = _gate(FakeStrategy([_verdict(True, REASON_FAST_SLOT), _verdict(False, REASON_SLOW_SLOT)]))
    first = gate.on_qblock_boundary(100, MID_PERIOD)
    assert first is not None and first.allowed
    result = gate.on_qblock_boundary(101, MID_PERIOD)
    assert result is not None
    assert not result.allowed and result.changed and not gate.participating
    # With participation off, a job is refused until the next boundary.
    assert not gate.on_job(MID_PERIOD).allowed


def test_a_gate_without_a_strategy_behaves_as_before():
    gate, _ = _gate()
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None and result.allowed and result.round is None


def test_round_strategy_feeds_the_budget_numbers_into_the_decision():
    _, pacer = _gate()
    refresher = SnapshotRefresher(HistoryStore(":memory:"), interval_s=3600.0)
    strategy = RoundStrategy(StrategyConfig(), refresher)
    d = strategy.decide(MID_PERIOD, pacer.decide(MID_PERIOD))
    assert d.join and d.reason == REASON_NO_DATA  # nothing recorded yet
    refresher.stop()


def test_a_skipped_round_is_logged_and_recorded(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(
        RoundStrategy, "decide", lambda self, now, budget: _verdict(False, REASON_SLOW_SLOT)
    )
    usage_db = str(tmp_path / "usage.db")
    with caplog.at_level(logging.INFO):
        sent = _run(
            monkeypatch, QuickSampler(), usage_db, str(tmp_path / "attempts"), expect_job=False
        )
    # No credits were granted, so the job the script sent was refused.
    assert not any(m.WhichOneof("msg") == "result" for m in sent)
    assert any(m.WhichOneof("msg") == "reject" for m in sent)
    assert any("skip (slow-slot)" in r.getMessage() for r in caplog.records)
    rows = {r["generation"]: r for r in HistoryStore(usage_db).rounds(since_ts=0, limit=10)}
    assert rows[2]["joined"] == 0 and rows[2]["reason"] == REASON_SLOW_SLOT
    assert rows[2]["expected_jobs"] == 165.0
    assert "p_win" not in rows[2].keys()
