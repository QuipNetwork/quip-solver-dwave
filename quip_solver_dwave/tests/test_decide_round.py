"""The single round decision: join now, or hold the funds for faster hours."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import pytest

from quip_solver_dwave.profile import SLOTS, Profile, SlotStats, Snapshot, slot_of
from quip_solver_dwave.strategy import (
    REASON_EXPLORE,
    REASON_FAST_SLOT,
    REASON_NO_DATA,
    REASON_SATURATED,
    REASON_SLOW_SLOT,
    StrategyConfig,
    decide_round,
    describe_round_decision,
    rounds_by_slot,
)

# 2026-09-07 13:00 UTC: September has 30 days, so day 7 is bin 5 and the
# slot is d06 13h. The period resets on the 9th.
NOW = datetime(2026, 9, 7, 13, tzinfo=timezone.utc).timestamp()
PERIOD_END = datetime(2026, 10, 9, tzinfo=timezone.utc).timestamp()
SLOT_NOW = slot_of(NOW)
NEXT_HOUR = SLOT_NOW + 1
ROUND_S = 600.0
ACCESS_S = 0.046
# 32400 s a month, spread flat: 0.0125 s of access per wall second, 7.5 s a
# round. A round at 1.1 jobs/s costs 30 s, so the funds cover about a
# quarter of the rounds left.
ACCRUAL_US_PER_S = 32_400e6 / (30 * 86_400)
HEADROOM_US = 7.6e6


def _snapshot(
    rates: Optional[Dict[int, float]] = None,
    *,
    default_rate: Optional[float] = 1.1,
    rtt_s: float = 3.0,
) -> Snapshot:
    rates = rates or {}
    slots = [
        SlotStats(jobs_per_s=rates.get(s, default_rate), rtt_s=rtt_s, queue_s=None, evidence=1000.0)
        for s in range(SLOTS)
    ]
    return Snapshot(
        built_at=NOW,
        profile=Profile(slots=slots, hour_factors=[1.0] * 24, day_factors=[1.0] * 28),
        round_length_s=ROUND_S,
        access_s_per_job=ACCESS_S,
    )


def _decide(snapshot, *, headroom_us=HEADROOM_US, config=None, explore_draw=0.5, period_end=PERIOD_END,
            accrual=ACCRUAL_US_PER_S):
    return decide_round(
        now=NOW,
        headroom_us=headroom_us,
        accrual_us_per_s=accrual,
        period_end=period_end,
        snapshot=snapshot,
        config=config or StrategyConfig(),
        explore_draw=explore_draw,
    )


def _jobs(rate: float) -> float:
    return rate * (ROUND_S - 3.0)


def test_rounds_by_slot_counts_the_rest_of_the_period_hour_by_hour():
    counts = rounds_by_slot(NOW, NOW + 2.5 * 3600, ROUND_S)
    assert counts == {SLOT_NOW: 6.0, NEXT_HOUR: 6.0, SLOT_NOW + 2: 3.0}
    # Starting mid-hour counts the partial hour, and the total is the
    # period's length in rounds.
    counts = rounds_by_slot(NOW + 1800, NOW + 2 * 3600, ROUND_S)
    assert counts[SLOT_NOW] == 3.0 and sum(counts.values()) == pytest.approx(9.0)


def test_no_snapshot_or_no_evidence_joins_with_no_data():
    for snap in (None, _snapshot(default_rate=None)):
        d = _decide(snap)
        assert d.join and d.reason == REASON_NO_DATA
        assert d.slot == SLOT_NOW
        # No history means no prediction was made, not a prediction of zero.
        assert d.expected_jobs is None and d.jobs_per_s is None


def test_expected_jobs_is_a_whole_round_at_this_slot_s_rate():
    # A joined round runs to its end, so the headroom does not cap it.
    d = _decide(_snapshot(), headroom_us=1e5)
    assert d.expected_jobs == pytest.approx(_jobs(1.1))  # 656.7


def test_uniform_rates_join_now():
    # Every round delivers the same, so this one is as good as any the funds
    # would otherwise buy later.
    d = _decide(_snapshot())
    assert d.join and d.reason == REASON_FAST_SLOT
    assert d.bar_jobs == pytest.approx(_jobs(1.1))
    assert 0 < d.rounds_funded < d.rounds_remaining


def test_a_slow_slot_is_skipped_for_the_funds_to_go_further():
    d = _decide(_snapshot({SLOT_NOW: 0.1}))
    assert not d.join and d.reason == REASON_SLOW_SLOT
    assert d.expected_jobs == pytest.approx(_jobs(0.1))
    assert d.bar_jobs == pytest.approx(_jobs(1.1))


def test_a_fast_slot_is_joined():
    d = _decide(_snapshot({SLOT_NOW: 3.0}))
    assert d.join and d.reason == REASON_FAST_SLOT
    assert d.bar_jobs == pytest.approx(_jobs(1.1))


def test_the_bar_weighs_slots_by_how_many_rounds_they_hold():
    # One hour a month at 5 jobs/s is six rounds: far fewer than the funds
    # cover, so the bar is set by the ordinary slots below it, not by the
    # rare fast one.
    d = _decide(_snapshot({NEXT_HOUR: 5.0, SLOT_NOW: 0.5}))
    assert not d.join and d.reason == REASON_SLOW_SLOT
    assert d.bar_jobs == pytest.approx(_jobs(1.1))


def test_the_bar_moves_up_when_the_funds_cover_fewer_rounds():
    # A tenth of the accrual: the funds cover only the fastest few slots, so
    # an ordinary slot no longer clears the bar.
    fast = {SLOT_NOW + k: 3.0 for k in range(1, 100)}
    ordinary = _decide(_snapshot(fast), accrual=ACCRUAL_US_PER_S / 10)
    assert not ordinary.join and ordinary.reason == REASON_SLOW_SLOT
    assert ordinary.bar_jobs == pytest.approx(_jobs(3.0))
    assert ordinary.rounds_funded < 100 * 6


def test_funds_that_cover_every_round_join_whatever_the_slot():
    d = _decide(_snapshot({SLOT_NOW: 0.1}), headroom_us=1e12)
    assert d.join and d.reason == REASON_SATURATED
    assert d.bar_jobs == 0.0 and d.rounds_funded == d.rounds_remaining


def test_the_last_round_of_the_period_is_joined():
    # One round left and funds for less than one round of it: the bar is
    # this round's own jobs, so it is joined rather than the money lost.
    d = _decide(_snapshot({SLOT_NOW: 0.1}), period_end=NOW + 500, headroom_us=1e5, accrual=0.0)
    assert d.join and d.reason == REASON_FAST_SLOT
    assert d.bar_jobs == pytest.approx(_jobs(0.1))
    assert d.rounds_remaining == pytest.approx(500 / ROUND_S)


def test_the_minimum_throughput_advantage_sets_the_bar_s_tolerance():
    # 0.9 jobs/s against a bar at 1.1: 22% under. The default 25% tolerance
    # joins; a 10% tolerance skips.
    snap = _snapshot({SLOT_NOW: 0.9})
    assert _decide(snap).join
    d = _decide(snap, config=StrategyConfig(min_throughput_advantage=0.1))
    assert not d.join and d.reason == REASON_SLOW_SLOT


def test_the_participation_chance_joins_regardless():
    snap = _snapshot({SLOT_NOW: 0.1})
    d = _decide(snap, config=StrategyConfig(participation_chance=0.5), explore_draw=0.4)
    assert d.join and d.reason == REASON_EXPLORE
    assert not _decide(snap, config=StrategyConfig(participation_chance=0.0), explore_draw=0.0).join


def test_verdict_order_is_explore_saturated_bar():
    snap = _snapshot({SLOT_NOW: 0.1})
    # explore beats saturated
    assert _decide(snap, headroom_us=1e12, explore_draw=0.0).reason == REASON_EXPLORE
    # saturated beats the bar
    assert _decide(snap, headroom_us=1e12).reason == REASON_SATURATED
    # the bar decides when neither applies
    assert _decide(snap).reason == REASON_SLOW_SLOT


def test_the_log_line_names_the_numbers():
    join = _decide(_snapshot({SLOT_NOW: 3.0}))
    line = describe_round_decision(340, join, HEADROOM_US)
    assert line.startswith("[QPU] qblock 340: join (fast-slot)")
    assert "1791 jobs at 3.00 jobs/s in d06 13h" in line and "bar 657" in line
    assert "funds cover" in line and "rounds left" in line and "headroom 8s" in line
    skip = _decide(_snapshot({SLOT_NOW: 0.1}))
    line = describe_round_decision(341, skip, HEADROOM_US)
    assert line.startswith("[QPU] qblock 341: skip (slow-slot)")
    assert "60 jobs at 0.10 jobs/s in d06 13h is under the bar of 657" in line
    covered = _decide(_snapshot(), headroom_us=1e12)
    line = describe_round_decision(342, covered, 1e12)
    assert "join (saturated)" in line and "every round is covered" in line
    assert "P(win)" not in line
