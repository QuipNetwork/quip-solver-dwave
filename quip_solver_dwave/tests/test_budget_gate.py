"""Tests for the qblock participation gate.

The gate is the rule that the QPU never half-joins a round: credits go out on a
qblock boundary or not at all, a joined round runs to its end, and the one
thing that parks credits mid-round is the period's allotment being spent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from quip_solver_dwave.budget import BudgetConfig, BudgetPacer
from quip_solver_dwave.session_loop import ParticipationGate
from quip_solver_dwave.usage import UsageLedger


def ts(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def gate_with(budget_s=3000.0, reset_day=9):
    led = UsageLedger(":memory:")
    pacer = BudgetPacer(
        BudgetConfig(budget_seconds=budget_s, reset_day=reset_day), led
    )
    return ParticipationGate(pacer), pacer


# Ten days into a 30-day period at 3000s/month: 1000s of allowance earned.
MID_PERIOD = ts(2026, 9, 19)


def test_a_fresh_gate_is_not_participating():
    # Nothing is granted until a qblock boundary says so, even with a full
    # allowance sitting there.
    gate, _ = gate_with()
    assert gate.participating is False


def test_a_job_arriving_before_any_boundary_is_refused():
    # The coordinator can only dispatch against credits we granted, but a stale
    # credit must not turn into a mid-round join.
    gate, _ = gate_with()
    result = gate.on_job(MID_PERIOD)
    assert result.allowed is False
    assert result.changed is False


def test_a_boundary_with_headroom_starts_participation():
    gate, _ = gate_with()
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert result.allowed is True
    assert result.changed is True  # caller grants credits on the change
    assert gate.participating is True


def test_a_boundary_without_headroom_sits_the_round_out():
    gate, pacer = gate_with()
    pacer.record_access_time(1500 * 1_000_000, MID_PERIOD)  # 500s past the line
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert result.allowed is False
    assert gate.participating is False


def test_the_same_generation_is_not_a_new_boundary():
    # Cancel can repeat a watermark; only an advance is a fresh qblock.
    gate, _ = gate_with()
    assert gate.on_qblock_boundary(100, MID_PERIOD) is not None
    assert gate.on_qblock_boundary(100, MID_PERIOD) is None
    assert gate.on_qblock_boundary(99, MID_PERIOD) is None
    assert gate.on_qblock_boundary(101, MID_PERIOD) is not None


def test_staying_in_across_a_boundary_does_not_re_grant():
    # Credits survive a reseed, so a second grant would over-fund the pipeline.
    gate, _ = gate_with()
    first = gate.on_qblock_boundary(100, MID_PERIOD)
    second = gate.on_qblock_boundary(101, MID_PERIOD)
    assert first is not None and first.changed is True
    assert second is not None and second.allowed is True
    assert second.changed is False  # no state flip -> caller grants nothing


def test_crossing_the_pacing_line_mid_qblock_keeps_mining():
    # The line is a target for the month, not a limit. A joined round runs
    # to its end however far past the line it goes; the next boundary is
    # where the overshoot is paid for.
    gate, pacer = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    pacer.record_access_time(1500 * 1_000_000, MID_PERIOD)  # 500s past the line
    result = gate.on_job(MID_PERIOD)
    assert result.allowed is True
    assert result.changed is False
    assert gate.participating is True
    # And that boundary sits the next round out.
    nxt = gate.on_qblock_boundary(101, MID_PERIOD)
    assert nxt is not None and nxt.allowed is False and nxt.changed is True


def test_spending_the_whole_allotment_mid_qblock_parks_credits():
    gate, pacer = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    pacer.record_access_time(3000 * 1_000_000, MID_PERIOD)  # the whole month, gone
    result = gate.on_job(MID_PERIOD)
    assert result.allowed is False
    assert result.changed is True  # caller logs the stop once
    assert result.decision.exhausted is True
    assert gate.participating is False


def test_the_stop_is_logged_once_not_per_rejected_job():
    gate, pacer = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    pacer.record_access_time(3000 * 1_000_000, MID_PERIOD)
    assert gate.on_job(MID_PERIOD).changed is True
    # Every later job in the same shut round is a silent refusal.
    assert gate.on_job(MID_PERIOD).changed is False
    assert gate.on_job(MID_PERIOD).changed is False


def test_recovery_waits_for_the_reset_and_a_boundary():
    gate, pacer = gate_with()
    gate.on_qblock_boundary(100, MID_PERIOD)
    pacer.record_access_time(3000 * 1_000_000, MID_PERIOD)
    gate.on_job(MID_PERIOD)
    assert gate.participating is False

    # Nothing in this period can restart it: the allotment is spent and the
    # line cannot climb past it.
    later = ts(2026, 9, 25)
    assert gate.on_job(later).allowed is False
    assert gate.on_qblock_boundary(101, later) is not None
    assert gate.participating is False

    # The first boundary after the reset does.
    result = gate.on_qblock_boundary(102, ts(2026, 10, 10))
    assert result is not None and result.allowed is True
    assert gate.participating is True


def test_a_sat_out_round_reports_the_wait():
    gate, pacer = gate_with()
    pacer.record_access_time(1500 * 1_000_000, MID_PERIOD)
    result = gate.on_qblock_boundary(100, MID_PERIOD)
    assert result is not None
    assert result.decision.seconds_until_headroom > 0
