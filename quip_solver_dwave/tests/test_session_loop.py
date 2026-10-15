"""Tests for session-loop helpers — generation-based cancellation."""

from __future__ import annotations

import logging

from quip_solver_dwave.session_loop import (
    _is_abandoned,
    capabilities_message,
    log_attempt,
    log_progress,
)
from quip_solver_dwave import ALGORITHM, BACKEND, FEATURES, MAX_EDGES, MAX_NODES


def test_mempool_generation_is_never_abandoned():
    # Generation 0 = mempool job, no PoW cancellation scope. It must survive any
    # watermark, or a reseed would wrongly drop live mempool work.
    assert _is_abandoned(0, 0) is False
    assert _is_abandoned(0, 5) is False
    assert _is_abandoned(0, 10_000) is False


def test_pow_generation_at_or_below_watermark_is_abandoned():
    assert _is_abandoned(3, 5) is True  # below
    assert _is_abandoned(5, 5) is True  # at the watermark


def test_pow_generation_above_watermark_survives():
    assert _is_abandoned(6, 5) is False  # newer than the reseed
    assert _is_abandoned(1, 0) is False  # no cancel yet (watermark 0)


def test_capabilities_message_matches_advertised_caps():
    # Must answer a GetCapabilities without touching the device, so it is a
    # pure function of the same static numbers Hello and --capabilities use.
    caps = capabilities_message()
    assert caps.backend == BACKEND
    assert caps.algorithm == ALGORITHM
    assert list(caps.supported_kinds) == [1]  # ISING_SAMPLE
    assert caps.max_nodes == MAX_NODES
    assert caps.max_edges == MAX_EDGES
    assert list(caps.features) == list(FEATURES)
    assert caps.protocol_version == 1
    assert caps.stream_width == 1


# First 8 bytes of a longer job id, matching the observed CPU miner line.
_JOB_ID = bytes.fromhex("1858b8caf35ae0df") + b"\x00\x01"


def test_attempt_line_matches_the_shared_format(caplog):
    # A finished attempt is info. Energy is milli internally and whole units
    # on the line. Wall and device use the shared duration buckets.
    with caplog.at_level(logging.INFO):
        log_attempt(
            _JOB_ID,
            energy_milli=-14_245_000,
            valid=0,
            total=106,
            wall_ms=21_000,
            device_ms=21_000,
        )
    assert (
        "[quip-solver-dwave] attempt 1858b8caf35ae0df..: energy -14245, valid 0/106 | 21.0s wall, 21.0s device"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.INFO


def test_rejected_attempt_line_matches_the_shared_format(caplog):
    with caplog.at_level(logging.WARNING):
        log_attempt(_JOB_ID, rejected="MALFORMED", wall_ms=21_000)
    assert (
        "[quip-solver-dwave] attempt 1858b8caf35ae0df..: rejected MALFORMED | 21.0s wall"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.WARNING


def test_cancelled_attempt_line_is_debug(caplog):
    with caplog.at_level(logging.DEBUG):
        log_attempt(_JOB_ID, cancelled=True, wall_ms=21_000)
    assert (
        "[quip-solver-dwave] attempt 1858b8caf35ae0df..: cancelled after 21.0s"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.DEBUG


def test_progress_line_matches_the_shared_format(caplog):
    # 10 jobs in 20 seconds is 0.5 jobs/s. Best and the requirement use
    # whole energy units, not milli.
    with caplog.at_level(logging.INFO):
        log_progress(
            jobs_done=10,
            elapsed_s=20.0,
            reads=106,
            sweeps=0,
            best_energy_milli=-14_245_000,
            max_energy_milli=-14_000_000,
            min_solutions=1,
        )
    assert (
        "[quip-solver-dwave] progress: 10 jobs | 0.5 jobs/s | reads=106 sweeps=0 | best=-14245 | requires energy<=-14000, solutions>=1"
        in caplog.text
    )
    assert caplog.records[0].levelno == logging.INFO


def _decision(headroom_s, until_s=0.0, spent_s=0.0, allowance_s=0.0, budget_s=3000.0):
    from quip_solver_dwave.budget import ParticipationDecision

    return ParticipationDecision(
        participate=headroom_s > 0,
        headroom_us=headroom_s * 1_000_000,
        allowance_us=allowance_s * 1_000_000,
        spent_us=spent_s * 1_000_000,
        period_start=0.0,
        period_end=0.0,
        seconds_until_headroom=until_s,
        budget_us=budget_s * 1_000_000,
        exhausted=spent_s >= budget_s,
    )


def test_joining_a_qblock_names_the_headroom_and_the_spend(caplog):
    from quip_solver_dwave.session_loop import _log_qblock_joined

    caplog.set_level(logging.DEBUG)
    _log_qblock_joined(204, _decision(600.0, spent_s=400.0, allowance_s=1000.0))
    assert "joining qblock 204" in caplog.text
    assert "600s of headroom" in caplog.text
    assert "spent 400s of 1000s" in caplog.text
    assert [r.levelname for r in caplog.records] == ["INFO"]


def test_sitting_out_a_qblock_reports_the_next_window(caplog):
    from quip_solver_dwave.session_loop import _log_qblock_sat_out

    caplog.set_level(logging.DEBUG)
    _log_qblock_sat_out(205, _decision(-300.0, until_s=21_600.0))
    assert "sitting out qblock 205" in caplog.text
    assert "300s past the budget line" in caplog.text
    assert "next window in 6h 0m" in caplog.text
    assert [r.levelname for r in caplog.records] == ["INFO"]


def test_spending_the_allotment_mid_qblock_logs_the_spend_and_the_reset(caplog):
    from quip_solver_dwave.session_loop import _log_allotment_spent

    caplog.set_level(logging.DEBUG)
    _log_allotment_spent(_decision(-2000.0, until_s=300.0, spent_s=3000.0, budget_s=3000.0))
    assert "period allotment spent mid-qblock" in caplog.text
    assert "3000s of 3000s" in caplog.text
    assert "after the reset in 5m 0s" in caplog.text
    assert [r.levelname for r in caplog.records] == ["INFO"]


def _pacer_at(spent_s, allowance_s, budget_s=3000.0, reset_day=9):
    """A pacer whose ledger and clock put spend at a chosen point on the line."""
    from datetime import datetime, timezone

    from quip_solver_dwave.budget import BudgetConfig, BudgetPacer, period_bounds
    from quip_solver_dwave.usage import UsageLedger

    led = UsageLedger(":memory:")
    pacer = BudgetPacer(BudgetConfig(budget_s, reset_day), led)
    start, end = period_bounds(
        datetime(2026, 9, 19, tzinfo=timezone.utc).timestamp(), reset_day
    )
    now = start + (end - start) * (allowance_s / budget_s)
    if spent_s:
        led.record(spent_s * 1_000_000, now=now)
    return pacer, now


def test_configure_states_the_quota_the_period_and_the_spend(caplog):
    from quip_solver_dwave.session_loop import _log_budget_configured

    caplog.set_level(logging.DEBUG)
    pacer, now = _pacer_at(spent_s=400.0, allowance_s=1000.0)
    _log_budget_configured(pacer, now)
    assert "budget 3000s per period" in caplog.text
    assert "resets day 9" in caplog.text
    assert "period 2026-09-09 -> 2026-10-09" in caplog.text
    assert "spent 400s of 1000s earned so far; jobs this period: 1" in caplog.text


def test_configure_says_why_it_waits_when_headroom_is_available(caplog):
    # Idle with budget in hand is the confusing case: the line must name the
    # qblock boundary as the reason, or it reads as a hang.
    from quip_solver_dwave.session_loop import _log_budget_configured

    caplog.set_level(logging.DEBUG)
    pacer, now = _pacer_at(spent_s=400.0, allowance_s=1000.0)
    _log_budget_configured(pacer, now)
    assert "waiting: 600s of headroom is available" in caplog.text
    assert "held until the next qblock boundary" in caplog.text
    assert "whole round rather than part of one" in caplog.text


def test_configure_estimates_the_wait_when_spend_is_past_the_line(caplog):
    from quip_solver_dwave.session_loop import _log_budget_configured

    caplog.set_level(logging.DEBUG)
    # 1200s spent against 1000s earned: 200s over, earned back at 3000s/30d.
    pacer, now = _pacer_at(spent_s=1200.0, allowance_s=1000.0)
    _log_budget_configured(pacer, now)
    assert "waiting: 200s past the budget line" in caplog.text
    assert "no credits are granted" in caplog.text
    assert "Next window in 48h 0m" in caplog.text
    assert "following qblock boundary" in caplog.text
