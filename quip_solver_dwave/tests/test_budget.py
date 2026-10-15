"""Tests for the QPU budget pacer and its quota-period arithmetic."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quip_solver_dwave.budget import (
    BudgetConfig,
    BudgetUnavailable,
    BudgetPacer,
    budget_from_backend_toml,
    parse_duration,
    period_bounds,
)
from quip_solver_dwave.usage import UsageLedger

DAY = 86400.0


def ts(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp()


def pacer(budget_s, reset_day=9, ledger=None):
    return BudgetPacer(
        BudgetConfig(budget_seconds=budget_s, reset_day=reset_day),
        ledger or UsageLedger(":memory:"),
    )


def test_parse_duration():
    assert parse_duration("30s") == 30.0
    assert parse_duration("5m") == 300.0
    assert parse_duration("2h") == 7200.0
    assert parse_duration("1d") == 86400.0
    assert parse_duration("1w") == 604800.0
    assert parse_duration("42") == 42.0


# ---------------------------------------------------------------- period math


def test_period_runs_from_reset_day_to_reset_day():
    start, end = period_bounds(ts(2026, 9, 20), reset_day=9)
    assert start == ts(2026, 9, 9)
    assert end == ts(2026, 10, 9)


def test_before_this_months_reset_the_period_opened_last_month():
    start, end = period_bounds(ts(2026, 9, 4, 16, 46), reset_day=9)
    assert start == ts(2026, 8, 9)
    assert end == ts(2026, 9, 9)


def test_the_reset_instant_itself_starts_the_new_period():
    start, _ = period_bounds(ts(2026, 9, 9), reset_day=9)
    assert start == ts(2026, 9, 9)


def test_period_wraps_across_the_year_boundary():
    start, end = period_bounds(ts(2026, 12, 20), reset_day=9)
    assert start == ts(2026, 12, 9)
    assert end == ts(2027, 1, 9)

    start, end = period_bounds(ts(2027, 1, 3), reset_day=9)
    assert start == ts(2026, 12, 9)
    assert end == ts(2027, 1, 9)


def test_reset_day_31_clamps_to_a_short_month():
    # February has no 31st: the reset lands on the 28th and the period still
    # runs edge to edge with no gap.
    start, end = period_bounds(ts(2026, 2, 10), reset_day=31)
    assert start == ts(2026, 1, 31)
    assert end == ts(2026, 2, 28)

    start, end = period_bounds(ts(2026, 3, 10), reset_day=31)
    assert start == ts(2026, 2, 28)
    assert end == ts(2026, 3, 31)


def test_reset_day_31_clamps_to_a_leap_february():
    start, end = period_bounds(ts(2028, 2, 10), reset_day=31)
    assert start == ts(2028, 1, 31)
    assert end == ts(2028, 2, 29)


def test_consecutive_periods_tile_without_gaps():
    # Walk a year at reset_day 31 (the worst case for clamping) and check that
    # each period's end is the next one's start.
    now = ts(2026, 1, 15)
    _, end = period_bounds(now, reset_day=31)
    for _ in range(14):
        start, next_end = period_bounds(end, reset_day=31)
        assert start == end, "period boundaries must tile with no gap"
        end = next_end


# ------------------------------------------------------------- even pacing


def test_allowance_is_the_elapsed_share_of_the_month():
    # 30-day period (9 Sep -> 9 Oct), 3000s of budget. A third of the way in,
    # a third of the budget has been earned.
    p = pacer(3000.0)
    start, end = period_bounds(ts(2026, 9, 20), reset_day=9)
    third = start + (end - start) / 3
    decision = p.decide(third)
    assert decision.allowance_us == pytest.approx(1000 * 1_000_000, rel=1e-9)


def test_fresh_period_has_no_allowance_yet():
    p = pacer(3000.0)
    decision = p.decide(ts(2026, 9, 9))
    assert decision.allowance_us == 0.0
    assert decision.participate is False


def test_spending_under_the_line_keeps_mining():
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    now = ts(2026, 9, 19)  # 10 of 30 days -> 1000s earned
    led.record(500 * 1_000_000, now=now)
    decision = p.decide(now)
    assert decision.participate is True
    assert decision.headroom_us == pytest.approx(500 * 1_000_000, rel=1e-6)
    assert decision.seconds_until_headroom == 0.0


def test_spending_past_the_line_shuts_the_gate():
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    now = ts(2026, 9, 19)  # 1000s earned
    led.record(1200 * 1_000_000, now=now)
    decision = p.decide(now)
    assert decision.participate is False
    assert decision.headroom_us == pytest.approx(-200 * 1_000_000, rel=1e-6)


def test_the_wait_is_how_long_the_line_takes_to_catch_up():
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    start, end = period_bounds(ts(2026, 9, 19), reset_day=9)
    rate_s_per_s = 3000.0 / (end - start)  # budget seconds earned per wall second
    now = ts(2026, 9, 19)
    led.record(1200 * 1_000_000, now=now)
    decision = p.decide(now)
    # 200s over the line, earned back at rate_s_per_s.
    assert decision.seconds_until_headroom == pytest.approx(200 / rate_s_per_s, rel=1e-6)


def test_a_blown_budget_waits_for_the_reset_not_forever():
    # Spend the whole month in a day: the line can never catch up inside this
    # period, so the honest ETA is the reset, not a number past the period end.
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    now = ts(2026, 9, 10)
    led.record(3000 * 1_000_000, now=now)
    decision = p.decide(now)
    assert decision.participate is False
    assert decision.seconds_until_headroom == pytest.approx(
        decision.period_end - now, rel=1e-9
    )


def test_the_reset_clears_last_periods_spend():
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    led.record(3000 * 1_000_000, now=ts(2026, 9, 10))
    assert p.decide(ts(2026, 10, 8)).participate is False
    # One day into the new period the old spend is out of scope and the fresh
    # allowance has started accruing.
    after = p.decide(ts(2026, 10, 10))
    assert after.spent_us == 0.0
    assert after.participate is True


def test_spend_recorded_before_this_period_does_not_count():
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    led.record(2000 * 1_000_000, now=ts(2026, 8, 20))  # previous period
    decision = p.decide(ts(2026, 9, 19))
    assert decision.spent_us == 0.0
    assert decision.participate is True


def test_even_distribution_holds_over_a_simulated_month():
    # Drive the gate at a burn rate well above the budget and confirm the month
    # ends within one qblock's spend of the budget, with the idle time spread
    # out rather than banked into one blackout.
    led = UsageLedger(":memory:")
    budget = 144_000.0  # ~80 min/day
    p = pacer(budget, ledger=led)
    start, end = period_bounds(ts(2026, 9, 20), reset_day=9)

    qblock_s = 300.0
    spend_per_qblock_us = 21.0 * 1_000_000  # 300s at the observed 0.07 s/s burn
    now = start
    participated = 0
    skipped = 0
    longest_gap = 0.0
    gap = 0.0
    while now < end:
        if p.decide(now).participate:
            led.record(spend_per_qblock_us, now=now)
            participated += 1
            gap = 0.0
        else:
            skipped += 1
            gap += qblock_s
            longest_gap = max(longest_gap, gap)
        now += qblock_s

    spent_s = led.spent_us_since(start) / 1_000_000
    # Never overshoot the month by more than the one qblock in flight.
    assert spent_s <= budget + 21.0
    # And actually use the allotment rather than under-spending it.
    assert spent_s >= budget - 21.0
    assert participated > skipped
    # The whole point: idle time arrives in short gaps, not a 6-hour blackout.
    assert longest_gap <= 2 * qblock_s


# ------------------------------------------------------------------- config


def test_budget_from_toml_reads_the_quota_shape(tmp_path):
    db = tmp_path / "usage.db"
    p = budget_from_backend_toml(
        f'budget = "40h"\nbudget_reset_day = 9\nusage_db = "{db}"\n'
    )
    assert p is not None
    assert p.config.budget_seconds == 144_000.0
    assert p.config.reset_day == 9
    assert p.config.usage_db == str(db)


def test_budget_from_toml_accepts_plain_seconds(tmp_path):
    p = budget_from_backend_toml(
        f'budget_seconds = 144000\nusage_db = "{tmp_path / "u.db"}"\n'
    )
    assert p is not None
    assert p.config.budget_seconds == 144_000.0
    assert p.config.reset_day == 1  # default


def test_no_budget_keys_means_no_pacer():
    assert budget_from_backend_toml("") is None
    assert budget_from_backend_toml("num_reads = 1\n") is None


def test_an_out_of_range_reset_day_refuses_to_start():
    with pytest.raises(BudgetUnavailable, match="budget_reset_day"):
        budget_from_backend_toml('budget = "40h"\nbudget_reset_day = 32\n')


def test_an_unopenable_ledger_refuses_to_start(tmp_path):
    # A path under a regular file cannot be a directory: the ledger cannot open,
    # and an unmetered miner must not mine.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    with pytest.raises(BudgetUnavailable, match="usage ledger"):
        budget_from_backend_toml(
            f'budget = "40h"\nusage_db = "{blocker / "sub" / "u.db"}"\n'
        )


def test_stats_report_the_period_and_the_headroom():
    led = UsageLedger(":memory:")
    p = pacer(3000.0, ledger=led)
    now = ts(2026, 9, 19)
    led.record(400 * 1_000_000, now=now)
    stats = p.stats(now)
    assert stats["budget_seconds"] == 3000.0
    assert stats["reset_day"] == 9
    assert stats["spent_seconds"] == pytest.approx(400.0)
    assert stats["allowance_seconds"] == pytest.approx(1000.0, rel=1e-6)
    assert stats["headroom_seconds"] == pytest.approx(600.0, rel=1e-6)
    assert stats["jobs_this_period"] == 1


def test_unquoted_duration_raises_rather_than_mining_unmetered():
    """`budget = 250m` is a TOML error, not a missing budget.

    Returning None here would mine unmetered on a typo — the exact failure the
    budget exists to prevent — so the parse error must be fatal and must name
    the fix.
    """
    with pytest.raises(BudgetUnavailable) as exc:
        budget_from_backend_toml('budget = 250m\nbudget_reset_day = 9\n')
    assert 'budget = "250m"' in str(exc.value)


def test_a_section_header_hides_the_budget_and_is_reported():
    """backend_toml is the flattened body of the miner entry, never a table.

    A pasted `[dwave]` header nests every key one level down, so the top-level
    lookups find nothing. That parses cleanly, so it cannot raise here; this
    pins the behaviour so the surprise is documented rather than discovered in
    production.
    """
    assert budget_from_backend_toml('[dwave]\nbudget = "250m"\n') is None


def test_malformed_toml_is_fatal_even_with_no_budget_key():
    with pytest.raises(BudgetUnavailable):
        budget_from_backend_toml("anneal_time_us = 80\nnum_reads = 32m\n")
