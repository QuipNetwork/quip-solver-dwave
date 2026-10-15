"""Tests for the durable QPU usage ledger."""

from __future__ import annotations

from quip_solver_dwave.usage import UsageLedger, hour_floor


def test_hour_floor_truncates_to_the_utc_hour():
    # 2026-09-04T16:46:05Z -> 2026-09-04T16:00:00Z
    assert hour_floor(1788799565.9) == 1788796800
    assert hour_floor(1788796800) == 1788796800  # already on the boundary


def test_records_accumulate_into_one_bucket_per_hour():
    led = UsageLedger(":memory:")
    base = 1788796800  # an hour boundary
    for _ in range(3):
        led.record(34_000, now=base + 10)
    led.record(58_000, now=base + 3600)  # next hour

    assert led.spent_us_since(base) == 3 * 34_000 + 58_000
    assert led.jobs_since(base) == 4
    # The second hour alone.
    assert led.spent_us_since(base + 3600) == 58_000
    assert led.jobs_since(base + 3600) == 1


def test_spend_before_the_period_start_is_excluded():
    led = UsageLedger(":memory:")
    base = 1788796800
    led.record(100_000, now=base - 3600)  # previous period
    led.record(34_000, now=base + 5)
    assert led.spent_us_since(base) == 34_000


def test_spend_survives_reopening_the_database(tmp_path):
    # The whole point of the ledger: a restart must not hand the miner a fresh
    # allotment. Write, close, reopen from disk, read the same total back.
    path = str(tmp_path / "usage.db")
    base = 1788796800
    led = UsageLedger(path)
    led.record(34_000, now=base)
    led.record(34_000, now=base)
    led.close()

    reopened = UsageLedger(path)
    assert reopened.spent_us_since(base) == 68_000
    assert reopened.jobs_since(base) == 2
    reopened.close()


def test_empty_ledger_reports_no_spend():
    led = UsageLedger(":memory:")
    assert led.spent_us_since(0) == 0.0
    assert led.jobs_since(0) == 0


def test_parent_directory_is_created(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "usage.db")
    led = UsageLedger(path)
    led.record(1_000, now=1788796800)
    assert led.spent_us_since(1788796800) == 1_000
    led.close()
