"""Day-of-month by hour-of-day estimates: slot math, recency, factors, shrinkage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quip_solver_dwave.profile import (
    DAY_BINS,
    HOURS_PER_DAY,
    SECONDS_PER_DAY,
    SLOTS,
    day_bin_of,
    recency_weight,
    slot_label,
    slot_of,
    slot_stats,
)


def _utc(y, mo, d, h=0):
    return datetime(y, mo, d, h, tzinfo=timezone.utc).timestamp()


# February 2026 has 28 days, so every calendar day is its own bin.
FEB_1 = _utc(2026, 2, 1)
DAY = SECONDS_PER_DAY


def _row(hour, *, jobs, busy_ms, rtt_ms_sum=0, sapi_ms_sum=0, sapi_jobs=0,
         access_us_sum=0, source="live"):
    return {
        "hour_start_s": int(hour),
        "source": source,
        "jobs": jobs,
        "busy_ms": busy_ms,
        "rtt_ms_sum": rtt_ms_sum,
        "sapi_ms_sum": sapi_ms_sum,
        "sapi_jobs": sapi_jobs,
        "access_us_sum": access_us_sum,
    }


def test_the_first_hour_of_a_month_is_slot_zero():
    assert SLOTS == DAY_BINS * HOURS_PER_DAY == 672
    assert slot_of(FEB_1) == 0
    assert slot_of(FEB_1 + 13 * 3600) == 13
    assert slot_of(_utc(2026, 2, 28, 13)) == 27 * 24 + 13
    assert slot_of(_utc(2026, 3, 1)) == 0
    assert slot_of(FEB_1 - 1) == 27 * 24 + 23  # Jan 31 23h, the last bin


def test_longer_months_squeeze_onto_the_same_28_bins():
    # 30 days: day 30 lands in the last bin, and days pair up along the way.
    assert day_bin_of(1, 30) == 0 and day_bin_of(30, 30) == 27
    assert day_bin_of(7, 30) == 5
    # 31 days: days 1 and 2 share bin 0, day 31 is bin 27.
    assert day_bin_of(2, 31) == 0 and day_bin_of(31, 31) == 27
    assert slot_of(_utc(2026, 10, 31, 23)) == SLOTS - 1
    assert slot_of(_utc(2026, 10, 2, 5)) == 5
    # Every month fills every bin: no bin is skipped by a short month.
    for days in (28, 29, 30, 31):
        assert {day_bin_of(d, days) for d in range(1, days + 1)} == set(range(DAY_BINS))


def test_slot_labels_name_the_normalized_day_and_hour():
    assert slot_label(0) == "d01 00h"
    assert slot_label(27 * 24 + 13) == "d28 13h"
    assert slot_label(SLOTS - 1) == "d28 23h"


def test_recency_weight_halves_every_half_life():
    assert recency_weight(0.0) == 1.0
    assert recency_weight(91 * DAY) == pytest.approx(0.5)
    assert recency_weight(182 * DAY) == pytest.approx(0.25)


def test_heavy_evidence_reports_the_cell_s_own_rate():
    rows = [_row(FEB_1, jobs=10_000, busy_ms=5_000_000)]  # 2 jobs/s
    profile = slot_stats(rows, now=FEB_1 + 7 * DAY)
    assert profile.slots[0].jobs_per_s == pytest.approx(2.0)
    assert profile.slots[0].evidence == pytest.approx(10_000 * recency_weight(7 * DAY))


def test_a_cell_without_evidence_inherits_the_global_estimate():
    rows = [_row(FEB_1, jobs=10_000, busy_ms=5_000_000)]
    profile = slot_stats(rows, now=FEB_1 + 7 * DAY)
    assert profile.slots[300].jobs_per_s == pytest.approx(2.0)
    assert profile.slots[300].evidence == 0.0


def test_the_hour_factor_carries_a_fast_hour_to_days_never_observed():
    # Days 1-3 at 13h run 4 jobs/s; the same days at 00h run 2 jobs/s. Day
    # 10 has no rows at all, so its 13h and 00h cells are pure predictions:
    # the hour factor says 13h beats 00h, and the day factor for day 10 is 1.
    rows = []
    for day in range(3):
        rows.append(_row(FEB_1 + day * DAY + 13 * 3600, jobs=20_000, busy_ms=5_000_000))
        rows.append(_row(FEB_1 + day * DAY, jobs=10_000, busy_ms=5_000_000))
    profile = slot_stats(rows, now=FEB_1 + 4 * DAY)
    assert profile.hour_factors[13] > 1.2 and profile.hour_factors[0] < 0.8
    assert profile.day_factors[9] == 1.0
    fast = profile.slots[9 * 24 + 13].jobs_per_s
    slow = profile.slots[9 * 24 + 0].jobs_per_s
    assert fast is not None and slow is not None and fast > 1.5 * slow
    assert profile.slots[9 * 24 + 13].evidence == 0.0


def test_the_day_factor_carries_a_slow_day_to_hours_never_observed():
    # Day 28 runs at half the rate of days 1-3 at the hours observed. Its
    # unobserved 20h cell is predicted below the same hour on day 10.
    rows = []
    for day in range(3):
        for hour in (2, 9, 15):
            rows.append(_row(FEB_1 + day * DAY + hour * 3600, jobs=10_000, busy_ms=5_000_000))
    for hour in (2, 9, 15):
        rows.append(_row(FEB_1 + 27 * DAY + hour * 3600, jobs=10_000, busy_ms=10_000_000))
    profile = slot_stats(rows, now=FEB_1 + 28 * DAY)
    assert profile.day_factors[27] < 0.8
    last_day = profile.slots[27 * 24 + 20].jobs_per_s
    mid_month = profile.slots[9 * 24 + 20].jobs_per_s
    assert last_day is not None and mid_month is not None and last_day < 0.8 * mid_month


def test_the_two_factors_combine_multiplicatively():
    # Hour 13 is 2x the other hours; day 28 is 0.5x the other days. Their
    # product predicts an unobserved (day 28, 13h) cell near the global
    # mean, not near either extreme.
    rows = []
    for day in range(6):
        rows.append(_row(FEB_1 + day * DAY + 13 * 3600, jobs=40_000, busy_ms=10_000_000))  # 4/s
        rows.append(_row(FEB_1 + day * DAY + 1 * 3600, jobs=20_000, busy_ms=10_000_000))  # 2/s
    rows.append(_row(FEB_1 + 27 * DAY + 1 * 3600, jobs=10_000, busy_ms=10_000_000))  # 1/s
    now = FEB_1 + 28 * DAY
    profile = slot_stats(rows, now=now)
    weights = [recency_weight(now - r["hour_start_s"]) for r in rows]
    everything = sum(w * r["jobs"] for w, r in zip(weights, rows)) / sum(
        w * r["busy_ms"] / 1000 for w, r in zip(weights, rows)
    )
    cell = profile.slots[27 * 24 + 13].jobs_per_s
    assert cell is not None
    assert cell == pytest.approx(everything * profile.hour_factors[13] * profile.day_factors[27], rel=1e-6)
    assert 1.0 < cell < 4.0


def test_thin_evidence_is_pulled_toward_the_prediction():
    rows = [
        _row(FEB_1, jobs=10_000, busy_ms=5_000_000),  # d01 00h, 2 jobs/s
        _row(FEB_1 + 2 * DAY, jobs=20, busy_ms=5_000),  # d03 00h, 4 jobs/s, 20 jobs
    ]
    profile = slot_stats(rows, now=FEB_1 + 3 * DAY)
    d03 = profile.slots[2 * 24].jobs_per_s
    assert d03 is not None and 2.0 < d03 < 2.5  # 20 jobs cannot outvote 200 pseudo-jobs


def test_older_months_count_less():
    rows = [
        _row(FEB_1, jobs=1000, busy_ms=1_000_000),  # 1 job/s, now
        _row(FEB_1 - 91 * DAY, jobs=1000, busy_ms=250_000),  # 4 jobs/s, half weight, Nov 2 00h
    ]
    profile = slot_stats(rows, now=FEB_1)
    # Same cell (both bin 0, hour 0). Weighted: 1500 jobs over 1125 busy
    # seconds, not 2000 over 1250.
    assert slot_of(FEB_1 - 91 * DAY) == 0
    assert profile.slots[0].jobs_per_s == pytest.approx(1500 / 1125)


def test_rows_outside_the_window_are_ignored():
    rows = [
        _row(FEB_1 - 200 * DAY, jobs=1000, busy_ms=1_000_000),
        _row(FEB_1 + 3600, jobs=1000, busy_ms=1_000_000),  # in the future
    ]
    profile = slot_stats(rows, now=FEB_1)
    assert profile.slots[0].jobs_per_s is None and profile.slots[0].rtt_s is None
    assert profile.hour_factors == [1.0] * HOURS_PER_DAY


def test_queue_wait_is_service_time_minus_access_time():
    rows = [_row(FEB_1, jobs=100, busy_ms=100_000, rtt_ms_sum=300_000,
                 sapi_ms_sum=290_000, sapi_jobs=100, access_us_sum=4_600_000)]
    profile = slot_stats(rows, now=FEB_1 + 3600)
    assert profile.slots[0].rtt_s == pytest.approx(3.0)
    assert profile.slots[0].queue_s == pytest.approx(2.9 - 0.046)


def test_seeded_rows_carry_throughput_but_no_round_trip():
    rows = [_row(FEB_1, jobs=100, busy_ms=100_000, access_us_sum=4_600_000, source="attempts")]
    profile = slot_stats(rows, now=FEB_1 + 3600)
    assert profile.slots[0].jobs_per_s == pytest.approx(1.0)
    assert profile.slots[0].rtt_s is None and profile.slots[0].queue_s is None


def test_round_trip_shrinks_by_its_own_evidence_not_the_total_job_count():
    # A cell can hold many seeded jobs (throughput only) alongside a few
    # live ones. Round trip has ten samples behind it, not 1010: shrinking
    # it with the seeded jobs' count would report it as if it were nearly
    # as certain as the throughput number, when it is not.
    mostly_seeded = _row(FEB_1, jobs=1000, busy_ms=500_000, source="attempts")
    a_few_live = _row(FEB_1, jobs=10, busy_ms=5_000, rtt_ms_sum=30_000, source="live")
    # Day 3 at the same hour: heavy evidence for a round trip far from the
    # thin cell's own 3.0 s/job.
    heavy = _row(FEB_1 + 2 * DAY, jobs=2000, busy_ms=1_000_000, rtt_ms_sum=2_000_000, source="live")
    profile = slot_stats([mostly_seeded, a_few_live, heavy], now=FEB_1 + 3 * DAY)
    d01 = profile.slots[0]
    # The rate's own evidence is the full 1010 jobs, so it stays near 2 jobs/s.
    assert d01.jobs_per_s == pytest.approx(2.0, rel=0.05)
    # The round trip's own evidence is 10 live jobs against 200 pseudo-jobs:
    # it lands close to the prediction's ~1.0 s/job, nowhere near the cell's
    # own 3.0 s/job average.
    assert d01.rtt_s is not None and d01.rtt_s < 1.5
