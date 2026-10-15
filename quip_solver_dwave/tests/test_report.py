"""The --profile report reads back what the history recorded."""

from __future__ import annotations

from datetime import datetime, timezone

from quip_solver_dwave.history import HistoryStore, JobSample
from quip_solver_dwave.report import render_profile

# 2026-09-07 is day 7 of a 30-day month: bin 5, so "d06".
MONDAY = datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp()


def _store() -> HistoryStore:
    store = HistoryStore(":memory:")
    # Enough jobs that the cell reports its own rate rather than shrinking
    # toward the prediction the slower hour below pulls down.
    for i in range(3000):
        store.record_job(
            JobSample(
                completed_at=MONDAY + 13 * 3600 + i,
                generation=2,
                rtt_ms=3000,
                access_us=46_000,
                inflight_at_submit=3,
                reads=48,
                best_energy_milli=-14_540_000,
                target_milli=-14_554_000,
                hits=0,
                sapi_ms=2900,
            ),
            round_start_ts=None,
        )
    store.add_busy([(int(MONDAY + 13 * 3600), 1_500_000)])  # 3000 jobs in 1500 s: 2.0 jobs/s
    # A slower evening hour, so the hour-of-day factors have something to say.
    for i in range(100):
        store.record_job(
            JobSample(
                completed_at=MONDAY + 20 * 3600 + i,
                generation=3,
                rtt_ms=6000,
                access_us=46_000,
                inflight_at_submit=3,
                reads=48,
                best_energy_milli=-14_540_000,
                target_milli=-14_554_000,
                hits=0,
            ),
            round_start_ts=None,
        )
    store.add_busy([(int(MONDAY + 20 * 3600), 100_000)])  # 100 jobs in 100 s: 1.0 jobs/s
    store.open_round(int(MONDAY + 13 * 3600), 2, joined=True, reason="budget", expected_jobs=165.0)
    store.close_round(int(MONDAY + 13 * 3600), int(MONDAY + 13 * 3600 + 600))
    # 2026-09-12: day 12 of 30 is bin 10, "d11".
    store.open_round(int(MONDAY + 5 * 86_400), 9, joined=False, reason="budget-sat-out", expected_jobs=None)
    return store


def test_the_report_shows_a_twenty_eight_by_twenty_four_grid():
    out = render_profile(_store(), now=MONDAY + 6 * 86_400)
    lines = out.splitlines()
    assert lines[0].startswith("QPU throughput by day of month")
    d06 = next(line for line in lines if line.startswith("d06 "))
    cells = d06.split()[1:]
    assert len(cells) == 24
    assert cells[13] == "2.0"
    assert cells[0] == "."  # no evidence in that cell
    grid_rows = [line for line in lines if line[:1] == "d" and line[1:3].isdigit() and line[3] == " "]
    assert len(grid_rows) == 56  # two grids of 28 days


def test_the_report_states_the_factors_behind_the_estimate():
    out = render_profile(_store(), now=MONDAY + 6 * 86_400)
    lines = out.splitlines()
    hours = [float(f) for f in lines[lines.index("Hour-of-day factors (x the global rate):") + 1].split()]
    assert len(hours) == 24
    assert hours[13] > 1.0 > hours[20] and hours[0] == 1.0  # 13h fast, 20h slow, 00h unseen
    days = lines[lines.index("Day-of-month factors (x the global rate):") + 1]
    assert days.startswith("     d01 ") and "d07 " in days


def test_the_report_lists_recent_rounds_with_expected_and_actual():
    out = render_profile(_store(), now=MONDAY + 6 * 86_400)
    assert "All: 1 rounds joined, 0 won" in out
    assert "Round length 600s, access 46 ms/job" in out
    rows = [line for line in out.splitlines() if " join " in line or " skip " in line]
    assert any("d06 13h" in r and "join" in r and " 165 " in r and "budget" in r for r in rows)
    assert any("d11 00h" in r and "skip" in r and "budget-sat-out" in r for r in rows)
    assert "P(win)" not in out and "Win model" not in out


def test_the_hits_column_is_wide_enough_for_a_joined_round_s_read_count():
    # A joined round's hits (reads at or below target) can run into five
    # digits; a four-character column runs it into the next column.
    store = HistoryStore(":memory:")
    start = int(MONDAY + 13 * 3600)
    store.open_round(start, 2, joined=True, reason="budget", expected_jobs=None)
    store.record_job(
        JobSample(
            completed_at=MONDAY + 13 * 3600 + 1,
            generation=2,
            rtt_ms=3000,
            access_us=46_000,
            inflight_at_submit=1,
            reads=1,
            best_energy_milli=-100,
            target_milli=-50,
            hits=12_345,
        ),
        round_start_ts=start,
    )
    out = render_profile(store, now=MONDAY + 6 * 86_400)
    row = next(line for line in out.splitlines() if "d06 13h" in line)
    assert "   12345" in row  # six-wide right-aligned: one pad space plus the two-space separator


def test_an_empty_history_still_renders():
    out = render_profile(HistoryStore(":memory:"), now=MONDAY)
    assert "All: 0 rounds joined, 0 won, 0 jobs" in out
    assert "Round length 600s, access 46 ms/job" in out
