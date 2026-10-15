"""Text report behind ``quip-dwave-qa --profile``.

Two 28x24 grids (throughput, D-Wave queue wait) with the hour-of-day and
day-of-month factors behind the throughput estimate, the round totals, and
the last rounds with what the strategy expected beside what happened. The
SQLite file stays the machine-readable surface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, List, Optional

from quip_solver_dwave.history import HistoryStore
from quip_solver_dwave.profile import (
    DAY_BINS,
    DEFAULT_WINDOW_DAYS,
    HOURS_PER_DAY,
    SECONDS_PER_DAY,
    Profile,
    SlotStats,
    build_snapshot,
    slot_label,
    slot_of,
)

RECENT_ROUNDS = 20


def _cell(value: Optional[float], evidence: float) -> str:
    # A cell borrowing everything from the prediction shows nothing of its own.
    if value is None or evidence < 1.0:
        return "   ."
    return f"{value:4.1f}"


def _grid(title: str, stats: List[SlotStats], pick: Callable[[SlotStats], Optional[float]]) -> List[str]:
    lines = [title, "     " + " ".join(f"{h:>4d}" for h in range(HOURS_PER_DAY))]
    for day in range(DAY_BINS):
        cells = []
        for hour in range(HOURS_PER_DAY):
            st = stats[day * HOURS_PER_DAY + hour]
            cells.append(_cell(pick(st), st.evidence))
        lines.append(f"d{day + 1:02d}  " + " ".join(cells))
    return lines


def _factors(profile: Profile) -> List[str]:
    lines = ["Hour-of-day factors (x the global rate):"]
    lines.append("     " + " ".join(f"{f:4.2f}" for f in profile.hour_factors))
    lines.append("Day-of-month factors (x the global rate):")
    for start in range(0, DAY_BINS, 7):
        lines.append(
            "     "
            + "  ".join(
                f"d{d + 1:02d} {f:4.2f}"
                for d, f in enumerate(profile.day_factors[start : start + 7], start=start)
            )
        )
    return lines


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%d %H:%M")


def render_profile(store: HistoryStore, now: float) -> str:
    since = now - DEFAULT_WINDOW_DAYS * SECONDS_PER_DAY
    snap = build_snapshot(store, now)
    stats = snap.slots
    lines: List[str] = []
    lines += _grid(
        "QPU throughput by day of month (normalized to 28) and UTC hour, jobs/s. "
        "'.' means no evidence in that cell.",
        stats,
        lambda s: s.jobs_per_s,
    )
    lines += _factors(snap.profile)
    lines.append("")
    lines += _grid(
        "D-Wave queue wait by day of month (normalized to 28) and UTC hour, seconds.",
        stats,
        lambda s: s.queue_s,
    )
    lines.append("")
    rounds = store.rounds(since_ts=since)
    joined = [r for r in rounds if r["joined"]]
    jobs = sum(int(r["jobs"]) for r in joined)
    wins = sum(int(r["won"]) for r in joined)
    lines.append(f"All: {len(joined)} rounds joined, {wins} won, {jobs} jobs")
    lines.append(
        f"Round length {snap.round_length_s:.0f}s, access {snap.access_s_per_job * 1000:.0f} ms/job"
    )
    lines.append("")
    lines.append(f"Last {RECENT_ROUNDS} rounds (UTC):")
    lines.append("start        slot     verdict  reason           expected  jobs    hits  won")
    for r in rounds[:RECENT_ROUNDS]:
        expected = "-" if r["expected_jobs"] is None else f"{r['expected_jobs']:.0f}"
        lines.append(
            f"{_utc(r['start_ts_s']):<12} {slot_label(slot_of(r['start_ts_s'])):<8} "
            f"{'join' if r['joined'] else 'skip':<8} {r['reason']:<16} {expected:>8}  "
            f"{r['jobs']:>4}  {r['hits']:>6}  {'W' if r['won'] else ''}"
        )
    return "\n".join(lines)
