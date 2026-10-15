"""Anneal schedules: the one way this miner tells the QPU how long to anneal.

SAPI takes the anneal either as ``annealing_time`` (one number) or as
``anneal_schedule`` (a list of ``[time_us, s]`` points), and refuses a problem
that carries both. A reverse anneal can only be written as a schedule, so every
job uses the schedule form and ``annealing_time`` is never sent. A forward
anneal of ``T`` microseconds is ``[[0, 0], [T, 1]]``, which D-Wave's own timing
model prices the same as ``annealing_time=T``.

``anneal_time_us`` keeps one meaning in both directions: the time a full ramp
of ``s`` from 0 to 1 takes, so it fixes the slope ``1 / T``. A reverse anneal
ramps from ``s = 1`` down to the reversal point at that slope, holds there for
the pause, and ramps back at the same slope.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

Schedule = List[List[float]]

# D-Wave's documented default anneal on Advantage2, for a sampler that does
# not publish ``default_annealing_time`` (the offline mock).
FALLBACK_ANNEAL_US = 20.0

# The reversal point and pause from D-Wave's reverse-anneal example schedule
# (docs.dwavequantum.com, "Annealing Implementation and Controls"). Starting
# values only: QUI-1387 step 7 measures what works on mining problems.
DEFAULT_REVERSAL_S = 0.5
DEFAULT_REVERSAL_PAUSE_US = 25

# SAPI takes times as floats. Rounding to the 0.01 us time grid keeps sums
# such as 0.1 * 20 from reaching the wire as 1.9999999999999996, and keeps
# every schedule point on a grid SAPI actually accepts.
_TIME_DECIMALS = 2


class ScheduleError(ValueError):
    """The requested anneal does not fit the solver's published limits."""


def _check_total(total_us: float, time_range: Optional[Sequence[float]]) -> None:
    if time_range is None:
        return
    low, high = float(time_range[0]), float(time_range[1])
    if not low <= total_us <= high:
        raise ScheduleError(
            f"a {total_us:g} us schedule is outside the solver's "
            f"annealing_time_range [{low:g}, {high:g}]"
        )


def forward_schedule(
    anneal_us: float, *, time_range: Optional[Sequence[float]] = None
) -> Schedule:
    """``[[0, 0], [T, 1]]``: the schedule form of ``annealing_time=T``."""
    total = round(float(anneal_us), _TIME_DECIMALS)
    if total <= 0:
        raise ScheduleError(f"anneal time must be positive, got {anneal_us!r}")
    _check_total(total, time_range)
    return [[0.0, 0.0], [total, 1.0]]


def reverse_schedule(
    anneal_us: float,
    reversal_s: float,
    pause_us: float,
    *,
    time_range: Optional[Sequence[float]] = None,
) -> Schedule:
    """Ramp from ``s = 1`` down to ``reversal_s``, pause, and ramp back.

    Each ramp takes ``(1 - reversal_s) * anneal_us``, so its slope is the
    forward anneal's ``1 / anneal_us``. D-Wave caps the slope at the inverse of
    the shortest anneal it allows, and ``anneal_us`` is checked against that
    same range, so a schedule that passes here cannot be too steep.
    """
    if not 0.0 < reversal_s < 1.0:
        raise ScheduleError(f"reversal point must be inside (0, 1), got {reversal_s!r}")
    if pause_us < 0:
        raise ScheduleError(f"pause must not be negative, got {pause_us!r}")
    if anneal_us <= 0:
        raise ScheduleError(f"anneal time must be positive, got {anneal_us!r}")
    if time_range is not None and float(anneal_us) < float(time_range[0]):
        raise ScheduleError(
            f"a {anneal_us:g} us ramp is steeper than the solver's fastest "
            f"anneal of {float(time_range[0]):g} us allows"
        )
    leg = (1.0 - reversal_s) * float(anneal_us)
    down = round(leg, _TIME_DECIMALS)
    hold = round(leg + float(pause_us), _TIME_DECIMALS)
    total = round(2.0 * leg + float(pause_us), _TIME_DECIMALS)
    if down <= 0.0:
        raise ScheduleError(
            f"reversal point {reversal_s!r} leaves no ramp at {anneal_us:g} us"
        )
    _check_total(total, time_range)
    points: Schedule = [[0.0, 1.0], [down, float(reversal_s)]]
    if hold > down:
        points.append([hold, float(reversal_s)])
    points.append([total, 1.0])
    return points
