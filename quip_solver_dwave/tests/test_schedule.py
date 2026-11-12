"""Anneal schedules: every job states its anneal as ``[time_us, s]`` points."""

from __future__ import annotations

import pytest

from quip_solver_dwave.schedule import ScheduleError, forward_schedule, reverse_schedule

ADVANTAGE2_RANGE = [0.5, 2000.0]


def test_a_forward_anneal_is_a_two_point_ramp_from_zero_to_one():
    assert forward_schedule(20) == [[0.0, 0.0], [20.0, 1.0]]


def test_a_forward_anneal_outside_the_solver_range_is_refused():
    with pytest.raises(ScheduleError, match="annealing_time_range"):
        forward_schedule(0.1, time_range=ADVANTAGE2_RANGE)
    with pytest.raises(ScheduleError, match="annealing_time_range"):
        forward_schedule(5000, time_range=ADVANTAGE2_RANGE)


def test_a_reverse_anneal_starts_and_ends_at_one_and_pauses_at_the_reversal_point():
    # D-Wave's rule for a reverse anneal: s starts and ends at 1.
    assert reverse_schedule(20, 0.5, 25) == [
        [0.0, 1.0],
        [10.0, 0.5],
        [35.0, 0.5],
        [45.0, 1.0],
    ]


def test_each_ramp_of_a_reverse_anneal_keeps_the_forward_slope():
    # anneal_time_us means one thing in both directions: a full 0-to-1 ramp
    # takes that long. A 0.1 move at 20 us per unit takes 2 us.
    points = reverse_schedule(20, 0.9, 80)
    (t0, s0), (t1, s1) = points[0], points[1]
    assert (s0 - s1) / (t1 - t0) == pytest.approx(1 / 20)
    (t2, s2), (t3, s3) = points[-2], points[-1]
    assert (s3 - s2) / (t3 - t2) == pytest.approx(1 / 20)


def test_schedule_times_reach_the_wire_without_float_noise():
    # (1 - 0.9) * 20 is 1.9999999999999996 in binary floating point.
    assert reverse_schedule(20, 0.9, 0) == [[0.0, 1.0], [2.0, 0.9], [4.0, 1.0]]


def test_a_zero_pause_leaves_out_the_repeated_point():
    # Time must strictly increase between points, so a zero-length hold
    # cannot be written as two points at the same time.
    times = [t for t, _ in reverse_schedule(20, 0.5, 0)]
    assert times == sorted(set(times))


@pytest.mark.parametrize("bad_s", [0.0, 1.0, -0.2, 1.5])
def test_a_reversal_point_outside_the_open_unit_interval_is_refused(bad_s):
    with pytest.raises(ScheduleError, match="reversal point"):
        reverse_schedule(20, bad_s, 25)


def test_a_reverse_anneal_longer_than_the_solver_allows_is_refused():
    with pytest.raises(ScheduleError, match="annealing_time_range"):
        reverse_schedule(20, 0.5, 1990, time_range=ADVANTAGE2_RANGE)


def test_a_reverse_anneal_shorter_than_the_solver_allows_is_refused():
    # 20 us at reversal 0.999 with no pause totals 0.04 us, well under the
    # chip's 0.5 us shortest anneal. A slope that passes the per-ramp check
    # can still add up to a schedule the solver's own range refuses.
    with pytest.raises(ScheduleError, match="annealing_time_range"):
        reverse_schedule(20, 0.999, 0, time_range=ADVANTAGE2_RANGE)


def test_a_reverse_anneal_below_the_time_grid_is_refused():
    # 7 us at reversal 0.999 totals 0.014 us: below D-Wave's 0.01 us grid,
    # and also below the solver's 0.5 us minimum.
    with pytest.raises(ScheduleError, match="annealing_time_range"):
        reverse_schedule(7, 0.999, 0, time_range=ADVANTAGE2_RANGE)


def test_a_ramp_steeper_than_the_solver_allows_is_refused():
    # The slope cap is the inverse of the shortest anneal in the range.
    with pytest.raises(ScheduleError, match="steeper"):
        reverse_schedule(0.1, 0.5, 25, time_range=ADVANTAGE2_RANGE)
