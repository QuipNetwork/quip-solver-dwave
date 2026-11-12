"""Metrics of the regime search."""

import numpy as np
import pytest

from quip_solver_dwave import regime_metrics as metrics


def test_match_sweeps_returns_the_first_rung_at_or_below_the_qpu_best():
    ladder = {512: -90.0, 2048: -100.0, 8192: -99.0, 32768: -120.0}
    assert metrics.match_sweeps(ladder, -100.0) == 2048.0
    assert metrics.match_sweeps(ladder, -110.0) == 32768.0


def test_match_sweeps_returns_none_for_an_unmatched_ladder():
    assert metrics.match_sweeps({512: -90.0, 2048: -95.0}, -100.0) is None


def test_match_sweeps_reads_the_rungs_in_ascending_order():
    assert metrics.match_sweeps({8192: -100.0, 512: -100.0}, -100.0) == 512.0


def test_matching_time_is_the_median_wall_time_of_the_matched_rung():
    assert metrics.matching_time(2048.0, {512: 0.1, 2048: 0.4}) == 0.4
    assert metrics.matching_time(None, {512: 0.1}) is None


def test_gate_window_measures_the_gap_in_robust_standard_deviations():
    deep = [-104.0, -102.0, -100.0, -98.0, -96.0]   # median -100, MAD 2
    lite = [-90.0] * 5
    qpu = [-95.0] * 5
    window = metrics.gate_window(lite, deep, qpu)
    assert window["spread"] == pytest.approx(1.4826 * 2.0)
    assert window["window"] == pytest.approx(10.0 / (1.4826 * 2.0))
    assert window["qpu_position"] == pytest.approx(0.5)


def test_spearman_is_one_for_a_monotone_relation():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert metrics.spearman(x, [v ** 3 for v in x]) == pytest.approx(1.0)
    assert metrics.spearman(x, [-v for v in x]) == pytest.approx(-1.0)


def test_partial_spearman_is_near_zero_when_the_control_explains_the_link():
    rng = np.random.default_rng(3)
    control = rng.normal(size=400)
    x = control + 0.3 * rng.normal(size=400)
    y = control + 0.3 * rng.normal(size=400)
    assert metrics.spearman(x, y) > 0.8
    assert abs(metrics.partial_spearman(x, y, control)) < 0.15


def test_partial_spearman_keeps_a_link_that_the_control_does_not_explain():
    rng = np.random.default_rng(4)
    control = rng.normal(size=400)
    shared = rng.normal(size=400)
    x = control + shared
    y = control + shared
    assert metrics.partial_spearman(x, y, control) > 0.8


def test_partial_spearman_is_not_a_number_when_the_control_explains_a_whole_series():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 1.0, 4.0, 3.0, 5.0]
    assert np.isnan(metrics.partial_spearman(x, x, x))      # three identical series gave exactly 1.0
    assert np.isnan(metrics.partial_spearman(x, y, x))      # the fit leaves nothing of x, and y keeps a residual


def test_partial_spearman_is_not_a_number_for_a_constant_series():
    assert np.isnan(metrics.partial_spearman([3.0] * 5, [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 1.0, 4.0, 3.0, 5.0]))


def test_gate_window_is_not_a_number_when_the_deep_energies_have_no_spread():
    window = metrics.gate_window([-90.0] * 5, [-100.0] * 5, [-95.0] * 5)
    assert window["spread"] == 0.0
    assert np.isnan(window["window"])
    assert window["qpu_position"] == pytest.approx(0.5)


def test_bootstrap_interval_brackets_the_statistic_and_is_reproducible():
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    y = x + rng.normal(size=200)
    low, high = metrics.bootstrap_interval(metrics.spearman, x, y, resamples=300, seed=1)
    assert low < metrics.spearman(x, y) < high
    assert (low, high) == metrics.bootstrap_interval(metrics.spearman, x, y, resamples=300, seed=1)


def test_wilson_interval_equals_the_closed_forms_at_a_share_of_zero_and_of_one():
    z_squared = 1.959963984540054 ** 2
    assert metrics.wilson_interval(0, 100) == (0.0, pytest.approx(z_squared / (100 + z_squared)))
    assert metrics.wilson_interval(100, 100) == (pytest.approx(100 / (100 + z_squared)), 1.0)


def test_wilson_interval_of_a_quarter_of_100_models_is_about_nine_points_each_way():
    low, high = metrics.wilson_interval(25, 100)
    assert low == pytest.approx(0.1755, abs=5e-5)
    assert high == pytest.approx(0.3430, abs=5e-5)


def test_wilson_interval_of_no_models_is_not_a_number():
    low, high = metrics.wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_time_condition_needs_ten_times_the_baseline_matching_time():
    assert metrics.time_condition(4.0, 0.4, 8192.0, 512.0) is not None
    assert metrics.time_condition(3.9, 0.4, 8192.0, 512.0) is None
    assert metrics.time_condition(None, 0.4, None, 512.0) is None
    assert metrics.time_condition(4.0, None, 8192.0, None) is None


def test_time_condition_names_what_made_the_matching_time():
    # 16 times the sweeps of the baseline: the model is harder.
    assert metrics.time_condition(4.0, 0.4, 8192.0, 512.0) == "matching sweeps"
    # 4 times the sweeps, so the cost of the kernel makes the rest of the time.
    assert metrics.time_condition(4.0, 0.4, 2048.0, 512.0) == "kernel cost alone"


def test_share_condition_needs_a_quarter_of_the_models():
    assert metrics.share_condition(25, 100)[0] == "unmatched share"
    assert metrics.share_condition(24, 100)[0] is None
    assert metrics.share_condition(0, 0) == (None, False)


def test_share_condition_is_uncertain_when_the_interval_spans_a_quarter():
    # With 100 models the interval spans 25% for a count of 17 to 33.
    assert metrics.share_condition(16, 100) == (None, False)
    assert metrics.share_condition(17, 100) == (None, True)
    assert metrics.share_condition(33, 100) == ("unmatched share", True)
    assert metrics.share_condition(34, 100) == ("unmatched share", False)
