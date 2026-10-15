from quip_solver_dwave.lite_filter import (
    FilterRow,
    ceiling_for_share,
    choose_sweeps,
    filter_row,
    low_energy_set,
)


def test_the_low_energy_set_is_the_lowest_one_percent_of_qpu_energies():
    attempts = {f"n{i:03d}": -i for i in range(200)}
    assert low_energy_set(attempts) == {"n199", "n198"}


def test_the_low_energy_set_keeps_every_nonce_tied_at_the_cut():
    attempts = {f"n{i:03d}": 0 for i in range(100)}
    attempts["n000"] = attempts["n001"] = -5
    assert low_energy_set(attempts) == {"n000", "n001"}


def test_a_ceiling_for_a_share_passes_that_weighted_share():
    lite = {"a": -40, "b": -30, "c": -20, "d": -10}
    weights = {"a": 1.0, "b": 1.0, "c": 4.0, "d": 4.0}
    # Total weight 10. The 20% ceiling must pass a and b only.
    assert ceiling_for_share(lite, weights, 0.2) == -30


def test_a_filter_row_counts_passes_misses_and_weighted_false_positives():
    lite = {"low1": -50, "low2": -5, "x": -45, "y": -1}
    weights = {"low1": 1.0, "low2": 1.0, "x": 10.0, "y": 10.0}
    row = filter_row(512, -40, lite, weights, {"low1", "low2"})
    assert (row.low_pass, row.low_miss) == (1, 1)
    assert row.false_positive_estimate == 10.0
    assert row.recall == 0.5
    assert abs(row.pass_share - 11.0 / 22.0) < 1e-12


def test_an_attempt_at_the_ceiling_passes():
    row = filter_row(512, -40, {"low": -40}, {"low": 1.0}, {"low"})
    assert row.low_pass == 1


def rows_with_recalls(recalls):
    return [
        FilterRow(sweeps, -1, 0.05, 0, 0, 0.0, recall)
        for sweeps, recall in zip([128, 256, 512, 1024, 2048, 4096], recalls)
    ]


def test_the_chosen_sweep_count_is_the_lowest_within_tolerance_of_the_longest():
    rows = rows_with_recalls([0.40, 0.55, 0.69, 0.70, 0.705, 0.71])
    assert choose_sweeps(rows).sweeps == 512


def test_the_longest_sweep_count_is_chosen_when_nothing_shorter_is_close():
    rows = rows_with_recalls([0.1, 0.2, 0.3, 0.4, 0.5, 0.71])
    assert choose_sweeps(rows).sweeps == 4096
