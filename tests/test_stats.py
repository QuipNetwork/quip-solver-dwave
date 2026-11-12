import numpy as np

from quip_solver_dwave.stats import ks_two_sample, spearman


def test_ks_does_not_separate_two_samples_of_one_distribution():
    rng = np.random.default_rng(1)
    d, p = ks_two_sample(rng.normal(size=400), rng.normal(size=400))
    assert d < 0.12 and p > 0.05


def test_ks_separates_two_shifted_distributions():
    rng = np.random.default_rng(1)
    d, p = ks_two_sample(rng.normal(size=400), rng.normal(loc=1.0, size=400))
    assert d > 0.3 and p < 0.001


def test_spearman_is_one_for_any_increasing_relation():
    x = np.arange(50.0)
    assert abs(spearman(x, np.exp(x / 10.0)) - 1.0) < 1e-12


def test_spearman_is_minus_one_for_a_decreasing_relation():
    x = np.arange(50.0)
    assert abs(spearman(x, -x**3) - (-1.0)) < 1e-12


def test_spearman_gives_tied_values_their_mean_rank():
    assert abs(spearman([1, 1, 2, 3], [1, 1, 2, 3]) - 1.0) < 1e-12
