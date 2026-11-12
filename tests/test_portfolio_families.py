"""Tests for quip_solver_dwave.portfolio_families."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quip_solver_dwave import portfolio_families as families


def _grouped_returns(rng: np.random.Generator, rows: int = 720) -> np.ndarray:
    """Two strongly correlated groups of three assets, with stock-style closed hours."""
    first, second = rng.standard_normal(rows), rng.standard_normal(rows)
    noise = 0.2 * rng.standard_normal((rows, 6))
    returns = np.column_stack([first] * 3 + [second] * 3) + noise
    returns[rows // 2 :, 5] = np.nan  # the last asset is closed for the second half
    return 0.01 * returns


def test_a_window_is_a_contiguous_slice_of_the_requested_length():
    returns = np.arange(100.0).reshape(50, 2)
    start, window = families.sample_window(returns, np.random.default_rng(1), 20)
    assert window.shape == (20, 2)
    assert np.array_equal(window, returns[start : start + 20])


def test_a_window_longer_than_the_history_is_refused():
    with pytest.raises(ValueError, match="history"):
        families.sample_window(np.zeros((10, 2)), np.random.default_rng(1), 11)


def test_the_block_bootstrap_keeps_every_row_whole():
    returns = _grouped_returns(np.random.default_rng(2))
    resampled = families.block_bootstrap(returns, np.random.default_rng(3), block=24)
    assert resampled.shape == returns.shape
    original = {tuple(np.nan_to_num(row, nan=9.0)) for row in returns}
    assert all(tuple(np.nan_to_num(row, nan=9.0)) in original for row in resampled)


def test_the_block_bootstrap_resamples_in_contiguous_blocks():
    returns = np.arange(96.0).reshape(96, 1)
    resampled = families.block_bootstrap(returns, np.random.default_rng(4), block=24)[:, 0]
    for start in range(0, 96, 24):
        steps = np.diff(resampled[start : start + 24]) % 96
        assert np.all(steps == 1.0)


@pytest.mark.parametrize("null", [families.null_iid_normal, families.null_shuffle])
def test_a_returns_null_keeps_the_missing_hours_and_removes_co_movement(null):
    returns = _grouped_returns(np.random.default_rng(5))
    result = null(returns, np.random.default_rng(6))
    assert np.array_equal(np.isnan(result), np.isnan(returns))
    before = abs(np.corrcoef(returns[:, 0], returns[:, 1])[0, 1])
    after = abs(np.corrcoef(result[:, 0], result[:, 1])[0, 1])
    assert before > 0.9 and after < 0.15


def test_the_shuffle_null_keeps_each_asset_exact_values():
    returns = _grouped_returns(np.random.default_rng(7))
    result = families.null_shuffle(returns, np.random.default_rng(8))
    for column in range(returns.shape[1]):
        observed = ~np.isnan(returns[:, column])
        assert np.array_equal(np.sort(result[observed, column]), np.sort(returns[observed, column]))


def test_the_iid_null_matches_each_asset_mean_and_volatility():
    returns = _grouped_returns(np.random.default_rng(9), rows=20000)
    result = families.null_iid_normal(returns, np.random.default_rng(10))
    np.testing.assert_allclose(np.nanstd(result, axis=0), np.nanstd(returns, axis=0), rtol=0.03)
    np.testing.assert_allclose(np.nanmean(result, axis=0), np.nanmean(returns, axis=0), atol=5e-4)


def test_the_rotation_null_keeps_variances_and_the_correlation_spectrum_roughly():
    sigma = np.cov(np.nan_to_num(_grouped_returns(np.random.default_rng(11))), rowvar=False)
    rotated = families.null_rotate_correlation(sigma, np.random.default_rng(12))
    np.testing.assert_allclose(np.diagonal(rotated), np.diagonal(sigma), rtol=1e-12)
    assert np.all(np.linalg.eigvalsh(rotated) > 0.0)
    assert np.allclose(rotated, rotated.T)
    grouped_before = abs(families.correlation(sigma)[0, 1])
    grouped_after = abs(families.correlation(rotated)[0, 1])
    assert grouped_before > 0.9 and grouped_after < grouped_before


def test_pair_overlaps_count_the_hours_both_assets_traded():
    returns = np.array([[1.0, np.nan, 1.0], [1.0, 1.0, np.nan], [1.0, 1.0, 1.0]])
    assert np.array_equal(families.pair_overlaps(returns), [[3, 2, 2], [2, 2, 1], [2, 1, 2]])


def test_significance_uses_each_pair_own_overlap():
    corr = np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 1.0]])
    overlaps = np.array([[720, 720, 100], [720, 720, 100], [100, 100, 100]])
    # z = atanh(0.2) * sqrt(n - 3): 5.4 over 720 hours, 2.0 over 100 hours.
    assert np.array_equal(
        families.significant_pairs(corr, overlaps),
        [[False, True, False], [True, False, False], [False, False, False]],
    )


def test_grouped_returns_have_significant_pairs_and_noise_has_almost_none():
    rng = np.random.default_rng(13)
    grouped = _grouped_returns(rng)
    noise = rng.standard_normal((720, 6))
    for returns, low, high in ((grouped, 0.35, 0.5), (noise, 0.0, 0.1)):
        corr = families.correlation(np.cov(np.nan_to_num(returns), rowvar=False))
        share = families.significant_pairs(corr, families.pair_overlaps(returns)).sum() / 30
        assert low <= share <= high


def test_modularity_is_high_for_two_groups_and_zero_for_noise():
    rng = np.random.default_rng(14)
    grouped_returns = _grouped_returns(rng)
    noise_returns = rng.standard_normal((720, 6))
    grouped = families.correlation(np.cov(np.nan_to_num(grouped_returns), rowvar=False))
    noise = families.correlation(np.cov(noise_returns, rowvar=False))
    grouped_mask = families.significant_pairs(grouped, families.pair_overlaps(grouped_returns))
    noise_mask = families.significant_pairs(noise, families.pair_overlaps(noise_returns))
    assert families.best_modularity(grouped, grouped_mask) > 0.4
    assert families.best_modularity(noise, noise_mask) < 0.05


def test_the_gap_to_the_second_best_selection_is_relative_to_the_best():
    q = np.diag([-1.0, -1.0, -0.9, 0.0])
    # Pairs: {0,1} = -2, then {0,2} = {1,2} = -1.9.
    assert families.second_best_gap(q, 2) == pytest.approx(0.05)


def test_near_degenerate_selections_are_counted_exactly():
    q = np.diag([-1.0, -1.0, -0.999, 0.0])
    # Pairs: {0,1} = -2, {0,2} = {1,2} = -1.999, the others are -1 or less deep.
    assert families.near_degenerate_count(q, 2, tolerance=0.001) == 3
    assert families.near_degenerate_count(q, 2, tolerance=0.0) == 1


def test_near_degenerate_count_matches_brute_force_on_a_random_qubo():
    rng = np.random.default_rng(15)
    q = rng.standard_normal((9, 9))
    q = (q + q.T) / 2
    energies = [
        float(q[np.ix_(subset, subset)].sum()) for subset in map(list, itertools.combinations(range(9), 4))
    ]
    best = min(energies)
    expected = sum(abs(e - best) <= 0.05 * abs(best) for e in energies)
    assert families.near_degenerate_count(q, 4, tolerance=0.05) == expected


def test_near_degenerate_count_declines_an_enumeration_that_is_too_large():
    assert families.near_degenerate_count(np.zeros((40, 40)), 20, tolerance=0.01) is None


def _factor_returns(rng: np.random.Generator, rows: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Eight assets on two factors, with stock-style closed hours on the last three."""
    loadings = np.array([[0.8, 0.0]] * 3 + [[0.6, 0.3]] * 2 + [[0.0, 0.7]] * 3)
    idiosyncratic = 1.0 - (loadings**2).sum(axis=1)
    standardized = rng.standard_normal((rows, 2)) @ loadings.T + rng.standard_normal((rows, 8)) * np.sqrt(idiosyncratic)
    returns = 0.01 * standardized
    returns[rng.random(rows) < 0.7, 5:] = np.nan
    return returns, loadings


def test_the_marchenko_pastur_rule_counts_only_eigenvalues_above_noise():
    rng = np.random.default_rng(16)
    returns, _ = _factor_returns(rng)
    sigma = np.cov(np.nan_to_num(returns), rowvar=False)
    assert families.marchenko_pastur_factors(families.correlation(sigma), 4000) == 2
    noise = np.cov(rng.standard_normal((4000, 8)), rowvar=False)
    assert families.marchenko_pastur_factors(families.correlation(noise), 4000) == 1


def test_the_tail_df_inverts_the_t_excess_kurtosis_within_bounds():
    assert families.t_tail_df(1.0) == pytest.approx(10.0)
    assert families.t_tail_df(100.0) == families.TAIL_DF_BOUNDS[0]
    assert families.t_tail_df(0.0) == families.TAIL_DF_BOUNDS[1]


def test_the_factor_fit_recovers_the_common_covariance():
    returns, loadings = _factor_returns(np.random.default_rng(17))
    observed = ~np.isnan(returns)
    sigma = np.ma.cov(np.ma.masked_invalid(returns), rowvar=False, allow_masked=True).filled(0.0)
    model = families.fit_factor_model(returns, sigma, 2)
    np.testing.assert_allclose(model.loadings @ model.loadings.T, loadings @ loadings.T, atol=0.1)
    np.testing.assert_allclose((model.loadings**2).sum(axis=1) + model.idiosyncratic, 1.0)
    assert model.scores.shape == (returns.shape[0], 2)
    assert np.array_equal(model.observed, observed)
    assert families.TAIL_DF_BOUNDS[0] <= model.tail_df <= families.TAIL_DF_BOUNDS[1]


def test_the_factor_fit_leaves_every_asset_an_idiosyncratic_floor():
    returns, _ = _factor_returns(np.random.default_rng(18))
    sigma = np.cov(np.nan_to_num(returns), rowvar=False)
    sigma[:3, :3] = 0.9999 * sigma[0, 0] + 0.0001 * np.diag(np.diagonal(sigma[:3, :3]))
    model = families.fit_factor_model(returns, sigma, 2)
    assert model.idiosyncratic.min() >= families.IDIOSYNCRATIC_FLOOR - 1e-12


def test_synthetic_assets_keep_the_source_structure_and_missing_hours():
    returns, _ = _factor_returns(np.random.default_rng(19))
    sigma = np.ma.cov(np.ma.masked_invalid(returns), rowvar=False, allow_masked=True).filled(0.0)
    model = families.fit_factor_model(returns, sigma, 2)
    synthetic, sources = families.factor_synthetic(
        model, np.random.default_rng(20), assets=80, rows=2000, block=24, loading_jitter=0.0, vol_jitter=0.0)
    assert synthetic.shape == (2000, 80) and sources.shape == (80,)
    assert set(sources.tolist()) == set(range(8))
    stock = np.isin(sources, [5, 6, 7])
    assert not np.isnan(synthetic[:, ~stock]).any()
    assert np.isnan(synthetic[:, stock]).mean() == pytest.approx(0.7, abs=0.05)
    copies = np.flatnonzero(sources == 6)
    for column in copies:
        assert np.array_equal(np.isnan(synthetic[:, column]), np.isnan(synthetic[:, copies[0]]))
    first, second = np.flatnonzero(sources == 0)[0], np.flatnonzero(sources == 1)[0]
    unrelated = np.flatnonzero(sources == 2)[0], np.flatnonzero(sources == 5)[0]
    assert np.corrcoef(synthetic[:, first], synthetic[:, second])[0, 1] > 0.5
    pair = synthetic[:, list(unrelated)]
    pair = pair[~np.isnan(pair).any(axis=1)]
    assert abs(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1]) < 0.15
    np.testing.assert_allclose(np.nanstd(synthetic, axis=0), model.vol[sources], rtol=0.15)


def test_synthetic_draws_are_reproducible_and_jitter_moves_the_loadings():
    returns, _ = _factor_returns(np.random.default_rng(21))
    sigma = np.cov(np.nan_to_num(returns), rowvar=False)
    model = families.fit_factor_model(returns, sigma, 2)

    def draw(jitter: float) -> np.ndarray:
        return families.factor_synthetic(model, np.random.default_rng(22), 12, 500, 24, jitter, jitter)[0]

    assert np.array_equal(draw(0.05), draw(0.05), equal_nan=True)
    assert not np.array_equal(draw(0.05), draw(0.0), equal_nan=True)
