"""Random portfolio models on a rigid topology, with and without real market structure.

Every model of basket size N shares one graph, the complete graph on N
selection variables, and differs only in its weights. The weights come from
hourly returns, and the returns come from one of two families:

- Family A resamples real returns. :func:`sample_window` takes a random
  contiguous window and :func:`block_bootstrap` resamples it in whole-row
  blocks, so co-movement, subgroups, heavy tails, volatility clustering, and
  closed-market hours all survive.
- Family B is synthetic, so N can exceed the real asset count.
  :func:`fit_factor_model` fits statistical factors to the real history, and
  :func:`factor_synthetic` draws new assets from them: each one resamples a
  fitted loading row with a small jitter, adds Student-t idiosyncratic noise
  with the fitted tail, and keeps its source asset's missing hours. The factor
  scores are block-bootstrapped in whole hours, so co-movement and volatility
  clustering survive.
- Family C removes the structure but keeps the marginals.
  :func:`null_iid_normal` and :func:`null_shuffle` act on the returns and keep
  every asset's missing hours. :func:`null_rotate_correlation` acts on the
  covariance and keeps each asset's variance and roughly the eigenvalue
  spectrum, but not which assets group together.

The diagnostics measure the structure instead of assuming it:
:func:`significant_pairs`, :func:`best_modularity`,
:func:`near_degenerate_count`, and :func:`second_best_gap`. They use each
pair's own overlap, because stock pairs share far fewer hours than crypto pairs.

Only numpy and scipy are needed, so the production backend's environment can
import this module.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

#: Largest exact enumeration :func:`near_degenerate_count` attempts.
ENUMERATION_LIMIT = 10_000_000
_CHUNK = 100_000
#: Fisher z above which a pair's correlation counts as significant.
SIGNIFICANCE_Z = 3.0
#: Smallest idiosyncratic share of a standardized asset's variance in family B.
IDIOSYNCRATIC_FLOOR = 0.05
#: Bounds on the fitted Student-t degrees of freedom: finite kurtosis, and close to normal.
TAIL_DF_BOUNDS = (5.0, 50.0)
#: Most refits :func:`fit_factor_model` makes before it keeps the communalities it has.
PRINCIPAL_AXIS_ITERATIONS = 500


def sample_window(returns: np.ndarray, rng: np.random.Generator, rows: int) -> tuple[int, np.ndarray]:
    """``(start, window)``: a random contiguous window of ``rows`` return rows and its first row."""
    if rows > returns.shape[0]:
        raise ValueError(f"a {rows}-row window does not fit a {returns.shape[0]}-row history")
    start = int(rng.integers(0, returns.shape[0] - rows + 1))
    return start, returns[start : start + rows]


def block_bootstrap(returns: np.ndarray, rng: np.random.Generator, block: int) -> np.ndarray:
    """Circular block bootstrap of whole rows, the same length as ``returns``.

    Each row keeps every asset's value for that hour, so cross-asset structure
    and the missing-hour pattern move together.
    """
    rows = returns.shape[0]
    starts = rng.integers(0, rows, size=math.ceil(rows / block))
    index = (starts[:, None] + np.arange(block)[None, :]).ravel()[:rows] % rows
    return returns[index]


def null_iid_normal(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent normal draws with each asset's observed mean and volatility."""
    result = np.full_like(returns, np.nan)
    for column in range(returns.shape[1]):
        observed = ~np.isnan(returns[:, column])
        values = returns[observed, column]
        result[observed, column] = rng.normal(values.mean(), values.std(), observed.sum())
    return result


def null_shuffle(returns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Each asset's observed values permuted independently among its observed hours."""
    result = returns.copy()
    for column in range(returns.shape[1]):
        observed = np.flatnonzero(~np.isnan(returns[:, column]))
        result[observed, column] = returns[rng.permutation(observed), column]
    return result


def correlation(sigma: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diagonal(sigma))
    return sigma / np.outer(scale, scale)


def null_rotate_correlation(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A covariance with the same variances and a Haar-rotated correlation spectrum.

    Rescaling the rotated matrix back to a unit diagonal moves the eigenvalues
    slightly, so the spectrum is kept approximately, not exactly.
    """
    values, _ = np.linalg.eigh(correlation(sigma))
    gaussian = rng.standard_normal(sigma.shape)
    q, r = np.linalg.qr(gaussian)
    q = q * np.sign(np.diagonal(r))
    rotated = correlation((q * values) @ q.T)
    scale = np.sqrt(np.diagonal(sigma))
    return rotated * np.outer(scale, scale)


@dataclass(frozen=True)
class FactorModel:
    """Statistical factors fitted to a real return history, on the standardized scale.

    ``loadings`` is (assets, factors) and ``scores`` is (hours, factors). Each
    asset's standardized variance is its squared loading norm plus
    ``idiosyncratic``. ``observed`` marks the hours each asset traded.
    """

    mean: np.ndarray
    vol: np.ndarray
    loadings: np.ndarray
    idiosyncratic: np.ndarray
    scores: np.ndarray
    observed: np.ndarray
    tail_df: float


def marchenko_pastur_factors(corr: np.ndarray, hours: float) -> int:
    """Eigenvalues of ``corr`` above the Marchenko-Pastur edge of pure noise over ``hours``, at least 1."""
    edge = (1.0 + math.sqrt(corr.shape[0] / hours)) ** 2
    return max(1, int((np.linalg.eigvalsh(corr) > edge).sum()))


def t_tail_df(excess_kurtosis: float) -> float:
    """Student-t degrees of freedom with this excess kurtosis, 6 / (df - 4), within :data:`TAIL_DF_BOUNDS`."""
    low, high = TAIL_DF_BOUNDS
    if excess_kurtosis <= 6.0 / (high - 4.0):
        return high
    return float(min(high, max(low, 4.0 + 6.0 / excess_kurtosis)))


def _cap_communality(loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Loadings shrunk to leave at least :data:`IDIOSYNCRATIC_FLOOR`, and the idiosyncratic share for unit variance."""
    communality = (loadings**2).sum(axis=1)
    loadings = loadings * np.sqrt(np.minimum(1.0, (1.0 - IDIOSYNCRATIC_FLOOR) / communality))[:, None]
    return loadings, 1.0 - (loadings**2).sum(axis=1)


def fit_factor_model(returns: np.ndarray, sigma: np.ndarray, factors: int) -> FactorModel:
    """Principal-axis factors of ``sigma``'s correlation, with Bartlett scores for every hour of ``returns``.

    ``sigma`` is the covariance estimated from ``returns``, so the loadings
    follow the production estimator. Principal-axis factoring starts from the
    squared multiple correlations and refits the reduced correlation until the
    communalities settle, so the idiosyncratic variances are fitted rather than
    left over from principal components. Each hour's scores are a weighted
    least squares fit over the assets that traded in it. The tail comes from the
    pooled excess kurtosis of the standardized residuals.
    """
    observed = ~np.isnan(returns)
    mean = np.nanmean(returns, axis=0)
    vol = np.sqrt(np.diagonal(sigma))
    corr = correlation(sigma)
    communality = np.clip(1.0 - 1.0 / np.diagonal(np.linalg.inv(corr)), 0.0, 1.0 - IDIOSYNCRATIC_FLOOR)
    loadings = np.zeros((corr.shape[0], factors))
    for _ in range(PRINCIPAL_AXIS_ITERATIONS):
        reduced = corr.copy()
        np.fill_diagonal(reduced, communality)
        values, vectors = np.linalg.eigh(reduced)
        top = np.argsort(values)[::-1][:factors]
        loadings, idiosyncratic = _cap_communality(vectors[:, top] * np.sqrt(np.clip(values[top], 0.0, None)))
        settled = np.abs(1.0 - idiosyncratic - communality).max() < 1e-8
        communality = 1.0 - idiosyncratic
        if settled:
            break
    loadings, idiosyncratic = _cap_communality(loadings)
    standardized = np.where(observed, (returns - mean) / vol, 0.0)
    scores = np.zeros((returns.shape[0], factors))
    residuals = []
    for hour in range(returns.shape[0]):
        traded = observed[hour]
        weighted = loadings[traded] / idiosyncratic[traded, None]
        scores[hour] = np.linalg.lstsq(weighted.T @ loadings[traded], weighted.T @ standardized[hour, traded],
                                       rcond=None)[0]
        fitted = loadings[traded] @ scores[hour]
        residuals.append((standardized[hour, traded] - fitted) / np.sqrt(idiosyncratic[traded]))
    pooled = np.concatenate(residuals)
    pooled = (pooled - pooled.mean()) / pooled.std()
    tail_df = t_tail_df(float((pooled**4).mean() - 3.0))
    return FactorModel(mean, vol, loadings, idiosyncratic, scores, observed, tail_df)


def factor_synthetic(model: FactorModel, rng: np.random.Generator, assets: int, rows: int, block: int,
                     loading_jitter: float, vol_jitter: float) -> tuple[np.ndarray, np.ndarray]:
    """``(returns, sources)``: ``assets`` synthetic assets over ``rows`` hours, and each one's source asset.

    Every synthetic asset copies a uniformly drawn real asset's loading row
    plus normal jitter of scale ``loading_jitter``, and its volatility times a
    lognormal jitter of scale ``vol_jitter``. Its idiosyncratic share makes the
    standardized variance 1, floored at :data:`IDIOSYNCRATIC_FLOOR`. The hours
    come from a random ``rows``-hour window, block-bootstrapped in ``block``-hour
    blocks; each asset keeps its source's missing hours in them.
    """
    sources = rng.integers(0, model.loadings.shape[0], size=assets)
    loadings = model.loadings[sources] + loading_jitter * rng.standard_normal((assets, model.loadings.shape[1]))
    loadings, idiosyncratic = _cap_communality(loadings)
    vol = model.vol[sources] * np.exp(vol_jitter * rng.standard_normal(assets))
    start, _ = sample_window(model.scores, rng, rows)
    hours = block_bootstrap(np.arange(start, start + rows)[:, None], rng, block)[:, 0]
    df = model.tail_df
    noise = rng.standard_t(df, size=(rows, assets)) * math.sqrt((df - 2.0) / df)
    standardized = model.scores[hours] @ loadings.T + noise * np.sqrt(idiosyncratic)
    returns = model.mean[sources] + vol * standardized
    returns[~model.observed[hours][:, sources]] = np.nan
    return returns, sources


def pair_overlaps(returns: np.ndarray) -> np.ndarray:
    """Hours in which both assets of each pair have a return; the diagonal is each asset's count."""
    observed = (~np.isnan(returns)).astype(np.int64)
    return observed.T @ observed


def significant_pairs(corr: np.ndarray, overlaps: np.ndarray, z: float = SIGNIFICANCE_Z) -> np.ndarray:
    """Pairs whose correlation is significant over that pair's own overlap (Fisher z).

    The backend estimates each correlation over the hours both assets traded,
    so a stock pair rests on far fewer hours than a crypto pair and needs a
    larger correlation to count. The diagonal is never significant.
    """
    clipped = np.clip(corr, -0.999999, 0.999999)
    score = np.abs(np.arctanh(clipped)) * np.sqrt(np.clip(overlaps - 3, 0, None))
    mask = score > z
    np.fill_diagonal(mask, False)
    return mask


def best_modularity(corr: np.ndarray, significant: np.ndarray) -> float:
    """Largest Newman modularity over the cuts of an average-linkage clustering.

    The graph weight is the positive correlation of each significant pair. A
    matrix with no significant positive pair has modularity 0.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    weights = np.where(significant & (corr > 0.0), corr, 0.0)
    total = weights.sum()
    n = corr.shape[0]
    if total <= 0.0 or n < 2:
        return 0.0
    degree = weights.sum(axis=1)
    distance = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    best = 0.0
    for groups in range(2, n + 1):
        labels = fcluster(tree, groups, criterion="maxclust")
        same = labels[:, None] == labels[None, :]
        best = max(best, float(((weights - np.outer(degree, degree) / total) * same).sum() / total))
    return best


def _selection_values(q: np.ndarray, k: int) -> np.ndarray | None:
    """QUBO value of every K-selection, or None beyond :data:`ENUMERATION_LIMIT`."""
    n = q.shape[0]
    if math.comb(n, k) > ENUMERATION_LIMIT:
        return None
    symmetric = (q + q.T) / 2.0
    values = []
    combinations = itertools.combinations(range(n), k)
    while True:
        chunk = np.fromiter(itertools.chain.from_iterable(itertools.islice(combinations, _CHUNK)), dtype=np.int64)
        if chunk.size == 0:
            break
        subsets = chunk.reshape(-1, k)
        values.append(symmetric[subsets[:, :, None], subsets[:, None, :]].sum(axis=(1, 2)))
    return np.concatenate(values)


def near_degenerate_count(q: np.ndarray, k: int, tolerance: float) -> int | None:
    """Exact count of K-selections whose QUBO value lies within ``tolerance`` of the best.

    ``tolerance`` is relative to the best value's magnitude. Returns None when
    the enumeration exceeds :data:`ENUMERATION_LIMIT`.
    """
    values = _selection_values(q, k)
    if values is None:
        return None
    best = values.min()
    return int((np.abs(values - best) <= tolerance * abs(best)).sum())


def second_best_gap(q: np.ndarray, k: int) -> float | None:
    """Relative gap between the best and second-best K-selection, or None beyond the limit."""
    values = _selection_values(q, k)
    if values is None or values.size < 2:
        return None
    first, second = np.partition(values, 1)[:2]
    return float((second - first) / abs(first))
