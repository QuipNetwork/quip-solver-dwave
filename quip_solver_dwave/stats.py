"""Small statistics for the measurement scripts. numpy only."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def ks_two_sample(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov: ``(D, p)`` from the asymptotic distribution.

    ``D`` is the largest gap between the two empirical distribution functions.
    The p-value is the Kolmogorov series with the small-sample correction of
    Numerical Recipes section 14.3, which is accurate from about 20 samples
    per arm.
    """
    xs, ys = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    grid = np.concatenate([xs, ys])
    cdf_x = np.searchsorted(xs, grid, side="right") / len(xs)
    cdf_y = np.searchsorted(ys, grid, side="right") / len(ys)
    d_stat = float(np.max(np.abs(cdf_x - cdf_y)))
    n_eff = len(xs) * len(ys) / (len(xs) + len(ys))
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * d_stat
    if lam < 1e-3:
        # The alternating series does not converge here, and its limit is 1.
        return d_stat, 1.0
    k = np.arange(1, 101)
    p_value = float(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2)))
    return d_stat, min(1.0, max(0.0, p_value))


def _mean_ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
    # Tied values share the mean of the ranks they span.
    _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return sums[inverse] / counts[inverse]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation, with mean ranks for ties."""
    ra, rb = _mean_ranks(a), _mean_ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])
