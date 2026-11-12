"""Metrics of the regime search. Every metric compares arms inside one cell.

Energies are lower-is-better. They are whole milli units for four cells and
floats for the clique cell, and nothing here depends on which.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binomtest, rankdata

# 1.4826 times the median absolute deviation estimates one standard deviation
# of a normal distribution.
MAD_TO_SIGMA = 1.4826

# The screening rule (specification section 9.5).
TIME_FACTOR = 10.0
UNMATCHED_THRESHOLD = 0.25
BY_SWEEPS = "matching sweeps"
BY_KERNEL_COST = "kernel cost alone"
BY_UNMATCHED = "unmatched share"


def match_sweeps(ladder: Dict[int, float], qpu_best: float) -> Optional[float]:
    """Sweeps at which the classical best reaches ``qpu_best``, or None.

    The default rule: the first rung, in ascending sweeps, whose best energy is
    at or below the QPU best energy. The maintainer chose this rule on
    2026-09-20 (specification, section 9.1). Each rung is its own run, so the
    ladder is not monotone, and one lucky run can match early.

    None marks an unmatched model, which is a model that no rung matches. The
    ladder, the report, and the screening rule all take that one definition
    from this function (specification, sections 8.2 and 9.5).
    """
    for sweeps in sorted(ladder):
        if ladder[sweeps] <= qpu_best:
            return float(sweeps)
    return None


def matching_time(sweeps: Optional[float], rung_wall_s: Dict[int, float]) -> Optional[float]:
    """Core-seconds of the matched rung: the median wall time of that rung in the cell."""
    if sweeps is None:
        return None
    return rung_wall_s[int(sweeps)]


def gate_window(best_lite: ArrayLike, best_deep: ArrayLike, qpu_best: ArrayLike) -> Dict[str, float]:
    """The room for a gate between a cheap solver and the deep classical level.

    ``window`` is the gap between the two medians in robust standard
    deviations of the deep energies. ``qpu_position`` places the median QPU
    best energy on that gap: 0 is the lite level and 1 is the deep level.
    ``window`` is not a number when the deep energies have no spread.
    """
    deep_values = np.asarray(best_deep, dtype=np.float64)
    lite = float(np.median(np.asarray(best_lite, dtype=np.float64)))
    deep = float(np.median(deep_values))
    qpu = float(np.median(np.asarray(qpu_best, dtype=np.float64)))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(deep_values - deep)))
    gap = lite - deep
    return {
        "e_lite": lite,
        "e_deep": deep,
        "e_qpu": qpu,
        "spread": spread,
        "window": gap / spread if spread > 0 else float("nan"),
        "qpu_position": (lite - qpu) / gap if gap != 0 else float("nan"),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b) / denominator if denominator > 0 else float("nan")


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    return _pearson(rankdata(x), rankdata(y))


def _residual(ranks: np.ndarray, design: np.ndarray) -> Optional[np.ndarray]:
    """What a straight-line fit on the control ranks leaves of ``ranks``, or None when it leaves nothing.

    A fit that explains a whole series leaves only rounding noise, and a correlation of noise means nothing.
    """
    residual = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
    return residual if residual @ residual > 1e-12 * (ranks @ ranks) else None


def partial_spearman(x: ArrayLike, y: ArrayLike, control: ArrayLike) -> float:
    """Rank correlation of ``x`` with ``y`` with ``control`` held fixed.

    Fit each rank series on the control ranks with a straight line, then
    correlate the residuals. The result is not a number when the fit leaves
    nothing of a series, as it does for a constant series and for a series
    with the ranks of the control.
    """
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(control)
    design = np.column_stack([np.ones_like(rc), rc])
    residual_x, residual_y = _residual(rx, design), _residual(ry, design)
    if residual_x is None or residual_y is None:
        return float("nan")
    return _pearson(residual_x, residual_y)


def bootstrap_interval(
    statistic: Callable[..., float], *series: ArrayLike, resamples: int = 2000, seed: int = 0
) -> Tuple[float, float]:
    """The 95% interval of ``statistic`` over resamples of the models."""
    data = [np.asarray(s, dtype=np.float64) for s in series]
    rng = np.random.default_rng(seed)
    count = len(data[0])
    values = []
    for _ in range(resamples):
        pick = rng.integers(0, count, count)
        value = statistic(*(s[pick] for s in data))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def wilson_interval(count: int, total: int) -> Tuple[float, float]:
    """The 95% Wilson interval of the share ``count / total``.

    The interval stays inside 0 to 1, and it keeps a width at a share of 0. That is the expected unmatched share of
    the control, and a bootstrap interval has no width there.
    """
    if total <= 0:
        return float("nan"), float("nan")
    interval = binomtest(count, total).proportion_ci(confidence_level=0.95, method="wilson")
    return float(interval.low), float(interval.high)


def _at_least(value: Optional[float], baseline: Optional[float], factor: float) -> bool:
    return value is not None and baseline is not None and baseline > 0 and value >= factor * baseline


def time_condition(
    median_time: Optional[float],
    baseline_median_time: Optional[float],
    median_sweeps: Optional[float],
    baseline_median_sweeps: Optional[float],
) -> Optional[str]:
    """The first condition of the screening rule: 10 times the baseline matching time, or more.

    Returns None when the condition does not hold. Otherwise the result names what made the time. It is
    ``matching sweeps`` when the matching sweeps are also 10 times the baseline or more. It is ``kernel cost alone``
    when they are not, because the cost of a sweep then makes the difference (specification section 9.5).
    """
    if not _at_least(median_time, baseline_median_time, TIME_FACTOR):
        return None
    return BY_SWEEPS if _at_least(median_sweeps, baseline_median_sweeps, TIME_FACTOR) else BY_KERNEL_COST


def share_condition(unmatched: int, models: int) -> Tuple[Optional[str], bool]:
    """The second condition of the screening rule: an unmatched share of 25% or more.

    Returns ``unmatched share`` when the condition holds and None when it does not, and whether that verdict is
    uncertain. It is uncertain when the 95% Wilson interval of the share spans 25%, because the interval then holds
    shares on both sides of the threshold.
    """
    if models <= 0:
        return None, False
    low, high = wilson_interval(unmatched, models)
    holds = unmatched >= UNMATCHED_THRESHOLD * models
    return BY_UNMATCHED if holds else None, low < UNMATCHED_THRESHOLD <= high
