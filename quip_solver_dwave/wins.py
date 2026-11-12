"""Estimate the share of qblocks a pipeline wins from its measured clears."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def poisson_win_share(round_seconds: Sequence[float], clears_per_second: float, latency_seconds: float) -> float:
    """Mean chance that the first clear lands before each round ends.

    Clears that count toward a round of length ``D`` come from candidates
    found in its first ``D - latency`` seconds, as a Poisson process. With no
    round, or with no clear rate, the chance is 0.
    """
    if len(round_seconds) == 0 or clears_per_second <= 0:
        return 0.0
    usable = np.clip(np.asarray(round_seconds, dtype=float) - latency_seconds, 0.0, None)
    return float(np.mean(1.0 - np.exp(-clears_per_second * usable)))


def replay_win_share(
    clears: Sequence[Tuple[float, float]], run_seconds: float, round_seconds: Sequence[float], step_seconds: float = 10.0
) -> float:
    """Share of round placements that a recorded run wins.

    ``clears`` holds ``(found, cleared)`` times in run seconds: when the
    screen first saw the model, and when a read below the threshold came
    back. A round of length ``D`` placed at run time ``t`` is won when some
    clear has ``found >= t`` and ``cleared <= t + D``. Every round length is
    placed at every ``step_seconds`` offset that fits the run.
    """
    if not clears:
        return 0.0
    order = np.argsort([found for found, _ in clears])
    found = np.asarray([clears[i][0] for i in order], dtype=float)
    cleared = np.asarray([clears[i][1] for i in order], dtype=float)
    # earliest[i]: first clear time among the clears found at or after found[i].
    earliest = np.minimum.accumulate(cleared[::-1])[::-1]
    earliest = np.append(earliest, np.inf)
    wins = placements = 0
    for length in round_seconds:
        starts = np.arange(0.0, run_seconds - length + 1e-9, step_seconds)
        if starts.size == 0:
            continue
        first = np.searchsorted(found, starts, side="left")
        wins += int((earliest[first] <= starts + length).sum())
        placements += int(starts.size)
    return wins / placements if placements else 0.0


def equilibrium_share(clears_per_second: float, field_clears_per_second: float) -> float:
    """Share of clearing power: the long-run share once the difficulty adjusts."""
    total = clears_per_second + field_clears_per_second
    return clears_per_second / total if total > 0 else 0.0
