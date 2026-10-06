"""Energy gates for the cascade: the rule that sets them and the tables that judge them.

A gate keeps a model when its best energy is at or below the gate. Energies
are whole energy units here. Lower is better.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def derive_gates(trace: np.ndarray, deepest: int, margin: int) -> List[int]:
    """One gate per leg from winner traces.

    ``trace[seed, model, leg]`` holds the best energy at the end of each leg,
    with the models ordered deepest chain energy first. A gate is the
    shallowest best over every seed of the ``deepest`` first models, plus
    ``margin``. Every one of those runs therefore passes its own gate.
    """
    if trace.ndim != 3 or trace.shape[1] < deepest or deepest < 1:
        raise ValueError(f"trace of shape {trace.shape} cannot set gates from {deepest} models")
    return [int(value) + margin for value in trace[:, :deepest, :].max(axis=(0, 1))]


def kept_runs(trace: np.ndarray, gates: Sequence[int]) -> np.ndarray:
    """``kept[seed, model, leg]``: the run is still alive after gate ``leg``."""
    if trace.shape[2] != len(gates):
        raise ValueError(f"{trace.shape[2]} legs against {len(gates)} gates")
    return np.logical_and.accumulate(trace <= np.asarray(gates), axis=2)


def cut_for_miss(winner_best: np.ndarray, miss: float) -> int:
    """The deepest cut that loses at most the share ``miss`` of the winner runs."""
    runs = np.sort(np.asarray(winner_best).reshape(-1))[::-1]  # shallowest first
    if runs.size == 0:
        raise ValueError("no winner runs")
    return int(runs[min(int(miss * runs.size), runs.size - 1)])


def gate_table(winner_best: np.ndarray, control_best: np.ndarray, misses: Sequence[float]) -> List[Dict[str, float]]:
    """For each allowed miss share: the cut, the winner runs it loses, and the control share it passes."""
    winners = np.asarray(winner_best).reshape(-1)
    controls = np.asarray(control_best).reshape(-1)
    rows = []
    for miss in misses:
        cut = cut_for_miss(winners, miss)
        rows.append(
            {
                "miss_allowed": miss,
                "cut": cut,
                "winners_lost": int((winners > cut).sum()),
                "winner_miss": float((winners > cut).mean()),
                "control_pass": float((controls <= cut).mean()) if controls.size else float("nan"),
            }
        )
    return rows
