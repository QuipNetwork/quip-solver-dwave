"""Tests for the energy ranking in scripts/portfolio_diversity_controls.py."""

from __future__ import annotations

import importlib.util
import itertools
import pathlib

import numpy as np

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "portfolio_diversity_controls.py"
spec = importlib.util.spec_from_file_location("portfolio_diversity_controls", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def _brute_force_rank(q: np.ndarray, k: int, support: tuple[int, ...]) -> int:
    def energy(selection: tuple[int, ...]) -> float:
        x = np.zeros(q.shape[0])
        x[list(selection)] = 1.0
        return float(x @ q @ x)

    target = energy(support)
    return 1 + sum(energy(s) < target - 1e-12 for s in itertools.combinations(range(q.shape[0]), k))


def test_rank_matches_brute_force_on_an_upper_triangular_qubo():
    rng = np.random.default_rng(3)
    q = np.triu(rng.normal(size=(7, 7)))
    for support in itertools.combinations(range(7), 3):
        assert controls._energy_rank(q, 3, support) == _brute_force_rank(q, 3, support)


def test_the_qubo_minimum_has_rank_one():
    q = np.diag([5.0, -1.0, -2.0, 4.0])
    assert controls._energy_rank(q, 2, (1, 2)) == 1
    assert controls._energy_rank(q, 2, (0, 3)) == 6


def test_an_arm_times_sampling_and_finish_separately_and_counts_distinct_baskets(monkeypatch):
    bits = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.int8)
    monkeypatch.setattr(controls, "_finals", lambda _, rows: {(0, 2): 2.0, (1, 2): 1.5} if rows is bits else {})
    arm = controls._arm(None, lambda: bits)
    assert arm["objective"] == 1.5
    assert arm["distinct_supports"] == 2
    assert arm["sampling_s"] >= 0.0 and arm["finish_s"] >= 0.0
