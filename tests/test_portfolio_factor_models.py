"""Tests for the pure draw planning in scripts/portfolio_factor_models.py."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # the script imports portfolio_family_models as a sibling
spec = importlib.util.spec_from_file_location("portfolio_factor_models", SCRIPTS / "portfolio_factor_models.py")
assert spec is not None and spec.loader is not None
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

RACES = [
    {"job_id": "a", "assets": ["X"] * 15, "sliders": {"riskPreference": 10.0, "holdCount": 5}},
    {"job_id": "b", "assets": ["X"] * 28, "sliders": {"riskPreference": 90.0, "holdCount": 9}},
]


def test_a_size_is_n_colon_k():
    assert models.parse_size("44:14") == (44, 14)
    for text in ("44", "44:2", "10:11", "a:b"):
        with pytest.raises(argparse.ArgumentTypeError):
            models.parse_size(text)


def test_draws_are_reproducible_and_numbered_across_sizes():
    first = models.plan_draws(RACES, [(32, 10), (44, 14)], 5, seed=3)
    assert first == models.plan_draws(RACES, [(32, 10), (44, 14)], 5, seed=3)
    assert first != models.plan_draws(RACES, [(32, 10), (44, 14)], 5, seed=4)
    assert [d["index"] for d in first] == list(range(10))
    assert [d["n_assets"] for d in first] == [32] * 5 + [44] * 5


def test_each_draw_takes_its_race_sliders_with_the_requested_hold_count():
    for draw in models.plan_draws(RACES, [(44, 14)], 20, seed=5):
        race = next(r for r in RACES if r["job_id"] == draw["race_job_id"])
        assert draw["sliders"] == {**race["sliders"], "holdCount": 14}
        assert draw["requested_k"] == 14
