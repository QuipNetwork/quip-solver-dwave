"""Tests for the pure draw planning in scripts/portfolio_family_models.py."""

from __future__ import annotations

import importlib.util
import pathlib

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "portfolio_family_models.py"
spec = importlib.util.spec_from_file_location("portfolio_family_models", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

UNIVERSE = [f"T{i:02d}" for i in range(28)]
RACES = [
    {"job_id": "small", "assets": UNIVERSE[:15], "sliders": {"holdCount": 5}},
    {"job_id": "large", "assets": UNIVERSE, "sliders": {"holdCount": 9}},
]


def test_draws_are_reproducible_from_the_seed():
    first = models.plan_draws(RACES, UNIVERSE, 20, seed=7, history_rows=2159)
    assert first == models.plan_draws(RACES, UNIVERSE, 20, seed=7, history_rows=2159)
    assert first != models.plan_draws(RACES, UNIVERSE, 20, seed=8, history_rows=2159)


def test_each_draw_takes_its_basket_size_and_sliders_from_one_race():
    for draw in models.plan_draws(RACES, UNIVERSE, 50, seed=1, history_rows=2159):
        race = next(r for r in RACES if r["job_id"] == draw["race_job_id"])
        assert draw["n_assets"] == len(race["assets"]) == len(set(draw["tickers"]))
        assert draw["sliders"] == race["sliders"]
        assert set(draw["tickers"]) <= set(UNIVERSE)


def test_every_window_fits_inside_the_history():
    draws = models.plan_draws(RACES, UNIVERSE, 200, seed=2, history_rows=800)
    assert all(0 <= d["window_start"] <= 800 - models.WINDOW_HOURS for d in draws)
    assert len({d["window_start"] for d in draws}) > 1
