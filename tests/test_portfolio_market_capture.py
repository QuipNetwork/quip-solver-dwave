"""Tests for scripts/portfolio_market_capture.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "portfolio_market_capture.py"
spec = importlib.util.spec_from_file_location("portfolio_market_capture", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def test_the_universe_is_every_asset_in_the_race_export(tmp_path):
    races = tmp_path / "races.json"
    races.write_text(json.dumps([{"assets": ["ETH", "BTC"]}, {"assets": ["BTC", "IONQ"]}]))
    assert capture.production_universe(races) == ["BTC", "ETH", "IONQ"]


def test_a_capture_missing_a_ticker_is_refused():
    body = {"bars": {"BTC": [{"t": "2026-09-26T16:00:00Z"}], "ETH": []}}
    with pytest.raises(ValueError, match="ETH, IONQ"):
        capture.bar_counts(body, ["BTC", "ETH", "IONQ"])


def test_bar_counts_are_per_ticker():
    body = {"bars": {"BTC": [{}, {}, {}], "IONQ": [{}]}}
    assert capture.bar_counts(body, ["BTC", "IONQ"]) == {"BTC": 3, "IONQ": 1}
