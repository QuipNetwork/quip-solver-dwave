"""Cascade projection: Poisson intervals, win shares, the clears of each scenario, and the call-rate ladder."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # cascade_projection imports round3_charts as a sibling


def _load_module():
    spec = importlib.util.spec_from_file_location("cascade_projection", SCRIPTS / "cascade_projection.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


projection = _load_module()
GATE = -14_700_000


def _events():
    """One card: two launches, the second with no stats; a kept and a dropped QPU model; an orphan deep event."""
    return [
        {"type": "start", "t": 0.0, "qpu_seconds": 7200.0, "window_s": 200000.0, "arms": [["gpu-256", 1]]},
        {"type": "survivor", "t": 1.0, "nonce": "g1", "arm": "gpu-256"},
        {"type": "deep", "t": 5.0, "nonce": "g1", "arm": "gpu-256", "program": "gpu-256", "status": "passed",
         "best": -14701, "seconds": 4.0},
        {"type": "survivor", "t": 2.0, "nonce": "q1", "arm": "qpu-seeded"},
        {"type": "qpu", "t": 3.0, "nonce": "q1", "arm": "qpu-seeded", "kept": True, "access_us": 50000, "reads": 32},
        {"type": "deep", "t": 9.0, "nonce": "q1", "arm": "qpu-seeded", "program": "seeded", "status": "passed",
         "best": -14650, "seconds": 6.0},
        {"type": "survivor", "t": 2.5, "nonce": "q2", "arm": "qpu-cold"},
        {"type": "qpu", "t": 3.5, "nonce": "q2", "arm": "qpu-cold", "kept": False, "access_us": 50000, "reads": 32},
        {"type": "deep", "t": 4.0, "nonce": "q2", "arm": "gpu-256", "program": "gpu-256", "status": "dropped",
         "best": -14400, "seconds": 0.02},
        {"type": "qpu_skip", "t": 3.6, "nonce": "q3", "arm": "qpu-cold", "reason": "pace"},
        {"type": "deep", "t": 7.0, "nonce": "orphan", "arm": "gpu-256", "program": "gpu-256", "status": "passed",
         "best": -14800, "seconds": 4.0},
        {"type": "stats", "t": 10.0, "salts": 1000, "passed": [100, 10, 3], "survivors": 3, "deep_done": 2, "drops": 1},
        {"type": "end", "t": 11.0},
        {"type": "start", "t": 12.0, "qpu_seconds": 15400.0, "window_s": 200000.0, "arms": [["qpu-seeded", 1]]},
    ]


def test_poisson_interval_is_exact_at_every_count():
    assert projection.poisson_interval(0) == pytest.approx((0.0, 3.689), abs=1e-3)
    assert projection.poisson_interval(1) == pytest.approx((0.0253, 5.572), abs=1e-3)
    assert projection.poisson_interval(6) == pytest.approx((2.2019, 13.0595), abs=1e-3)  # not the normal 1.2, 10.8
    assert projection.poisson_interval(100) == pytest.approx((81.36, 121.63), abs=1e-2)


def test_win_share_is_zero_without_a_rate_and_rises_with_it():
    assert projection.win_share(0.0, 10.0, [400.0, 500.0]) == 0.0
    assert 0.0 < projection.win_share(1 / 1000, 10.0, [400.0, 500.0]) < projection.win_share(1 / 100, 10.0, [400.0, 500.0])


def test_first_clears_counts_qpu_arms_only_when_kept_and_ignores_orphans():
    events = _events()
    assert projection.first_clears(events, {"gpu-256"}, GATE) == {"g1": (1.0, 5.0)}
    assert projection.first_clears(events, {"cold", "qpu-seeded"}, GATE) == {}
    assert projection.first_clears(events, {"gpu-256", "cold", "qpu-seeded"}, GATE) == {"g1": (1.0, 5.0)}


def test_project_with_zero_clears_uses_the_interval_and_the_fallback_latency():
    result = projection.project({}, run_seconds=3600.0, round_lengths=[400.0, 500.0], latency_default=120.0)
    assert result["clears"] == 0 and result["latency_s"] == 120.0
    assert result["point"]["win_share"] == 0.0 and result["low"]["win_share"] == 0.0
    assert result["high"]["clears"] == pytest.approx(3.689, abs=1e-3) and result["high"]["win_share"] > 0.0


def test_last_stats_per_launch_skips_a_launch_without_stats():
    stats = projection.last_stats_per_launch(_events())
    assert [s["salts"] for s in stats] == [1000]


def test_utilization_ladder_and_idle_share():
    result = projection.utilization({"card0": _events()}, run_seconds=100.0, service_rate=5.0, monthly_seconds=32400.0)
    ladder = result["ladder"]
    assert ladder["achieved"] == pytest.approx(2 / 100)
    assert ladder["QPU access limit"] == pytest.approx(20.0)
    assert ladder["pace cap of this run"] == pytest.approx(15400.0 / 200000.0 / 0.05)
    assert ladder["budget, 32,400 s per month"] == pytest.approx(32400.0 / (30 * 86400) / 0.05)
    assert result["idle_share_of_service"] == pytest.approx(1 - 0.02 / 5.0)
    assert result["qpu"]["kept"] == 1 and result["qpu"]["skips"] == 1 and result["qpu"]["gpu_256_passed"] == 2


def test_main_writes_projection_json_and_three_charts(tmp_path):
    card = tmp_path / "card0"
    card.mkdir()
    (card / "events.jsonl").write_text("\n".join(json.dumps(e) for e in _events()), encoding="utf-8")
    wins = {"gate_milli": GATE, "run_seconds": 100.0, "salts": 1000, "qpu_access_s": 0.1, "qpu_skips": 1,
            "survivors": {"gpu-256": 1, "qpu-cold": 1, "qpu-seeded": 1}, "rounds": 2, "median_round_s": 450.0,
            "arms": {"gpu": {"clears": 1, "median_find_to_clear_s": 4.0},
                     "qpu-cold": {"clears": 0, "median_find_to_clear_s": None},
                     "qpu-seeded": {"clears": 0, "median_find_to_clear_s": None}}}
    (tmp_path / "wins.json").write_text(json.dumps(wins), encoding="utf-8")
    rounds = {"block_seconds": 6.0, "rounds": [{"blocks": 70, "gate_milli": GATE}, {"blocks": 80, "gate_milli": GATE}]}
    (tmp_path / "rounds.json").write_text(json.dumps(rounds), encoding="utf-8")
    out = tmp_path / "out"
    assert projection.main(["--events", str(card / "events.jsonl"), "--wins", str(tmp_path / "wins.json"),
                            "--rounds", str(tmp_path / "rounds.json"), "--out", str(out)]) == 0
    written = json.loads((out / "projection.json").read_text(encoding="utf-8"))
    assert written["scenarios"]["pipeline"]["clears"] == 1 and written["scenarios"]["qpu-only"]["clears"] == 0
    assert written["funnel"][0] == ["salts", 1000] and written["funnel"][-1] == ["clears", 1]
    assert ["finished deep legs, QPU arms", 1] in written["funnel"]
    assert ["finished deep legs, gpu-256", 2] in written["funnel"]
    for name in ("funnel.svg", "deep-energy.svg", "qpu-ladder.svg"):
        assert (out / name).exists()


def test_deep_energy_colors_every_qpu_arm_leg_in_the_accent_color():
    legs = [("gpu-256", "gpu-256", -14650.0), ("chain-16x131072-b0250", "qpu-seeded", -14660.0),
            ("cold-1048576", "cold", -14670.0)]
    fig = projection.chart_deep_energy(legs, GATE)
    colors = {ln.get_label(): ln.get_color() for ln in fig.axes[0].get_lines() if ln.get_label().startswith("program:")}
    assert colors["program:chain-16x131072-b0250"] == projection.QPU
    assert colors["program:cold-1048576"] == projection.QPU
    assert colors["program:gpu-256"] == projection.GRAY
