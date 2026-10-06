#!/usr/bin/env python3
"""Projections from the QPU-seeded cascade run, and the QPU capacity the cards leave idle.

Reads the sidecar journal of each card, the win estimate that ``cascade_wins.py``
wrote from them, and the chain's recent rounds. Writes ``projection.json`` and
three charts into ``--out``.

Two projections come from the same run. ``qpu-only`` counts a clear only when
the QPU gate kept the model: what the QPU adds by itself. ``pipeline`` counts
every clear of the pipeline as it runs, from any arm. Each gives the clear rate
with its 95% interval, the Poisson win share over the recent round lengths, the
equilibrium share, and wins per day. With few clears the interval is the
result, not the point estimate.

The call-rate ladder lists QPU calls per second: what the monthly budget
sustains, this run's pace cap, the achieved rate, the survivor flow to the QPU
arms, the flow of all survivors, the QPU's measured service rate, and the
access-time limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent))
# A sibling script, so it must follow the path insert. Pyright resolves it through extraPaths in pyproject.toml.
from round3_charts import GRAY, INK, LIGHT, QPU, save, style  # noqa: E402

QPU_ARMS: Set[str] = {"cold", "qpu-seeded"}
ALL_ARMS: Set[str] = {"gpu-256", "cold", "qpu-seeded"}
SURVIVOR_QPU_ARMS = ("qpu-cold", "qpu-seeded")
Clear = Tuple[float, float]
Leg = Tuple[str, str, float]  # program, arm, best energy


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def poisson_cdf(count: int, mean: float) -> float:
    """P(X <= count) for a Poisson mean."""
    return sum(math.exp(k * math.log(mean) - mean - math.lgamma(k + 1)) for k in range(count + 1)) if mean > 0 else 1.0


def poisson_interval(clears: int) -> Tuple[float, float]:
    """Garwood's exact 95% interval on the mean of a Poisson count, by bisection on the Poisson CDF.

    The lower bound is the mean at which a count of ``clears`` or more has probability 0.025, and the upper bound
    is the mean at which a count of ``clears`` or fewer has probability 0.025. A count of 0 has no lower bound.
    """

    def solve(target: float, count: int) -> float:
        low, high = 0.0, clears + 10.0 * math.sqrt(clears + 1.0) + 10.0
        for _ in range(200):
            mid = (low + high) / 2
            if poisson_cdf(count, mid) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    lower = solve(0.975, clears - 1) if clears > 0 else 0.0
    return lower, solve(0.025, clears)


def win_share(rate: float, latency_s: float, round_lengths: Sequence[float]) -> float:
    """Mean chance that a clear found in a round's first ``D - latency`` seconds lands before the round ends."""
    if rate <= 0.0 or not round_lengths:
        return 0.0
    usable = np.clip(np.asarray(round_lengths, dtype=float) - latency_s, 0.0, None)
    return float(np.mean(1.0 - np.exp(-rate * usable)))


def first_clears(events: Sequence[Mapping[str, Any]], arms: Set[str], gate_milli: int) -> Dict[str, Clear]:
    """``nonce -> (found, cleared)`` for the first clear of each model whose deep leg ran in ``arms``.

    A clear is a deep leg that was not dropped, with its best below the gate, on a model that a ``survivor``
    event found. A QPU-arm leg counts only when the QPU gate kept the model, as ``cascade_wins.py`` counts it.
    """
    found = {e["nonce"]: e["t"] for e in events if e["type"] == "survivor"}
    kept = {e["nonce"] for e in events if e["type"] == "qpu" and e["kept"]}
    first: Dict[str, Clear] = {}
    for event in events:
        if event["type"] != "deep" or event["arm"] not in arms or event["nonce"] not in found:
            continue
        if event["status"] == "dropped" or event["best"] * 1000 >= gate_milli:
            continue
        if event["arm"] in QPU_ARMS and event["nonce"] not in kept:
            continue
        nonce = event["nonce"]
        if nonce not in first or event["t"] < first[nonce][1]:
            first[nonce] = (found[nonce], event["t"])
    return first


def project(clears: Mapping[str, Clear], run_seconds: float, round_lengths: Sequence[float],
            latency_default: float) -> Dict[str, Any]:
    """Point and 95% interval rows for one scenario."""
    count = len(clears)
    latencies = sorted(cleared - found for found, cleared in clears.values())
    latency = latencies[len(latencies) // 2] if latencies else latency_default
    mean_round = float(np.mean(round_lengths))
    field_rate = 1.0 / mean_round

    def row(clear_count: float) -> Dict[str, float]:
        rate = clear_count / run_seconds if run_seconds else 0.0
        share = win_share(rate, latency, round_lengths)
        return {"clears": clear_count, "clears_per_hour": rate * 3600.0, "win_share": share,
                "equilibrium_share": rate / (rate + field_rate) if rate + field_rate else 0.0,
                "wins_per_day": share * 86400.0 / mean_round}

    low, high = poisson_interval(count)
    return {"clears": count, "latency_s": latency, "point": row(count), "low": row(low), "high": row(high)}


def last_stats_per_launch(events: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The last ``stats`` event of each launch. A launch with no ``stats`` contributes nothing."""
    result: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for event in events:
        if event["type"] == "start":
            if current is not None:
                result.append(current)
            current = None
        elif event["type"] == "stats":
            current = dict(event)
    if current is not None:
        result.append(current)
    return result


def last_start(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return [e for e in events if e["type"] == "start"][-1]


def utilization(events_by_card: Mapping[str, Sequence[Mapping[str, Any]]], run_seconds: float,
                service_rate: float, monthly_seconds: float) -> Dict[str, Any]:
    """QPU counts, the call-rate ladder, the idle share, and the cards that would fill the service rate."""
    events = [e for card in events_by_card.values() for e in card]
    calls = [e for e in events if e["type"] == "qpu"]
    access_s = float(np.median([e["access_us"] for e in calls])) / 1e6 if calls else 0.05
    survivors = Counter(e["arm"] for e in events if e["type"] == "survivor")
    gpu_legs = [e for e in events if e["type"] == "deep" and e["arm"] == "gpu-256"]
    gpu_passed = sum(1 for e in gpu_legs if e["status"] != "dropped")
    finished = Counter(e["program"] for e in events if e["type"] == "deep" and e["status"] != "dropped")
    cap_seconds = sum(last_start(card)["qpu_seconds"] for card in events_by_card.values())
    window_s = max(last_start(card)["window_s"] for card in events_by_card.values())
    achieved = len(calls) / run_seconds
    all_flow = sum(survivors.values()) / run_seconds
    ladder = {
        "budget, 32,400 s per month": monthly_seconds / (30 * 86400.0) / access_s,
        "pace cap of this run": cap_seconds / window_s / access_s,
        "achieved": achieved,
        "survivors sent to the QPU arms": sum(survivors[a] for a in SURVIVOR_QPU_ARMS) / run_seconds,
        "all survivors, both cards": all_flow,
        "QPU service rate, 16 in flight": service_rate,
        "QPU access limit": 1.0 / access_s,
    }
    per_card = all_flow / len(events_by_card)
    kept = sum(1 for e in calls if e["kept"])
    return {
        "qpu": {"calls": len(calls), "kept": kept, "keep_share": kept / len(calls) if calls else 0.0,
                "skips": sum(1 for e in events if e["type"] == "qpu_skip"), "access_s_per_call": access_s,
                "gpu_256_legs": len(gpu_legs), "gpu_256_passed": gpu_passed,
                "gpu_256_pass_share": gpu_passed / len(gpu_legs) if gpu_legs else 0.0,
                "finished_legs_by_program": dict(sorted(finished.items()))},
        "ladder": ladder,
        "idle_share_of_service": 1.0 - achieved / service_rate,
        "cards_to_fill_service": service_rate / per_card if per_card else math.inf,
    }


def funnel(events_by_card: Mapping[str, Sequence[Mapping[str, Any]]], wins: Mapping[str, Any],
           clears: int) -> List[Tuple[str, int]]:
    """Counts at each stage, summed over the last stats of every launch of every card."""
    passed = [0, 0, 0]
    for card in events_by_card.values():
        for stats in last_stats_per_launch(card):
            for index in range(3):
                passed[index] += stats["passed"][index]
    events = [e for card in events_by_card.values() for e in card]
    calls = [e for e in events if e["type"] == "qpu"]
    finished = [e for e in events if e["type"] == "deep" and e["status"] != "dropped"]
    # The pipeline branches at the handoff: the gpu-256 arm never calls the QPU, so its legs are listed apart.
    return [("salts", int(wins["salts"])), ("after 8 sweeps", passed[0]), ("after 16 sweeps", passed[1]),
            ("after 64 sweeps", passed[2]), ("QPU calls", len(calls)),
            ("QPU kept", sum(1 for e in calls if e["kept"])),
            ("finished deep legs, QPU arms", sum(1 for e in finished if e["arm"] in QPU_ARMS)),
            ("finished deep legs, gpu-256", sum(1 for e in finished if e["arm"] == "gpu-256")), ("clears", clears)]


def chart_funnel(stages: Sequence[Tuple[str, int]]) -> Figure:
    """Salts to clears, on a log scale, with the count at each stage."""
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(stages) + 1))
    y = list(range(len(stages)))
    counts = [max(count, 0.5) for _, count in stages]
    ax.barh(y, counts, color=[QPU if "QPU" in label else GRAY for label, _ in stages], height=0.6)
    for yy, (label, count) in zip(y, stages):
        ax.text(max(count, 0.5), yy, f" {count:,}", va="center", fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_yticks(y, [label for label, _ in stages], fontsize=8)
    ax.invert_yaxis()
    style(ax, xlabel="count, log scale")
    return fig


def chart_deep_energy(legs: Sequence[Leg], gate_milli: int) -> Figure:
    """Best energy of every finished deep leg, by program, against the chain threshold. QPU-arm legs are blue."""
    programs = sorted({program for program, _, _ in legs})
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(programs) + 1))
    rng = np.random.default_rng(0)
    for yy, program in enumerate(programs):
        bests = [best for name, _, best in legs if name == program]
        arms = {arm for name, arm, _ in legs if name == program}
        ax.plot(bests, yy + rng.uniform(-0.15, 0.15, len(bests)), "o", ms=3, alpha=0.5, ls="none",
                color=QPU if arms & QPU_ARMS else GRAY, label=f"program:{program}")
    ax.axvline(gate_milli / 1000.0, color=INK, lw=0.8, ls="--")
    ax.annotate("chain threshold", (gate_milli / 1000.0, -0.7), xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=INK, ha="left", va="center")
    ax.set_yticks(list(range(len(programs))), programs, fontsize=8)
    ax.set_ylim(len(programs) - 0.5, -1.1)  # inverted, with room above the first row for the threshold label
    style(ax, xlabel="best energy of the leg (lower is better)")
    return fig


def chart_ladder(ladder: Mapping[str, float]) -> Figure:
    """QPU calls per second at each rung. The achieved rate is in the accent color."""
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(ladder) + 1))
    labels = list(ladder)
    y = list(range(len(labels)))
    ax.barh(y, [ladder[label] for label in labels], height=0.6,
            color=[QPU if label == "achieved" else LIGHT for label in labels])
    for yy, label in zip(y, labels):
        ax.text(ladder[label], yy, f" {ladder[label]:.2f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    style(ax, xlabel="QPU calls per second")
    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--events", required=True, action="extend", nargs="+", help="events.jsonl of each card")
    parser.add_argument("--wins", required=True, help="output of cascade_wins.py on the same journals")
    parser.add_argument("--rounds", required=True, help="rounds.json from chain_fetch.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--service-rate", type=float, default=5.28,
                        help="QPU calls per second the throughput test reached at 16 in flight")
    parser.add_argument("--monthly-seconds", type=float, default=32400.0, help="the monthly QPU allotment")
    args = parser.parse_args(argv)
    wins = json.loads(Path(args.wins).read_text(encoding="utf-8"))
    rounds = json.loads(Path(args.rounds).read_text(encoding="utf-8"))
    lengths = [row["blocks"] * rounds["block_seconds"] for row in rounds["rounds"]]
    events_by_card = {Path(p).parent.name: read_jsonl(Path(p)) for p in dict.fromkeys(args.events)}
    events = sorted((e for card in events_by_card.values() for e in card), key=lambda e: e["t"])
    gate = int(wins["gate_milli"])
    run_seconds = float(wins["run_seconds"])

    def fallback_latency(arms: Iterable[str]) -> float:
        wanted = set(arms)
        seconds = sorted(e["seconds"] for e in events if e["type"] == "deep" and e["arm"] in wanted
                         and e["status"] != "dropped")
        return seconds[len(seconds) // 2] if seconds else 0.0

    scenarios = {}
    for name, arms in (("qpu-only", QPU_ARMS), ("pipeline", ALL_ARMS)):
        clears = first_clears(events, arms, gate)
        scenarios[name] = {"arms": sorted(arms), **project(clears, run_seconds, lengths, fallback_latency(arms))}
    use = utilization(events_by_card, run_seconds, args.service_rate, args.monthly_seconds)
    stages = funnel(events_by_card, wins, scenarios["pipeline"]["clears"])
    summary = {"gate_milli": gate, "run_seconds": run_seconds, "salts": wins["salts"],
               "qpu_access_s": wins["qpu_access_s"], "rounds": len(lengths), "mean_round_s": float(np.mean(lengths)),
               "median_round_s": sorted(lengths)[len(lengths) // 2], "scenarios": scenarios, **use,
               "funnel": [[label, count] for label, count in stages]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "projection.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    legs = [(e["program"], e["arm"], float(e["best"])) for e in events
            if e["type"] == "deep" and e["status"] != "dropped"]
    save(chart_funnel(stages), args.out / "funnel.svg")
    save(chart_deep_energy(legs, gate), args.out / "deep-energy.svg")
    save(chart_ladder(use["ladder"]), args.out / "qpu-ladder.svg")
    for name, scenario in scenarios.items():
        point, high = scenario["point"], scenario["high"]
        print(f"{name:>9}: {scenario['clears']} clears, win share {100 * point['win_share']:.1f}% "
              f"(95% upper {100 * high['win_share']:.1f}%), {point['wins_per_day']:.1f} wins per day "
              f"(upper {high['wins_per_day']:.1f})")
    print(f"QPU: {use['qpu']['calls']} calls, kept {100 * use['qpu']['keep_share']:.2f}%, "
          f"achieved {use['ladder']['achieved']:.2f} calls/s, "
          f"idle share of service {100 * use['idle_share_of_service']:.0f}%, "
          f"cards to fill the service rate {use['cards_to_fill_service']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
