#!/usr/bin/env python3
"""Turn pipeline runs and the chain's recent rounds into a win estimate for each arm.

A clear is a deep leg that was not dropped and whose best energy is below the
chain threshold. Its find time is when the screen first reported the model. The estimate places
every recent round length at every offset of the run and counts the rounds in
which a clear was found and finished inside the round.

Each --events file is the journal of one card. Its ``t`` counts from the
``started_at`` of its first ``start`` event, so the journals share one clock.
The run is the time in which at least one card was up.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from quip_miner_dwave import regime_io, wins

# The ``arm`` of the ``deep`` events that each arm counts as its deep legs.
ARMS = {
    "gpu": {"gpu-256"},
    "qpu-cold": {"cold"},
    "qpu-seeded": {"qpu-seeded"},
}


class Launch(NamedTuple):
    """One launch of a sidecar: its span in absolute seconds, and the last ``salts`` count of its miner."""

    start: float
    end: float
    salts: int


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """One JSON value for each line. A line that does not parse raises, because a skipped line of a journal could
    be a ``start``."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def ends_launch(event: Dict[str, Any]) -> bool:
    """An ``end``, or an ``error`` of the launch itself. The sidecar writes ``OutDirBusy`` for a second launch that
    gave up on the lock of the out-dir, while the first launch still runs."""
    return event["type"] == "end" or (event["type"] == "error" and not event["message"].startswith("OutDirBusy"))


def read_launches(events: Sequence[Dict[str, Any]]) -> List[Launch]:
    """A launch runs from a ``start`` event to the next ``end`` or ``error``. One that a later ``start`` finds open
    stops at the event before it, and one that the journal leaves open stops at its last event. ``salts`` is the
    count in the last ``stats`` event of the launch, because the miner counts from its own launch."""
    launches: List[Launch] = []
    start: Optional[float] = None
    salts, last = 0, 0.0
    for event in events:
        kind = event["type"]
        if kind == "start":
            if start is not None:
                launches.append(Launch(start, last, salts))
            start, salts = event["t"], 0
        elif start is not None and kind == "stats":
            salts = event["salts"]
        elif start is not None and ends_launch(event):
            launches.append(Launch(start, event["t"], salts))
            start = None
        last = event["t"]
    if start is not None:
        launches.append(Launch(start, last, salts))
    return launches


def read_journal(path: str) -> Tuple[List[Dict[str, Any]], List[Launch]]:
    """The events of one journal with ``t`` in absolute seconds, and its launches."""
    events = read_jsonl(path)
    origin = next((datetime.fromisoformat(e["started_at"]).timestamp() for e in events if e["type"] == "start"), None)
    if origin is None:
        raise ValueError(f"{path} holds no start event, so its times have no origin")
    events = [{**event, "t": origin + event["t"]} for event in events]
    return events, read_launches(events)


def union(spans: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """The spans merged into disjoint ones, in order."""
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def run_time(t: float, run: Sequence[Tuple[float, float]]) -> float:
    """Seconds of ``run`` before the absolute time ``t``: the clock of the run, which has no gaps."""
    return sum(max(0.0, min(t, end) - start) for start, end in run)


def arm_clears(events: Sequence[Dict[str, Any]], arm: str, gate_milli: int) -> List[Tuple[float, float]]:
    """``(found, cleared)`` seconds of the first clear of each model in ``arm``.

    A deep program that ends below the threshold clears, unless it ended ``dropped``: a gate stopped it, or its lease
    ended before it pushed. The miner ends a program as ``hit`` when the live target of the round admits its best so
    far, and as ``passed`` after its last leg; one ``deep`` event reports the whole program, with the best that it
    pushed, at the time of the push. A hit above the threshold stopped early at a live target looser than the
    threshold, so its best is not final and it does not count. The ``gpu-256`` program has two legs, the gated leg
    and then the cold leg, and its one event is the clear of the ``gpu`` arm. A QPU survivor that the sidecar skipped
    runs on ``gpu-256`` and counts there. The ``qpu-cold`` arm counts its cold program, and the ``qpu-seeded`` arm
    its seeded program, a single leg or a chain of legs, only when the QPU gate kept the model.
    """
    found = {e["nonce"]: e["t"] for e in events if e["type"] == "survivor"}
    qpu_kept = {e["nonce"] for e in events if e["type"] == "qpu" and e["kept"]}
    first: Dict[str, float] = {}
    for event in events:
        if event["type"] != "deep" or event["arm"] not in ARMS[arm] or event["nonce"] not in found:
            continue
        eligible = arm == "gpu" or event["nonce"] in qpu_kept
        if eligible and event["status"] != "dropped" and event["best"] * 1000 < gate_milli:
            first[event["nonce"]] = min(first.get(event["nonce"], float("inf")), event["t"])
    return sorted((found[nonce], cleared) for nonce, cleared in first.items())


def qpu_access_seconds(paths: Sequence[str]) -> float:
    """QPU access time in the spend journals beside the events files. A card with none counts 0 s."""
    spent_us = 0
    for path in paths:
        out_dir = Path(path).parent
        if (out_dir / regime_io.SPEND_JOURNAL).exists():
            spent_us += regime_io.read_spend(out_dir)[0]
        else:
            print(f"cascade_wins: no {regime_io.SPEND_JOURNAL} beside {path}, so it counts 0 s of QPU access",
                  file=sys.stderr)
    return spent_us / 1e6


def read_attempts(paths: Iterable[str]) -> Tuple[List[Tuple[int, int]], int]:
    """``(ts_ms, energy_milli)`` of each row of the attempts files, and the count of lines that hold no row.

    The coordinator skips such a line too (``dashboard.rs:364-373``), because writers that overlap can glue two rows
    on one line or cut one short. A line holds no row when it is not an object with a time and an energy that are
    numbers. A row from before ``raw_best_energy_milli`` existed holds only ``best_energy_milli``.
    """
    rows: List[Tuple[int, int]] = []
    unreadable = 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                energy = row["raw_best_energy_milli"] if "raw_best_energy_milli" in row else row["best_energy_milli"]
                rows.append((int(row["ts_ms"]), int(energy)))
            except (ValueError, KeyError, TypeError):
                unreadable += 1
    return rows, unreadable


def attempt_summary(paths: Iterable[str], gate_milli: int) -> Dict[str, Any]:
    """What the coordinator recorded, as a cross-check of the clears of the journals.

    A row is one validated result (``AttemptRecord`` in ``crates/quip-coordinator/src/attempt.rs``), with ``ts_ms`` in
    milliseconds since the epoch. ``raw_best_energy_milli`` is its true best. ``best_energy_milli`` holds ``i64::MAX``
    when the result missed the gate that the coordinator held then, and this script judges against its own gate. The
    sentinel never passes a gate, so a legacy row that holds only ``best_energy_milli`` counts the same way.
    ``attempts_unreadable`` counts the lines that held no row.
    """
    rows, unreadable = read_attempts(paths)
    times = sorted(datetime.fromtimestamp(ts_ms / 1000, UTC) for ts_ms, _ in rows)
    return {
        "attempts": len(rows),
        "attempts_below_gate": sum(energy < gate_milli for _, energy in rows),
        "attempts_unreadable": unreadable,
        "first_attempt_at": times[0].isoformat() if times else None,
        "last_attempt_at": times[-1].isoformat() if times else None,
    }


def summarize(events: Sequence[Dict[str, Any]], launches: Sequence[Launch], rounds: Dict[str, Any],
              gate_milli: int, qpu_access_s: float) -> Dict[str, Any]:
    run = union((launch.start, launch.end) for launch in launches)
    run_seconds = sum(end - start for start, end in run)
    lengths = [row["blocks"] * rounds["block_seconds"] for row in rounds["rounds"]]
    field_rate = 1.0 / (sum(lengths) / len(lengths))
    event_us = sum(e["access_us"] for e in events if e["type"] == "qpu")
    arms = {}
    for arm in ARMS:
        clears = arm_clears(events, arm, gate_milli)
        legs = sum(1 for e in events if e["type"] == "deep" and e["arm"] in ARMS[arm])
        latency = sorted(cleared - found for found, cleared in clears)
        rate = len(clears) / run_seconds if run_seconds else 0.0
        on_run_clock = [(run_time(found, run), run_time(cleared, run)) for found, cleared in clears]
        arms[arm] = {
            "clears": len(clears),
            "deep_legs": legs,
            "clears_per_hour": rate * 3600,
            "median_find_to_clear_s": latency[len(latency) // 2] if latency else None,
            "replay_win_share": wins.replay_win_share(on_run_clock, run_seconds, lengths),
            "poisson_win_share": wins.poisson_win_share(lengths, rate, latency[len(latency) // 2] if latency else 0.0),
            "equilibrium_share": wins.equilibrium_share(rate, field_rate),
        }
    survivors = Counter(e["arm"] for e in events if e["type"] == "survivor")
    skips = sum(1 for e in events if e["type"] == "qpu_skip")
    return {"gate_milli": gate_milli, "run_seconds": run_seconds, "salts": sum(launch.salts for launch in launches),
            "qpu_access_s": qpu_access_s, "qpu_event_access_s": event_us / 1e6, "qpu_skips": skips,
            "survivors": dict(sorted(survivors.items())), "rounds": len(lengths),
            "median_round_s": sorted(lengths)[len(lengths) // 2], "arms": arms}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--events", required=True, action="extend", nargs="+",
                        help="events.jsonl of one card; give one for each card")
    parser.add_argument("--attempts", action="extend", nargs="+",
                        help="attempts.jsonl of the coordinator, as a path or a glob; a cross-check of the clears")
    parser.add_argument("--rounds", required=True, help="rounds.json from chain_fetch.py")
    parser.add_argument("--gate-milli", type=int, help="threshold to judge clears; default is the median recent gate")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    attempt_paths: set[str] = set()
    for pattern in args.attempts or []:
        if not (matches := glob.glob(pattern)):
            parser.error(f"--attempts {pattern!r} matches no file")
        attempt_paths.update(str(Path(match).resolve()) for match in matches)
    rounds = json.loads(Path(args.rounds).read_text(encoding="utf-8"))
    gates = sorted(row["gate_milli"] for row in rounds["rounds"])
    gate = args.gate_milli if args.gate_milli is not None else gates[len(gates) // 2]
    journals = list(dict.fromkeys(str(Path(path).resolve()) for path in args.events))  # a journal counts once
    events: List[Dict[str, Any]] = []
    launches: List[Launch] = []
    for path in journals:
        journal_events, journal_launches = read_journal(path)
        events += journal_events
        launches += journal_launches
    events.sort(key=lambda event: event["t"])
    summary = summarize(events, launches, rounds, gate, qpu_access_seconds(journals))
    if attempt_paths:
        summary.update(attempt_summary(sorted(attempt_paths), gate))
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"threshold {gate} milli, run {summary['run_seconds']:.0f} s, {summary['salts']} salts, "
          f"QPU {summary['qpu_access_s']:.1f} s, median round {summary['median_round_s']:.0f} s")
    by_arm = ", ".join(f"{arm} {count}" for arm, count in summary["survivors"].items()) or "none"
    print(f"survivors {by_arm}; QPU skips {summary['qpu_skips']}")
    for arm, row in summary["arms"].items():
        print(f"{arm:>11}: {row['clears']:>3} clears from {row['deep_legs']:>4} deep legs, "
              f"{row['clears_per_hour']:.1f} per hour, replay win share {100 * row['replay_win_share']:.1f}%, "
              f"Poisson {100 * row['poisson_win_share']:.1f}%, equilibrium {100 * row['equilibrium_share']:.1f}%")
    if attempt_paths:
        print(f"coordinator: {summary['attempts']} attempts, {summary['attempts_below_gate']} below the gate, "
              f"{summary['attempts_unreadable']} unreadable lines, "
              f"from {summary['first_attempt_at']} to {summary['last_attempt_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
