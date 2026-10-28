#!/usr/bin/env python3
"""Set cascade gates from winner traces, and judge a QPU gate from captures.

``msa``  Reads a trace of chain winners through open gates and writes the
         engine's programs file with a gated screen program. The trace and
         the gates are in whole energy units, as the engine counts them.
``qpu``  Reads QPU captures of winners and of control models and writes the
         table of cuts by allowed winner loss. The captures and the cuts are
         in milli energy, as chain records hold them, and the table names
         that unit in its ``unit`` key. Divide a cut by 1,000 to compare it
         with an engine energy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from quip_miner_dwave import gates

REHEAT_BETA = 0.25
MISSES = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]


def load_trace(path: str, order: List[str]) -> np.ndarray:
    """``trace[seed, model, leg]`` with the models in ``order``."""
    position = {nonce: index for index, nonce in enumerate(order)}
    cells: Dict[int, Dict[int, List[int]]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["nonce"] not in position:
                raise SystemExit(f"trace nonce {row['nonce'][:16]} is not in the qblocks file")
            legs = [int(row[key]) for key in sorted((k for k in row if k.startswith("best_")),
                                                    key=lambda k: int(k[5:]))]
            cells.setdefault(int(row["seed"]), {})[position[row["nonce"]]] = legs
    seeds = sorted(cells)
    models = sorted(cells[seeds[0]])
    if any(sorted(cells[seed]) != models for seed in seeds):
        raise SystemExit("the trace does not hold every model at every seed")
    return np.array([[cells[seed][model] for model in models] for seed in seeds])


def screen_program(sweeps: List[int], gate_values: List[int], lanes: int) -> dict:
    legs = []
    for index, (count, gate) in enumerate(zip(sweeps, gate_values)):
        leg = {"start": "fresh" if index == 0 else "continue", "sweeps": count, "gate": gate}
        if index:
            leg["from_beta"] = REHEAT_BETA
        legs.append(leg)
    return {"name": "screen", "words": lanes // 64, "observe_every": 0, "legs": legs}


def run_msa(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.qblocks).read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda row: row["energy_milli"])
    trace = load_trace(args.trace, [row["nonce"] for row in rows])
    marks = [int(mark) for mark in args.legs.split(",")]
    if len(marks) != trace.shape[2]:
        raise SystemExit(f"--legs names {len(marks)} checkpoints and the trace holds {trace.shape[2]}")
    gate_values = gates.derive_gates(trace, args.deepest, args.margin)
    kept = gates.kept_runs(trace, gate_values)
    for index, mark in enumerate(marks):
        print(f"gate at {mark:>4} sweeps: {gate_values[index]:>7}  winner runs kept {100 * kept[:, :, index].mean():.2f}%")
    sweeps = [marks[0]] + [b - a for a, b in zip(marks, marks[1:])]
    payload = {"lanes": args.lanes, "deepest": args.deepest, "margin": args.margin,
               "winners": int(trace.shape[1]), "seeds": int(trace.shape[0]),
               "programs": [screen_program(sweeps, gate_values, args.lanes)]}
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


def best_of_captures(directory: str) -> np.ndarray:
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise SystemExit(f"{directory} holds no captures")
    return np.array([int(np.min(np.load(path)["energies_milli"])) for path in files])


def run_qpu(args: argparse.Namespace) -> int:
    winners = best_of_captures(args.winners)
    controls = best_of_captures(args.controls) if args.controls else np.array([])
    table = gates.gate_table(winners, controls, MISSES)
    for row in table:
        print(f"lose at most {100 * row['miss_allowed']:>5.1f}% of winners: cut {row['cut']}, "
              f"lost {row['winners_lost']}, controls passed {100 * row['control_pass']:.2f}%")
    payload = {"unit": "milli", "winners": int(winners.size), "controls": int(controls.size), "table": table}
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    msa = modes.add_parser("msa")
    msa.add_argument("--trace", required=True)
    msa.add_argument("--qblocks", required=True)
    msa.add_argument("--deepest", type=int, default=100)
    msa.add_argument("--margin", type=int, default=10)
    msa.add_argument("--lanes", type=int, choices=(64, 128), required=True)
    msa.add_argument("--legs", default="8,16,64,256", help="cumulative sweeps at each checkpoint of the trace")
    msa.add_argument("--out", required=True)
    qpu = modes.add_parser("qpu")
    qpu.add_argument("--winners", required=True, help="directory of <nonce>.npz captures of chain winners")
    qpu.add_argument("--controls", help="directory of captures of screen survivors that are not winners")
    qpu.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return run_msa(args) if args.mode == "msa" else run_qpu(args)


if __name__ == "__main__":
    sys.exit(main())
