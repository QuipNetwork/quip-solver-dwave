#!/usr/bin/env python3
"""Run MSA at high sweeps on the step 5 nonce lists: step 6 of QUI-1387.

``run`` replays each nonce list through ``quip-coordinator drive --source
list`` with quip-solver-cuda as the miner, in chunks of at most 10,000 because
the driver holds a whole list in memory, and merges the reports into one CSV.

``compare`` reads that CSV. The low-energy set is what the QPU filter keeps.
The false-positive set is what MSA-lite passes and the QPU filter rules out.
If the low-energy set finishes lower, the QPU filter adds value over MSA-lite
alone. If the two finish alike, it does not, and the value of the QPU has to
come from seeding.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

from quip_solver_dwave.stats import ks_two_sample

HEADER = ["nonce", "set", "heavy_best_milli", "reads", "sweeps", "wall_ms"]
CHUNK = 10_000
THRESHOLDS = (-14_500_000, -14_540_000, -14_560_000, -14_580_000, -14_600_000, -14_620_000)
NO_SOLUTION = 9223372036854775807  # i64::MAX sentinel the driver uses for "no solution"


def read_nonces(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def done_nonces(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(row["nonce"], row["set"]) for row in csv.DictReader(fh)}


def run(args: argparse.Namespace) -> int:
    sets = {"low-energy": read_nonces(args.low), "false-positive": read_nonces(args.false_positive)}
    done = done_nonces(args.out)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    skipped = 0
    with open(args.out, "a", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        if new_file:
            writer.writerow(HEADER)
        for name, nonces in sets.items():
            if args.cap and len(nonces) > args.cap:
                nonces = random.Random(args.seed).sample(nonces, args.cap)
            todo = [n for n in nonces if (n, name) not in done]
            for start in range(0, len(todo), args.chunk):
                chunk = todo[start : start + args.chunk]
                with tempfile.TemporaryDirectory() as tmp:
                    list_path, report_path = f"{tmp}/list.jsonl", f"{tmp}/report.jsonl"
                    with open(list_path, "w", encoding="utf-8") as fh:
                        for nonce in chunk:
                            fh.write(json.dumps({"nonce": nonce}) + "\n")
                    command = [
                        args.coordinator, "drive", "--miner", args.miner, "--source", "list",
                        "--list", list_path, "--topology", args.spec,
                        "--num-reads", str(args.reads), "--num-sweeps", str(args.sweeps),
                        # The report's best energy is the best of the diverse
                        # selection, which keeps min(rows, min_solutions) rows.
                        # Asking for every read makes it the true minimum.
                        "--min-solutions", str(args.reads),
                        "--report", report_path,
                    ]
                    if args.device is not None:
                        command += ["--device", str(args.device)]
                    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
                    with open(report_path, encoding="utf-8") as fh:
                        for line in fh:
                            row = json.loads(line)
                            if row.get("aggregate"):
                                continue
                            if row["best_energy_milli"] == NO_SOLUTION:
                                skipped += 1
                            # A no-solution row is still written, at the sentinel
                            # value, so done_nonces sees it and a resume does not
                            # re-spend GPU time reproducing it. compare's
                            # read_energies already filters the sentinel back out.
                            writer.writerow([
                                row["job_id"], name, row["best_energy_milli"],
                                row["reads"], row["sweeps"], row["wall_ms"],
                            ])
                out.flush()
                print(f"{name}: {min(start + args.chunk, len(todo))} of {len(todo)}")
    if skipped:
        print(f"wrote {skipped} no-solution rows at the sentinel (sentinel {NO_SOLUTION})", file=sys.stderr)
    return 0


def read_energies(path: str) -> Tuple[Dict[str, List[int]], int]:
    """Read ``heavy_best_milli`` per set, ignoring rows at the no-solution sentinel."""
    by_set: Dict[str, List[int]] = {"low-energy": [], "false-positive": []}
    ignored = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            energy = int(row["heavy_best_milli"])
            if energy == NO_SOLUTION:
                ignored += 1
                continue
            by_set[row["set"]].append(energy)
    return by_set, ignored


def compare(args: argparse.Namespace) -> int:
    by_set, ignored = read_energies(args.out)
    if ignored:
        print(f"ignored {ignored} no-solution rows (sentinel {NO_SOLUTION})", file=sys.stderr)
    print(f"{'set':<15} {'n':>6} {'mean':>12} {'median':>12} " + " ".join(f"{t // 1000:>8}" for t in THRESHOLDS))
    for name, energies in by_set.items():
        if not energies:
            print(f"{name:<15} {0:6d}")
            continue
        counts = " ".join(f"{sum(e <= t for e in energies):8d}" for t in THRESHOLDS)
        print(f"{name:<15} {len(energies):6d} {statistics.mean(energies):12.0f} {statistics.median(energies):12.0f} {counts}")
    if not by_set["low-energy"] or not by_set["false-positive"]:
        print("skipping mean gap and KS test: one or both sets are empty")
        return 0
    d_stat, p_value = ks_two_sample(by_set["low-energy"], by_set["false-positive"])
    gap = statistics.mean(by_set["low-energy"]) - statistics.mean(by_set["false-positive"])
    print(f"mean gap (low-energy minus false-positive): {gap:+.0f} milli   KS D={d_stat:.4f} p={p_value:.2g}")
    print("the threshold columns count models at or below that energy, in units of 1,000 milli")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("mode", choices=["run", "compare"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--low")
    parser.add_argument("--false-positive")
    parser.add_argument("--spec")
    parser.add_argument("--coordinator", default="/home/carback1/quip-miner/target/release/quip-coordinator")
    parser.add_argument("--miner", default="/home/carback1/quip-miner-cuda/target/release/quip-cuda-msa")
    parser.add_argument("--reads", type=int, default=128)
    parser.add_argument("--sweeps", type=int, default=131072)
    parser.add_argument("--cap", type=int, default=4000, help="sample each set down to this; 0 keeps all")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--device", type=int)
    parser.add_argument("--seed", type=int, default=1387)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
