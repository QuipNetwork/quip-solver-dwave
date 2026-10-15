#!/usr/bin/env python3
"""Seed MSA-heavy with QPU reads and sweep the start beta: step 7 of QUI-1387.

For every captured model the script runs, on the CPU kernel at 64 reads:

  cold     a cold start at each sweep count. The reference.
  seeded   a start from the QPU reads, with the best MSA-lite reads of the same
           model in the lanes the QPU reads leave free, entering the beta
           ladder at a fraction of the way up. 0.5 is the geometric midpoint.

Inside a seeded job the fill lanes are the control: they show whether a lane
that starts from a QPU read finishes lower than a lane that continues from
MSA-lite. Across jobs, the summary reports, for each start fraction, the
fewest sweeps at which the seeded run reaches the energy of the longest cold
run. The start fraction that needs the fewest sweeps is the step 7 answer.

Both arms use one kernel, so the comparison measures the seed and not the
kernel. The run appends and skips what the CSV already holds, so it resumes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import quip_msa

from quip_solver_dwave import replay
from quip_solver_dwave.msa import load_kernel, pack_lanes

HEADER = [
    "nonce", "arm", "start_fraction", "sweeps", "best_milli",
    "qpu_lanes", "qpu_lane_best_milli", "fill_lane_best_milli", "wall_ms",
    "fill_wall_ms",
]


def done_rows(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((row["nonce"], row["arm"], row["start_fraction"], int(row["sweeps"])))
            except (TypeError, ValueError, KeyError):
                continue
    return done


def ensure_trailing_newline(path: str) -> None:
    """If ``path`` exists and its last byte is not a newline, append one.

    Protects a resumed run against gluing its first new row onto a partial
    tail left by a killed prior run.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) != b"\n":
            fh.write(b"\n")


def expected_keys(nonce: str, sweeps_list: List[int], fractions: List[float]) -> List[tuple]:
    keys = []
    for sweeps in sweeps_list:
        keys.append((nonce, "cold", "", sweeps))
        for fraction in fractions:
            keys.append((nonce, "seeded", f"{fraction:.2f}", sweeps))
    return keys


def run(args: argparse.Namespace) -> int:
    spec = replay.load_spec(args.spec)
    kernel = load_kernel()
    done = done_rows(args.out)
    lock = threading.Lock()
    new_file = not os.path.exists(args.out)
    ensure_trailing_newline(args.out)
    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(HEADER)

    def one(path: str) -> None:
        nonce = os.path.basename(path)[: -len(".npz")]
        if all(key in done for key in expected_keys(nonce, args.sweeps, args.fractions)):
            return
        reads = np.load(path)
        h, j = replay.model_from_nonce(spec, nonce)
        hot, cold = quip_msa.default_beta_range(h, spec.dense_edges, j)
        seed = int(nonce[:15], 16)
        order = np.argsort(reads["energies_milli"], kind="stable")[: args.qpu_lanes]
        qpu_spins = reads["spins"][order]
        qpu_energies = reads["energies_milli"][order]
        if args.qpu_lanes >= 64:
            lite_spins = lite_energies = None
            fill_wall_ms = 0.0
        else:
            fill_start = time.perf_counter()
            lite_spins, lite_energies = kernel.sample(
                h, spec.dense_edges, j, num_sweeps=args.lite_sweeps, num_reads=64, seed=seed
            )
            fill_wall_ms = (time.perf_counter() - fill_start) * 1000
        states, qpu_count = pack_lanes(qpu_spins, qpu_energies, lite_spins, lite_energies)
        rows: List[list] = []
        for sweeps in args.sweeps:
            if (nonce, "cold", "", sweeps) not in done:
                start = time.perf_counter()
                _, e = kernel.sample(h, spec.dense_edges, j, num_sweeps=sweeps, num_reads=64, seed=seed + 1)
                rows.append([
                    nonce, "cold", "", sweeps, int(e.min()), 0, "", "",
                    f"{(time.perf_counter() - start) * 1000:.0f}", "0",
                ])
            for fraction in args.fractions:
                key = f"{fraction:.2f}"
                if (nonce, "seeded", key, sweeps) in done:
                    continue
                start = time.perf_counter()
                _, e = kernel.sample(
                    h, spec.dense_edges, j, num_sweeps=sweeps, num_reads=64, seed=seed + 2,
                    initial_spins=states, start_beta=hot * (cold / hot) ** fraction,
                )
                wall = (time.perf_counter() - start) * 1000
                fill = e[qpu_count : len(states)]
                rows.append([
                    nonce, "seeded", key, sweeps, int(e[: len(states)].min()), qpu_count,
                    int(e[:qpu_count].min()) if qpu_count else "",
                    int(fill.min()) if len(fill) else "", f"{wall:.0f}", f"{fill_wall_ms:.0f}",
                ])
        with lock:
            writer.writerows(rows)
            out.flush()

    paths = sorted(glob.glob(f"{args.reads_dir}/*.npz"))[: args.limit or None]
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            list(pool.map(one, paths))
    finally:
        out.close()
    return 0


def summarise(args: argparse.Namespace) -> int:
    cold: Dict[Tuple[str, int], int] = {}
    seeded: Dict[Tuple[str, str, int], dict] = {}
    with open(args.out, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "cold":
                cold[(r["nonce"], int(r["sweeps"]))] = int(r["best_milli"])
            else:
                seeded[(r["nonce"], r["start_fraction"], int(r["sweeps"]))] = r
    sweeps_list = sorted({s for _, s in cold})
    longest = sweeps_list[-1]
    nonces = sorted({n for n, _ in cold})
    fractions = sorted({f for _, f, _ in seeded})
    skipped_pairs = 0

    print("median best energy, milli")
    print(f"{'arm':<12} " + " ".join(f"{s:>10}" for s in sweeps_list))
    cold_row = []
    for s in sweeps_list:
        vals = [cold[(n, s)] for n in nonces if (n, s) in cold]
        cold_row.append(f"{statistics.median(vals):10.0f}" if vals else f"{'-':>10}")
    print(f"{'cold':<12} " + " ".join(cold_row))
    for f in fractions:
        seeded_row = []
        for s in sweeps_list:
            vals = [int(seeded[(n, f, s)]["best_milli"]) for n in nonces if (n, f, s) in seeded]
            seeded_row.append(f"{statistics.median(vals):10.0f}" if vals else f"{'-':>10}")
        print(f"{'seeded ' + f:<12} " + " ".join(seeded_row))

    print(f"\nmedian seeded wall time at {longest} sweeps, ms")
    print(f"{'fraction':<10} {'without fill':>14} {'with fill':>12}")
    for f in fractions:
        vals, vals_with_fill = [], []
        for n in nonces:
            row = seeded.get((n, f, longest))
            if row is None:
                continue
            wall = float(row["wall_ms"])
            vals.append(wall)
            vals_with_fill.append(wall + float(row.get("fill_wall_ms") or 0))
        if vals:
            print(f"{f:<10} {statistics.median(vals):14.0f} {statistics.median(vals_with_fill):12.0f}")
        else:
            print(f"{f:<10} {'-':>14} {'-':>12}")

    print(f"\nfewest sweeps at which a seeded run reaches the cold energy at {longest} sweeps")
    print(f"{'fraction':<10} {'reached':>8} {'median sweeps':>14} {'qpu lane wins':>14} {'fill lane wins':>15}")
    for f in fractions:
        needed, qpu_wins, fill_wins = [], 0, 0
        for n in nonces:
            if (n, longest) not in cold:
                skipped_pairs += 1
                continue
            target = cold[(n, longest)]
            hit = []
            incomplete = False
            for s in sweeps_list:
                row = seeded.get((n, f, s))
                if row is None:
                    incomplete = True
                    continue
                if int(row["best_milli"]) <= target:
                    hit.append(s)
            if incomplete:
                skipped_pairs += 1
            if hit:
                needed.append(hit[0])
            row = seeded.get((n, f, longest))
            if row is None:
                continue
            if row["qpu_lane_best_milli"] and row["fill_lane_best_milli"]:
                qpu_best = int(row["qpu_lane_best_milli"])
                fill_best = int(row["fill_lane_best_milli"])
                qpu_wins += qpu_best < fill_best
                fill_wins += fill_best < qpu_best
        median = f"{statistics.median(needed):.0f}" if needed else "-"
        print(f"{f:<10} {len(needed):8d} {median:>14} {qpu_wins:14d} {fill_wins:15d}")

    by_bucket: Dict[int, List[int]] = defaultdict(list)
    best_fraction = args.ceiling_fraction
    for n in nonces:
        if (n, longest) not in cold:
            skipped_pairs += 1
            continue
        row = seeded.get((n, best_fraction, longest))
        if row is None:
            skipped_pairs += 1
            continue
        reads = np.load(f"{args.reads_dir}/{n}.npz")
        bucket = int(reads["energies_milli"].min()) // 20_000 * 20_000
        gain = cold[(n, longest)] - int(row["best_milli"])
        by_bucket[bucket].append(gain)
    print(f"\nseeded gain over cold at {longest} sweeps and start fraction {best_fraction}, by QPU best energy")
    print(f"{'QPU best, from':>15} {'models':>7} {'median gain':>12} {'share that gain':>16}")
    for bucket in sorted(by_bucket):
        gains = by_bucket[bucket]
        print(f"{bucket:15d} {len(gains):7d} {statistics.median(gains):12.0f} {sum(g > 0 for g in gains) / len(gains):16.1%}")
    print("a positive gain means the seeded run finished lower. The QPU energy ceiling is the")
    print("highest bucket whose median gain is still positive.")
    if skipped_pairs:
        print(f"\nskipped {skipped_pairs} incomplete pairs (partial CSV)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("mode", choices=["run", "summarise"])
    parser.add_argument("--spec")
    parser.add_argument("--reads-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sweeps", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
    parser.add_argument("--lite-sweeps", type=int, default=512)
    parser.add_argument(
        "--qpu-lanes", type=int, default=32,
        help="lanes seeded from the best QPU reads, lowest energy first; the rest are "
        "filled with MSA-lite states (1..64)",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ceiling-fraction", type=lambda s: f"{float(s):.2f}", default="0.50")
    args = parser.parse_args()
    if not 1 <= args.qpu_lanes <= 64:
        parser.error("--qpu-lanes must be between 1 and 64")
    return run(args) if args.mode == "run" else summarise(args)


if __name__ == "__main__":
    sys.exit(main())
