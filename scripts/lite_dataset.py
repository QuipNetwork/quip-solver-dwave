#!/usr/bin/env python3
"""Run MSA-lite over recorded attempts: the step 5 data set of QUI-1387.

Every nonce is rebuilt into its model and annealed cold at 64 reads, once per
sweep count. The output is one CSV line per nonce and sweep count, with the
best energy of the 64 reads. A second CSV records the rate in models/s that
each sweep count reached, which is the number that sizes the machine.

Two selections:

  --sample N   the whole low-energy set (the lowest 1% of QPU energies) plus N
               other attempts drawn at random. Use this with every sweep count.
  --all        every unique nonce. Use this with the one chosen sweep count.

The run appends and skips what the CSV already holds, so it resumes.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

from quip_solver_dwave import replay
from quip_solver_dwave.lite_filter import low_energy_set
from quip_solver_dwave.msa import load_kernel

HEADER = ["nonce", "sweeps", "lite_best_milli", "lite_wall_ms"]
RATE_HEADER = ["sweeps", "threads", "models", "seconds", "models_per_s"]


def already_done(path: str) -> Set[Tuple[str, int]]:
    if not os.path.exists(path):
        return set()
    done: Set[Tuple[str, int]] = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add((row["nonce"], int(row["sweeps"])))
            except (TypeError, ValueError):
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


def select(attempts: Dict[str, int], sample: int, seed: int) -> List[str]:
    low = low_energy_set(attempts)
    rest = sorted(set(attempts) - low)
    picked = random.Random(seed).sample(rest, min(sample, len(rest)))
    return sorted(low) + picked


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rates-out", required=True)
    parser.add_argument("--sweeps", type=int, nargs="+", default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1387)
    pick = parser.add_mutually_exclusive_group(required=True)
    pick.add_argument("--sample", type=int, help="low-energy set plus this many random others")
    pick.add_argument("--all", action="store_true", help="every unique nonce")
    args = parser.parse_args()

    spec = replay.load_spec(args.spec)
    attempts = replay.load_attempts(args.attempts)
    nonces = sorted(attempts) if args.all else select(attempts, args.sample, args.seed)
    kernel = load_kernel()
    done = already_done(args.out)
    lock = threading.Lock()

    new_file = not os.path.exists(args.out)
    ensure_trailing_newline(args.out)
    out = open(args.out, "a", newline="", encoding="utf-8")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(HEADER)
    new_rates = not os.path.exists(args.rates_out)
    ensure_trailing_newline(args.rates_out)
    rates = open(args.rates_out, "a", newline="", encoding="utf-8")
    rate_writer = csv.writer(rates)
    if new_rates:
        rate_writer.writerow(RATE_HEADER)

    def one(nonce: str, sweeps: int) -> None:
        h, j = replay.model_from_nonce(spec, nonce)
        start = time.perf_counter()
        _, energies = kernel.sample(
            h, spec.dense_edges, j, num_sweeps=sweeps, num_reads=64,
            seed=int(nonce[:15], 16),
        )
        wall_ms = (time.perf_counter() - start) * 1000.0
        with lock:
            writer.writerow([nonce, sweeps, int(energies.min()), f"{wall_ms:.2f}"])

    try:
        for sweeps in args.sweeps:
            todo = [n for n in nonces if (n, sweeps) not in done]
            if not todo:
                continue
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                list(pool.map(lambda n: one(n, sweeps), todo))
            seconds = time.perf_counter() - start
            out.flush()
            rate_writer.writerow(
                [sweeps, args.threads, len(todo), f"{seconds:.1f}", f"{len(todo) / seconds:.1f}"]
            )
            rates.flush()
            print(f"{sweeps:5d} sweeps: {len(todo)} models in {seconds:.0f} s, {len(todo) / seconds:.1f} models/s")
    finally:
        out.close()
        rates.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
