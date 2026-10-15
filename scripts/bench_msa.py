#!/usr/bin/env python3
"""The QUI-1387 step 4 gate: can the Python path feed the pipeline?

Two numbers decide whether MSA-lite can stay in Python or the miner has to be
ported to Rust:

  FIXED     what one call costs with almost no sweeps: graph build, colouring
            lookup, 64 lane extractions and 64 energy re-scores. No sweep count
            can go under this.
  RATE      models per second at each MSA-lite sweep count, with one job per
            thread. The kernel releases the GIL, so threads are real cores.

The target from QUI-1387 is 100 models/s on the 4 cores of qpu-1, at 64 reads.
Run this on qpu-1. A number from a workstation says nothing about that host.

The graph is a Zephyr Z12 lattice (4800 nodes, 45864 couplers), which is the
Advantage2 layout before the chip's missing qubits are taken out. Production
is 4577 nodes and 41514 couplers, so rates here read about 10% low.
No QPU and no network are used.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import List, Tuple

import numpy as np
from dwave.graphs import zephyr_graph

from quip_solver_dwave.msa import LANES, load_kernel

LITE_SWEEPS = (128, 256, 512, 1024, 2048)
TARGET_MODELS_PER_S = 100.0


def zephyr_problem() -> Tuple[np.ndarray, np.ndarray]:
    """``(h, edges)`` for Z12, with dense indices in sorted label order."""
    graph = zephyr_graph(12)
    index = {label: i for i, label in enumerate(sorted(graph.nodes))}
    edges = np.array([(index[u], index[v]) for u, v in graph.edges], dtype=np.int64)
    return np.zeros(len(index)), edges


def draw_j(rng: np.random.Generator, count: int) -> np.ndarray:
    """One mining model's couplings: +1 or -1 on every edge."""
    return rng.choice(np.array([-1.0, 1.0]), size=count)


def fixed_cost_ms(kernel, h, edges, repeats: int) -> float:
    """Median wall time of a one-sweep, 64-read call."""
    rng = np.random.default_rng(0)
    kernel.sample(h, edges, draw_j(rng, len(edges)), num_sweeps=1, num_reads=LANES)
    times = []
    for seed in range(repeats):
        j = draw_j(rng, len(edges))
        t0 = time.perf_counter()
        kernel.sample(h, edges, j, num_sweeps=1, num_reads=LANES, seed=seed)
        times.append(time.perf_counter() - t0)
    return 1000.0 * float(np.median(times))


def rate(kernel, h, edges, sweeps: int, threads: int, seconds: float) -> Tuple[float, float]:
    """``(models per second, median best energy in milli)`` over ``seconds``."""
    deadline = time.perf_counter() + seconds
    done: List[int] = []
    best: List[int] = []
    lock = threading.Lock()

    def work(worker: int) -> None:
        rng = np.random.default_rng(1000 + worker)
        count, energies = 0, []
        while time.perf_counter() < deadline:
            j = draw_j(rng, len(edges))
            _, energy = kernel.sample(
                h, edges, j, num_sweeps=sweeps, num_reads=LANES, seed=count
            )
            energies.append(int(energy.min()))
            count += 1
        with lock:
            done.append(count)
            best.extend(energies)

    t0 = time.perf_counter()
    pool = [threading.Thread(target=work, args=(w,)) for w in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - t0
    return sum(done) / elapsed, float(np.median(best))


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--threads", type=int, default=4, help="cores to use; qpu-1 has 4")
    parser.add_argument("--seconds", type=float, default=10.0, help="per sweep count")
    parser.add_argument("--repeats", type=int, default=20, help="calls for the fixed cost")
    args = parser.parse_args()

    kernel = load_kernel()
    h, edges = zephyr_problem()
    print(f"Z12: {len(h)} nodes, {len(edges)} couplers, {LANES} reads, {args.threads} threads")

    fixed = fixed_cost_ms(kernel, h, edges, args.repeats)
    ceiling = args.threads * 1000.0 / fixed
    print(f"fixed cost per call : {fixed:7.2f} ms  (ceiling {ceiling:6.1f} models/s at any sweep count)")

    print(f"{'sweeps':>7} {'models/s':>9} {'median best (milli)':>20}")
    rates = {}
    for sweeps in LITE_SWEEPS:
        per_s, energy = rate(kernel, h, edges, sweeps, args.threads, args.seconds)
        rates[sweeps] = per_s
        print(f"{sweeps:>7} {per_s:>9.1f} {energy:>20.0f}")

    meets = [s for s, r in rates.items() if r >= TARGET_MODELS_PER_S]
    if meets:
        print(f"target {TARGET_MODELS_PER_S:g} models/s met up to {max(meets)} sweeps")
        return 0
    print(f"target {TARGET_MODELS_PER_S:g} models/s not met at any sweep count")
    return 1


if __name__ == "__main__":
    sys.exit(main())
