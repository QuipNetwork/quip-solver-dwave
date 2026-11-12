#!/usr/bin/env python3
"""Run the classical arm and the seed arm for one cell of the regime search.

The ladder runs the kernel cold at each rung, 64 reads, one job on each worker
thread. Every energy is scored here from the spins. The kernel's own energies
are never read: a cold job with couplings that the multi-spin path does not
take falls back to the scalar kernel without a signal, so a guard checks the
kernel path once before any run is timed.

A model counts as unmatched when no rung brings the classical best energy
down to its QPU best energy. A model that counts as unmatched at one anneal
time or at both gets one more run at ``--extra-rung``. That step and the seed
arm need the QPU captures, so run this script again after
``regime_capture.py``. Both CSV files append and skip what they hold. The
seed arm reads the capture of 80 µs, and it says so when a model has none.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import numpy as np

from quip_solver_dwave import regime_io, regime_metrics, regimes, replay
from quip_solver_dwave.msa import load_kernel, pack_lanes

LADDER_HEADER = ["nonce", "sweeps", "best_energy", "wall_s", "kernel_label", "k_share"]
SEED_HEADER = ["nonce", "qpu_lanes", "lite_lanes", "qpu_lane_best", "lite_lane_best", "wall_s"]
MAX_UNIT_BONDS = 63


def rng_seed(nonce: str, sweeps: int) -> int:
    """One kernel seed for each run: a rung is its own run, not a longer copy of the rung before."""
    return (int(nonce[:15], 16) + sweeps) % (1 << 63)


def guard_kernel_path(kernel: object, label: str, model: regimes.Model) -> None:
    observed = regimes.observed_kernel_label(kernel, model.kernel_h, model.kernel_edges, model.kernel_j)
    if observed != label:
        raise SystemExit(f"the kernel serves this cell with {observed}, and the cell expects {label}")
    if label == regimes.CPU_MSA:
        bonds = regimes.max_bonds_per_spin(len(model.kernel_h), model.kernel_edges)
        if bonds > MAX_UNIT_BONDS:
            raise SystemExit(f"a spin has {bonds} unit bonds, over the kernel limit of {MAX_UNIT_BONDS}")


def done_keys(path: Path, fields: Tuple[str, ...]) -> Set[Tuple[str, ...]]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {tuple(row[f] for f in fields) for row in csv.DictReader(handle) if all(row.get(f) for f in fields)}


def open_csv(path: Path, header: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    handle = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if new:
        writer.writerow(header)
    return handle, writer


def qpu_best(cell_dir: Path, nonce: str) -> Dict[int, float]:
    """The QPU best energy of a model at each captured anneal time."""
    best: Dict[int, float] = {}
    for folder in sorted(cell_dir.glob("qpu-*")):
        path = folder / f"{nonce}.npz"
        if path.exists():
            with np.load(path) as capture:
                best[int(folder.name.split("-")[1])] = float(capture["energies"].min())
    return best


def read_ladder(path: Path) -> Dict[str, Dict[int, float]]:
    ladder: Dict[str, Dict[int, float]] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ladder.setdefault(row["nonce"], {})[int(row["sweeps"])] = float(row["best_energy"])
    return ladder


def needs_extra_rung(ladder: Dict[int, float], rungs: List[int], targets: Dict[int, float]) -> bool:
    """Whether a model gets the run at the extra rung: it counts as unmatched at one anneal time or at both.

    ``ladder`` holds the best energy of each finished run of the model. ``targets`` holds its QPU best energy at
    each captured anneal time. The answer is no until every rung has run. The report asks ``match_sweeps`` the same
    question, so both take one definition of an unmatched model (specification sections 8.2 and 9.1).
    """
    finished = {sweeps: ladder[sweeps] for sweeps in rungs if sweeps in ladder}
    if len(finished) != len(rungs):
        return False
    return any(regime_metrics.match_sweeps(finished, target) is None for target in targets.values())


def seed_row(
    kernel: Any, beta_range: Callable[..., Tuple[float, float]], model: regimes.Model, nonce: str,
    capture_path: Path, reads: int, lite_sweeps: int, seed_sweeps: int,
) -> List[object]:
    """One row of ``seed.csv``: QPU reads against lite states, in one kernel run.

    Half the lanes start from the best QPU reads. ``pack_lanes`` drops a duplicate state, because two lanes that
    start equal stay equal, and lite states fill the rest. The run starts at the geometric midpoint of the beta
    range. The kernel returns ``reads`` rows also for fewer start states, so each best energy reads its own lanes.
    """
    with np.load(capture_path) as capture:
        order = np.argsort(capture["energies"], kind="stable")[: reads // 2]
        qpu_spins, qpu_energies = capture["spins"][order], capture["energies"][order]
    lite_spins, _ = kernel.sample(
        model.kernel_h, model.kernel_edges, model.kernel_j,
        num_sweeps=lite_sweeps, num_reads=reads, seed=rng_seed(nonce, lite_sweeps),
    )
    states, qpu_count = pack_lanes(qpu_spins, qpu_energies, lite_spins, regimes.score(model, lite_spins), lanes=reads)
    hot, cold = beta_range(model.kernel_h, model.kernel_edges, model.kernel_j)
    start = time.perf_counter()
    spins, _ = kernel.sample(
        model.kernel_h, model.kernel_edges, model.kernel_j,
        num_sweeps=seed_sweeps, num_reads=reads, seed=rng_seed(nonce, seed_sweeps) + 1,
        initial_spins=states, start_beta=math.sqrt(hot * cold),
    )
    wall = time.perf_counter() - start
    energies = regimes.score(model, spins)
    lite_lanes = len(states) - qpu_count
    return [
        nonce, qpu_count, lite_lanes,
        repr(float(energies[:qpu_count].min())) if qpu_count else "",
        repr(float(energies[qpu_count:len(states)].min())) if lite_lanes else "",
        f"{wall:.4f}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--cell", required=True, choices=regimes.CELL_NAMES)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--embeddings-dir")
    parser.add_argument("--models", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260920)
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--rungs", type=int, nargs="+", default=[512, 2048, 8192, 32768, 131072])
    parser.add_argument("--extra-rung", type=int, default=524288)
    parser.add_argument("--seed-sweeps", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--clique-holdings", type=int, default=regimes.CLIQUE_HOLDINGS)
    args = parser.parse_args()

    spec = replay.load_spec(args.spec)
    cell_dir = Path(args.round_dir) / args.cell
    nonces = regime_io.cell_nonces(cell_dir, args.models, args.rng_seed)

    try:
        label, model_of = regime_io.model_source(args.cell, spec, args.embeddings_dir, args.clique_holdings)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        regime_io.check_manifest(
            cell_dir, **regime_io.cell_identity(args.cell, spec, args.clique_holdings, args.embeddings_dir),
            reads=args.reads, ladder_rungs=sorted(args.rungs), extra_rung=args.extra_rung,
            seed_sweeps=args.seed_sweeps, threads=args.threads,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    kernel = load_kernel()
    guard_kernel_path(kernel, label, model_of(nonces[0]))
    lock = threading.Lock()

    # ---- the ladder, and the extra rung for a model that no rung matches
    ladder_path = cell_dir / "ladder.csv"
    ladder_done = done_keys(ladder_path, ("nonce", "sweeps"))
    known = read_ladder(ladder_path)
    ladder_file, ladder_writer = open_csv(ladder_path, LADDER_HEADER)

    def run_ladder(nonce: str) -> None:
        rungs = list(args.rungs)
        if needs_extra_rung(known.get(nonce, {}), args.rungs, qpu_best(cell_dir, nonce)):
            rungs.append(args.extra_rung)
        rungs = [sweeps for sweeps in rungs if (nonce, str(sweeps)) not in ladder_done]
        if not rungs:
            return
        model = model_of(nonce)
        for sweeps in rungs:
            start = time.perf_counter()
            spins, _ = kernel.sample(
                model.kernel_h, model.kernel_edges, model.kernel_j,
                num_sweeps=sweeps, num_reads=args.reads, seed=rng_seed(nonce, sweeps),
            )
            wall = time.perf_counter() - start
            best = regimes.score(model, spins).min()
            # The clique cell also reports the share of reads that hold exactly K assets.
            k_share = ""
            if args.cell == "clique-portfolio":
                k_share = f"{float(((spins > 0).sum(axis=1) == args.clique_holdings).mean()):.4f}"
            with lock:
                ladder_writer.writerow([nonce, sweeps, repr(float(best)), f"{wall:.4f}", label, k_share])
                ladder_file.flush()
                ladder_done.add((nonce, str(sweeps)))
                known.setdefault(nonce, {})[sweeps] = float(best)

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            list(pool.map(run_ladder, nonces))
            # A second pass picks up the extra rung for models whose top rung just finished.
            list(pool.map(run_ladder, nonces))
    finally:
        ladder_file.close()

    # ---- the seed arm: QPU reads against lite states, in one kernel run
    if label != regimes.CPU_MSA:
        print(f"{args.cell}: no seed arm, because a seeded job needs the {regimes.CPU_MSA} path")
        return 0
    import quip_msa  # pyright: ignore[reportMissingModuleSource]

    without = sum(not (cell_dir / "qpu-80" / f"{nonce}.npz").exists() for nonce in nonces)
    if without:
        print(f"{args.cell}: {without} of {len(nonces)} models have no capture at 80 µs, so the seed arm skips them")
    seed_path = cell_dir / "seed.csv"
    seed_done = done_keys(seed_path, ("nonce",))
    seed_file, seed_writer = open_csv(seed_path, SEED_HEADER)
    lite_sweeps = min(args.rungs)

    def run_seed(nonce: str) -> None:
        capture_path = cell_dir / "qpu-80" / f"{nonce}.npz"
        if (nonce,) in seed_done or not capture_path.exists():
            return
        row = seed_row(
            kernel, quip_msa.default_beta_range, model_of(nonce), nonce, capture_path,
            args.reads, lite_sweeps, args.seed_sweeps,
        )
        with lock:
            seed_writer.writerow(row)
            seed_file.flush()

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            list(pool.map(run_seed, nonces))
    finally:
        seed_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
