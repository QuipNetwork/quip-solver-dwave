#!/usr/bin/env python3
"""Capture QPU reads for one cell of the regime search: the quantum arm.

Each model goes through the miner's own ``OceanSampler``, so the defect
clamping, the anneal schedule, and the reconstruction of clamped qubits are
the production code. One ``.npz`` file for each model and anneal time holds
the reads in kernel numbering and their energies, scored here from the spins.
It also holds the coupling extremes of the submitted problem, which the report
compares with the ranges of the solver.

``--dry-run`` prints the model count and the estimated spend and submits
nothing. A real run needs ``--yes``. The spend journal of the round is the
record of what every cell has spent. The script writes the estimate of a job
into the journal before the submission, and the real charge directly after
the answer. Before each submission it reads the journal again, and it stops
when the spend plus the estimate of the next job would pass
``--max-qpu-seconds``. Run one cell for each process: the sampler caches its
submission plan by problem size.

Before any spend, the script compares the qubits and the couplers of the cell
with the live chip. It stops on a difference, unless ``--allow-drift`` says
that the production clamping may handle it.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

from quip_solver_dwave import regime_io, regimes, replay
from quip_solver_dwave.ocean import OceanSampler

# Access time of one job in microseconds: QPU-WINS-ANALYSIS.md.
PROGRAMMING_US = 33_769
READOUT_US = 211


class OfflineSampler:
    """The stand-in for ``--mock``: Ocean's compiled annealer at a few sweeps.

    The miner's own ``MockSampler`` uses the pure Python reference annealer,
    which needs minutes for a few hundred qubits. The reads here only have to
    be valid spins, so that every later step runs on real files.
    """

    def __init__(self) -> None:
        from dwave.samplers import SimulatedAnnealingSampler

        self._sampler = SimulatedAnnealingSampler()
        self.properties: dict = {}

    def sample_ising(self, h, j, **kwargs):
        return self._sampler.sample_ising(h, j, num_reads=int(kwargs.get("num_reads") or 1), num_sweeps=20)


def connect(mock: bool) -> Any:
    """The sampler of a run. Only the real one reaches the QPU."""
    return OceanSampler(sampler=OfflineSampler(), mock=True) if mock else OceanSampler(mock=False)


def estimate_us(reads: int, anneal_us: int) -> int:
    return PROGRAMMING_US + reads * (anneal_us + READOUT_US)


def check_budget(already_us: int, planned_us: int, max_qpu_seconds: float) -> None:
    total = (already_us + planned_us) / 1e6
    if total > max_qpu_seconds:
        raise SystemExit(
            f"refusing to run: {already_us / 1e6:.1f} s spent plus about {planned_us / 1e6:.1f} s planned "
            f"is {total:.1f} s, over the --max-qpu-seconds limit of {max_qpu_seconds:.1f} s"
        )


def pending(cell_dir: Path, nonces: List[str], anneal_times: List[int], reads: int) -> List[Tuple[str, int]]:
    todo = []
    for anneal in anneal_times:
        for nonce in nonces:
            path = cell_dir / f"qpu-{anneal}" / f"{nonce}.npz"
            if not path.exists():
                todo.append((nonce, anneal))
                continue
            with np.load(path) as capture:
                if len(capture["spins"]) != reads or len(capture["energies"]) != reads:
                    raise SystemExit(f"{path} has a wrong row count; resolve the saved answer before resuming")
    return todo


def drift(
    model: regimes.Model, live_nodes: Sequence[int], live_edges: Sequence[Tuple[int, int]]
) -> Tuple[int, int]:
    """The count of qubits and the count of couplers of a model that the live chip lacks."""
    nodes = {int(q) for q in live_nodes}
    edges = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in live_edges}
    lost_qubits = sum(int(q) not in nodes for q in model.qpu_nodes)
    lost_couplers = sum((min(u, v), max(u, v)) not in edges for u, v in model.qpu_edges.tolist())
    return lost_qubits, lost_couplers


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--cell", required=True, choices=regimes.CELL_NAMES)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--embeddings-dir")
    parser.add_argument("--models", type=int, default=100)
    parser.add_argument("--rng-seed", type=int, default=20260920)
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--anneal-us", type=int, nargs="+", default=[80, 400])
    parser.add_argument("--clique-holdings", type=int, default=regimes.CLIQUE_HOLDINGS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    parser.add_argument("--max-qpu-seconds", type=float, default=75.0)
    parser.add_argument("--allow-drift", action="store_true", help="run on a chip that lacks qubits or couplers")
    parser.add_argument("--mock", action="store_true", help="use the offline SA mock and spend nothing")
    args = parser.parse_args()
    if not math.isfinite(args.max_qpu_seconds) or args.max_qpu_seconds <= 0:
        parser.error("--max-qpu-seconds must be a finite positive number")

    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    # One capture owns the round through reservation and charging. Re-reading
    # the journal alone cannot prevent two processes spending the same allowance.
    with open(round_dir / ".capture.lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another capture owns this round; wait for it to finish") from exc
        return run_capture(args, parser)


def run_capture(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    spec = replay.load_spec(args.spec)
    round_dir = Path(args.round_dir)
    cell_dir = round_dir / args.cell
    nonces = regime_io.cell_nonces(cell_dir, args.models, args.rng_seed)

    try:
        _, model_of = regime_io.model_source(args.cell, spec, args.embeddings_dir, args.clique_holdings)
    except ValueError as exc:
        parser.error(str(exc))
    identity = regime_io.cell_identity(args.cell, spec, args.clique_holdings, args.embeddings_dir)
    try:
        regime_io.check_manifest(cell_dir, **identity, reads=args.reads, anneal_us=args.anneal_us, mock=args.mock)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    todo = pending(cell_dir, nonces, args.anneal_us, args.reads)
    planned = sum(estimate_us(args.reads, anneal) for _, anneal in todo)
    already, open_jobs = regime_io.read_spend(round_dir)
    print(
        f"{args.cell}: {len(todo)} captures to run, about {planned / 1e6:.1f} s of QPU access; "
        f"the round has spent {already / 1e6:.1f} s"
    )
    if open_jobs:
        print(f"jobs with no recorded charge: {open_jobs}. The round counts each one at its estimate")
    if args.dry_run or not todo:
        return 0
    if not args.mock:
        check_budget(already, planned, args.max_qpu_seconds)
        if not args.yes:
            print("pass --yes to spend it")
            return 2

    sampler = connect(args.mock)
    run_us = 0
    try:
        sampler.ensure_connected()
        first = model_of(todo[0][0])
        if not args.mock:
            lost_qubits, lost_couplers = drift(first, sampler.live_nodes, sampler.live_edges)
            if (lost_qubits or lost_couplers) and not args.allow_drift:
                raise SystemExit(
                    f"the live chip lacks {lost_qubits} of the qubits and {lost_couplers} of the couplers of this "
                    "cell. Run step 0 again on the live graph, or pass --allow-drift to clamp them"
                )
        sampler.set_session_topology(
            [int(q) for q in first.qpu_nodes], [(int(u), int(v)) for u, v in first.qpu_edges.tolist()]
        )
        for count, (nonce, anneal) in enumerate(todo, 1):
            if not args.mock:
                # Another process can spend between two jobs, so the journal is read again each time.
                spent, _ = regime_io.read_spend(round_dir)
                estimate = estimate_us(args.reads, anneal)
                if (spent + estimate) / 1e6 > args.max_qpu_seconds:
                    print(
                        f"stopping before capture {count} of {len(todo)}: the round has spent {spent / 1e6:.2f} s, "
                        "and the next job would pass the --max-qpu-seconds cap",
                        file=sys.stderr,
                    )
                    return 3
                regime_io.append_spend(round_dir, regime_io.SUBMIT, args.cell, nonce, anneal, estimate)
            model = first if nonce == todo[0][0] else model_of(nonce)
            result = sampler.sample(
                model.qpu_nodes, model.qpu_h, model.qpu_edges, model.qpu_j,
                num_reads=args.reads, anneal_time_us=anneal, nonce_seed=bytes.fromhex(nonce),
                label=f"quip-regime-{args.cell}",
            )
            access_us = 0 if args.mock else int(result.device_access_time_us)
            if not args.mock:
                # The charge goes to the journal before any step that can fail.
                regime_io.append_spend(round_dir, regime_io.CHARGE, args.cell, nonce, anneal, access_us)
                run_us += access_us
            spins, broken = regimes.decode_reads(model, result.spins, result.variables)
            defects = result.defect_info
            out_dir = cell_dir / f"qpu-{anneal}"
            out_dir.mkdir(parents=True, exist_ok=True)
            regime_io.atomic_savez(
                out_dir / f"{nonce}.npz",
                spins=spins,
                energies=regimes.score(model, spins),
                access_us=np.int64(access_us),
                anneal_us=np.int64(anneal),
                mock=np.bool_(args.mock),
                clamped=np.int64(len(defects.fixed_spins) if defects is not None else 0),
                removed_couplers=np.int64(len(defects.removed_edges) if defects is not None else 0),
                broken_chain_share=np.float64(broken),
                chain_strength=np.float64(model.chain_strength),
                **{name: np.float64(value) for name, value in regimes.coupling_extremes(model).items()},
            )
            if len(spins) != args.reads:
                raise SystemExit(
                    f"{out_dir / nonce}.npz holds {len(spins)} rows, and the run asked for {args.reads} reads. "
                    "The answer format of the solver has changed, so the run stops"
                )
            if count % 25 == 0:
                print(f"{count} of {len(todo)}  this run has spent {run_us / 1e6:.1f} s")
    finally:
        sampler.close()
    print(f"{args.cell}: captured {len(todo)}, QPU access spent in this run {run_us / 1e6:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
