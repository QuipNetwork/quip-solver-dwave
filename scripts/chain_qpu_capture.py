#!/usr/bin/env python3
"""Sample chain models on the QPU and keep every read, with overlapped calls and a spend journal.

The run goes through the production ``OceanSampler``, so the two qubits and
40 couplers the chip lacks are clamped as in production. Energies are
recomputed from the spins on the full, unscaled model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quip_miner_dwave import regime_io, replay

RESERVATION_MARGIN = 1.2


def estimate_us(reads: int, anneal_us: int) -> float:
    """Access time of one job on a 4.5k-qubit native problem, fitted to six measured settings."""
    return 34_640.0 + reads * (anneal_us + 158.2)


def full_energies_milli(spins: np.ndarray, dense_edges: np.ndarray, h: np.ndarray, j: np.ndarray) -> np.ndarray:
    s = spins.astype(np.int64)
    field = s @ np.rint(h * 1000).astype(np.int64)
    coupling = (s[:, dense_edges[:, 0]] * s[:, dense_edges[:, 1]]) @ np.rint(j * 1000).astype(np.int64)
    return field + coupling


def read_nonces(path: str) -> List[str]:
    """One nonce per line, as plain hex or as a JSON object with a ``nonce`` key."""
    nonces = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            nonces.append(json.loads(line)["nonce"] if line.startswith("{") else line)
    return nonces


def pending_jobs(nonces: Sequence[str], repeats: int, out_dir: Path, reads: int, anneal_us: int) -> List[Dict[str, Any]]:
    reservation = int(estimate_us(reads, anneal_us) * RESERVATION_MARGIN)
    jobs = []
    for nonce in nonces:
        for repeat in range(repeats):
            key = f"{nonce}-r{repeat}"
            if not (out_dir / f"{key}.npz").exists():
                jobs.append({"key": key, "nonce": nonce, "reservation_us": reservation})
    return jobs


def capture(
    jobs: Sequence[Dict[str, Any]],
    sample: Callable[[Dict[str, Any]], int],
    journal_dir: Path,
    cell: str,
    anneal_us: int,
    cap_seconds: float,
    in_flight: int,
    save: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[int, int]:
    """Run ``jobs`` with at most ``in_flight`` at once. Returns ``(status, captured)``.

    ``sample`` sends one job to the QPU and returns the access time in
    microseconds. ``save``, when given, then decodes the answer and writes its
    files. Each job writes a submit line with its reservation before ``sample``
    and a charge line directly after it, before ``save``, so a failure in
    ``save`` cannot lose the charge. A job that reports no access time is
    charged its reservation, because a charge of 0 would leave the cap with
    nothing to count, and the run stops as it does after a failure. Status 3
    means the next reservation would pass the cap. After a failure no new job
    starts, and every job in flight still finishes, so its charge reaches the
    journal before the error rises.
    """
    queue = deque(jobs)
    running: Dict[Future, Dict[str, Any]] = {}
    lock = threading.Lock()
    status, captured = 0, 0
    error: Optional[BaseException] = None

    def run(job: Dict[str, Any]) -> None:
        access_us = int(sample(job))
        # SAPI sends no timing record when something is wrong. A charge of 0 would leave the cap nothing to count,
        # so the reservation stands in for it and the run stops.
        charge_us = access_us if access_us > 0 else int(job["reservation_us"])
        with lock:
            regime_io.append_spend(journal_dir, regime_io.CHARGE, cell, job["key"], anneal_us, charge_us)
        if access_us <= 0:
            raise SystemExit(f"{job['key']}: the solver reported no QPU access time, so the journal charges its "
                             f"reservation of {charge_us} us and the run stops")
        if save is not None:
            save(job)

    with ThreadPoolExecutor(max_workers=in_flight, thread_name_prefix="chain-qpu") as pool:
        while True:
            while error is None and status == 0 and queue and len(running) < in_flight:
                job = queue[0]
                with lock:
                    spent, _ = regime_io.read_spend(journal_dir)
                    if (spent + job["reservation_us"]) / 1e6 > cap_seconds:
                        print(f"stopping before {job['key']}: {spent / 1e6:.2f} s charged plus reserved, "
                              f"cap {cap_seconds:.1f} s", file=sys.stderr)
                        status = 3
                        break
                    regime_io.append_spend(journal_dir, regime_io.SUBMIT, cell, job["key"], anneal_us,
                                           int(job["reservation_us"]))
                queue.popleft()
                running[pool.submit(run, job)] = job
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                running.pop(future)
                failure = future.exception()
                if failure is None:
                    captured += 1
                elif error is None:
                    error = failure
    if error is not None:
        raise error
    return status, captured


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--nonces", required=True, help="one hex nonce per line, or a qblocks.jsonl file")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--anneal-us", type=int, default=80)
    parser.add_argument("--j-scale", type=float, default=1.0,
                        help="multiply the couplings and turn SAPI scaling off; 1.0 keeps SAPI scaling")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--in-flight", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N nonces; 0 is all")
    parser.add_argument("--cap-seconds", type=float, required=True, help="approved QPU access time for this run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
        parser.error("--cap-seconds must be a finite positive number")

    spec = replay.load_spec(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nonces = read_nonces(args.nonces)
    if args.limit:
        nonces = nonces[: args.limit]
    jobs = pending_jobs(nonces, args.repeats, out_dir, args.reads, args.anneal_us)
    estimate_s = len(jobs) * estimate_us(args.reads, args.anneal_us) / 1e6
    spent, open_jobs = regime_io.read_spend(out_dir)
    print(f"{len(jobs)} jobs to run, about {estimate_s:.1f} s of QPU access; "
          f"{spent / 1e6:.1f} s already charged or reserved here, {open_jobs} open reservations")
    if args.dry_run:
        return 0
    if open_jobs:
        raise SystemExit("refusing: the journal holds open reservations. A person must settle them first")
    if not args.yes:
        print("pass --yes to spend it")
        return 2

    from quip_miner_dwave.ocean import OceanSampler

    sampler = OceanSampler(mock=False, submit_workers=args.in_flight)
    sampler.ensure_connected()
    sampler.set_session_topology([int(n) for n in spec.nodes], [(int(u), int(v)) for u, v in spec.edges])
    cell = f"chain-r{args.reads}-s{args.j_scale:g}"

    def sample(job: Dict[str, Any]) -> int:
        h, j = replay.model_from_nonce(spec, job["nonce"])
        started = time.monotonic()
        result = sampler.sample(
            spec.nodes, h, spec.edges, j * args.j_scale,
            num_reads=args.reads, anneal_time_us=args.anneal_us, nonce_seed=bytes.fromhex(job["nonce"]),
            label="quip-chain-capture", auto_scale=None if args.j_scale == 1.0 else False,
        )
        # The answer waits on its job until ``save`` runs, which is after ``capture`` has journaled the charge.
        job["answer"] = (h, j, result, time.monotonic() - started)
        return int(result.device_access_time_us)

    def save(job: Dict[str, Any]) -> None:
        h, j, result, wall_s = job.pop("answer")
        spins = replay.spec_order(result.spins, result.variables, spec)
        regime_io.atomic_savez(
            out_dir / f"{job['key']}.npz",
            spins=spins,
            energies_milli=full_energies_milli(spins, spec.dense_edges, h, j),
            access_us=np.int64(result.device_access_time_us),
            wall_s=np.float64(wall_s),
            reads=np.int64(args.reads), anneal_us=np.int64(args.anneal_us), j_scale=np.float64(args.j_scale),
        )

    try:
        status, captured = capture(jobs, sample, out_dir, cell, args.anneal_us, args.cap_seconds, args.in_flight,
                                   save=save)
    finally:
        close = getattr(sampler, "close", None)
        if close:
            close()
    spent, _ = regime_io.read_spend(out_dir)
    print(f"captured {captured} of {len(jobs)} jobs; {spent / 1e6:.2f} s charged in this directory")
    return status


if __name__ == "__main__":
    sys.exit(main())
