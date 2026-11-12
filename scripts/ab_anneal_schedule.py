#!/usr/bin/env python3
"""A/B the anneal-schedule form against the anneal-time form, on real hardware.

The miner now states every forward anneal as ``anneal_schedule=[[0, 0], [T, 1]]``
instead of ``annealing_time=T`` (or nothing, for the solver default). This
script checks that the change is invisible: same cost, same energies.

  TIME      what the miner sent before: ``annealing_time`` when an override is
            given, and no anneal parameter at all otherwise.
  SCHEDULE  what it sends now: the same anneal written as a two-point schedule.

Every model runs once per arm, and the arm that goes first alternates from
model to model, so drift on the QPU lands on both arms equally.

The check passes when both of these hold:

  - the median billed ``qpu_access_time`` of the two arms differs by under 1%
  - a two-sample Kolmogorov-Smirnov test on the best energy per job does not
    separate the arms at the 5% level

``--dry-run`` stops after the free part: it reads the solver's metadata and
asks D-Wave's timing model to price both forms. Nothing is submitted.

A full run at the defaults spends about 18 s of QPU access time (400 jobs at
about 46 ms). That spend does not pass through the miner's usage ledger.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
from dwave.system import DWaveSampler

from quip_solver_dwave.schedule import forward_schedule
from quip_solver_dwave.stats import ks_two_sample

ACCESS_TOLERANCE = 0.01
KS_ALPHA = 0.05
ARMS = ("time", "schedule")


def arm_params(arm: str, anneal_us: float, override: bool) -> Dict[str, Any]:
    """The anneal parameters one arm sends."""
    if arm == "schedule":
        return {"anneal_schedule": forward_schedule(anneal_us)}
    return {"annealing_time": anneal_us} if override else {}


def build_model(base: DWaveSampler, seed: int) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
    """A production-shaped model: h = 0 and J = +/-1 on every live coupler."""
    rng = np.random.default_rng(seed)
    edges = [(int(u), int(v)) for u, v in base.edgelist]
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(edges))
    h = {int(n): 0.0 for n in base.nodelist}
    j = {edge: float(sign) for edge, sign in zip(edges, signs)}
    return h, j


def preflight(solver: Any, reads: int, anneal_us: float, override: bool) -> bool:
    """Price both forms with D-Wave's own timing model. Costs nothing."""
    estimates = {
        arm: solver.estimate_qpu_access_time(
            num_qubits=len(solver.nodes),
            num_reads=reads,
            **arm_params(arm, anneal_us, override),
        )
        for arm in ARMS
    }
    for arm, micros in estimates.items():
        print(f"estimate  {arm:<8}: {micros / 1000:8.3f} ms per job")
    same = abs(estimates["time"] - estimates["schedule"]) < 1e-6
    print(f"estimates equal    : {same}")
    return same


def run(solver: Any, base: DWaveSampler, args: argparse.Namespace, anneal_us: float) -> List[Dict[str, Any]]:
    """Submit every model once per arm, ``args.window`` problems in flight."""
    override = args.anneal_time_us > 0
    rows: List[Dict[str, Any]] = []
    pending: Deque[Tuple[int, str, Any]] = deque()

    def collect() -> None:
        model, arm, future = pending.popleft()
        rows.append(
            {
                "model": model,
                "arm": arm,
                "access_us": int(future.timing["qpu_access_time"]),
                "best_energy": float(min(future.energies)),
            }
        )

    for model in range(args.jobs):
        h, j = build_model(base, args.seed + model)
        order = ARMS if model % 2 == 0 else ARMS[::-1]
        for arm in order:
            future = solver.sample_ising(
                h,
                j,
                num_reads=args.reads,
                label=f"quip-ab-{arm}",
                **arm_params(arm, anneal_us, override),
            )
            pending.append((model, arm, future))
            if len(pending) >= args.window:
                collect()
    while pending:
        collect()
    return rows


def verdict(rows: List[Dict[str, Any]]) -> bool:
    by_arm = {arm: [r for r in rows if r["arm"] == arm] for arm in ARMS}
    medians = {
        arm: statistics.median(r["access_us"] for r in by_arm[arm]) for arm in ARMS
    }
    gap = abs(medians["time"] - medians["schedule"]) / medians["time"]
    ks_stat, ks_p = ks_two_sample(
        [r["best_energy"] for r in by_arm["time"]],
        [r["best_energy"] for r in by_arm["schedule"]],
    )
    for arm in ARMS:
        energies = [r["best_energy"] for r in by_arm[arm]]
        print(
            f"{arm:<8}: jobs {len(energies):4d}  median access "
            f"{medians[arm] / 1000:8.3f} ms  median best energy "
            f"{statistics.median(energies):10.1f}"
        )
    access_ok = gap < ACCESS_TOLERANCE
    energy_ok = ks_p >= KS_ALPHA
    print(f"median access gap  : {gap * 100:.3f}%  (pass under {ACCESS_TOLERANCE * 100:g}%)  {access_ok}")
    print(f"KS on best energy  : D={ks_stat:.4f} p={ks_p:.4f}  (pass at p >= {KS_ALPHA})  {energy_ok}")
    return access_ok and energy_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=200, help="models, each run once per arm")
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument(
        "--anneal-time-us",
        type=float,
        default=0.0,
        help="anneal override; 0 compares against the solver default, as production runs",
    )
    parser.add_argument("--window", type=int, default=8, help="problems kept in flight")
    parser.add_argument("--seed", type=int, default=1387)
    parser.add_argument("--out", default="ab_anneal_schedule.csv")
    parser.add_argument("--dry-run", action="store_true", help="metadata and estimates only")
    parser.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    args = parser.parse_args()

    base = DWaveSampler(request_timeout=(60, 300))
    try:
        solver = base.solver
        default_us = float(solver.properties["default_annealing_time"])
        anneal_us = args.anneal_time_us or default_us
        print(f"{solver.name}: {len(solver.nodes)} qubits, anneal {anneal_us:g} us")

        if not preflight(solver, args.reads, anneal_us, args.anneal_time_us > 0):
            print("the timing model prices the two forms differently; not submitting")
            return 1
        if args.dry_run:
            return 0

        per_job_us = solver.estimate_qpu_access_time(
            num_qubits=len(solver.nodes), num_reads=args.reads
        )
        spend_s = 2 * args.jobs * per_job_us / 1e6
        print(f"this run submits {2 * args.jobs} jobs and spends about {spend_s:.1f} s of QPU access")
        if not args.yes:
            print("pass --yes to spend it")
            return 2

        rows = run(solver, base, args, anneal_us)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["model", "arm", "access_us", "best_energy"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.out}")
        return 0 if verdict(rows) else 1
    finally:
        base.close()


if __name__ == "__main__":
    sys.exit(main())
