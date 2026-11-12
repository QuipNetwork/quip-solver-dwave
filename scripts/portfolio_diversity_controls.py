"""Read-diversity controls for the portfolio family models that the exact oracle solved.

The Round 2 CPU arms reach the select-QUBO minimum, but that basket is often not
the best final portfolio after production's repair and weighting finish. This
script measures how much read diversity closes the gap, for every model in the
family index. Gaps and ranks need the optimum, so they are recorded only for the
models in ``exact-oracle.json``; the others record objectives alone:

* ``random``: uniformly random reads at the production read budget.
* ``neal-final-<m>x``: neal at the production reads and sweeps, annealed from a
  tenth of the hot end of its default beta range to ``m`` times the hot end.
  The default range ends many orders of magnitude colder than its hot end, and
  every read then freezes onto the same basket.
* ``optimum_energy_rank``: the rank of the oracle's best basket among all
  K-baskets ordered by select-QUBO energy (1 means the QUBO minimum).

Every read goes through qpo's ``greedy_project`` and ``weights_for_support``,
the finish every Round 2 arm uses. Every arm records its sampling and finish
times and the number of distinct baskets its reads project to. Each worker runs
on its own core from the parent's affinity set, so run the script under
``taskset`` with at least ``--workers`` cores. Run it with the qpo reference
environment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from quip_solver_dwave import portfolio_replication as portfolio
from quip_solver_dwave import round2_io

READS = 500
SWEEPS = 500
FINAL_BETA_MULTIPLES = (1, 3, 10, 30)
SEED_MIX = 0xD1CE
MATERIAL_TOLERANCE = 0.005


def _finals(problem: Any, bits: np.ndarray) -> dict[tuple[int, ...], float]:
    from qpo.projection import (  # pyright: ignore[reportMissingImports]
        greedy_project,
        weights_for_support,
    )

    rho = problem.rho if problem.frustration_beta else None
    scored: dict[tuple[int, ...], float] = {}
    for row in bits:
        support = tuple(greedy_project(row, problem, rho))
        if support not in scored:
            scored[support] = float(problem.objective(weights_for_support(support, problem)))
    return scored


def _best_final(problem: Any, bits: np.ndarray) -> float:
    return min(_finals(problem, bits).values())


def _arm(problem: Any, sample: Any) -> dict[str, Any]:
    """Time one arm's sampling and finish, and keep its best objective."""
    started = time.perf_counter()
    bits = sample()
    sampled = time.perf_counter()
    scored = _finals(problem, bits)
    finished = time.perf_counter()
    return {"objective": min(scored.values()), "sampling_s": sampled - started, "finish_s": finished - sampled,
            "distinct_supports": len(scored)}


def _pin_worker(cores: Any) -> None:
    os.sched_setaffinity(0, {cores.get()})


def _energy_rank(q: np.ndarray, k: int, optimum: Sequence[int]) -> int:
    symmetric = (q + q.T) / 2.0
    combos = np.asarray(list(itertools.combinations(range(q.shape[0]), k)), dtype=np.int64)
    energies = symmetric[combos[:, :, None], combos[:, None, :]].sum(axis=(1, 2))
    target = symmetric[np.ix_(list(optimum), list(optimum))].sum()
    return 1 + int((energies < target - 1e-12 * abs(target)).sum())


def control(task: dict[str, Any]) -> dict[str, Any]:
    import dimod
    import neal  # pyright: ignore[reportMissingImports]

    manifest, arrays = round2_io.read_bundle(Path(task["fixture"]))
    problem = portfolio.load_problem(manifest, arrays)
    qubo, _, _, _, _ = portfolio.encode_to_ising(problem)
    # neal rejects seeds of 2**31 and above, whatever its error message says.
    seed = (zlib.crc32(task["name"].encode()) ^ SEED_MIX) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    best = task["best_objective"]
    arms = {"random": _arm(problem, lambda: rng.integers(0, 2, size=(READS, qubo.n), dtype=np.int8))}
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
    hot = float(neal.default_beta_range(bqm)[0])

    def anneal(multiple: int) -> np.ndarray:
        response = neal.SimulatedAnnealingSampler().sample(
            bqm, beta_range=(hot / 10.0, hot * multiple), beta_schedule_type="geometric",
            num_reads=READS, num_sweeps=SWEEPS, seed=seed,
        )
        variables = list(response.variables)
        return response.record.sample[:, [variables.index(i) for i in range(qubo.n)]]

    for multiple in FINAL_BETA_MULTIPLES:
        arms[f"neal-final-{multiple}x"] = _arm(problem, lambda multiple=multiple: anneal(multiple))
    for arm in arms.values():
        arm["gap"] = None if best is None else (arm["objective"] - best) / abs(best)
    rank = None if best is None else _energy_rank(
        np.asarray(qubo.Q, dtype=np.float64), problem.cardinality_k, task["best_support"])
    return {"name": task["name"], "best_objective": best, "optimum_energy_rank": rank, "arms": arms}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True, help="directory holding index.json and exact-oracle.json")
    parser.add_argument("--fixtures", type=Path, required=True, help="portfolio family fixture directory")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    index = json.loads((args.families / "index.json").read_text(encoding="utf-8"))
    oracle = json.loads((args.families / "exact-oracle.json").read_text(encoding="utf-8"))
    solved = {m["name"]: m for m in oracle["models"]}
    tasks = [
        {"name": r["name"], "fixture": str(args.fixtures / r["name"]),
         "best_objective": solved[r["name"]]["best_objective"] if r["name"] in solved else None,
         "best_support": solved[r["name"]]["best_support"] if r["name"] in solved else None}
        for r in index["models"]
    ]
    available = sorted(os.sched_getaffinity(0))
    if len(available) < args.workers:
        raise SystemExit(f"{args.workers} workers need {args.workers} cores; the affinity set has {len(available)}")
    cores = multiprocessing.Manager().Queue()
    for core in available[:args.workers]:
        cores.put(core)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_pin_worker, initargs=(cores,)) as pool:
        results = list(pool.map(control, tasks))
    summary = {
        "reads": READS, "sweeps": SWEEPS, "final_beta_multiples": list(FINAL_BETA_MULTIPLES),
        "timing": "perf_counter per arm, one worker per core", "worker_cores": available[:args.workers],
        "material_tolerance": MATERIAL_TOLERANCE, "oracle_limit": oracle["limit"], "models": results,
    }
    (args.families / "diversity-controls.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"models": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
