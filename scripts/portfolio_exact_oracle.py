"""Exact best final objective of each portfolio family model, by enumeration.

For every model whose K-selections number at most ``--limit``, this scores
every selection with qpo's production weighting (``weights_for_support``) and
objective, and records the best value and how many selections come within
0.5% of it. The result is the optimum a solver arm's final portfolio can reach,
so arms can be compared against the optimum as well as against each other.
Run it with the qpo reference environment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from quip_solver_dwave import portfolio_replication as portfolio

MATERIAL_TOLERANCE = 0.005


def solve(task: dict[str, Any]) -> dict[str, Any]:
    from qpo.projection import (  # pyright: ignore[reportMissingImports]
        weights_for_support,
    )

    with np.load(task["arrays"]) as saved:
        gamma, w_max, w_min, beta = (float(x) for x in saved["scalars"])
        problem = portfolio.make_problem(saved["mu"], saved["Sigma"], gamma, w_min, w_max, task["k"], beta)
        reference_best = float(saved["reference_objectives"].min())
    objectives = []
    best = math.inf
    best_support: tuple[int, ...] | None = None
    for support in itertools.combinations(range(problem.N), task["k"]):
        objective = float(problem.objective(weights_for_support(support, problem)))
        if objective < best:
            best, best_support = objective, support
        objectives.append(objective)
    values = np.asarray(objectives)
    if best > reference_best + 1e-12 * abs(reference_best):
        raise RuntimeError(f"{task['name']}: a production-scored read beats the enumerated optimum")
    return {
        "name": task["name"], "selections": int(values.size), "best_objective": best,
        "best_support": list(best_support or ()), "reference_read_best": reference_best,
        "within_material": int((np.abs(values - best) <= MATERIAL_TOLERANCE * abs(best)).sum()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    index = json.loads((args.families / "index.json").read_text(encoding="utf-8"))
    tasks = [
        {"name": r["name"], "arrays": str(args.families / r["arrays_file"]), "k": r["cardinality_k"]}
        for r in index["models"] if math.comb(r["n_assets"], r["cardinality_k"]) <= args.limit
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(solve, tasks))
    summary = {"limit": args.limit, "material_tolerance": MATERIAL_TOLERANCE, "models": results}
    (args.families / "exact-oracle.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"models_solved": len(results), "of": len(index["models"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
