"""Check that qpo's portfolio pipeline agrees with the production backend on every family model.

``scripts/portfolio_family_models.py`` saves, for every model, the production
backend's selection QUBO and its final objective for 64 random reads. This
script rebuilds each problem with qpo (the pinned benchmark the Round 2 CPU
arms score through), then compares the QUBO and every read's final objective.
Run it with the qpo reference environment. It exits nonzero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from quip_solver_dwave import portfolio_replication as portfolio

RELATIVE_TOLERANCE = 1e-9


def compare(record: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    from qpo.encoders import encode_select  # pyright: ignore[reportMissingImports]

    gamma, w_max, w_min, beta = (float(x) for x in arrays["scalars"])
    problem = portfolio.make_problem(
        arrays["mu"], arrays["Sigma"], gamma, w_min, w_max, record["cardinality_k"], beta, record["tickers"]
    )
    q = np.asarray(encode_select(problem).Q, dtype=np.float64)
    q_gap = float(np.abs(q - arrays["Q"]).max() / np.abs(arrays["Q"]).max())
    reads = portfolio.per_read_diagnostics(problem, arrays["reference_spins"])["records"]
    ours = np.asarray([read["objective"] for read in reads], dtype=np.float64)
    reference = arrays["reference_objectives"]
    objective_gap = float((np.abs(ours - reference) / np.maximum(np.abs(reference), 1e-300)).max())
    return {
        "name": record["name"], "qubo_relative_gap": q_gap, "objective_relative_gap": objective_gap,
        "agrees": q_gap <= RELATIVE_TOLERANCE and objective_gap <= RELATIVE_TOLERANCE,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True, help="portfolio family model directory")
    args = parser.parse_args(argv)
    index = json.loads((args.families / "index.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for record in index["models"]:
        with np.load(args.families / record["arrays_file"]) as saved:
            results.append(compare(record, {name: saved[name] for name in saved.files}))
    failures = [r for r in results if not r["agrees"]]
    summary = {
        "models": len(results), "disagreements": len(failures),
        "worst_qubo_relative_gap": max(r["qubo_relative_gap"] for r in results),
        "worst_objective_relative_gap": max(r["objective_relative_gap"] for r in results),
        "tolerance": RELATIVE_TOLERANCE, "failures": failures[:20],
    }
    (args.families / "qpo-equivalence.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
