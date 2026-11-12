"""Score a portfolio QPU capture and compare it with the CPU arms and diversity controls.

Every captured read goes through qpo's ``greedy_project`` and
``weights_for_support``, the finish the CPU arms and controls use. The script
also records the chain-break evidence of every job: the share of reads with at
least one broken chain, and the mean broken-chain fraction per read. Run it with
the qpo reference environment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from portfolio_diversity_controls import _finals

from quip_solver_dwave import portfolio_replication as portfolio
from quip_solver_dwave import round2_io


def score_job(meta_path: Path, fixtures: Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job = meta["job"]
    manifest, arrays = round2_io.read_bundle(fixtures / job["name"])
    problem = portfolio.load_problem(manifest, arrays)
    with np.load(meta_path.with_suffix(".npz")) as saved:
        bits = np.asarray(saved["bits"])
        occurrences = np.asarray(saved["num_occurrences"])
        breaks = np.asarray(saved["chain_break_fraction"])
    reads = int(occurrences.sum())
    scored = _finals(problem, bits)
    return {
        "name": job["name"], "variant": job["variant"], "anneal_us": job["anneal_us"],
        "objective": min(scored.values()), "distinct_supports": len(scored),
        "broken_read_share": float(occurrences[breaks > 0].sum() / reads),
        "mean_chain_break_fraction": float((occurrences * breaks).sum() / reads),
        "qpu_access_time_us": meta["qpu_access_time_us"], "chain_strength": meta["chain_strength"],
    }


def cpu_objectives(cpu_root: Path, fixtures: Path) -> dict[str, dict[str, float]]:
    names = {
        record["model_hash"]: record["name"]
        for record in json.loads((fixtures / "fixtures.json").read_text(encoding="utf-8"))["fixtures"]
    }
    found: dict[str, dict[str, float]] = {}
    for path in cpu_root.glob("*/*/result.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        found.setdefault(names[result["model_hash"]], {})[result["requested_kernel"]] = result["objective"]
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True, help="Round 2 portfolio CPU run directory")
    args = parser.parse_args(argv)
    jobs = [score_job(path, args.fixtures) for path in sorted((args.round_dir / "jobs").glob("*.json"))]
    arms = cpu_objectives(args.cpu, args.fixtures)
    controls = json.loads((args.families / "diversity-controls.json").read_text(encoding="utf-8"))["models"]
    for model in controls:
        for arm, value in model["arms"].items():
            arms.setdefault(model["name"], {})[arm] = value["objective"]
    for job in jobs:
        arms.setdefault(job["name"], {})[f"qpu-{job['anneal_us']}us"] = job["objective"]
    oracle = {
        model["name"]: model["best_objective"]
        for model in json.loads((args.families / "exact-oracle.json").read_text(encoding="utf-8"))["models"]
    }
    captured = sorted({job["name"] for job in jobs})
    models = []
    for name in captured:
        best = oracle.get(name, min(arms[name].values()))
        models.append({
            "name": name, "reference": "exact" if name in oracle else "best-known", "best_objective": best,
            "arms": {arm: {"objective": value, "gap": (value - best) / abs(best)}
                     for arm, value in sorted(arms[name].items())},
        })
    summary = {"jobs": jobs, "models": models}
    (args.round_dir / "analysis.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"jobs": len(jobs), "models": len(models)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
