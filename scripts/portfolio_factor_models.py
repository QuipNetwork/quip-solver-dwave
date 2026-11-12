"""Build family B portfolio models: synthetic assets from factors fitted to the real history.

Family A can draw at most as many assets as the stored bars hold. Family B fits
statistical factors to the whole real history, with the production covariance
estimator, and draws any number N of synthetic assets from them with
:func:`portfolio_families.factor_synthetic`. Each size takes an explicit N and
hold count K. Each draw takes the remaining sliders from a random production
race, a random 720-hour window of factor scores, and a block-bootstrap seed.

The factor count is the number of correlation eigenvalues above the
Marchenko-Pastur edge, over the median hours a pair of real assets shares. A
synthetic asset takes its source asset's class, so the backend's variance floor
treats it like its source.

The covariance, expected return, sliders, beta, and selection QUBO come from
the production trading-game backend, called as ``portfolio_family_models.py``
calls them. Run this script with the backend's reference environment and
``--backend-src`` pointing at its source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import portfolio_family_models as base  # pyright: ignore[reportMissingImports]

from quip_solver_dwave import portfolio_families as families

VARIANT = "factor"
SEED = 20260930
HISTORY_HOURS = 2159
LOADING_JITTER = 0.05
VOL_JITTER = 0.1


def parse_size(text: str) -> tuple[int, int]:
    """``"N:K"`` as ``(N, K)``."""
    n, _, k = text.partition(":")
    if not (n.isdigit() and k.isdigit()) or not 3 <= int(k) <= int(n):
        raise argparse.ArgumentTypeError(f"{text!r} is not N:K with 3 <= K <= N")
    return int(n), int(k)


def plan_draws(races: Sequence[dict[str, Any]], sizes: Sequence[tuple[int, int]], count: int,
               seed: int) -> list[dict[str, Any]]:
    """``count`` seeded draws per size, numbered across sizes. Pure, so it runs without the backend."""
    rng = np.random.default_rng(seed)
    draws = []
    for n, k in sizes:
        for _ in range(count):
            race = races[int(rng.integers(len(races)))]
            draws.append({
                "index": len(draws), "race_job_id": race["job_id"],
                "sliders": {**race["sliders"], "holdCount": k}, "n_assets": n, "requested_k": k,
                "synthetic_seed": int(rng.integers(2**63)), "reference_seed": int(rng.integers(2**63)),
            })
    return draws


def build(args: argparse.Namespace) -> dict[str, Any]:
    bk = base._backend(args.backend_src)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    raw = (args.provenance.parent / provenance["bars_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["bars_sha256"]:
        raise SystemExit("the stored bars do not match their provenance hash")
    universe = provenance["tickers"]
    history = base._source(bk, json.loads(raw)).hourly_returns(universe, HISTORY_HOURS)
    universe_classes = [bk.get_asset(t).asset_class for t in universe]
    sigma = bk.covariance(history, universe_classes)
    overlaps = families.pair_overlaps(history)
    pair_hours = float(np.median(overlaps[~np.eye(len(universe), dtype=bool)]))
    factors = args.factors or families.marchenko_pastur_factors(families.correlation(sigma), pair_hours)
    model = families.fit_factor_model(history, sigma, factors)
    races = json.loads(args.races.read_text(encoding="utf-8"))
    draws = plan_draws(races, args.sizes, args.count, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for draw in draws:
        rng = np.random.default_rng(draw["synthetic_seed"])
        returns, sources = families.factor_synthetic(
            model, rng, draw["n_assets"], base.WINDOW_HOURS, base.BLOCK_HOURS, LOADING_JITTER, VOL_JITTER)
        tickers = [f"F{j:02d}-{universe[s]}" for j, s in enumerate(sources)]
        classes = [universe_classes[s] for s in sources]
        problem = base.build_problem(bk, draw["sliders"], tickers, returns, classes=classes)
        if problem.cardinality_k != draw["requested_k"]:
            raise SystemExit(f"draw {draw['index']}: the backend set K = {problem.cardinality_k}, "
                             f"not the requested {draw['requested_k']}")
        qubo = bk.encode_select(problem)
        q = np.asarray(qubo.Q, dtype=np.float64)
        spins = np.random.default_rng(draw["reference_seed"]).choice(
            np.array([-1, 1], dtype=np.int8), size=(base.REFERENCE_READS, len(q)))
        name = f"draw{draw['index']:03d}-{VARIANT}"
        record = {
            **draw, "name": name, "variant": VARIANT, "family": "B", "tickers": tickers,
            "source_tickers": [universe[s] for s in sources], "cardinality_k": problem.cardinality_k,
            "gamma": problem.gamma, "frustration_beta": problem.frustration_beta,
            "diagnostics": base._diagnostics(problem, q, returns),
        }
        reference = (spins, base.reference_objectives(bk, problem, qubo, spins))
        base._write_model(args.out, name, problem, q, returns, reference, record)
        records.append(record)
    index = {
        "schema": "portfolio-families-b-v1",
        "bars_file": provenance["bars_file"], "bars_sha256": provenance["bars_sha256"],
        "races_sha256": hashlib.sha256(args.races.read_bytes()).hexdigest(),
        "backend_src": str(args.backend_src), "seed": args.seed, "window_hours": base.WINDOW_HOURS,
        "block_hours": base.BLOCK_HOURS, "degeneracy_tolerance": base.DEGENERACY_TOLERANCE,
        "sizes": [list(size) for size in args.sizes],
        "factor_model": {
            "history_hours": HISTORY_HOURS, "median_pair_hours": pair_hours, "factors": factors,
            "factor_rule": "given" if args.factors else "marchenko-pastur",
            "tail_df": model.tail_df, "loading_jitter": LOADING_JITTER, "vol_jitter": VOL_JITTER,
            "explained_share": float((model.loadings**2).sum() / len(universe)),
        },
        "variants": [VARIANT], "models": records,
    }
    temporary = args.out / "index.json.tmp"
    temporary.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.out / "index.json")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True, help="bars provenance JSON")
    parser.add_argument("--races", type=Path, required=True, help="production race export JSON")
    parser.add_argument("--backend-src", type=Path, required=True, help="qtw-tradinggame backend/src")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=parse_size, nargs="+", required=True, help="basket sizes as N:K")
    parser.add_argument("--count", type=int, default=12, help="draws per size")
    parser.add_argument("--factors", type=int, help="factor count instead of the Marchenko-Pastur rule")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    index = build(args)
    print(json.dumps({"models": len(index["models"]), "factors": index["factor_model"]["factors"],
                      "tail_df": index["factor_model"]["tail_df"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
