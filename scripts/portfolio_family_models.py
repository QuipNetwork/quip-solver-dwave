"""Build rigid-topology portfolio models from captured assets-api bars.

Each base draw takes a random production race for the basket size N, the hold
count K, and the sliders. It adds a random N-asset subset of the production
universe, a random 720-hour window, and a block-bootstrap seed. From that one
draw it builds four paired models that differ only in market structure:

    real     family A: block-bootstrapped real returns
    iid      family C: independent normal returns, same means and volatilities
    shuffle  family C: each asset's returns permuted in time independently
    rotate   family C: the real covariance with a Haar-rotated correlation

The covariance, expected return, sliders, beta, and selection QUBO come from
the production trading-game backend (qtw-tradinggame commit 3d6acc6), called
the way ``backend.orchestration.job`` calls them. Run this script with the
backend's reference environment and ``--backend-src`` pointing at its source.

``--check`` rebuilds one basket through the backend's own
``AssetsApiSource.hourly_returns`` on the stored bars and asserts that the
QUBO matches this script's path with no bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from quip_solver_dwave import portfolio_families as families

VARIANTS = ("real", "iid", "shuffle", "rotate")
WINDOW_HOURS = 720
BLOCK_HOURS = 24
DEGENERACY_TOLERANCE = 0.005
SEED = 20260926
CHECK_BASKETS = 10
#: Random reads per model scored by the production finish, for cross-checking other scorers.
REFERENCE_READS = 64
REFERENCE_SEED_MIX = 0x5EED


def plan_draws(races: Sequence[dict[str, Any]], universe: Sequence[str], count: int, seed: int,
               history_rows: int) -> list[dict[str, Any]]:
    """The seeded base draws. Pure, so it runs without the backend."""
    rng = np.random.default_rng(seed)
    draws = []
    for index in range(count):
        race = races[int(rng.integers(len(races)))]
        n = len(race["assets"])
        subset = sorted(str(t) for t in rng.choice(list(universe), size=n, replace=False))
        draws.append({
            "index": index,
            "race_job_id": race["job_id"],
            "sliders": race["sliders"],
            "n_assets": n,
            "tickers": subset,
            "window_start": int(rng.integers(0, history_rows - WINDOW_HOURS + 1)),
            "bootstrap_seed": int(rng.integers(2**63)),
            "null_seed": int(rng.integers(2**63)),
        })
    return draws


def _backend(backend_src: Path) -> argparse.Namespace:
    """The production backend functions this script calls, from ``backend_src``."""
    sys.path.insert(0, str(backend_src))
    # The backend is reference code with its own environment, outside this project.
    from backend import config  # pyright: ignore[reportMissingImports]
    from backend.api.schemas import (  # pyright: ignore[reportMissingImports]
        SliderValues,
    )
    from backend.financial.basket import (  # pyright: ignore[reportMissingImports]
        get_asset,
    )
    from backend.financial.estimators.covariance import (  # pyright: ignore[reportMissingImports]
        covariance,
    )
    from backend.financial.estimators.expected_return import (  # pyright: ignore[reportMissingImports]
        expected_return,
    )
    from backend.financial.prices.assets_api import (  # pyright: ignore[reportMissingImports]
        AssetsApiSource,
    )
    from backend.financial.qubo_encoder import (  # pyright: ignore[reportMissingImports]
        encode_select,
        resolve_frustration_beta,
    )
    from backend.financial.slider_map import (  # pyright: ignore[reportMissingImports]
        map_sliders,
    )
    from backend.financial.types import (  # pyright: ignore[reportMissingImports]
        PortfolioProblem,
    )
    from backend.solvers.sampling import (  # pyright: ignore[reportMissingImports]
        select_solution,
    )

    return argparse.Namespace(
        config=config, SliderValues=SliderValues, get_asset=get_asset, covariance=covariance,
        expected_return=expected_return, AssetsApiSource=AssetsApiSource, encode_select=encode_select,
        resolve_frustration_beta=resolve_frustration_beta, map_sliders=map_sliders,
        PortfolioProblem=PortfolioProblem, select_solution=select_solution,
    )


def _source(bk, body: dict[str, Any]):
    """The backend's assets-api source, answering from the stored bars instead of the network."""
    source = bk.AssetsApiSource(base_url="http://stored-bars.invalid")

    def stored(path: str, params: dict[str, Any], **_: Any) -> dict[str, Any]:
        tickers = params["tickers"].split(",")
        return {"interval": body["interval"], "bars": {t: body["bars"][t] for t in tickers}}

    source._get = stored  # pyright: ignore[reportAttributeAccessIssue]
    return source


def build_problem(bk, sliders: dict[str, Any], tickers: Sequence[str], returns: np.ndarray,
                  sigma: np.ndarray | None = None, classes: Sequence[str] | None = None):
    """The production problem for these returns, as ``backend.orchestration.job`` builds it.

    ``classes`` gives the asset classes of tickers the backend does not list, such as synthetic ones.
    """
    params = bk.map_sliders(bk.SliderValues.model_validate(sliders), len(tickers))
    if classes is None:
        classes = [bk.get_asset(t).asset_class for t in tickers]
    problem = bk.PortfolioProblem(
        mu=bk.expected_return(returns, bk.config.MU_WINDOW_HOURS),
        Sigma=bk.covariance(returns, classes) if sigma is None else sigma,
        gamma=params.gamma, w_max=params.w_max, w_min=params.w_min, asset_tickers=list(tickers),
        cardinality_k=params.cardinality_k, n_units_M=params.n_units_M, u_min_units=params.u_min_units,
    )
    problem.frustration_beta = bk.resolve_frustration_beta(problem)
    return problem


def _diagnostics(problem, q: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    corr = families.correlation(problem.Sigma)
    significant = families.significant_pairs(corr, families.pair_overlaps(returns))
    off_diagonal = ~np.eye(len(corr), dtype=bool)
    return {
        "significant_pair_share": float(significant[off_diagonal].mean()),
        "modularity": families.best_modularity(corr, significant),
        "mean_abs_offdiagonal_correlation": float(np.abs(corr[off_diagonal]).mean()),
        "near_degenerate_selections": families.near_degenerate_count(
            q, problem.cardinality_k, DEGENERACY_TOLERANCE
        ),
        "second_best_gap": families.second_best_gap(q, problem.cardinality_k),
    }


class _Reads:
    """The smallest sampler response ``backend.solvers.sampling`` accepts: one read."""

    def __init__(self, bits: np.ndarray) -> None:
        self._sample = {i: int(b) for i, b in enumerate(bits)}

    def samples(self) -> list[dict[int, int]]:
        return [self._sample]


def reference_objectives(bk, problem, qubo, spins: np.ndarray) -> np.ndarray:
    """Production's final objective for each read on its own: repair to K, weight, score."""
    objectives = []
    for row in spins:
        weights, _ = bk.select_solution(_Reads((row + 1) // 2), qubo, problem)
        objectives.append(problem.objective(weights))
    return np.asarray(objectives, dtype=np.float64)


def _write_model(out: Path, name: str, problem, q: np.ndarray, returns: np.ndarray,
                 reference: tuple[np.ndarray, np.ndarray], record: dict[str, Any]) -> None:
    arrays = {
        "Q": q, "mu": problem.mu, "Sigma": problem.Sigma, "returns": returns,
        "reference_spins": reference[0], "reference_objectives": reference[1],
        "scalars": np.asarray([problem.gamma, problem.w_max, problem.w_min, problem.frustration_beta]),
    }
    path = out / f"{name}.npz"
    np.savez(path, **arrays)
    record["arrays_file"] = path.name
    record["arrays_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["qubo_sha256"] = hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    bk = _backend(args.backend_src)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    raw = (args.provenance.parent / provenance["bars_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance["bars_sha256"]:
        raise SystemExit("the stored bars do not match their provenance hash")
    body = json.loads(raw)
    universe = provenance["tickers"]
    history = _source(bk, body).hourly_returns(universe, 2159)
    races = [r for r in json.loads(args.races.read_text(encoding="utf-8")) if len(r["assets"]) >= args.min_assets]
    draws = plan_draws(races, universe, args.count, args.seed, history.shape[0])

    args.out.mkdir(parents=True, exist_ok=True)
    models = []
    for draw in draws:
        columns = [universe.index(t) for t in draw["tickers"]]
        window = history[draw["window_start"] : draw["window_start"] + WINDOW_HOURS][:, columns]
        real = families.block_bootstrap(window, np.random.default_rng(draw["bootstrap_seed"]), BLOCK_HOURS)
        null_rng = np.random.default_rng(draw["null_seed"])
        returns = {
            "real": real,
            "iid": families.null_iid_normal(real, null_rng),
            "shuffle": families.null_shuffle(real, null_rng),
            "rotate": real,  # the rotation null changes only the covariance
        }
        real_problem = build_problem(bk, draw["sliders"], draw["tickers"], real)
        built = {
            "real": real_problem,
            "iid": build_problem(bk, draw["sliders"], draw["tickers"], returns["iid"]),
            "shuffle": build_problem(bk, draw["sliders"], draw["tickers"], returns["shuffle"]),
            "rotate": build_problem(
                bk, draw["sliders"], draw["tickers"], real,
                sigma=families.null_rotate_correlation(real_problem.Sigma, null_rng),
            ),
        }
        reference_rng = np.random.default_rng(draw["null_seed"] ^ REFERENCE_SEED_MIX)
        for variant in VARIANTS:
            problem = built[variant]
            qubo = bk.encode_select(problem)
            q = np.asarray(qubo.Q, dtype=np.float64)
            spins = reference_rng.choice(np.array([-1, 1], dtype=np.int8), size=(REFERENCE_READS, len(q)))
            name = f"draw{draw['index']:03d}-{variant}"
            record = {
                **draw, "name": name, "variant": variant, "family": "A" if variant == "real" else "C",
                "cardinality_k": problem.cardinality_k, "gamma": problem.gamma,
                "frustration_beta": problem.frustration_beta,
                "diagnostics": _diagnostics(problem, q, returns[variant]),
            }
            reference = (spins, reference_objectives(bk, problem, qubo, spins))
            _write_model(args.out, name, problem, q, returns[variant], reference, record)
            models.append(record)
    index = {
        "schema": "portfolio-families-v3",
        "bars_file": provenance["bars_file"], "bars_sha256": provenance["bars_sha256"],
        "races_sha256": hashlib.sha256(args.races.read_bytes()).hexdigest(),
        "backend_src": str(args.backend_src), "seed": args.seed, "min_assets": args.min_assets, "window_hours": WINDOW_HOURS,
        "block_hours": BLOCK_HOURS, "degeneracy_tolerance": DEGENERACY_TOLERANCE,
        "variants": list(VARIANTS), "models": models,
    }
    temporary = args.out / "index.json.tmp"
    temporary.write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.out / "index.json")
    return index


def check(args: argparse.Namespace) -> None:
    """The backend's own latest-window path and this script's path give the same QUBO."""
    bk = _backend(args.backend_src)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    body = json.loads((args.provenance.parent / provenance["bars_file"]).read_bytes())
    universe = provenance["tickers"]
    races = json.loads(args.races.read_text(encoding="utf-8"))
    baskets = list({tuple(race["assets"]): race for race in races}.values())[:CHECK_BASKETS]
    source = _source(bk, body)
    history = source.hourly_returns(universe, 2159)
    mismatched = []
    for race in baskets:
        tickers = list(race["assets"])
        direct = build_problem(bk, race["sliders"], tickers, source.hourly_returns(tickers, WINDOW_HOURS))
        window = history[-WINDOW_HOURS:][:, [universe.index(t) for t in tickers]]
        ours = build_problem(bk, race["sliders"], tickers, window)
        if not np.array_equal(bk.encode_select(direct).Q, bk.encode_select(ours).Q):
            mismatched.append(race["job_id"])
    print(json.dumps({"baskets": len(baskets), "mismatched": mismatched}))
    if mismatched:
        raise SystemExit("the script's window path does not reproduce the backend's QUBO")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True, help="bars provenance JSON")
    parser.add_argument("--races", type=Path, required=True, help="production race export JSON")
    parser.add_argument("--backend-src", type=Path, required=True, help="qtw-tradinggame backend/src")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-assets", type=int, default=0, help="draw only from races with at least this many assets")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        check(args)
        return 0
    if args.out is None:
        parser.error("--out is required unless --check is given")
    index = build(args)
    print(json.dumps({"models": len(index["models"]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
