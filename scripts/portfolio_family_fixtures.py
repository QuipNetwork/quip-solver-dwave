"""Write every portfolio family model as a Round 2 portfolio fixture bundle.

The Round 2 portfolio runner (``scripts/round2_cpu.py portfolio``) scores every
solver arm through qpo's select, repair, and weighting finish, which
``scripts/portfolio_qpo_equivalence.py`` showed to agree exactly with the
production backend on these models. This script rebuilds each model's problem
with qpo and writes it with :func:`portfolio_replication.write_portfolio_fixture`.
The fixture writer re-derives beta with qpo's repository-default ramp and
refuses any model whose beta differs. Run it with the qpo reference environment.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from quip_solver_dwave import portfolio_replication as portfolio

PROVENANCE = {
    "real": "assets-api-block-bootstrap",
    "iid": "null-iid-normal",
    "shuffle": "null-time-shuffle",
    "rotate": "null-correlation-rotation",
    "factor": "fitted-factor-synthetic",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True, help="portfolio family model directory")
    parser.add_argument("--out", type=Path, required=True, help="fixture bundle directory")
    parser.add_argument(
        "--allow-dirty", action="store_true", help="record, instead of refuse, uncommitted source changes"
    )
    args = parser.parse_args(argv)
    index = json.loads((args.families / "index.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for record in index["models"]:
        with np.load(args.families / record["arrays_file"]) as saved:
            gamma, w_max, w_min, beta = (float(x) for x in saved["scalars"])
            problem = portfolio.make_problem(
                saved["mu"], saved["Sigma"], gamma, w_min, w_max, record["cardinality_k"], beta, record["tickers"]
            )
        path = args.out / record["name"]
        model_hash = portfolio.write_portfolio_fixture(
            path, problem, "repository-default-ramp", provenance=PROVENANCE[record["variant"]],
            allow_dirty=args.allow_dirty,
        )
        written.append({"name": record["name"], "variant": record["variant"], "model_hash": model_hash})
    (args.out / "fixtures.json").write_text(
        json.dumps({"families_index_schema": index["schema"], "fixtures": written}, indent=1), encoding="utf-8"
    )
    print(json.dumps({"fixtures": len(written), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
