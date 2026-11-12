#!/usr/bin/env python3
"""Write the Task 3 portfolio-scoring fixture bundles.

Task 3 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: builds
the 4/18/28-asset synthetic-test fixtures the task brief calls for (exhaustive
tests use the 4-asset problem directly; this script is what actually freezes
all three onto disk) and writes each one, twice, under two distinct
diversification-beta ramps so a manuscript-ramp fixture is never confused with
one built from P's live repository-default ramp
(``qpo.encoders.resolve_frustration_beta``). The two ramps are different
formulas that happen to agree only at n=28 (both reach a fraction of 0.4);
see ``quip_solver_dwave.portfolio_replication`` for the derivations.

Requires P's pinned environment on the path (``qpo``, ``dimod``), so it must
run under P's interpreter, e.g.:

    PYTHONPATH=<this repo> <P>/.venv/bin/python scripts/portfolio_replication.py

Historical replication stays blocked (Task 1 found no historical portfolio
market capture), so every fixture here is built from a deterministic,
seeded synthetic instance -- never live market data -- and carries
``provenance = "synthetic-test"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quip_solver_dwave import portfolio_replication as pr  # noqa: E402

DEFAULT_ROOT = Path("/home/carback1/quip-data/regimes/round2/bundles-portfolio-fixtures")

#: (n_assets, cardinality_k), matching the brief's "4 assets for exhaustive
#: tests, then 18 and 28 for integration tests," with K a plausible fraction
#: of N (the same shape as P's own tests/conftest.py::problem_factory).
FIXTURES: Tuple[Tuple[int, int], ...] = ((4, 2), (18, 6), (28, 9))


def write_fixtures(root: Path, *, allow_dirty: bool) -> None:
    for n, k in FIXTURES:
        manuscript_problem = pr.synthetic_one_factor_problem(n, k, seed=n)
        manuscript_problem.frustration_beta = pr.manuscript_ramp_beta(manuscript_problem)
        manuscript_problem.invalidate_cache()
        manuscript_path = root / f"n{n}-k{k}" / "manuscript-ramp"
        manuscript_hash = pr.write_portfolio_fixture(
            manuscript_path,
            manuscript_problem,
            beta_mode="manuscript-ramp",
            allow_dirty=allow_dirty,
        )
        print(
            f"n={n} k={k} manuscript-ramp beta={manuscript_problem.frustration_beta:.6g} -> "
            f"{manuscript_path} ({manuscript_hash})"
        )

        default_problem = pr.synthetic_one_factor_problem(n, k, seed=n)
        default_problem.frustration_beta = pr.repository_default_beta(default_problem)
        default_problem.invalidate_cache()
        default_path = root / f"n{n}-k{k}" / "repository-default-ramp"
        default_hash = pr.write_portfolio_fixture(
            default_path,
            default_problem,
            beta_mode="repository-default-ramp",
            allow_dirty=allow_dirty,
        )
        print(
            f"n={n} k={k} repository-default-ramp beta={default_problem.frustration_beta:.6g} "
            f"-> {default_path} ({default_hash})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "record export provenance even if this repo's tree has uncommitted changes "
            "(d_head would then not identify the code that wrote the bundles)"
        ),
    )
    args = parser.parse_args(argv)
    args.root.mkdir(parents=True, exist_ok=True)
    write_fixtures(args.root, allow_dirty=args.allow_dirty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
