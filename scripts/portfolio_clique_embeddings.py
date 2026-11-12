"""Place one fixed clique embedding for every production portfolio basket size.

Production solved every race with the D-Wave clique sampler, so every basket
of N assets used the same clique embedding of size N: a rigid topology. This
script places that embedding on the working graph once per N, with the
``minorminer.busclique.find_clique_embedding`` call the clique sampler makes,
and writes one file per size with the chain lengths. ``--sizes`` places cliques
for sizes no production race used, such as family B's. Nothing here uses QPU time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from embed_probe import find_clique  # pyright: ignore[reportMissingImports]

from quip_solver_dwave import embeddings, regime_io, replay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="topology spec JSON of the working graph")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--races", type=Path, help="production race export JSON: one clique per basket size")
    source.add_argument("--sizes", type=int, nargs="+", help="explicit basket sizes, such as family B's")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zephyr-m", type=int, default=embeddings.ZEPHYR_M)
    parser.add_argument("--zephyr-t", type=int, default=embeddings.ZEPHYR_T)
    args = parser.parse_args(argv)

    spec = replay.load_spec(args.spec)
    edges = embeddings.edge_set(spec.edges.tolist())
    topology_hash = regime_io.spec_topology_hash(spec)
    if args.sizes:
        sizes = sorted(set(args.sizes))
    else:
        sizes = sorted({len(race["assets"]) for race in json.loads(args.races.read_text(encoding="utf-8"))})
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for size in sizes:
        chains = find_clique(size, spec, args.zephyr_m, args.zephyr_t)
        embeddings.validate_clique_embedding(chains, edges)
        lengths = [len(chain) for chain in chains]
        regime_io.atomic_write_json(
            args.out / f"clique-{size}.json", {"n": size, "topology_hash": topology_hash, "chains": chains}
        )
        summary.append({
            "n": size, "shortest_chain": min(lengths), "longest_chain": max(lengths), "qubits": sum(lengths),
        })
    regime_io.atomic_write_json(
        args.out / "summary.json", {"spec": str(args.spec), "topology_hash": topology_hash, "sizes": summary}
    )
    for row in summary:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
