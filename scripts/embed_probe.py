#!/usr/bin/env python3
"""Place the regime-search lattices and the clique on the working graph: step 0.

The lattices use the formula maps of ``quip_solver_dwave.embeddings``, shifted
to the place with the fewest vacancies. A spin on a dead qubit, or outside the
graph, becomes a vacancy. The default diamond lattice is larger than the chip,
so the chip cuts it to its own shape and every working qubit hosts a spin. The
default clique is the largest one that ``minorminer.busclique`` places over a
fixed list of seeds, so the chip also sets the asset count of the clique cell.
Nothing here takes QPU access time. ``--solver-properties`` connects to the
solver API to read the coupling and anneal ranges, and that connection is the
only network call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from quip_solver_dwave import embeddings, lattices, regime_io, replay

PROPERTY_NAMES = (
    "chip_id",
    "j_range",
    "extended_j_range",
    "h_range",
    "per_qubit_coupling_range",
    "per_group_coupling_range",
    "annealing_time_range",
    "default_annealing_time",
)
# Seeds that the search for the largest clique tries. On the working graph of 2026-09-20, 4 of these 256 seeds
# reach the largest clique that any seed found.
CLIQUE_SEEDS = 256


def place_diamond(
    dims: lattices.Dims, nodes: Set[int], edges: Set[Tuple[int, int]], m: int, t: int
) -> embeddings.Placement:
    lattice_edges = lattices.diamond_edges(dims)
    return embeddings.best_placement(
        lambda shift: embeddings.diamond_labels(dims, shift, m, t),
        lattice_edges, np.arange(len(lattice_edges)), nodes, edges, max_shift=2 * m,
    )


def place_cubic_dimer(
    dims: lattices.Dims, nodes: Set[int], edges: Set[Tuple[int, int]], m: int, t: int
) -> embeddings.Placement:
    bonds, _ = lattices.cubic_logical_bonds(dims)
    lattice_edges, _, made_by = lattices.cubic_dimer_edges(dims, np.full(len(bonds), 1000, dtype=np.int64))
    return embeddings.best_placement(
        lambda shift: embeddings.cubic_dimer_labels(dims, shift, m, t),
        lattice_edges, made_by, nodes, edges, max_shift=2 * m,
    )


def clique_rank(found: Dict[int, Sequence[int]]) -> Tuple[int, int, int]:
    """Larger is better: more variables, then a shorter longest chain, then fewer qubits."""
    lengths = [len(chain) for chain in found.values()]
    return len(lengths), -max(lengths, default=0), -sum(lengths)


def largest_clique(graph: Any, seeds: int) -> Dict[int, Sequence[int]]:
    """The best clique that ``busclique`` places over ``seeds`` seeds, by :func:`clique_rank`.

    The search of ``busclique`` is seeded, and on a graph with defects the seed
    changes the largest clique that it finds. ``busgraph_cache`` keeps one
    cache for each seed. The search without a cache is not an option, because
    it fails with ``std::bad_alloc`` for a clique of 5 to 8 variables.
    """
    from minorminer.busclique import busgraph_cache

    best: Dict[int, Sequence[int]] = {}
    for seed in range(seeds):
        found = busgraph_cache(graph, seed=seed).largest_clique()
        if clique_rank(found) > clique_rank(best):
            best = found
    return best


def find_clique(
    size: Optional[int], spec: replay.TopologySpec, m: int, t: int, seeds: int = CLIQUE_SEEDS
) -> List[List[int]]:
    """Chains of a clique on the working graph, defects included.

    Without a ``size`` the clique is the largest one that ``busclique`` places
    over ``seeds`` seeds, so the chip sets the size of the clique cell.
    """
    from dwave.graphs import zephyr_graph
    from minorminer.busclique import find_clique_embedding

    graph = zephyr_graph(
        m,
        t,
        node_list=[int(n) for n in spec.nodes],
        edge_list=[(int(u), int(v)) for u, v in spec.edges],
    )
    if size is None:
        found = largest_clique(graph, seeds)
    else:
        found = find_clique_embedding(size, graph)
        if not found:
            raise SystemExit(f"no clique of {size} variables embeds in the working graph")
    return [[int(q) for q in found[i]] for i in range(len(found))]


def solver_properties() -> dict:
    from quip_solver_dwave.ocean import OceanSampler

    sampler = OceanSampler(mock=False)
    try:
        sampler.ensure_connected()
        properties = getattr(sampler.sampler, "properties", {})
        return {name: properties.get(name) for name in PROPERTY_NAMES}
    finally:
        sampler.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--spec", required=True, help="topology spec JSON of the working graph")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--diamond", type=int, nargs=3, default=[48, 48, 16], metavar=("LX", "LY", "LZ"))
    parser.add_argument("--cubic-dimer", type=int, nargs=3, default=[12, 12, 8], metavar=("LX", "LY", "LZ"))
    parser.add_argument("--clique", type=int, help="variables of the clique (default: the largest clique that fits)")
    parser.add_argument("--zephyr-m", type=int, default=embeddings.ZEPHYR_M, help="grid size of the Zephyr graph")
    parser.add_argument("--zephyr-t", type=int, default=embeddings.ZEPHYR_T, help="wires for each Zephyr line")
    parser.add_argument("--solver-properties", action="store_true", help="also read the solver ranges (network)")
    args = parser.parse_args()

    spec = replay.load_spec(args.spec)
    nodes = {int(n) for n in spec.nodes}
    edges = embeddings.edge_set(spec.edges.tolist())
    topology_hash = regime_io.spec_topology_hash(spec)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for lattice, dims_list, placer in (
        ("diamond", args.diamond, place_diamond),
        ("cubic-dimer", args.cubic_dimer, place_cubic_dimer),
    ):
        dims = (dims_list[0], dims_list[1], dims_list[2])
        placement = placer(dims, nodes, edges, args.zephyr_m, args.zephyr_t)
        regime_io.save_placement(out / f"{lattice}.json", lattice, dims, placement, topology_hash)
        placed = int(placement.alive.sum())
        inside = int((placement.labels >= 0).sum())
        print(
            f"{lattice} {dims}: shift {placement.shift}, {placed} spins and {int(placement.keep.sum())} edges placed, "
            f"{inside - placed} of the {inside} spins inside the graph are vacant ({(inside - placed) / inside:.1%})"
        )

    chains = find_clique(args.clique, spec, args.zephyr_m, args.zephyr_t)
    embeddings.validate_clique_embedding(chains, edges)
    lengths = [len(chain) for chain in chains]
    regime_io.atomic_write_json(
        out / regime_io.CLIQUE_FILE, {"n": len(chains), "topology_hash": topology_hash, "chains": chains}
    )
    print(f"clique {len(chains)}: chains of {min(lengths)} to {max(lengths)} qubits, {sum(lengths)} qubits in total")

    if args.solver_properties:
        properties = solver_properties()
        regime_io.atomic_write_json(out.parent / "solver-properties.json", properties)
        print("solver properties:", json.dumps(properties))
    return 0


if __name__ == "__main__":
    sys.exit(main())
