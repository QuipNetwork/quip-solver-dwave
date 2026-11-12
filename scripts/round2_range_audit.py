#!/usr/bin/env python3
"""Audit the legal physical scale of Round 2 bundles against the live graph.

Task 6 of docs/superpowers/plans/2026-09-22-regime-search-round2.md. For
every bundled model, ``audit`` embeds the physical model (Round 1's fixed
embeddings for clique-portfolio; the bundle's own physical qubit labels for
the other four cells) and calls
:func:`quip_solver_dwave.physical_range.audit_scale` against a properties
file and a topology file. Both files can be the saved snapshots under
``quip-data/regimes`` or an archive that ``fetch-live`` wrote, so the same
audit runs against saved data first and live metadata second.

``fetch-live`` only reads solver metadata (``properties``, ``nodes``,
``undirected_edges``) through Ocean. It never calls ``sample`` or
``sample_ising``: no QPU problem is submitted and no spend occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quip_solver_dwave import __version__, physical_range, regime_io, regimes, replay, round2_io
from quip_solver_dwave.embeddings import ZEPHYR_M, ZEPHYR_T, edge_set, embed_clique

SAVED_SPEC = Path("/home/carback1/quip-data/regimes/topology-live-2026-09-21.spec.json")
SAVED_PROPERTIES = Path("/home/carback1/quip-data/regimes/solver-properties.json")
EMBEDDINGS_DIR = Path("/home/carback1/quip-data/regimes/embeddings")
BUNDLES_ROOT = Path("/home/carback1/quip-data/regimes/round2/bundles")
OUTPUT_ROOT = Path("/home/carback1/quip-data/regimes/round2/range-audit")
D_REPO = Path(__file__).resolve().parents[1]

_chains_cache: Dict[str, List[List[int]]] = {}


def git_facts(repo: Path) -> Tuple[str, bool]:
    """HEAD and dirty state of ``repo``. Raises if either git call fails; never returns None.

    "Dirty" means a tracked file differs from HEAD (staged or unstaged) --
    ``--untracked-files=no``, so a stray scratch file in the working tree does
    not trip this: an untracked file is never part of a commit, so it cannot
    make ``d_head`` misidentify the code that produced an audit.
    """
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if head.returncode != 0:
        raise RuntimeError(f"could not determine {repo}'s HEAD commit: {head.stderr.strip()}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, capture_output=True, text=True
    )
    if status.returncode != 0:
        raise RuntimeError(f"could not determine {repo}'s dirty state: {status.stderr.strip()}")
    return head.stdout.strip(), bool(status.stdout.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_topology_json(path: Path) -> Tuple[List[int], List[List[int]]]:
    """``(nodes, edges)`` of any JSON that carries those two keys: a coordinator spec or a live archive."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [int(n) for n in raw["nodes"]]
    edges = [[int(a), int(b)] for a, b in raw["edges"]]
    return nodes, edges


def _clique_chains(spec: replay.TopologySpec, embeddings_dir: Path) -> List[List[int]]:
    key = str(embeddings_dir)
    cached = _chains_cache.get(key)
    if cached is not None:
        return cached
    chains = regime_io.load_chains(embeddings_dir / regime_io.CLIQUE_FILE, regime_io.spec_topology_hash(spec))
    _chains_cache[key] = chains
    return chains


def physical_arrays_for(
    cell: str, manifest: Dict[str, Any], arrays: Dict[str, np.ndarray], spec: replay.TopologySpec, embeddings_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """``(nodes, h, edges, j, chain_strength)`` on physical qubit labels.

    ``edges`` are physical qubit-label pairs, the convention
    :func:`physical_range.audit_scale` and
    :func:`quip_solver_dwave.embeddings.embed_clique` share. Native, diamond,
    and cubic-dimer bundles already carry physical labels through
    ``variable_labels``; clique-portfolio carries the logical portfolio
    model and is embedded here with Round 1's fixed chains and chain
    prefactor of 3.0 — never re-derived.
    """
    if cell == "clique-portfolio":
        chains = _clique_chains(spec, embeddings_dir)
        working = edge_set(spec.edges.tolist())
        nodes, h, edges, j, chain_strength = embed_clique(
            arrays["h"], arrays["edges"], arrays["j"], chains, working, prefactor=3.0
        )
        return nodes, h, edges, j, chain_strength
    nodes = np.asarray(manifest["variable_labels"], dtype=np.int64)
    edges = nodes[arrays["edges"]]
    return nodes, arrays["h"], edges, arrays["j"], None


def _scale_key(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "")


def _summarize(
    cell: str, nonce: str, model_hash: str, chain_strength: Optional[float],
    chain_strength_ratio: Optional[float], result: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "cell": cell, "nonce": nonce, "model_hash": model_hash, "status": result["status"],
    }
    if result["status"] == "blocked":
        row["reason"] = result["reason"]
        return row
    row.update({
        "j_range_used": result["j_range_used"],
        "ceiling": result["ceiling"],
        "all_zero": result["all_zero"],
        "limiting_constraint_kind": result["limiting_constraint"]["kind"] if result["limiting_constraint"] else None,
        "chain_strength": chain_strength,
        "chain_strength_ratio": chain_strength_ratio,
    })
    return row


def write_model_output(
    output_root: Path, cell: str, nonce: str, result: Dict[str, Any],
    chain_strength: Optional[float], chain_strength_ratio: Optional[float],
) -> None:
    model_dir = output_root / cell / nonce
    model_dir.mkdir(parents=True, exist_ok=True)
    if result["status"] == "blocked":
        regime_io.atomic_write_json(model_dir / "manifest.json", {"cell": cell, "nonce": nonce, **result})
        return

    manifest = {
        "cell": cell,
        "nonce": nonce,
        "status": result["status"],
        "j_range_used": result["j_range_used"],
        "ceiling": result["ceiling"],
        "all_zero": result["all_zero"],
        "limiting_constraint": result["limiting_constraint"],
        "coefficient_stats": result["coefficient_stats"],
        "group_sums": result["group_sums"],
        # Round 1's fixed chain prefactor (3.0) held constant; chain couplers
        # scale uniformly with everything else. The ratio is against the
        # *logical* portfolio coefficients, before embedding.
        "chain_strength": chain_strength,
        "chain_strength_ratio_to_logical": chain_strength_ratio,
        "scales": [
            {"requested": entry["requested"], "submitted_scale": entry["submitted_scale"]}
            for entry in result["scales"]
        ],
    }
    regime_io.atomic_write_json(model_dir / "manifest.json", manifest)

    arrays = {}
    for entry in result["scales"]:
        suffix = _scale_key(entry["requested"])
        arrays[f"h_{suffix}"] = entry["h"]
        arrays[f"j_{suffix}"] = entry["j"]
    regime_io.atomic_savez(model_dir / "arrays.npz", **arrays)


def run_audit(args: argparse.Namespace) -> int:
    d_head, dirty = git_facts(D_REPO)
    if dirty and not args.allow_dirty:
        print(
            f"{D_REPO} has uncommitted changes; refusing to audit from a dirty tree "
            "(outputs would not record the code that produced them). Commit or stash first, "
            "or pass --allow-dirty.",
            file=sys.stderr,
        )
        return 1

    spec = replay.load_spec(str(SAVED_SPEC))  # fixed: only used to load Round 1's chains/embeddings
    nodes, edges = load_topology_json(args.graph_topology)
    graph = physical_range.build_zephyr_graph(nodes, edges, m=ZEPHYR_M, t=ZEPHYR_T)
    properties = json.loads(args.properties.read_text(encoding="utf-8"))

    # Keep index rows for cells this run does not touch, so a --cells subset
    # run never drops the other cells' entries from the summary index.
    index_path = args.output_root / "index.json"
    rows: List[Dict[str, Any]] = []
    if index_path.exists():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        rows = [row for row in previous["rows"] if row["cell"] not in args.cells]

    for cell in args.cells:
        cell_dir = args.bundles_root / cell
        if not cell_dir.is_dir():
            print(f"{cell_dir} does not exist; skipping {cell}", file=sys.stderr)
            continue
        nonces = sorted(p.name for p in cell_dir.iterdir() if p.is_dir())
        for nonce in nonces:
            manifest, arrays_npz = round2_io.read_bundle(cell_dir / nonce)
            phys_nodes, h, phys_edges, j, chain_strength = physical_arrays_for(
                cell, manifest, arrays_npz, spec, args.embeddings_dir
            )
            chain_strength_ratio = None
            if chain_strength is not None:
                logical_abs_max = max(
                    float(np.abs(arrays_npz["h"]).max(initial=0.0)),
                    float(np.abs(arrays_npz["j"]).max(initial=0.0)),
                )
                if logical_abs_max > 0.0:
                    chain_strength_ratio = chain_strength / logical_abs_max
            result = physical_range.audit_scale(phys_nodes, h, phys_edges, j, graph, properties)
            write_model_output(args.output_root, cell, nonce, result, chain_strength, chain_strength_ratio)
            rows.append(_summarize(cell, nonce, manifest["hash"], chain_strength, chain_strength_ratio, result))
        print(f"{cell}: audited {len(nonces)} models")

    rows.sort(key=lambda row: (row["cell"], row["nonce"]))
    run_manifest = {
        "d_head": d_head,
        "d_dirty": dirty,
        "package_versions": {"quip_solver_dwave": __version__},
        "interpreter": {"path": sys.executable, "version": platform.python_version()},
        "properties_source": {"path": str(args.properties), "sha256": sha256_file(args.properties)},
        "graph_topology_source": {"path": str(args.graph_topology), "sha256": sha256_file(args.graph_topology)},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": len(rows),
        "rows": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_write_json(args.output_root / "index.json", run_manifest)
    print(f"wrote {args.output_root / 'index.json'} ({len(rows)} models)")
    return 0


def run_fetch_live(args: argparse.Namespace) -> int:
    from dwave.cloud import Client

    with Client.from_config() as client:
        solver = client.get_solver(name=args.solver)
        properties = dict(solver.properties)
        live_nodes = sorted(int(q) for q in solver.nodes)
        live_edges = sorted([sorted([int(a), int(b)]) for a, b in solver.undirected_edges])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_root / f"live-metadata-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_write_json(out_dir / "properties.json", properties)
    regime_io.atomic_write_json(out_dir / "topology.json", {"nodes": live_nodes, "edges": live_edges})

    saved_properties = json.loads(args.saved_properties.read_text(encoding="utf-8"))
    saved_nodes, saved_edges = load_topology_json(args.saved_topology_spec)

    compared_keys = (
        "chip_id", "h_range", "j_range", "extended_j_range",
        "per_qubit_coupling_range", "per_group_coupling_range",
        "annealing_time_range", "default_annealing_time",
    )
    property_drift = {
        key: {"saved": saved_properties.get(key), "live": properties.get(key)}
        for key in compared_keys
        if saved_properties.get(key) != properties.get(key)
    }
    saved_node_set, live_node_set = set(saved_nodes), set(live_nodes)
    saved_edge_set = {(min(a, b), max(a, b)) for a, b in saved_edges}
    live_edge_set = {(min(a, b), max(a, b)) for a, b in live_edges}
    topology_drift = {
        "nodes_only_saved": sorted(saved_node_set - live_node_set)[:10],
        "nodes_only_live": sorted(live_node_set - saved_node_set)[:10],
        "n_nodes_only_saved": len(saved_node_set - live_node_set),
        "n_nodes_only_live": len(live_node_set - saved_node_set),
        "edges_only_saved": [list(e) for e in sorted(saved_edge_set - live_edge_set)[:10]],
        "edges_only_live": [list(e) for e in sorted(live_edge_set - saved_edge_set)[:10]],
        "n_edges_only_saved": len(saved_edge_set - live_edge_set),
        "n_edges_only_live": len(live_edge_set - saved_edge_set),
    }
    topology_matches = topology_drift["n_nodes_only_saved"] == topology_drift["n_nodes_only_live"] == 0 and (
        topology_drift["n_edges_only_saved"] == topology_drift["n_edges_only_live"] == 0
    )

    d_head, dirty = git_facts(D_REPO)
    drift = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "solver": args.solver,
        "chip_id": properties.get("chip_id"),
        "d_head": d_head,
        "d_dirty": dirty,
        "properties_match_saved": not property_drift,
        "property_drift": property_drift,
        "topology_matches_saved": topology_matches,
        "topology_drift": topology_drift,
    }
    regime_io.atomic_write_json(out_dir / "drift.json", drift)

    print(f"wrote {out_dir}")
    print(f"properties match saved: {drift['properties_match_saved']}")
    print(f"topology matches saved: {drift['topology_matches_saved']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit bundled models against a properties+topology snapshot.")
    audit.add_argument("--cells", nargs="+", default=["diamond-pm1", "clique-portfolio"], choices=regimes.CELL_NAMES)
    audit.add_argument("--bundles-root", type=Path, default=BUNDLES_ROOT)
    audit.add_argument("--properties", type=Path, default=SAVED_PROPERTIES)
    audit.add_argument("--graph-topology", type=Path, default=SAVED_SPEC)
    audit.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    audit.add_argument("--output-root", type=Path, default=OUTPUT_ROOT / "saved")
    audit.add_argument("--allow-dirty", action="store_true")
    audit.set_defaults(func=run_audit)

    fetch = sub.add_parser(
        "fetch-live",
        help="Archive live solver properties and topology (metadata only; never samples).",
    )
    fetch.add_argument("--solver", default="Advantage2_system1")
    fetch.add_argument("--saved-properties", type=Path, default=SAVED_PROPERTIES)
    fetch.add_argument("--saved-topology-spec", type=Path, default=SAVED_SPEC)
    fetch.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    fetch.set_defaults(func=run_fetch_live)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"round2_range_audit failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
