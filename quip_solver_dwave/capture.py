"""Capture a live QPU working graph as a coordinator topology spec.

The network publishes one topology and every miner reduces its own chip's
defects against it (see ``defects``). That published graph is itself a capture
of some chip on some day: D-Wave recalibrates, qubits and couplers drop out and
return, so a capture starts drifting the moment it is taken.

This module turns the connected solver's working graph into the JSON the
coordinator reads (``seed-chain --topology``), and diffs it against the spec in
force so an operator can see how far the two have moved apart before deciding
to republish.

The on-chain topology hash is deliberately not computed here. The coordinator
owns that definition, and a second implementation would be free to disagree
with it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# What the live network runs: h pinned to zero, J at the ends of the range.
# Both are network policy rather than chip properties, so they are inputs here,
# not something the QPU can be asked for.
DEFAULT_ALLOWED_H_MILLI = [0]
DEFAULT_ALLOWED_J_MILLI = [-1000, 1000]


def canonical_edges(edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """Undirected edges as sorted, deduplicated ``[min, max]`` pairs.

    A solver reports both directions of every coupler; the spec carries each
    once, which is also how the coordinator hashes them.
    """
    seen = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}
    return [[u, v] for u, v in sorted(seen)]


def capture_spec(
    nodes: Sequence[int],
    edges: Iterable[Tuple[int, int]],
    *,
    allowed_h_milli: Sequence[int] = DEFAULT_ALLOWED_H_MILLI,
    allowed_j_milli: Sequence[int] = DEFAULT_ALLOWED_J_MILLI,
) -> Dict[str, Any]:
    """Build a topology spec document from a live working graph."""
    node_list = sorted({int(n) for n in nodes})
    node_set = set(node_list)
    edge_list = [e for e in canonical_edges(edges) if e[0] in node_set and e[1] in node_set]
    return {
        "nodes": node_list,
        "edges": edge_list,
        "allowed_h_milli": [int(v) for v in allowed_h_milli],
        "allowed_j_milli": [int(v) for v in allowed_j_milli],
    }


def compare_specs(current: Dict[str, Any], captured: Dict[str, Any]) -> Dict[str, int]:
    """Diff the spec in force against a fresh capture.

    ``missing_*`` is what the published topology asks for and this chip no
    longer has — the part every job must reduce away. ``unused_*`` is hardware
    the published topology does not reach.
    """
    cur_nodes = {int(n) for n in current.get("nodes", [])}
    new_nodes = {int(n) for n in captured.get("nodes", [])}
    cur_edges = {tuple(e) for e in canonical_edges(current.get("edges", []))}
    new_edges = {tuple(e) for e in canonical_edges(captured.get("edges", []))}
    return {
        "published_nodes": len(cur_nodes),
        "live_nodes": len(new_nodes),
        "missing_nodes": len(cur_nodes - new_nodes),
        "unused_nodes": len(new_nodes - cur_nodes),
        "published_edges": len(cur_edges),
        "live_edges": len(new_edges),
        "missing_edges": len(cur_edges - new_edges),
        "unused_edges": len(new_edges - cur_edges),
    }


def format_comparison(diff: Dict[str, int]) -> str:
    """Render a diff for an operator, with the reduction cost spelled out."""
    def pct(part: int, whole: int) -> float:
        return 100.0 * part / whole if whole else 0.0

    return "\n".join(
        [
            f"nodes: published {diff['published_nodes']}, live {diff['live_nodes']}",
            f"  missing from this chip: {diff['missing_nodes']} "
            f"({pct(diff['missing_nodes'], diff['published_nodes']):.2f}%) — clamped per job",
            f"  live but unpublished:   {diff['unused_nodes']} — hardware the network does not use",
            f"edges: published {diff['published_edges']}, live {diff['live_edges']}",
            f"  missing from this chip: {diff['missing_edges']} "
            f"({pct(diff['missing_edges'], diff['published_edges']):.2f}%) — removed per job, scored unoptimized",
            f"  live but unpublished:   {diff['unused_edges']} — couplers the network does not use",
        ]
    )


def write_spec(spec: Dict[str, Any], path: str) -> None:
    """Write the spec as JSON the coordinator's --topology flag accepts."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
        f.write("\n")


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
