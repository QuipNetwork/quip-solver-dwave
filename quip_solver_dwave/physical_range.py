"""Legal physical scale for a model on the live graph.

Task 6 of docs/superpowers/plans/2026-09-22-regime-search-round2.md.
:func:`audit_scale` finds the largest positive multiplier a physical
model's coefficients can carry without breaking the solver's ``h_range``,
its J range (the extended range when the solver documents one), or a
Zephyr graph's per-group coupling bound, then returns the coefficients
scaled to 0.5, 0.75, and 1.0 of that ceiling, each revalidated after
multiplication. A missing solver property or a graph outside the Zephyr
family blocks the audit instead of guessing a default.

``nodes`` are physical qubit labels, parallel to ``h``. ``edges`` are
physical qubit-label pairs (not array positions), parallel to ``j`` — the
same convention :func:`quip_solver_dwave.embeddings.embed_clique` already
returns, so a caller with an embedded model has nothing left to translate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

SCALE_FRACTIONS = (0.5, 0.75, 1.0)
_J_RANGE_KEYS = ("extended_j_range", "j_range")


def signed_scale_limit(values: Sequence[float], lower: float, upper: float) -> float:
    """The largest positive scale that keeps every signed ``value`` inside ``[lower, upper]``."""
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < 0 < upper):
        raise ValueError("range must have finite bounds around zero")
    limits = []
    for value in values:
        if not math.isfinite(value):
            raise ValueError("coefficient must be finite")
        if value > 0:
            limits.append(upper / value)
        elif value < 0:
            limits.append(lower / value)
    return min(limits, default=math.inf)


def build_zephyr_graph(nodes: Sequence[int], edges: Sequence[Sequence[int]], m: int, t: int = 4) -> nx.Graph:
    """An attributed Zephyr(``m``, ``t``) graph over a working node/edge set.

    Only ``family`` and ``rows`` drive ``dwave.system.coupling_groups``; the
    rest match the attributes a real chip graph carries.
    """
    graph = nx.Graph()
    graph.add_nodes_from(int(q) for q in nodes)
    graph.add_edges_from((int(a), int(b)) for a, b in edges)
    graph.graph.update(family="zephyr", rows=m, columns=m, tile=t, labels="int")
    return graph


_group_cache: Dict[int, List[List[Tuple[int, int]]]] = {}


def _groups_for(graph: nx.Graph) -> List[List[Tuple[int, int]]]:
    """Ocean's coupling groups of ``graph``, cached by object identity.

    The group structure is a pure function of the graph, and one graph
    object is audited against many models in one run.
    """
    cached = _group_cache.get(id(graph))
    if cached is not None:
        return cached
    from dwave.system.coupling_groups import coupling_groups

    groups = [list(group) for group in coupling_groups(graph)]
    _group_cache[id(graph)] = groups
    return groups


def _range_tuple(value: Any) -> Optional[Tuple[float, float]]:
    """``(lower, upper)`` if ``value`` is a finite two-element range around zero, else ``None``."""
    if not (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    ):
        return None
    lower, upper = float(value[0]), float(value[1])
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < 0 < upper):
        return None
    return lower, upper


def _blocked(reason: str, **extra: Any) -> Dict[str, Any]:
    return {"status": "blocked", "reason": reason, **extra}


@dataclass(frozen=True)
class _Candidate:
    kind: str
    label: Any
    value: float
    bound: str
    limit: float


def _best(values: Sequence[float], labels: Sequence[Any], lower: float, upper: float, kind: str) -> Optional[_Candidate]:
    """The tightest ``signed_scale_limit`` candidate among ``values``, and which one produced it."""
    best: Optional[_Candidate] = None
    for value, label in zip(values, labels):
        if not math.isfinite(value):
            raise ValueError(f"{kind} coefficient must be finite")
        if value > 0:
            limit, bound = upper / value, "upper"
        elif value < 0:
            limit, bound = lower / value, "lower"
        else:
            continue
        if best is None or limit < best.limit:
            best = _Candidate(kind, label, float(value), bound, float(limit))
    return best


def _shrink_until_valid(scale: float, violates: Callable[[float], bool], max_tries: int = 1000) -> float:
    """Move ``scale`` toward zero one representable float at a time until ``violates`` clears."""
    tries = 0
    while violates(scale):
        scale = math.nextafter(scale, 0.0)
        tries += 1
        if tries > max_tries:
            raise ValueError("could not find a submitted scale that satisfies every bound")
    return scale


def _coefficient_stats(h: np.ndarray, j: np.ndarray) -> Dict[str, Any]:
    def stats(arr: np.ndarray) -> Dict[str, Any]:
        if arr.size == 0:
            return {"min": 0.0, "max": 0.0, "abs_max": 0.0, "n_nonzero": 0}
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "abs_max": float(np.abs(arr).max()),
            "n_nonzero": int(np.count_nonzero(arr)),
        }

    return {"h": stats(h), "j": stats(j)}


def _requested_scales(
    h: np.ndarray,
    j: np.ndarray,
    ceiling: float,
    all_zero: bool,
    h_range: Tuple[float, float],
    j_range: Tuple[float, float],
    group_range: Optional[Tuple[float, float]],
    qubit_range: Optional[Tuple[float, float]],
    group_members: List[List[float]],
    qubit_members: Dict[int, List[float]],
) -> List[Dict[str, Any]]:
    if all_zero:
        return [{"requested": 1.0, "submitted_scale": 1.0, "h": h.copy(), "j": j.copy()}]

    def violates(scale: float) -> bool:
        if h.size and (np.any(h * scale < h_range[0]) or np.any(h * scale > h_range[1])):
            return True
        if j.size and (np.any(j * scale < j_range[0]) or np.any(j * scale > j_range[1])):
            return True
        if group_range is not None:
            for values in group_members:
                if not values:
                    continue
                total = sum(v * scale for v in values)
                if total < group_range[0] or total > group_range[1]:
                    return True
        if qubit_range is not None:
            for values in qubit_members.values():
                total = sum(v * scale for v in values)
                if total < qubit_range[0] or total > qubit_range[1]:
                    return True
        return False

    base = math.nextafter(ceiling, 0.0)
    results = []
    for fraction in SCALE_FRACTIONS:
        scale = _shrink_until_valid(fraction * base, violates)
        results.append({"requested": fraction, "submitted_scale": scale, "h": h * scale, "j": j * scale})
    return results


def audit_scale(
    nodes: np.ndarray,
    h: np.ndarray,
    edges: np.ndarray,
    j: np.ndarray,
    graph: nx.Graph,
    properties: Dict[str, Any],
) -> Dict[str, Any]:
    """The legal positive scale of a physical model, and its scaled coefficients.

    Returns a blocked audit — never a guessed default — when ``graph`` is
    not a Zephyr graph, when ``properties`` lacks ``h_range`` or a J range,
    when a Zephyr graph's properties lack ``per_group_coupling_range``, or
    when the model touches a qubit or coupler the graph does not have
    (topology drift against the placement that produced the model).
    """
    if graph.graph.get("family") != "zephyr":
        return _blocked(f"unrecognized topology: expected zephyr, got {graph.graph.get('family')!r}")

    h_range = _range_tuple(properties.get("h_range"))
    if h_range is None:
        return _blocked("missing required property: h_range")

    j_range_name: Optional[str] = None
    j_range: Optional[Tuple[float, float]] = None
    for name in _J_RANGE_KEYS:
        j_range = _range_tuple(properties.get(name))
        if j_range is not None:
            j_range_name = name
            break
    if j_range is None:
        return _blocked("missing required property: j_range or extended_j_range")

    group_range_present = properties.get("per_group_coupling_range")
    if group_range_present is None:
        return _blocked("missing required property: per_group_coupling_range for a zephyr topology")
    group_range = _range_tuple(group_range_present)
    if group_range is None:
        return _blocked("malformed property: per_group_coupling_range")

    qubit_range_present = properties.get("per_qubit_coupling_range")
    qubit_range: Optional[Tuple[float, float]] = None
    if qubit_range_present is not None:
        qubit_range = _range_tuple(qubit_range_present)
        if qubit_range is None:
            return _blocked("malformed property: per_qubit_coupling_range")

    nodes = np.asarray(nodes, dtype=np.int64)
    h = np.asarray(h, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), dtype=np.int64)
    j = np.asarray(j, dtype=np.float64)
    if h.shape != nodes.shape:
        raise ValueError(f"h must align with nodes; got {h.shape} and {nodes.shape}")
    if j.shape != (edges.shape[0],):
        raise ValueError(f"j must align with edges; got {j.shape} and {edges.shape}")

    missing_nodes = [int(q) for q in nodes if not graph.has_node(int(q))]
    missing_edges = [[int(a), int(b)] for a, b in edges.tolist() if not graph.has_edge(int(a), int(b))]
    if missing_nodes or missing_edges:
        return _blocked(
            "placement uses qubits or couplers outside the live graph (topology drift)",
            missing_nodes=missing_nodes[:10],
            missing_edges=missing_edges[:10],
        )

    candidates = []
    h_best = _best(h.tolist(), nodes.tolist(), h_range[0], h_range[1], "h")
    if h_best is not None:
        candidates.append(h_best)

    edge_pairs = [(int(a), int(b)) for a, b in edges.tolist()]
    j_best = _best(j.tolist(), [[a, b] for a, b in edge_pairs], j_range[0], j_range[1], "j")
    if j_best is not None:
        candidates.append(j_best)

    edge_j: Dict[Tuple[int, int], float] = {}
    for (a, b), value in zip(edge_pairs, j.tolist()):
        key = (a, b) if a < b else (b, a)
        edge_j[key] = edge_j.get(key, 0.0) + value

    groups = _groups_for(graph)
    group_members: List[List[float]] = []
    for group in groups:
        values = []
        for p, q in group:
            key = (p, q) if p < q else (q, p)
            value = edge_j.get(key)
            if value is not None:
                values.append(value)
        group_members.append(values)
    group_sums = [sum(values) for values in group_members]
    group_best = _best(group_sums, list(range(len(groups))), group_range[0], group_range[1], "group")
    if group_best is not None:
        candidates.append(group_best)

    qubit_members: Dict[int, List[float]] = {}
    if qubit_range is not None:
        for (a, b), value in zip(edge_pairs, j.tolist()):
            qubit_members.setdefault(a, []).append(value)
            qubit_members.setdefault(b, []).append(value)
        qubit_labels = list(qubit_members.keys())
        qubit_sums = [sum(qubit_members[label]) for label in qubit_labels]
        qubit_best = _best(qubit_sums, qubit_labels, qubit_range[0], qubit_range[1], "qubit_sum")
        if qubit_best is not None:
            candidates.append(qubit_best)

    all_zero = not np.any(h) and not np.any(j)
    if candidates:
        winner = min(candidates, key=lambda c: c.limit)
        ceiling = winner.limit
        limiting_constraint: Optional[Dict[str, Any]] = {
            "kind": winner.kind, "label": winner.label, "value": winner.value, "bound": winner.bound,
        }
    else:
        ceiling = math.inf
        limiting_constraint = None

    scales = _requested_scales(
        h, j, ceiling, all_zero, h_range, j_range, group_range, qubit_range, group_members, qubit_members
    )

    return {
        "status": "ok",
        "j_range_used": j_range_name,
        "ceiling": ceiling,
        "all_zero": all_zero,
        "limiting_constraint": limiting_constraint,
        "coefficient_stats": _coefficient_stats(h, j),
        "group_sums": {
            "count": len(group_sums),
            "min": float(min(group_sums)) if group_sums else 0.0,
            "max": float(max(group_sums)) if group_sums else 0.0,
        },
        "scales": scales,
    }
