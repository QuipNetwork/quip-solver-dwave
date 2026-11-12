"""Embeddings of the regime-search cells onto the working graph.

A lattice cell uses one physical qubit for each spin. The map is a formula on
Zephyr coordinates, shifted to the place with the fewest vacancies. A spin
whose qubit is dead or outside the graph becomes a vacancy, as in the largest
lattice of King et al. The clique cell uses a chain of qubits for each
variable.

The validators raise ``ValueError`` with the first fault they find, because a
wrong embedding makes the QPU solve a different problem without any error.

Zephyr geometry, for a graph ``Z(m, t)``: a vertical qubit ``(0, w, k, j, z)``
sits in column ``w`` and spans rows ``r`` and ``r + 1`` with ``r = 2z + j``. A
horizontal qubit ``(1, w, k, j, z)`` sits in row ``w`` and spans columns ``s``
and ``s + 1`` with ``s = 2z + j``. A vertical and a horizontal qubit share a
coupler when the row is one of the two spanned rows and the column is one of
the two spanned columns, for every pair of wire numbers ``k``. Two qubits of
the same line ``(u, w, k)`` share a coupler when ``2z + j`` differs by 1 or 2.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, cast

import numpy as np

from quip_solver_dwave import lattices

Edge = Tuple[int, int]
Shift = Tuple[int, int]

ZEPHYR_M = 12
ZEPHYR_T = 4


def zephyr_label(u: int, w: int, k: int, j: int, z: int, m: int = ZEPHYR_M, t: int = ZEPHYR_T) -> int:
    """The linear qubit label of a Zephyr coordinate, or -1 outside the graph."""
    if not (0 <= w <= 2 * m and 0 <= k < t and 0 <= z < m and j in (0, 1)):
        return -1
    return (((u * (2 * m + 1) + w) * t + k) * 2 + j) * m + z


def diamond_labels(dims: lattices.Dims, shift: Shift, m: int = ZEPHYR_M, t: int = ZEPHYR_T) -> np.ndarray:
    """The qubit label of each diamond site on an ideal ``Z(m, t)``, or -1.

    Seen from above, the diamond lattice is a square grid turned by 45
    degrees: a site of an even level is a vertical qubit, a site of an odd
    level is a horizontal qubit, and the four grid neighbours of a site are
    exactly the qubits that cross it. The four levels that share one grid
    point take the four wire numbers, so a depth of 16 uses every qubit of
    the region.
    """
    ly = dims[1]
    labels = []
    for x, y, level in lattices.diamond_sites(dims).tolist():
        if level // 4 >= t:
            raise ValueError(f"a depth of {dims[2]} needs more than {t} wires for each line")
        if level % 2 == 0:
            column = (x + y - 1) // 2 + shift[0]
            row = (x - y - 1) // 2 + ly // 2 + shift[1]
            labels.append(zephyr_label(0, column, level // 4, row % 2, row // 2, m, t))
        else:
            row = (x - y) // 2 + ly // 2 + shift[1]
            column = (x + y) // 2 - 1 + shift[0]
            labels.append(zephyr_label(1, row, level // 4, column % 2, column // 2, m, t))
    return np.asarray(labels, dtype=np.int64)


def cubic_dimer_labels(dims: lattices.Dims, shift: Shift, m: int = ZEPHYR_M, t: int = ZEPHYR_T) -> np.ndarray:
    """The qubit label of each cubic dimer spin on an ideal ``Z(m, t)``, or -1.

    The first spin of site ``(x, y, z)`` is a vertical qubit and its x bonds
    run along that qubit's line. The second spin is a horizontal qubit and its
    y bonds run along that line. Layer ``z`` picks the wire ``z // 2`` and one
    of the two crossing positions, so the depth is 2 t at most.
    """
    lx, ly, lz = dims
    if lz > 2 * t:
        raise ValueError(f"a depth of {lz} needs more than {t} wires for each line")
    n = lx * ly * lz
    labels = np.full(2 * n, -1, dtype=np.int64)
    for z, y, x in itertools.product(range(lz), range(ly), range(lx)):
        site = x + y * lx + z * lx * ly
        row = 2 * x + shift[1]
        column = 2 * y + shift[0]
        labels[site] = zephyr_label(0, column + z % 2, z // 2, row % 2, row // 2, m, t)
        labels[site + n] = zephyr_label(1, row + z % 2, z // 2, column % 2, column // 2, m, t)
    return labels


@dataclass(frozen=True)
class Placement:
    """A lattice on the working graph. Arrays use the full lattice numbering.

    ``labels[i]`` is the qubit of spin ``i``. ``alive[i]`` is False for a
    vacancy. ``keep[e]`` is False for a lattice edge that the problem drops.
    """

    shift: Shift
    labels: np.ndarray
    alive: np.ndarray
    keep: np.ndarray

    @property
    def vacant_spins(self) -> int:
        return int((~self.alive).sum())

    @property
    def dropped_edges(self) -> int:
        return int((~self.keep).sum())


def place_lattice(
    labels: np.ndarray,
    edges: np.ndarray,
    made_by: np.ndarray,
    working_nodes: Set[int],
    working_edges: Set[Edge],
    shift: Shift,
) -> Placement:
    """Apply the dead qubits and couplers of the working graph to an ideal map.

    ``made_by[e]`` groups the edges: -1 marks a dimer, and any other value
    names the logical bond of the edge. A dimer that cannot run makes both of
    its spins vacant. A logical bond runs only when all of its edges run. A
    spin that keeps no edge has nothing to solve, so it is vacant too.
    """
    alive = np.array([int(q) in working_nodes for q in labels], dtype=bool)

    def runs(u: int, v: int) -> bool:
        a, b = int(labels[u]), int(labels[v])
        return bool(alive[u] and alive[v]) and (min(a, b), max(a, b)) in working_edges

    for (u, v), group in zip(edges.tolist(), made_by.tolist()):
        if group == -1 and not runs(u, v):
            alive[u] = alive[v] = False
    ok = np.array([runs(u, v) for u, v in edges.tolist()], dtype=bool)
    broken = {group for group, fine in zip(made_by.tolist(), ok.tolist()) if group != -1 and not fine}
    keep = ok & ~np.isin(made_by, list(broken))
    touched = np.zeros(len(labels), dtype=bool)
    touched[edges[keep].ravel()] = True
    return Placement(shift, labels, alive & touched, keep)


def best_placement(
    label_fn: Callable[[Shift], np.ndarray],
    edges: np.ndarray,
    made_by: np.ndarray,
    working_nodes: Set[int],
    working_edges: Set[Edge],
    max_shift: int = 2 * ZEPHYR_M,
) -> Placement:
    """The shift with the fewest vacancies, then the fewest dropped edges.

    A site that falls outside the graph is a vacancy like any other, so a
    lattice larger than the graph is cut to the shape of the graph.
    """
    best: Placement | None = None
    for across in range(-max_shift, max_shift + 1):
        for down in range(-max_shift, max_shift + 1):
            candidate = place_lattice(
                label_fn((across, down)), edges, made_by, working_nodes, working_edges, (across, down)
            )
            if best is None or (candidate.vacant_spins, candidate.dropped_edges) < (
                best.vacant_spins,
                best.dropped_edges,
            ):
                best = candidate
    if best is None or not best.alive.any():
        raise ValueError("no spin of the lattice lands on the graph at any shift")
    return best


def edge_set(edges: Iterable[Sequence[int]]) -> Set[Edge]:
    """Undirected edges as ``(low, high)`` pairs."""
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}


def validate_lattice_embedding(
    n_spins: int, lattice_edges: np.ndarray, mapping: Sequence[int], working_edges: Set[Edge]
) -> None:
    """``mapping[i]`` is the physical qubit of spin ``i``."""
    if len(mapping) != n_spins:
        raise ValueError(f"the map covers {len(mapping)} spins; the lattice has {n_spins}")
    if len(set(int(q) for q in mapping)) != n_spins:
        raise ValueError("the map is not injective: two spins share a qubit")
    for u, v in lattice_edges.tolist():
        a, b = int(mapping[u]), int(mapping[v])
        if (min(a, b), max(a, b)) not in working_edges:
            raise ValueError(f"bond ({u}, {v}) maps to ({a}, {b}), which is not a working coupler")


def validate_clique_embedding(chains: Sequence[Sequence[int]], working_edges: Set[Edge]) -> None:
    """``chains[i]`` holds the physical qubits of variable ``i``."""
    owner: Dict[int, int] = {}
    for index, chain in enumerate(chains):
        if not chain:
            raise ValueError(f"chain {index} is empty")
        for qubit in chain:
            if int(qubit) in owner:
                raise ValueError(f"qubit {qubit} is in chain {owner[int(qubit)]} and in chain {index}")
            owner[int(qubit)] = index
    inside: List[Set[Edge]] = [set() for _ in chains]
    linked: Set[Edge] = set()
    for a, b in working_edges:
        if a in owner and b in owner:
            ca, cb = owner[a], owner[b]
            if ca == cb:
                inside[ca].add((a, b))
            else:
                linked.add((min(ca, cb), max(ca, cb)))
    for index, chain in enumerate(chains):
        reached = {int(chain[0])}
        frontier = [int(chain[0])]
        while frontier:
            here = frontier.pop()
            for a, b in inside[index]:
                other = b if a == here else a if b == here else None
                if other is not None and other not in reached:
                    reached.add(other)
                    frontier.append(other)
        if len(reached) != len(chain):
            raise ValueError(f"chain {index} is not connected")
    count = len(chains)
    missing = [(i, j) for i in range(count) for j in range(i + 1, count) if (i, j) not in linked]
    if missing:
        raise ValueError(f"{len(missing)} chain pairs share no working coupler; first: {missing[0]}")


def embed_clique(
    h: np.ndarray,
    edges: np.ndarray,
    j: np.ndarray,
    chains: Sequence[Sequence[int]],
    working_edges: Set[Edge],
    prefactor: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """The logical problem on its chains: ``(nodes, h, edges, j, chain_strength)``.

    The chain strength is Ocean's uniform torque compensation. The portfolio
    benchmark measured 17% broken chains at a prefactor of 2 and 0.42% at 3.
    """
    import dimod
    from dwave.embedding import embed_bqm
    from dwave.embedding.chain_strength import uniform_torque_compensation

    logical = dimod.BinaryQuadraticModel.from_ising(
        {i: float(b) for i, b in enumerate(h)},
        {(int(u), int(v)): float(b) for (u, v), b in zip(edges.tolist(), j)},
    )
    embedding = {i: [int(q) for q in chain] for i, chain in enumerate(chains)}
    used = {q for chain in embedding.values() for q in chain}
    adjacency: Dict[int, Set[int]] = {q: set() for q in used}
    for a, b in working_edges:
        if a in used and b in used:
            adjacency[a].add(b)
            adjacency[b].add(a)
    strength = float(uniform_torque_compensation(logical, embedding, prefactor))
    target = embed_bqm(logical, embedding, adjacency, chain_strength=strength)
    # dimod types a variable as Hashable. Every label here is a qubit number.
    linear = {cast(int, q): float(bias) for q, bias in target.linear.items()}
    quadratic = {
        (min(cast(int, u), cast(int, v)), max(cast(int, u), cast(int, v))): float(bias)
        for (u, v), bias in target.quadratic.items()
    }
    nodes = np.asarray(sorted(linear), dtype=np.int64)
    pairs = sorted(quadratic)
    return (
        nodes,
        np.asarray([linear[int(q)] for q in nodes], dtype=np.float64),
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray([quadratic[pair] for pair in pairs], dtype=np.float64),
        strength,
    )


def resolve_chains(
    spins: np.ndarray, variables: Sequence[int], chains: Sequence[Sequence[int]]
) -> Tuple[np.ndarray, float]:
    """Logical reads by majority vote, and the share of broken chains.

    ``spins`` is ``(reads, len(variables))`` of +1/-1 on physical labels. A
    tie goes to the first qubit of the chain, so the result is deterministic.
    """
    column = {int(label): i for i, label in enumerate(variables)}
    physical = np.asarray(spins, dtype=np.int64)
    logical = np.empty((physical.shape[0], len(chains)), dtype=np.int8)
    broken = 0
    for index, chain in enumerate(chains):
        block = physical[:, [column[int(q)] for q in chain]]
        total = block.sum(axis=1)
        logical[:, index] = np.where(total == 0, block[:, 0], np.sign(total))
        broken += int((np.abs(total) != len(chain)).sum())
    return logical, broken / (physical.shape[0] * len(chains))
