"""Defect-qubit clamping and full-topology reconstruction (v0.2 port).

Offline qubits are fixed to deterministic spins (seeded from a nonce) and their
couplings are absorbed into neighbor biases so the reduced problem stays
energy-consistent with the full topology.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class DefectInfo:
    """Metadata needed to reconstruct a full-topology sample from a reduced one."""

    fixed_spins: Dict[int, int]
    energy_offset: float
    removed_edges: Dict[Tuple[int, int], float]


def live_topology(
    nodes: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Canonical live (non-defective) node/edge orderings."""
    defective_set = set(defective_qubits)
    live_nodes = [n for n in nodes if n not in defective_set]
    live_edges = [
        (u, v)
        for (u, v) in edges
        if u not in defective_set
        and v not in defective_set
        and (u, v) not in defective_edges
    ]
    return live_nodes, live_edges


def clamp_fixed_variables(
    h: Dict[int, float],
    j: Dict[Tuple[int, int], float],
    nonce_seed: Union[int, bytes],
    defective_qubits: Sequence[int],
    defective_edges: "set[Tuple[int, int]]",
    start_state: Optional[Dict[int, int]] = None,
) -> Tuple[
    Dict[int, float],
    Dict[Tuple[int, int], float],
    Dict[int, int],
    float,
    Dict[Tuple[int, int], float],
]:
    """Clamp defective qubits; return reduced h/J + reconstruction metadata.

    A missing qubit takes its spin from ``start_state`` when the job is a warm
    start, and from the nonce-seeded draw otherwise. The reduced problem folds
    each clamped spin into its neighbours' biases, so clamping to anything but
    the start state's own spin would hand the QPU a start state that disagrees
    with the problem it is asked to refine.
    """
    defective_set = set(defective_qubits)
    fixed_spins: Dict[int, int] = {}
    if isinstance(nonce_seed, (bytes, bytearray)):
        nonce_seed = int.from_bytes(nonce_seed, "big")
    rng = np.random.default_rng(nonce_seed)
    if start_state is not None:
        for qubit in defective_qubits:
            spin = start_state.get(qubit)
            if spin is None:
                # This defect sits outside the job's own graph, so the warm
                # start carries no spin for it: clamp it exactly as the cold
                # (seedless) path would.
                fixed_spins[qubit] = int(2 * rng.integers(2) - 1)
            else:
                fixed_spins[qubit] = 1 if spin >= 0 else -1
    else:
        for qubit in defective_qubits:
            fixed_spins[qubit] = int(2 * rng.integers(2) - 1)

    h_reduced = {k: v for k, v in h.items() if k not in defective_set}
    for (u, v), j_val in j.items():
        if u in defective_set and v not in defective_set:
            h_reduced[v] = h_reduced.get(v, 0.0) + j_val * fixed_spins[u]
        elif v in defective_set and u not in defective_set:
            h_reduced[u] = h_reduced.get(u, 0.0) + j_val * fixed_spins[v]

    j_reduced = {
        (u, v): val
        for (u, v), val in j.items()
        if u not in defective_set
        and v not in defective_set
        and (u, v) not in defective_edges
    }

    energy_offset = 0.0
    for k, s_k in fixed_spins.items():
        energy_offset += h.get(k, 0.0) * s_k
    for (u, v), j_val in j.items():
        if u in defective_set and v in defective_set:
            energy_offset += j_val * fixed_spins[u] * fixed_spins[v]

    removed_edges = {
        (u, v): val
        for (u, v), val in j.items()
        if u not in defective_set
        and v not in defective_set
        and (u, v) in defective_edges
    }
    return h_reduced, j_reduced, fixed_spins, energy_offset, removed_edges


def prepare_problem(
    h: Dict[int, float],
    j: Dict[Tuple[int, int], float],
    *,
    defective_qubits: Sequence[int] = (),
    defective_edges: Optional[set] = None,
    nonce_seed: Union[int, bytes, None] = None,
    start_state: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], Optional[DefectInfo]]:
    """Apply defect clamping when the live graph is missing qubits or couplers.

    The seed picks spins for clamped qubits, so it is required only when there
    are qubits to clamp. Missing couplers alone still have to be removed: a
    coupler the QPU does not have makes SAPI reject the whole problem with
    ``ProblemStructureError``, which is what a seedless early return used to
    cause on a chip whose qubits all match but whose couplers do not.

    ``start_state`` (qubit label to spin) replaces the seed for a warm start:
    the clamped spins are read from it, so no seed is needed.
    """
    de = defective_edges or set()
    if not (defective_qubits or de):
        return h, j, None
    if defective_qubits and nonce_seed is None and start_state is None:
        raise ValueError(
            "clamping defective qubits needs a nonce seed; "
            f"{len(defective_qubits)} qubits are missing from the live graph"
        )
    h_r, j_r, fixed, offset, removed = clamp_fixed_variables(
        h,
        j,
        nonce_seed if nonce_seed is not None else 0,
        defective_qubits,
        de,
        start_state=start_state,
    )
    return h_r, j_r, DefectInfo(fixed, offset, removed)


def reconstruct_sample(
    reduced: Dict[int, int],
    reduced_energy: float,
    defect_info: Optional[DefectInfo],
) -> Tuple[Dict[int, int], float]:
    """Reinsert clamped spins and correct energy for the full topology."""
    if defect_info is None:
        return reduced, reduced_energy
    full = dict(reduced)
    full.update(defect_info.fixed_spins)
    energy = reduced_energy + defect_info.energy_offset
    for (u, v), j_val in defect_info.removed_edges.items():
        energy += j_val * full[u] * full[v]
    return full, energy


def reconstruct_samples(
    spins: np.ndarray,
    variables: List[int],
    energies: List[float],
    defect_info: Optional[DefectInfo],
) -> Tuple[np.ndarray, List[int], List[float]]:
    """Reinsert clamped spins for a whole batch of reads at once.

    The batch form of :func:`reconstruct_sample`, over the array the sampler
    returned rather than a dict per read. A clamped qubit holds one spin for
    every read, so it becomes a constant column appended to the array, and the
    energy correction for each removed edge is one vectorised term.
    """
    if defect_info is None:
        return spins, variables, energies
    # No shortcut on empty fixed_spins/removed_edges: energy_offset still
    # applies, and skipping it here silently drops the clamped contribution.

    n_reads = spins.shape[0]
    fixed = list(defect_info.fixed_spins.items())
    if fixed:
        block = np.empty((n_reads, len(fixed)), dtype=np.int8)
        for col, (_, spin) in enumerate(fixed):
            block[:, col] = 1 if spin >= 0 else -1
        spins = np.hstack([spins, block])
        variables = list(variables) + [int(q) for q, _ in fixed]

    corrected = np.asarray(energies, dtype=np.float64) + defect_info.energy_offset
    if defect_info.removed_edges:
        col_of = {v: i for i, v in enumerate(variables)}
        for (u, v), j_val in defect_info.removed_edges.items():
            corrected += (
                j_val
                * spins[:, col_of[u]].astype(np.float64)
                * spins[:, col_of[v]].astype(np.float64)
            )
    return spins, variables, [float(e) for e in corrected]
