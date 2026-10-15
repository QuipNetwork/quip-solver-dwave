"""Rebuild the model of a recorded attempt from its nonce.

The coordinator sends h and J with every job, so the miner never draws a
model itself. The measurement scripts do: the attempts log records only the
nonce. The draw is the protocol's own (``quip_msa.draw_ising`` calls
``quip_protocol::chacha8::draw_ising_milli``), and it walks the topology's
edges in order, so a spec with the right edges in the wrong order draws the
wrong model.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from quip_solver_dwave.msa import MsaUnavailable


@dataclass(frozen=True)
class TopologySpec:
    nodes: np.ndarray
    edges: np.ndarray
    dense_edges: np.ndarray
    allowed_h_milli: List[int]
    allowed_j_milli: List[int]


def load_spec(path: Union[str, Path]) -> TopologySpec:
    """Read a quip-coordinator topology spec (``nodes``, ``edges``, allowed sets)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = np.asarray(raw["nodes"], dtype=np.int64)
    edges = np.asarray(raw["edges"], dtype=np.int64).reshape(-1, 2)
    index = {int(label): i for i, label in enumerate(nodes)}
    unknown = [int(label) for label in np.unique(edges) if int(label) not in index]
    if unknown:
        raise ValueError(f"edges name nodes the spec does not list: {unknown[:5]}")
    dense = np.array([[index[int(u)], index[int(v)]] for u, v in edges], dtype=np.int64)
    return TopologySpec(
        nodes=nodes,
        edges=edges,
        dense_edges=dense.reshape(-1, 2),
        allowed_h_milli=[int(v) for v in raw["allowed_h_milli"]],
        allowed_j_milli=[int(v) for v in raw["allowed_j_milli"]],
    )


def model_from_nonce(spec: TopologySpec, nonce_hex: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(h, j)`` in energy units for the model this nonce draws on ``spec``."""
    try:
        import quip_msa
    except ImportError as exc:
        raise MsaUnavailable(
            "quip_msa is not installed; build it from quip-solver-cpu/py with "
            "`maturin develop --release`"
        ) from exc
    nonce = bytes.fromhex(nonce_hex.removeprefix("0x"))
    return quip_msa.draw_ising(
        nonce, len(spec.nodes), len(spec.edges), spec.allowed_h_milli, spec.allowed_j_milli
    )


def load_attempts(path: Union[str, Path]) -> Dict[str, int]:
    """Nonce to its lowest recorded QPU energy, from a ``fetch_attempts.py`` CSV."""
    best: Dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            energy = int(row["raw_best_energy_milli"])
            nonce = row["nonce"]
            if nonce not in best or energy < best[nonce]:
                best[nonce] = energy
    return best


def spec_order(spins: np.ndarray, variables: Sequence[int], spec: TopologySpec) -> np.ndarray:
    """Reorder read columns from the sampler's variable order into spec node order."""
    column = {int(label): i for i, label in enumerate(variables)}
    missing = [int(label) for label in spec.nodes if int(label) not in column]
    if missing:
        raise ValueError(f"the reads lack spec nodes: {missing[:5]}")
    take = [column[int(label)] for label in spec.nodes]
    return np.ascontiguousarray(spins[:, take], dtype=np.int8)
