"""Files of the regime search: placements, chains, nonces, manifests, the spend journal, and atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from quip_solver_dwave import lattices, regimes
from quip_solver_dwave.embeddings import Placement, edge_set
from quip_solver_dwave.replay import TopologySpec

PathLike = Union[str, Path]

LATTICE_FILES = {"diamond-pm1": "diamond.json", "cubic-dimer-pm1": "cubic-dimer.json"}
CLIQUE_FILE = "clique.json"
MANIFEST_FILE = "manifest.json"
SPEND_JOURNAL = "spend.jsonl"
SUBMIT = "submit"
CHARGE = "charge"


def atomic_savez(path: PathLike, **arrays: Any) -> None:
    """Write an npz through a temp file in the same directory, then replace.

    A kill in the middle leaves only the temp file, which a resume check for
    ``<nonce>.npz`` does not see. ``capture_qpu_reads.py`` has the same helper.
    """
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    try:
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)  # pyright: ignore[reportArgumentType]
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path: PathLike, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        Path(tmp).write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_placement(path: PathLike, lattice: str, dims: lattices.Dims, placement: Placement, topology_hash: str) -> None:
    atomic_write_json(path, {
        "lattice": lattice,
        "dims": list(dims),
        "shift": list(placement.shift),
        "topology_hash": topology_hash,
        "labels": placement.labels.tolist(),
        "alive": placement.alive.astype(int).tolist(),
        "keep": placement.keep.astype(int).tolist(),
    })


def archive_topology_digest(nodes: Sequence[int], edges: Iterable[Tuple[int, int]]) -> bytes:
    """32-byte blake2b of the sorted nodes and the sorted ``(min, max)`` edges.

    The identity the regime manifests and embedding files record. It is the
    digest the miner advertised before quip-solver-core 0.0.11, kept here
    because the archived cells under ``~/quip-data/regimes`` carry it. It is
    not the FAM-19 wire hash (``quip_solver_core.session.native_topology_hash``).
    """
    sorted_nodes = sorted(int(n) for n in nodes)
    sorted_edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in edges)
    h = hashlib.blake2b(digest_size=32)
    h.update(len(sorted_nodes).to_bytes(4, "little"))
    for n in sorted_nodes:
        h.update(n.to_bytes(4, "little"))
    h.update(len(sorted_edges).to_bytes(4, "little"))
    for u, v in sorted_edges:
        h.update(u.to_bytes(4, "little"))
        h.update(v.to_bytes(4, "little"))
    return h.digest()


def spec_topology_hash(spec: TopologySpec) -> str:
    """The hash of the working graph, as step 0 writes it into each embedding file."""
    nodes = sorted(int(n) for n in spec.nodes)
    return archive_topology_digest(nodes, sorted(edge_set(spec.edges.tolist()))).hex()


def _load_embedding(path: PathLike, topology_hash: str) -> Dict[str, Any]:
    """An embedding file of step 0. It must be for the topology of ``--spec``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw["topology_hash"] != topology_hash:
        raise ValueError(
            f"{path} is for the topology {raw['topology_hash']}, and --spec gives {topology_hash}; run step 0 again"
        )
    return raw


def load_placement(path: PathLike, topology_hash: str) -> "tuple[lattices.Dims, Placement]":
    raw = _load_embedding(path, topology_hash)
    dims = (int(raw["dims"][0]), int(raw["dims"][1]), int(raw["dims"][2]))
    placement = Placement(
        (int(raw["shift"][0]), int(raw["shift"][1])),
        np.asarray(raw["labels"], dtype=np.int64),
        np.asarray(raw["alive"], dtype=bool),
        np.asarray(raw["keep"], dtype=bool),
    )
    return dims, placement


def load_chains(path: PathLike, topology_hash: str) -> List[List[int]]:
    raw = _load_embedding(path, topology_hash)
    return [[int(q) for q in chain] for chain in raw["chains"]]


def load_cell(name: str, spec: TopologySpec, embeddings_dir: Optional[PathLike]) -> regimes.Cell:
    """A cell that draws from the protocol stream. The clique cell has no ``Cell``."""
    if name in regimes.NATIVE_CELLS:
        return regimes.native_cell(name, spec)
    if embeddings_dir is None:
        raise ValueError(f"{name} needs --embeddings-dir")
    dims, placement = load_placement(Path(embeddings_dir) / LATTICE_FILES[name], spec_topology_hash(spec))
    if name == "diamond-pm1":
        return regimes.diamond_cell(dims, placement)
    return regimes.cubic_dimer_cell(dims, placement)


def model_source(
    name: str, spec: TopologySpec, embeddings_dir: Optional[PathLike], clique_holdings: int
) -> Tuple[str, Callable[[str], regimes.Model]]:
    """The kernel label of a cell, and the function that builds the model of a nonce.

    The clique cell has one asset for each chain of the embedding that step 0 wrote.
    """
    if name != "clique-portfolio":
        cell = load_cell(name, spec, embeddings_dir)
        return cell.kernel_label, lambda nonce: regimes.build_model(cell, nonce)
    if embeddings_dir is None:
        raise ValueError(f"{name} needs --embeddings-dir")
    chains = load_chains(Path(embeddings_dir) / CLIQUE_FILE, spec_topology_hash(spec))
    working = edge_set(spec.edges.tolist())
    return regimes.CPU_SA, lambda nonce: regimes.build_clique_model(nonce, chains, working, clique_holdings)


def cell_nonces(cell_dir: PathLike, count: int, rng_seed: int) -> List[str]:
    """The nonce list of a cell. The first call writes it, and later calls must agree with it."""
    path = Path(cell_dir) / "nonces.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["rng_seed"] != rng_seed or len(raw["nonces"]) != count:
            raise ValueError(
                f"{path} holds seed {raw['rng_seed']} and {len(raw['nonces'])} nonces; refusing to mix runs"
            )
        return list(raw["nonces"])
    nonces = regimes.make_nonces(count, rng_seed)
    Path(cell_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"rng_seed": rng_seed, "nonces": nonces})
    return nonces


def cell_identity(
    name: str, spec: TopologySpec, clique_holdings: int, embeddings_dir: Optional[PathLike] = None
) -> Dict[str, Any]:
    """Record the ordered draw inputs and the embedding, as well as the working graph."""
    ordered = json.dumps({"nodes": spec.nodes.tolist(), "edges": spec.dense_edges.tolist()}, sort_keys=True)
    identity: Dict[str, Any] = {
        "topology_hash": spec_topology_hash(spec),
        "model_order_hash": hashlib.sha256(ordered.encode()).hexdigest(),
    }
    if name not in regimes.NATIVE_CELLS:
        if embeddings_dir is None:
            raise ValueError(f"{name} needs --embeddings-dir")
        filename = CLIQUE_FILE if name == "clique-portfolio" else LATTICE_FILES[name]
        payload = _load_embedding(Path(embeddings_dir) / filename, identity["topology_hash"])
        identity["embedding_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if name == "clique-portfolio":
        identity["clique_holdings"] = clique_holdings
    return identity


def check_manifest(cell_dir: PathLike, **fields: Any) -> None:
    """Record what defines the data of a cell on first use, and refuse a later run that differs.

    The capture records the topology, the holdings of the clique cell, the reads, the anneal times, and the kind of
    sampler. The ladder records the topology and the holdings. A script adds the fields that the file lacks.
    """
    path = Path(cell_dir) / MANIFEST_FILE
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for name, value in fields.items():
        if name in saved and saved[name] != value:
            raise ValueError(
                f"{name} is {saved[name]} in {path} and {value} in this run; refusing to mix runs. "
                "Use a new round directory, or remove the cell to start it again"
            )
    if any(name not in saved for name in fields):
        Path(cell_dir).mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {**saved, **fields})


def append_spend(round_dir: PathLike, event: str, cell: str, nonce: str, anneal_us: int, micros: int) -> None:
    """Append one line to the spend journal of the round, and force it to the disk.

    The capture writes a ``submit`` line with the estimate before each job, and a ``charge`` line with the real
    access time directly after the answer. A failure after the answer therefore cannot lose the charge.
    """
    path = Path(round_dir) / SPEND_JOURNAL
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"event": event, "cell": cell, "nonce": nonce, "anneal_us": anneal_us, "us": micros}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_spend(round_dir: PathLike) -> Tuple[int, int]:
    """Access time that the round has spent, in microseconds, and the count of jobs with no recorded charge.

    A job with no recorded charge counts at its estimate, because the QPU may have run it. A charge settles the
    latest open job of its capture, so a capture that ran twice counts twice.
    """
    path = Path(round_dir) / SPEND_JOURNAL
    if not path.exists():
        return 0, 0
    charged = 0
    open_jobs: Dict[Tuple[str, str, int], List[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        key = (entry["cell"], entry["nonce"], entry["anneal_us"])
        if entry["event"] == SUBMIT:
            open_jobs.setdefault(key, []).append(int(entry["us"]))
        elif entry["event"] == CHARGE:
            open_jobs[key].pop()
            charged += int(entry["us"])
        else:
            raise ValueError(f"{path} holds a line with the unknown event {entry['event']}")
    estimates = [us for jobs in open_jobs.values() for us in jobs]
    return charged + sum(estimates), len(estimates)
