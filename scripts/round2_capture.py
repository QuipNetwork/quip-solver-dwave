#!/usr/bin/env python3
"""Auditable QPU capture for Round 2: an immutable manifest, an explicit
approval, and a capture run that can only spend against both.

Four subcommands share one manifest file:

``manifest``  Reads a job-spec JSON (cells, nonces, requested scales, anneal
              times, reads, an order seed, the bundle and range-audit roots,
              the solver properties, and a reservation margin) and writes
              every job's identity -- model, embedding, and solver-property
              hashes, the scaled coefficients' hash, and a seeded submission
              order -- to one file. The manifest's own sha256, taken over the
              file's raw bytes, is its identity: nothing inside the file
              claims to be that hash, so nothing in it can go stale against
              itself.
``approve``   Records a cap in microseconds against one named manifest hash,
              with an approver. There is no default cap.
``dry-run``   Prints per-cell counts, access-time estimates, and the round's
              current charged-plus-reserved spend, and submits nothing.
``capture``   Runs a manifest's jobs, in their seeded order, against an
              approval that names this exact manifest hash and cap. Follows
              the capture sequence Round 1 established in
              ``regime_capture.py``: round lock, preflight against the live
              topology, reserve, submit, charge before decode, decode, save.
              Physical scaling comes from Task 6
              (``round2_range_audit.physical_arrays_for`` and its
              ``arrays.npz``); every submission passes ``auto_scale=False``.

This runner never submits without an approval naming its own manifest hash
and a cap, never retries a job whose outcome is unknown, and never estimates
a cap from a solver whose timing properties cannot bound one.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quip_solver_dwave import regime_io, regimes, replay, round2_io

FAILED_FILE = "failed_captures.json"
MODE_MARKER_FILE = "capture-mode.json"


def _load_round2_range_audit():
    """``round2_range_audit.py`` as a module: ``scripts/`` is not a package.

    Reuses its ``physical_arrays_for`` rather than re-deriving Task 6's
    embedding of a physical model, which is the one place that logic lives.
    """
    path = Path(__file__).resolve().parent / "round2_range_audit.py"
    spec = importlib.util.spec_from_file_location("round2_range_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


round2_range_audit = _load_round2_range_audit()


# --------------------------------------------------------------- primitives


def estimate_access_us(properties: Dict[str, Any], num_qubits: int, num_reads: int, anneal_us: int) -> float:
    """The access-time upper bound of one job, from the solver's own timing model.

    Wraps ``dwave.cloud.solver.StructuredSolver.estimate_qpu_access_time``,
    which reads only ``properties['problem_timing_data']`` -- no live
    connection, no sampler. Raises when the given properties cannot bound an
    access time, so a manifest can report that limit before approval rather
    than promise an unenforceable hard cap.
    """
    from dwave.cloud.solver import StructuredSolver

    class _PropertyHolder:
        """Duck-types the one attribute ``estimate_qpu_access_time`` reads.

        Building a real ``StructuredSolver`` needs a live client connection;
        the method only ever touches ``self.properties``, so a stand-in with
        just that attribute lets the estimate run fully offline.
        """

        def __init__(self, properties: Dict[str, Any]) -> None:
            self.properties = properties

    stand_in = _PropertyHolder(properties)
    try:
        return float(
            StructuredSolver.estimate_qpu_access_time(
                stand_in,  # type: ignore[arg-type]  # duck-typed stand-in; see _PropertyHolder
                num_qubits=num_qubits, num_reads=num_reads, annealing_time=float(anneal_us),
            )
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"cannot bound QPU access time from the given properties: {exc}") from exc


def coefficient_hash(h: np.ndarray, j: np.ndarray) -> str:
    """The identity of one job's exact physical coefficients, in submission order."""
    return hashlib.sha256(
        (round2_io.array_hash(np.asarray(h)) + round2_io.array_hash(np.asarray(j))).encode()
    ).hexdigest()


def request_identity_hash(nodes: np.ndarray, h: np.ndarray, edges: np.ndarray, j: np.ndarray) -> str:
    """The identity of the exact arrays handed to the sampler for one job.

    Preflight refuses any job whose physical graph the live chip cannot run
    without clamping, so this is also what reached SAPI: nothing here is
    reduced by defect clamping.
    """
    parts = "".join(
        round2_io.array_hash(np.asarray(a)) for a in (nodes, h, edges, j)
    )
    return hashlib.sha256(parts.encode()).hexdigest()


def _scale_suffix(fraction: float) -> str:
    return f"{fraction:.2f}".replace(".", "")


# ------------------------------------------------------------------ context


@dataclass(frozen=True)
class Context:
    """Where a job's physical model and the solver's properties come from."""

    bundles_root: Path
    range_audit_dir: Path
    spec: Optional[replay.TopologySpec]
    embeddings_dir: Optional[Path]
    properties: Dict[str, Any]
    properties_hash: str


def _bundle_and_physical(cell: str, nonce: str, ctx: Context):
    """A bundle's manifest and arrays, plus its physical nodes/edges (Task 6's mapping)."""
    manifest, arrays = round2_io.read_bundle(ctx.bundles_root / cell / nonce)
    phys_nodes, _, phys_edges, _, chain_strength = round2_range_audit.physical_arrays_for(
        cell, manifest, arrays, ctx.spec, ctx.embeddings_dir
    )
    return manifest, arrays, phys_nodes, phys_edges, chain_strength


def _range_audit_entry(cell: str, nonce: str, ctx: Context) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    model_dir = ctx.range_audit_dir / cell / nonce
    ra_manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if ra_manifest.get("status") != "ok":
        raise ValueError(f"{model_dir}: the physical-range audit is blocked: {ra_manifest.get('reason')}")
    with np.load(model_dir / "arrays.npz") as npz:
        arrays = {name: np.asarray(npz[name]) for name in npz.files}
    return ra_manifest, arrays


def _scaled_hj(
    ra_manifest: Dict[str, Any], ra_arrays: Dict[str, np.ndarray], requested_scale: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    for entry in ra_manifest["scales"]:
        if math.isclose(entry["requested"], requested_scale, rel_tol=0.0, abs_tol=1e-9):
            suffix = _scale_suffix(requested_scale)
            return float(entry["submitted_scale"]), ra_arrays[f"h_{suffix}"], ra_arrays[f"j_{suffix}"]
    raise ValueError(f"requested scale {requested_scale} is not one the range audit recorded")


def _clique_chains_for(ctx: Context) -> List[List[int]]:
    if ctx.spec is None or ctx.embeddings_dir is None:
        raise ValueError("clique-portfolio needs spec_path and embeddings_dir in the job spec")
    topology_hash = regime_io.spec_topology_hash(ctx.spec)
    return regime_io.load_chains(Path(ctx.embeddings_dir) / regime_io.CLIQUE_FILE, topology_hash)


_NATIVE_CELLS = ("native-pm1", "native-125")


def _embedding_hash(cell: str, ctx: Context) -> Optional[str]:
    """The embedding's identity, recomputed from the file on disk right now (I5).

    A bundle's own ``identity.embedding_hash`` is part of the bundle's own
    hashed content, so comparing it against itself proves nothing: it can
    only ever equal itself, which made the old drift check tautological.
    This instead re-reads the chain (or placement) file fresh, so a file
    that changed after the bundle was written -- while the bundle's stored
    copy of its hash did not -- is caught. Native cells have no embedding.
    """
    if cell in _NATIVE_CELLS:
        return None
    if ctx.spec is None or ctx.embeddings_dir is None:
        raise ValueError(f"{cell} needs spec_path and embeddings_dir in the job spec")
    identity = regime_io.cell_identity(cell, ctx.spec, regimes.CLIQUE_HOLDINGS, ctx.embeddings_dir)
    return identity.get("embedding_hash")


def _assert_arrays_pair(cell: str, nonce: str, phys_nodes, phys_edges, h, j) -> None:
    """Physical nodes must parallel ``h``, and physical edges must parallel ``j`` (I5).

    ``physical_arrays_for`` recomputes physical nodes/edges from whatever
    chain (or placement) file is on disk right now; the scaled ``h``/``j``
    come from Task 6's ``arrays.npz``, written earlier and unconditionally.
    Nothing else ties those two sources together positionally, so a changed
    chain file with the same topology hash could pair Task 6's coefficients
    with the wrong qubits without changing any hash this runner checks.
    """
    if len(phys_nodes) != len(h) or len(phys_edges) != len(j):
        raise ValueError(
            f"{cell}/{nonce}: physical arrays do not pair (nodes={len(phys_nodes)} vs h={len(h)}, "
            f"edges={len(phys_edges)} vs j={len(j)}); refusing to submit a mismatched problem"
        )


# ---------------------------------------------------------------- manifest


def build_job(
    cell: str, nonce: str, requested_scale: float, anneal_us: int, reads: int, margin: float, ctx: Context
) -> Dict[str, Any]:
    manifest, _arrays, phys_nodes, phys_edges, _chain_strength = _bundle_and_physical(cell, nonce, ctx)
    ra_manifest, ra_arrays = _range_audit_entry(cell, nonce, ctx)
    submitted_scale, h_scaled, j_scaled = _scaled_hj(ra_manifest, ra_arrays, requested_scale)
    _assert_arrays_pair(cell, nonce, phys_nodes, phys_edges, h_scaled, j_scaled)
    embedding_hash = _embedding_hash(cell, ctx)
    model_hash = manifest["hash"]
    estimate_us = estimate_access_us(ctx.properties, len(phys_nodes), reads, anneal_us)
    capture_key = round2_io.run_id({
        "cell": cell, "nonce": nonce, "model_hash": model_hash,
        "requested_scale": requested_scale, "anneal_us": anneal_us, "reads": reads,
        "embedding_hash": embedding_hash, "solver_property_hash": ctx.properties_hash,
    })
    return {
        "cell": cell, "nonce": nonce, "model_hash": model_hash,
        "requested_scale": requested_scale, "submitted_scale": submitted_scale,
        "anneal_us": anneal_us, "reads": reads, "num_qubits": len(phys_nodes),
        "embedding_hash": embedding_hash, "solver_property_hash": ctx.properties_hash,
        "coefficient_hash": coefficient_hash(h_scaled, j_scaled),
        "ceiling": ra_manifest["ceiling"], "limiting_constraint": ra_manifest["limiting_constraint"],
        "chain_strength": ra_manifest.get("chain_strength"),
        "chain_strength_ratio_to_logical": ra_manifest.get("chain_strength_ratio_to_logical"),
        "estimate_us": estimate_us, "reservation_us": estimate_us * margin,
        "capture_key": capture_key,
    }


def _randomized_job_order(jobs: Sequence[Dict[str, Any]], seed: int) -> List[str]:
    """Shuffle settings within model blocks, then flatten shuffled whole blocks."""
    blocks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for job in jobs:
        blocks.setdefault((str(job["cell"]), str(job["nonce"])), []).append(job)

    rng = random.Random(seed)
    shuffled_blocks = list(blocks.values())
    rng.shuffle(shuffled_blocks)
    for block in shuffled_blocks:
        rng.shuffle(block)
    return [str(job["capture_key"]) for block in shuffled_blocks for job in block]


def build_manifest(job_spec: Dict[str, Any]) -> Dict[str, Any]:
    properties_path = Path(job_spec["properties_path"])
    properties = json.loads(properties_path.read_text(encoding="utf-8"))
    properties_hash = hashlib.sha256(properties_path.read_bytes()).hexdigest()
    spec_path = job_spec.get("spec_path")
    spec = replay.load_spec(spec_path) if spec_path else None
    embeddings_dir = job_spec.get("embeddings_dir")
    ctx = Context(
        bundles_root=Path(job_spec["bundles_root"]),
        range_audit_dir=Path(job_spec["range_audit_dir"]),
        spec=spec,
        embeddings_dir=Path(embeddings_dir) if embeddings_dir else None,
        properties=properties,
        properties_hash=properties_hash,
    )
    margin = float(job_spec["reservation_margin"])
    reads = int(job_spec["reads"])

    jobs = []
    for cell in job_spec["cells"]:
        for nonce in job_spec["nonces"][cell]:
            for requested_scale in job_spec["requested_scales"]:
                for anneal_us in job_spec["anneal_us"]:
                    jobs.append(
                        build_job(cell, nonce, float(requested_scale), int(anneal_us), reads, margin, ctx)
                    )
    jobs.sort(key=lambda job: (job["cell"], job["nonce"], job["requested_scale"], job["anneal_us"]))

    keys = [job["capture_key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("two jobs in this job spec produced the same capture key; refusing to build a manifest")

    order_seed = int(job_spec["order_seed"])
    job_order = _randomized_job_order(jobs, order_seed)
    job_order_hash = hashlib.sha256(json.dumps(job_order).encode()).hexdigest()

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reservation_margin": margin,
        "reads": reads,
        "sources": {
            "bundles_root": str(job_spec["bundles_root"]),
            "range_audit_dir": str(job_spec["range_audit_dir"]),
            "properties_path": str(properties_path),
            "properties_hash": properties_hash,
            "spec_path": str(spec_path) if spec_path else None,
            "embeddings_dir": str(embeddings_dir) if embeddings_dir else None,
        },
        "jobs": jobs,
        "job_order_seed": order_seed,
        "job_order": job_order,
        "job_order_hash": job_order_hash,
    }


def cmd_manifest(args: argparse.Namespace) -> int:
    job_spec = json.loads(Path(args.job_spec).read_text(encoding="utf-8"))
    manifest = build_manifest(job_spec)
    regime_io.atomic_write_json(args.out, manifest)
    manifest_hash = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"wrote {args.out} ({len(manifest['jobs'])} jobs)")
    print(f"manifest hash: {manifest_hash}")
    return 0


# ----------------------------------------------------------------- approval


def cmd_approve(args: argparse.Namespace) -> int:
    manifest_bytes = Path(args.manifest).read_bytes()
    actual_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_hash != args.manifest_hash:
        raise SystemExit(
            f"--manifest-hash {args.manifest_hash} does not match {args.manifest}, which hashes to "
            f"{actual_hash}; refusing to approve a manifest under the wrong identity"
        )
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
        raise SystemExit("--cap-seconds must be a finite positive number")
    approval = {
        "manifest_hash": actual_hash,
        "cap_seconds": args.cap_seconds,
        "approver": args.approver,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    regime_io.atomic_write_json(args.out, approval)
    print(f"wrote {args.out}: {args.cap_seconds:.1f} s approved by {args.approver} against {actual_hash}")
    return 0


def check_approval(manifest_path: Path, approval_path: Path) -> float:
    """The approved cap in seconds, or a refusal naming why this run may not spend."""
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if not approval_path.exists():
        raise SystemExit(f"no approval at {approval_path}; refusing to submit without one")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approved_hash = approval.get("manifest_hash")
    if approved_hash != manifest_hash:
        raise SystemExit(
            f"{approval_path} approves manifest hash {approved_hash}, and {manifest_path} hashes to "
            f"{manifest_hash}; refusing to submit against a different manifest"
        )
    cap_seconds = approval.get("cap_seconds")
    if cap_seconds is None:
        raise SystemExit(f"{approval_path} names no cap; refusing to submit with no cap")
    cap_seconds = float(cap_seconds)
    # approve enforces this before writing the file, but an approval can also
    # be edited or written by hand; json round-trips NaN and Infinity even
    # though they are not valid JSON, and a NaN cap disables every comparison
    # against it silently.
    if not math.isfinite(cap_seconds) or cap_seconds <= 0:
        raise SystemExit(f"{approval_path} names a cap of {cap_seconds}, which is not a finite positive number")
    return cap_seconds


# ------------------------------------------------------------------ dry-run


def cmd_dry_run(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    round_dir = Path(args.round_dir)
    spent, open_jobs = regime_io.read_spend(round_dir)

    by_cell: Dict[str, Dict[str, float]] = {}
    total_reservation_us = 0.0
    for job in manifest["jobs"]:
        bucket = by_cell.setdefault(job["cell"], {"count": 0, "estimate_us": 0.0, "reservation_us": 0.0})
        bucket["count"] += 1
        bucket["estimate_us"] += job["estimate_us"]
        bucket["reservation_us"] += job["reservation_us"]
        total_reservation_us += job["reservation_us"]

    for cell, bucket in sorted(by_cell.items()):
        print(
            f"{cell}: {int(bucket['count'])} jobs, about {bucket['estimate_us'] / 1e6:.1f} s estimated, "
            f"{bucket['reservation_us'] / 1e6:.1f} s reserved"
        )
    print(f"total reservation across the manifest: {total_reservation_us / 1e6:.1f} s")
    print(
        f"the round has {spent / 1e6:.1f} s charged plus reserved, with {open_jobs} job(s) that have no "
        "recorded charge"
    )
    print(
        "the cap this runner enforces is checked against these estimate-based reservations, not a "
        "guaranteed hard bound on what the QPU will actually charge"
    )
    return 0


# ------------------------------------------------------------------ capture


def drift(
    phys_nodes: Sequence[int], phys_edges: Sequence[Tuple[int, int]],
    live_nodes: Sequence[int], live_edges: Sequence[Tuple[int, int]],
) -> Tuple[int, int]:
    """The count of qubits and the count of couplers of a model that the live chip lacks."""
    nodes = {int(q) for q in live_nodes}
    edges = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in live_edges}
    lost_qubits = sum(int(q) not in nodes for q in phys_nodes)
    lost_couplers = sum((min(int(u), int(v)), max(int(u), int(v))) not in edges for u, v in phys_edges)
    return lost_qubits, lost_couplers


def journal_capture_status(round_dir: Path) -> Tuple[set, set]:
    """``(open_keys, charged_keys)`` from one pass over the spend journal.

    ``open_keys`` have a ``submit`` line and no matching ``charge`` -- an
    uncertain outcome: a process killed by SIGKILL, OOM, or power loss
    between the reservation and the charge leaves exactly this (I1a).

    ``charged_keys`` have at least one ``charge`` line, real or the
    unknown-charge branch's reservation-amount substitute. A key can be
    charged and still have no saved reads: a kill between the charge and
    ``atomic_savez`` finishing leaves the journal looking settled with
    nothing to show for it, on the normal (non-mock) success path this
    runner otherwise has no special-cased guard for. The caller cross-checks
    this against each job's output file.
    """
    path = round_dir / regime_io.SPEND_JOURNAL
    if not path.exists():
        return set(), set()
    open_count: Dict[str, int] = {}
    charged: set = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        # This runner always journals under the capture key, not the nonce.
        key = entry["nonce"]
        if entry["event"] == regime_io.SUBMIT:
            open_count[key] = open_count.get(key, 0) + 1
        elif entry["event"] == regime_io.CHARGE:
            open_count[key] = open_count.get(key, 0) - 1
            charged.add(key)
    open_keys = {key for key, count in open_count.items() if count > 0}
    return open_keys, charged


def load_failed(round_dir: Path) -> Dict[str, Any]:
    path = round_dir / FAILED_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def record_failed(round_dir: Path, capture_key: str, reason: str) -> None:
    """Make one capture key's uncertain outcome durable, so it is never retried automatically."""
    failed = load_failed(round_dir)
    failed[capture_key] = {"reason": reason, "at": datetime.now(timezone.utc).isoformat()}
    regime_io.atomic_write_json(round_dir / FAILED_FILE, failed)


def check_mode_marker(round_dir: Path, mock: bool) -> None:
    """Refuse to mix mock and real captures in one round directory."""
    path = round_dir / MODE_MARKER_FILE
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if bool(saved.get("mock")) != mock:
            raise SystemExit(
                f"{path} says mock is {saved.get('mock')}, and this run passes --mock={mock}; "
                "a mock and a real capture must not share a round directory"
            )
        return
    regime_io.atomic_write_json(path, {"mock": mock})


def validate_and_load(manifest: Dict[str, Any], ctx: Context) -> Dict[str, Dict[str, Any]]:
    """Recompute every job's physical model and hash-check it against the manifest.

    Raises before any reservation if a bundle, an embedding, the solver
    properties, or the range-audit's scaled coefficients have changed since
    the manifest was built.
    """
    if ctx.properties_hash != manifest["sources"]["properties_hash"]:
        raise SystemExit(
            f"the solver properties at {manifest['sources']['properties_path']} have changed since the "
            "manifest was built; refusing to submit against a different snapshot"
        )

    loaded: Dict[str, Dict[str, Any]] = {}
    for job in manifest["jobs"]:
        bundle_manifest, bundle_arrays, phys_nodes, phys_edges, chain_strength = _bundle_and_physical(
            job["cell"], job["nonce"], ctx
        )
        if bundle_manifest["hash"] != job["model_hash"]:
            raise SystemExit(f"{job['cell']}/{job['nonce']}: the bundle has changed since the manifest was built")
        embedding_hash = _embedding_hash(job["cell"], ctx)
        if embedding_hash != job["embedding_hash"]:
            raise SystemExit(
                f"{job['cell']}/{job['nonce']}: the embedding file has changed since the manifest was built"
            )
        ra_manifest, ra_arrays = _range_audit_entry(job["cell"], job["nonce"], ctx)
        submitted_scale, h_scaled, j_scaled = _scaled_hj(ra_manifest, ra_arrays, job["requested_scale"])
        actual_hash = coefficient_hash(h_scaled, j_scaled)
        if actual_hash != job["coefficient_hash"]:
            raise SystemExit(
                f"{job['cell']}/{job['nonce']} at scale {job['requested_scale']}: the scaled coefficients "
                "have changed since the manifest was built; refusing to submit an unaudited problem"
            )
        try:
            _assert_arrays_pair(job["cell"], job["nonce"], phys_nodes, phys_edges, h_scaled, j_scaled)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        chains = _clique_chains_for(ctx) if job["cell"] == "clique-portfolio" else None
        loaded[job["capture_key"]] = {
            "bundle_arrays": bundle_arrays,
            "phys_nodes": phys_nodes,
            "phys_edges": phys_edges,
            "h_scaled": h_scaled,
            "j_scaled": j_scaled,
            "submitted_scale": submitted_scale,
            "chains": chains,
            "chain_strength": chain_strength,
        }
    return loaded


def preflight(manifest: Dict[str, Any], loaded: Dict[str, Dict[str, Any]], live_nodes, live_edges, mock: bool) -> None:
    """Refuse the whole run, before any reservation, if a live chip lacks part of a model.

    No ``--allow-drift`` escape exists in this runner: a captured Round 2
    problem that clamped a defect would not be the audited physical model.
    """
    if mock:
        return
    checked = set()
    for job in manifest["jobs"]:
        key = (job["cell"], job["nonce"])
        if key in checked:
            continue
        checked.add(key)
        data = loaded[job["capture_key"]]
        lost_qubits, lost_couplers = drift(data["phys_nodes"], data["phys_edges"], live_nodes, live_edges)
        if lost_qubits or lost_couplers:
            raise SystemExit(
                f"{job['cell']}/{job['nonce']}: the live chip lacks {lost_qubits} of the qubits and "
                f"{lost_couplers} of the couplers this model needs; this runner has no --allow-drift"
            )


def _output_path(round_dir: Path, job: Dict[str, Any]) -> Path:
    suffix = _scale_suffix(job["requested_scale"])
    return round_dir / job["cell"] / f"scale-{suffix}" / f"qpu-{job['anneal_us']}" / f"{job['nonce']}.npz"


def _output_matches(out_path: Path, capture_key: str) -> bool:
    """Whether a file already at ``out_path`` is really this job's capture.

    The path alone (cell/scale/anneal/nonce) does not carry the model,
    embedding, or solver-property hashes a capture key does, so a resume
    must not trust a file left over from before one of those changed
    (M1) -- for example a properties snapshot update between manifest runs.
    """
    with np.load(out_path) as existing:
        return str(existing["capture_key"]) == capture_key


def _model_for(job: Dict[str, Any], data: Dict[str, Any]) -> regimes.Model:
    bundle_arrays = data["bundle_arrays"]
    return regimes.Model(
        kernel_h=bundle_arrays["h"], kernel_edges=bundle_arrays["edges"], kernel_j=bundle_arrays["j"],
        qpu_nodes=data["phys_nodes"], qpu_h=data["h_scaled"], qpu_edges=data["phys_edges"], qpu_j=data["j_scaled"],
        score_edges=bundle_arrays["edges"], score_j=bundle_arrays["j"], score_h=bundle_arrays["h"],
        chains=data["chains"], chain_strength=float(data["chain_strength"] or 0.0),
    )


class OfflineSampler:
    """The stand-in for ``--mock``: Ocean's compiled annealer at a few sweeps.

    Matches ``scripts/regime_capture.py``'s ``OfflineSampler``. The miner's
    own ``MockSampler`` defaults to dimod's ``ExactSolver``, which enumerates
    every state and cannot run a realistic Round 2 model (a 40-qubit problem
    already asks it for 40.0 TiB). The reads here only have to be valid
    spins, so every later step in the pipeline runs on real files.
    """

    def __init__(self) -> None:
        from dwave.samplers import SimulatedAnnealingSampler

        self._sampler = SimulatedAnnealingSampler()
        self.properties: dict = {}

    def sample_ising(self, h, j, **kwargs):
        return self._sampler.sample_ising(h, j, num_reads=int(kwargs.get("num_reads") or 1), num_sweeps=20)


def connect(mock: bool) -> Any:
    """The sampler of a run. Only the real one reaches the QPU."""
    from quip_solver_dwave.ocean import OceanSampler

    return OceanSampler(sampler=OfflineSampler(), mock=True) if mock else OceanSampler(mock=False)


def run_capture(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cap_seconds = check_approval(manifest_path, Path(args.approval))

    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    check_mode_marker(round_dir, args.mock)

    sources = manifest["sources"]
    properties_path = Path(sources["properties_path"])
    properties = json.loads(properties_path.read_text(encoding="utf-8"))
    properties_hash = hashlib.sha256(properties_path.read_bytes()).hexdigest()
    spec = replay.load_spec(sources["spec_path"]) if sources.get("spec_path") else None
    embeddings_dir = Path(sources["embeddings_dir"]) if sources.get("embeddings_dir") else None
    ctx = Context(
        bundles_root=Path(sources["bundles_root"]),
        range_audit_dir=Path(sources["range_audit_dir"]),
        spec=spec,
        embeddings_dir=embeddings_dir,
        properties=properties,
        properties_hash=properties_hash,
    )

    # Every model is hash-checked against the manifest before the first
    # reservation, real run or mock.
    loaded = validate_and_load(manifest, ctx)

    if not args.mock and not args.yes:
        print(f"{len(manifest['jobs'])} jobs manifested; pass --yes to spend against the approved cap")
        return 2

    by_key = {job["capture_key"]: job for job in manifest["jobs"]}
    failed = load_failed(round_dir)
    # Computed once: within this run, a key only gains a journal line when
    # this loop reserves or charges it, and both are checked before that
    # happens. Empty for a mock round, since a mock run journals nothing.
    open_keys, charged_keys = journal_capture_status(round_dir)

    capture_setup_started_ns = time.monotonic_ns()
    sampler = connect(args.mock)
    run_us = 0
    submitted = 0
    try:
        sampler.ensure_connected()
        preflight(manifest, loaded, sampler.live_nodes, sampler.live_edges, args.mock)
        if manifest["job_order"]:
            first = loaded[manifest["job_order"][0]]
            sampler.set_session_topology(
                [int(q) for q in first["phys_nodes"]],
                [(int(u), int(v)) for u, v in first["phys_edges"]],
            )
        capture_setup_finished_ns = time.monotonic_ns()
        if capture_setup_finished_ns < capture_setup_started_ns:
            raise ValueError("invalid monotonic timestamp order for capture setup")
        capture_setup_s = (capture_setup_finished_ns - capture_setup_started_ns) / 1e9
        for count, capture_key in enumerate(manifest["job_order"], 1):
            job = by_key[capture_key]
            if capture_key in failed:
                raise SystemExit(
                    f"{job['cell']}/{job['nonce']}: capture key {capture_key} previously failed "
                    f"({failed[capture_key]['reason']}); this runner never retries an uncertain submission "
                    f"automatically. Resolve it and remove it from {round_dir / FAILED_FILE} before running again"
                )
            if capture_key in open_keys:
                raise SystemExit(
                    f"{job['cell']}/{job['nonce']}: capture key {capture_key} has an open reservation in "
                    f"{round_dir / regime_io.SPEND_JOURNAL} with no matching charge, from a run that did not "
                    "finish (killed, crashed, or lost power mid-submission). This runner never retries an "
                    "uncertain submission automatically. Confirm with D-Wave whether it ran, then append a "
                    "charge line or a failed-capture record before running again"
                )
            out_path = _output_path(round_dir, job)
            if out_path.exists() and _output_matches(out_path, capture_key):
                continue
            if capture_key in charged_keys:
                # The journal shows this key paid for, but no output file
                # matches it: a run charged it and then a kill, a crash, or
                # lost power struck before atomic_savez finished. Unlike an
                # open reservation, D-Wave already ran this and was already
                # paid; the reads are lost, not merely uncertain, so this
                # refuses rather than spending on it a second time.
                raise SystemExit(
                    f"{job['cell']}/{job['nonce']}: capture key {capture_key} has a charge in "
                    f"{round_dir / regime_io.SPEND_JOURNAL} with no output file to show for it, from a run "
                    "that did not finish saving its reads (killed, crashed, or lost power after the charge). "
                    "This runner never retries an uncertain submission automatically. The QPU already ran and "
                    "was already paid for; recover the reads by hand if possible, or accept the loss and "
                    f"record capture key {capture_key} as failed before running again"
                )
            data = loaded[capture_key]
            job_started_ns = time.monotonic_ns()
            if not args.mock:
                spent, _ = regime_io.read_spend(round_dir)
                if (spent + job["reservation_us"]) / 1e6 > cap_seconds:
                    print(
                        f"stopping before job {count} of {len(manifest['job_order'])}: the round has "
                        f"{spent / 1e6:.2f} s charged plus reserved, and this job's reservation would pass "
                        f"the approved cap of {cap_seconds:.1f} s",
                        file=sys.stderr,
                    )
                    return 3
                regime_io.append_spend(
                    round_dir, regime_io.SUBMIT, job["cell"], capture_key, job["anneal_us"],
                    int(job["reservation_us"]),
                )
            submitted_at_utc = datetime.now(timezone.utc).isoformat()
            submitted_ns = time.monotonic_ns()
            try:
                result = sampler.sample(
                    data["phys_nodes"], data["h_scaled"], data["phys_edges"], data["j_scaled"],
                    num_reads=job["reads"], anneal_time_us=job["anneal_us"],
                    nonce_seed=bytes.fromhex(job["nonce"]), label=args.label, auto_scale=False,
                )
                sampler_finished_ns = time.monotonic_ns()
            except BaseException as exc:
                if not args.mock:
                    record_failed(round_dir, capture_key, f"{type(exc).__name__}: {exc}")
                raise
            access_us = 0 if args.mock else int(result.device_access_time_us)
            if not args.mock:
                if access_us <= 0:
                    # SAPI accepted the problem but reported no usable timing
                    # (answer._access_time_us's 0 for missing timing info).
                    # Charging 0 would close the reservation and silently
                    # undercount every cap check after it, so this charges
                    # the conservative reservation amount instead and marks
                    # the key failed rather than trust an unknown number.
                    #
                    # record_failed goes first, before the closing charge
                    # line: both facts (this needs a substitute charge, and
                    # this key must never be retried) are already known here,
                    # and the charge write is what makes journal_capture_status
                    # (I1a) stop treating the key as open. Writing the charge
                    # first would leave a window where a kill, lost power, or
                    # record_failed's own write failing between the two left
                    # the key looking closed (a matched submit+charge) but
                    # not failed -- the failed-capture check would then miss
                    # it, though the charged-but-no-output check below would
                    # still catch it, just without this reason attached.
                    record_failed(round_dir, capture_key, "unknown charge: the solver reported no access time")
                    regime_io.append_spend(
                        round_dir, regime_io.CHARGE, job["cell"], capture_key, job["anneal_us"],
                        int(job["reservation_us"]),
                    )
                    raise SystemExit(
                        f"{job['cell']}/{job['nonce']}: the solver reported no QPU access time for capture "
                        f"key {capture_key}; charged the reservation amount ({job['reservation_us'] / 1e6:.2f} "
                        "s) and marked it failed. This runner never retries an uncertain submission "
                        "automatically"
                    )
                # The charge goes to the journal before decoding or any file write.
                regime_io.append_spend(round_dir, regime_io.CHARGE, job["cell"], capture_key, job["anneal_us"], access_us)
                run_us += access_us
            try:
                model = _model_for(job, data)
                spins, broken = regimes.decode_reads(model, result.spins, result.variables)
                if len(spins) != job["reads"]:
                    raise SystemExit(
                        f"{job['cell']}/{job['nonce']} holds {len(spins)} rows, and the job asked for "
                        f"{job['reads']} reads. The answer format of the solver has changed, so the run stops"
                    )
                request_hash = request_identity_hash(
                    data["phys_nodes"], data["h_scaled"], data["phys_edges"], data["j_scaled"]
                )
                energies = regimes.score(model, spins)
                scored_usable_ns = time.monotonic_ns()
                timing_points = (job_started_ns, submitted_ns, sampler_finished_ns, scored_usable_ns)
                if timing_points != tuple(sorted(timing_points)):
                    raise ValueError("invalid monotonic timestamp order for captured answer")
                timing_known = result.submitted_on_s is not None and result.solved_on_s is not None
                timing_reason = (
                    "" if timing_known else
                    "mock or non-cloud sampler; SAPI's submitted_on/solved_on/sapi_ms need a real cloud Future"
                )
                sapi_ms = result.extra.get("sapi_ms")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                regime_io.atomic_savez(
                    out_path,
                    spins=spins,
                    raw_spins=np.asarray(result.spins),
                    raw_variables=np.asarray(result.variables, dtype=np.int64),
                    energies=energies,
                    access_us=np.int64(access_us),
                    device_access_s=np.float64(access_us / 1e6),
                    capture_setup_s=np.float64(capture_setup_s),
                    job_setup_s=np.float64((submitted_ns - job_started_ns) / 1e9),
                    sampler_call_s=np.float64((sampler_finished_ns - submitted_ns) / 1e9),
                    decode_score_s=np.float64((scored_usable_ns - sampler_finished_ns) / 1e9),
                    end_to_end_s=np.float64((scored_usable_ns - job_started_ns) / 1e9),
                    job_started_monotonic_s=np.float64(job_started_ns / 1e9),
                    submitted_monotonic_s=np.float64(submitted_ns / 1e9),
                    scored_usable_monotonic_s=np.float64(scored_usable_ns / 1e9),
                    submitted_at_utc=submitted_at_utc,
                    model_setup_cache_status="cached",
                    cold_model_setup_s=np.float64(np.nan),
                    cold_model_setup_unavailable_reason=(
                        "physical model and embedding were prepared before capture; "
                        "cold cache construction was not measured in this process"
                    ),
                    anneal_us=np.int64(job["anneal_us"]),
                    mock=np.bool_(args.mock),
                    requested_scale=np.float64(job["requested_scale"]),
                    submitted_scale=np.float64(job["submitted_scale"]),
                    auto_scale=np.bool_(False),
                    coefficient_hash=job["coefficient_hash"],
                    request_bytes_hash=request_hash,
                    ceiling=np.float64(job["ceiling"]),
                    limiting_constraint=json.dumps(job["limiting_constraint"]),
                    chain_strength=np.float64(job["chain_strength"] or 0.0),
                    chain_strength_ratio_to_logical=np.float64(
                        job["chain_strength_ratio_to_logical"]
                        if job["chain_strength_ratio_to_logical"] is not None else np.nan
                    ),
                    broken_chain_share=np.float64(broken),
                    # SAPI reports no scaling metadata for a submission that
                    # already carries auto_scale=False. If a future solver ever
                    # returns one, thread it through here instead of this notice.
                    returned_scaling_present=np.bool_(False),
                    returned_scaling_reason=(
                        "auto_scale=False; the OceanSampler wrapper does not expose any returned scaling "
                        "metadata for this submission"
                    ),
                    submitted_on_s=np.float64(result.submitted_on_s) if result.submitted_on_s is not None else np.nan,
                    solved_on_s=np.float64(result.solved_on_s) if result.solved_on_s is not None else np.nan,
                    sapi_ms=np.float64(sapi_ms) if sapi_ms is not None else np.nan,
                    timing_unavailable_reason=timing_reason,
                    capture_key=capture_key,
                    model_hash=job["model_hash"],
                )
            except BaseException as exc:
                if not args.mock:
                    record_failed(round_dir, capture_key, f"{type(exc).__name__}: {exc}")
                raise
            submitted += 1
    finally:
        sampler.close()
    print(f"captured {submitted} of {len(manifest['job_order'])} jobs; this run spent {run_us / 1e6:.2f} s")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    # One capture owns the round through reservation and charging, matching
    # regime_capture.py's lock: re-reading the journal alone cannot stop two
    # processes from spending the same allowance.
    with open(round_dir / ".capture.lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another capture owns this round; wait for it to finish") from exc
        return run_capture(args)


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="Build the immutable job manifest from a job-spec JSON.")
    manifest.add_argument("--job-spec", required=True)
    manifest.add_argument("--out", required=True)
    manifest.set_defaults(func=cmd_manifest)

    approve = sub.add_parser("approve", help="Record an explicit cap against one named manifest hash.")
    approve.add_argument("--manifest", required=True)
    approve.add_argument("--manifest-hash", required=True)
    approve.add_argument("--cap-seconds", type=float, required=True)
    approve.add_argument("--approver", required=True)
    approve.add_argument("--out", required=True)
    approve.set_defaults(func=cmd_approve)

    dry_run = sub.add_parser("dry-run", help="Print planned spend against a manifest; submits nothing.")
    dry_run.add_argument("--manifest", required=True)
    dry_run.add_argument("--round-dir", required=True)
    dry_run.set_defaults(func=cmd_dry_run)

    capture = sub.add_parser("capture", help="Run a manifest's jobs, in order, against its approval.")
    capture.add_argument("--manifest", required=True)
    capture.add_argument("--approval", required=True)
    capture.add_argument("--round-dir", required=True)
    capture.add_argument("--label", default="quip-regime-round2")
    capture.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    capture.add_argument("--mock", action="store_true", help="use the offline mock sampler and spend nothing")
    capture.set_defaults(func=cmd_capture)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
