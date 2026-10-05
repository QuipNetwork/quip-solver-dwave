"""QPU capture of portfolio family models through production's own D-Wave call.

Production solved every race with ``DWaveCliqueSampler.sample_qubo`` on the
backend's select QUBO, at 500 reads, a uniform-torque-compensation chain
strength with prefactor 3.0, and no spin-reversal transforms. This script makes
that same call on the family models, at several anneal times, and saves every
read with its chain-break fraction and the access time D-Wave charged.
``manifest --prefactor`` sets other chain-strength prefactors for a chain test;
each job records its own, and a job without one uses production's 3.0.

The clique sampler places its own embedding. Before the first submission, and
again for every job, this script recomputes that embedding and refuses to spend
unless its chains are exactly the fixed chains in ``clique-<N>.json``: one rigid
topology for every model of a basket size.

``manifest`` writes the immutable job list, ``round2_capture.py approve``
records a cap against its hash, and ``capture`` spends against that approval.
Spend accounting reuses ``round2_capture``'s journal: a reservation before each
submission, a charge after it, and no automatic retry of an uncertain outcome.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import itertools
import json
import random
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from quip_miner_dwave import regime_io, replay


def _load_round2_capture():
    path = Path(__file__).resolve().parent / "round2_capture.py"
    spec = importlib.util.spec_from_file_location("round2_capture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["round2_capture"] = module
    spec.loader.exec_module(module)
    return module


capture_tools = _load_round2_capture()

CHAIN_STRENGTH_PREFACTOR = 3.0
# Ocean types QUBO keys as pairs of any hashable label; these labels are 0..n-1.
QuboDict = dict[tuple[Any, Any], float]
LABEL = "quip-portfolio-families"
SCRIPT_PATH = Path(__file__).resolve()


# ---------------------------------------------------------------- models


def production_qubo_dict(q: np.ndarray) -> QuboDict:
    """``QuboMatrix.to_dict`` of the production backend, which fixes the variable order 0..n-1."""
    qdict: QuboDict = {}
    n = q.shape[0]
    for i in range(n):
        qdict[(i, i)] = float(q[i, i])
        for j in range(i + 1, n):
            value = float(q[i, j] + q[j, i])
            if value != 0.0:
                qdict[(i, j)] = value
    return qdict


def load_q(path: Path) -> np.ndarray:
    with np.load(path) as saved:
        return np.asarray(saved["Q"], dtype=np.float64)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def job_prefactor(job: dict[str, Any]) -> float:
    """The job's chain-strength prefactor; jobs from manifests that predate the option used production's."""
    return float(job.get("chain_strength_prefactor", CHAIN_STRENGTH_PREFACTOR))


def chain_set(chains: Iterable[Sequence[int]]) -> frozenset[frozenset[int]]:
    return frozenset(frozenset(int(q) for q in chain) for chain in chains)


# -------------------------------------------------------------- manifest


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    index = json.loads((args.families / "index.json").read_text(encoding="utf-8"))
    records = {record["name"]: record for record in index["models"]}
    properties_bytes = args.properties.read_bytes()
    properties = json.loads(properties_bytes)
    properties = properties.get("properties", properties)
    jobs = []
    for draw in args.draws:
        for variant in args.variants:
            name = f"{draw}-{variant}"
            record = records[name]
            n = int(record["n_assets"])
            clique_path = args.cliques / f"clique-{n}.json"
            clique = json.loads(clique_path.read_text(encoding="utf-8"))
            qubits = sum(len(chain) for chain in clique["chains"])
            model_hash = sha256_file(args.families / record["arrays_file"])
            clique_hash = sha256_file(clique_path)
            for anneal_us, prefactor in itertools.product(args.anneal_us, args.prefactor):
                estimate_us = capture_tools.estimate_access_us(properties, qubits, args.reads, anneal_us)
                capture_key = hashlib.sha256(json.dumps({
                    "name": name, "model_hash": model_hash, "anneal_us": anneal_us, "reads": args.reads,
                    "clique_hash": clique_hash,
                    "solver": args.solver, "prefactor": prefactor,
                }, sort_keys=True).encode()).hexdigest()
                jobs.append({
                    "name": name, "arrays_file": record["arrays_file"], "draw": draw, "variant": variant,
                    "n_assets": n,
                    "cardinality_k": int(record["cardinality_k"]), "model_hash": model_hash,
                    "anneal_us": int(anneal_us), "reads": args.reads, "num_qubits": qubits,
                    "clique_file": str(clique_path), "clique_hash": clique_hash, "estimate_us": estimate_us,
                    "reservation_us": estimate_us * args.margin, "capture_key": capture_key,
                    "chain_strength_prefactor": prefactor,
                })
    keys = [job["capture_key"] for job in jobs]
    if len(set(keys)) != len(keys):
        raise SystemExit("two jobs share a capture key; refusing to build the manifest")
    order = list(keys)
    random.Random(args.order_seed).shuffle(order)
    return {
        "schema": "portfolio-qpu-capture-v1", "created_at": datetime.now(UTC).isoformat(),
        "solver": args.solver, "label": LABEL, "chain_strength_prefactors": list(args.prefactor),
        "capture_script_sha256": sha256_file(SCRIPT_PATH),
        "families": str(args.families), "families_index_sha256": sha256_file(args.families / "index.json"),
        "spec_path": str(args.spec), "properties_path": str(args.properties),
        "properties_sha256": hashlib.sha256(properties_bytes).hexdigest(),
        "reservation_margin": args.margin, "order_seed": args.order_seed,
        "estimate_total_us": sum(job["estimate_us"] for job in jobs),
        "jobs": jobs, "job_order": order,
    }


def cmd_manifest(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    regime_io.atomic_write_json(args.out, manifest)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({
        "jobs": len(manifest["jobs"]), "estimate_s": manifest["estimate_total_us"] / 1e6,
        "reservation_s": sum(job["reservation_us"] for job in manifest["jobs"]) / 1e6,
        "manifest_hash": digest,
    }))
    return 0


# --------------------------------------------------------------- samplers


class MockCliqueSampler:
    """An offline stand-in with the clique sampler's embedding and answer shape. It spends nothing."""

    def __init__(self, spec: replay.TopologySpec) -> None:
        from dwave.graphs import zephyr_graph
        from dwave.samplers import SimulatedAnnealingSampler

        from quip_miner_dwave import embeddings

        self.target_graph = zephyr_graph(
            embeddings.ZEPHYR_M, embeddings.ZEPHYR_T,
            node_list=[int(n) for n in spec.nodes], edge_list=[(int(u), int(v)) for u, v in spec.edges],
        )
        self.solver_name = "mock"
        self._sampler = SimulatedAnnealingSampler()

    def sample_qubo(self, qdict: QuboDict, **kwargs: Any):
        sampleset = self._sampler.sample_qubo(qdict, num_reads=int(kwargs["num_reads"]), num_sweeps=20, seed=1)
        return sampleset, {"timing": {}}

    def close(self) -> None:
        pass


class LiveCliqueSampler:
    """Production's sampler, pinned to one named solver."""

    def __init__(self, solver: str) -> None:
        from dwave.system import DWaveCliqueSampler

        self._sampler = DWaveCliqueSampler(solver=solver)
        self.solver_name = str(self._sampler.child.solver.name)
        if self.solver_name != solver:
            raise SystemExit(f"asked for solver {solver}, and the client connected to {self.solver_name}")

    @property
    def target_graph(self):
        return self._sampler.target_graph

    def sample_qubo(self, qdict: QuboDict, *, prefactor: float, **kwargs: Any):
        from dwave.embedding.chain_strength import uniform_torque_compensation

        sampleset = self._sampler.sample_qubo(
            qdict, chain_strength=partial(uniform_torque_compensation, prefactor=prefactor),
            label=LABEL, **kwargs,
        )
        sampleset.resolve()
        return sampleset, dict(sampleset.info)

    def close(self) -> None:
        self._sampler.close()


def chain_strength_for(qdict: QuboDict, embedding: dict[int, list[int]],
                       prefactor: float = CHAIN_STRENGTH_PREFACTOR) -> float:
    """The chain strength ``DWaveCliqueSampler`` passes down, which its sample set does not report.

    The clique sampler converts the QUBO to spin form, then ``FixedEmbeddingComposite`` calls the
    chain-strength callable on that model and the embedding.
    """
    import dimod
    from dwave.embedding.chain_strength import uniform_torque_compensation

    spin = dimod.BinaryQuadraticModel.from_qubo(qdict).change_vartype(dimod.SPIN, inplace=False)
    return float(uniform_torque_compensation(spin, embedding, prefactor=prefactor))


def embedding_for(sampler: Any, qdict: QuboDict) -> dict[int, list[int]]:
    """The embedding ``DWaveCliqueSampler.sample`` computes for this QUBO, by the same call."""
    import dimod
    from minorminer.busclique import find_clique_embedding

    bqm = dimod.BinaryQuadraticModel.from_qubo(qdict)
    n = len(bqm.variables)
    if list(bqm.variables) != list(range(n)):
        raise SystemExit("the QUBO's variables are not 0..n-1 in order, as production's are")
    found = find_clique_embedding(bqm.variables, sampler.target_graph, use_cache=True)
    return {i: [int(q) for q in found[i]] for i in range(n)}


# ---------------------------------------------------------------- capture


def _output_paths(round_dir: Path, job: dict[str, Any]) -> tuple[Path, Path]:
    stem = f"{job['name']}-{job['anneal_us']}us"
    if job_prefactor(job) != CHAIN_STRENGTH_PREFACTOR:
        stem += f"-p{job_prefactor(job):g}"
    jobs = round_dir / "jobs"
    return jobs / f"{stem}.npz", jobs / f"{stem}.json"


def _saved(round_dir: Path, job: dict[str, Any]) -> bool:
    arrays, meta = _output_paths(round_dir, job)
    if not (arrays.exists() and meta.exists()):
        return False
    return json.loads(meta.read_text(encoding="utf-8")).get("capture_key") == job["capture_key"]


def run_capture(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cap_seconds = capture_tools.check_approval(args.manifest, args.approval)
    round_dir = args.round_dir
    round_dir.mkdir(parents=True, exist_ok=True)
    capture_tools.check_mode_marker(round_dir, args.mock)
    if sha256_file(SCRIPT_PATH) != manifest["capture_script_sha256"]:
        raise SystemExit("this capture script changed after the manifest was built")
    families = Path(manifest["families"])
    if sha256_file(families / "index.json") != manifest["families_index_sha256"]:
        raise SystemExit("the family index changed after the manifest was built")
    if sha256_file(manifest["properties_path"]) != manifest["properties_sha256"]:
        raise SystemExit("the solver properties file changed after the manifest was built")
    by_key = {job["capture_key"]: job for job in manifest["jobs"]}
    qdicts: dict[str, QuboDict] = {}
    for job in manifest["jobs"]:
        path = families / job["arrays_file"]
        if sha256_file(path) != job["model_hash"]:
            raise SystemExit(f"{job['name']}: the model file changed after the manifest was built")
        if sha256_file(job["clique_file"]) != job["clique_hash"]:
            raise SystemExit(f"{job['name']}: {job['clique_file']} changed after the manifest was built")
        qdicts[job["name"]] = production_qubo_dict(load_q(path))
    failed = capture_tools.load_failed(round_dir)
    open_keys, charged_keys = capture_tools.journal_capture_status(round_dir)
    sampler: MockCliqueSampler | LiveCliqueSampler = (
        MockCliqueSampler(replay.load_spec(manifest["spec_path"])) if args.mock
        else LiveCliqueSampler(manifest["solver"])
    )
    try:
        # Every basket size is checked against its fixed clique before anything is spent.
        for job in manifest["jobs"]:
            fixed = json.loads(Path(job["clique_file"]).read_text(encoding="utf-8"))["chains"]
            if chain_set(embedding_for(sampler, qdicts[job["name"]]).values()) != chain_set(fixed):
                raise SystemExit(
                    f"{job['name']}: the clique sampler's embedding on {sampler.solver_name} is not the fixed "
                    f"clique in {job['clique_file']}; refresh the topology before capturing"
                )
        if not args.mock and not args.yes:
            print(f"preflight passed on {sampler.solver_name}: every job embeds on its fixed clique. "
                  f"Pass --yes to spend {len(by_key)} jobs against the approved cap of {cap_seconds:.1f} s")
            return 2
        for count, key in enumerate(manifest["job_order"], 1):
            job = by_key[key]
            if key in failed:
                raise SystemExit(f"{job['name']}: capture key {key} failed before ({failed[key]['reason']})")
            if key in open_keys:
                raise SystemExit(f"{job['name']}: capture key {key} has an open reservation with no charge")
            if _saved(round_dir, job):
                continue
            if key in charged_keys:
                raise SystemExit(f"{job['name']}: capture key {key} was charged and has no saved reads")
            qdict = qdicts[job["name"]]
            embedding = embedding_for(sampler, qdict)
            if not args.mock:
                spent, _ = regime_io.read_spend(round_dir)
                if (spent + job["reservation_us"]) / 1e6 > cap_seconds:
                    print(f"stopping before job {count}: {spent / 1e6:.2f} s charged or reserved, and this job's "
                          f"reservation would pass the approved cap of {cap_seconds:.1f} s", file=sys.stderr)
                    return 3
                regime_io.append_spend(round_dir, regime_io.SUBMIT, job["name"], key, job["anneal_us"],
                                       int(job["reservation_us"]))
            submitted_at = datetime.now(UTC).isoformat()
            started = time.perf_counter()
            try:
                sampleset, info = sampler.sample_qubo(
                    qdict, num_reads=job["reads"], annealing_time=job["anneal_us"], prefactor=job_prefactor(job))
            except BaseException as exc:
                if not args.mock:
                    capture_tools.record_failed(round_dir, key, f"{type(exc).__name__}: {exc}")
                raise
            wall_s = time.perf_counter() - started
            access_us = int(info.get("timing", {}).get("qpu_access_time") or 0)
            charge_known = access_us > 0
            if not args.mock:
                # With no reported access time, the reservation stands in for the charge.
                charge_us = access_us if charge_known else int(job["reservation_us"])
                regime_io.append_spend(round_dir, regime_io.CHARGE, job["name"], key, job["anneal_us"], charge_us)
            record = sampleset.record
            variables = list(sampleset.variables)
            bits = np.asarray(record.sample[:, [variables.index(i) for i in range(len(variables))]], dtype=np.int8)
            occurrences = np.asarray(record.num_occurrences, dtype=np.int64)
            if int(occurrences.sum()) != job["reads"]:
                raise SystemExit(f"{job['name']}: {int(occurrences.sum())} reads came back, {job['reads']} asked")
            names = record.dtype.names or ()
            breaks = (np.asarray(record.chain_break_fraction, dtype=np.float64) if "chain_break_fraction" in names
                      else np.zeros(len(record), dtype=np.float64))
            arrays_path, meta_path = _output_paths(round_dir, job)
            arrays_path.parent.mkdir(parents=True, exist_ok=True)
            regime_io.atomic_savez(
                arrays_path, bits=bits, num_occurrences=occurrences, chain_break_fraction=breaks,
                energy=np.asarray(record.energy, dtype=np.float64),
            )
            regime_io.atomic_write_json(meta_path, {
                "capture_key": key, "job": job, "solver": sampler.solver_name, "mock": args.mock,
                "capture_script_sha256": manifest["capture_script_sha256"],
                "submitted_at": submitted_at, "wall_s": wall_s, "qpu_access_time_us": access_us,
                "charge_known": charge_known,
                "timing": info.get("timing", {}), "problem_id": info.get("problem_id"),
                "chain_strength": chain_strength_for(qdict, embedding, job_prefactor(job)),
                "embedding": {str(v): chain for v, chain in embedding.items()},
            })
            if not args.mock and not charge_known:
                # Saved reads plus a failed key: the reads stay usable, and the key never reruns.
                capture_tools.record_failed(round_dir, key, "unknown charge: the solver reported no access time")
                raise SystemExit(f"{job['name']}: no QPU access time reported; saved the reads and charged "
                                 "the reservation")
            print(json.dumps({"job": count, "name": job["name"], "anneal_us": job["anneal_us"],
                              "access_us": access_us,
                              "broken_read_share": float(occurrences[breaks > 0].sum() / job["reads"])}))
    finally:
        sampler.close()
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    args.round_dir.mkdir(parents=True, exist_ok=True)
    with open(args.round_dir / ".capture.lock", "a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another capture owns this round; wait for it to finish") from exc
        return run_capture(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--families", type=Path, required=True)
    manifest.add_argument("--cliques", type=Path, required=True)
    manifest.add_argument("--spec", type=Path, required=True)
    manifest.add_argument("--properties", type=Path, required=True)
    manifest.add_argument("--draws", nargs="+", required=True)
    manifest.add_argument("--variants", nargs="+", default=["real", "iid", "shuffle", "rotate"])
    manifest.add_argument("--anneal-us", type=int, nargs="+", default=[20, 80, 400])
    manifest.add_argument("--reads", type=int, default=500)
    manifest.add_argument("--prefactor", type=float, nargs="+", default=[CHAIN_STRENGTH_PREFACTOR],
                          help="chain-strength prefactors; each one adds a job per model and anneal time")
    manifest.add_argument("--margin", type=float, default=1.1)
    manifest.add_argument("--order-seed", type=int, default=20260926)
    manifest.add_argument("--solver", default="Advantage2_system1")
    manifest.add_argument("--out", type=Path, required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--manifest", type=Path, required=True)
    capture.add_argument("--approval", type=Path, required=True)
    capture.add_argument("--round-dir", type=Path, required=True)
    capture.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    capture.add_argument("--mock", action="store_true", help="use the offline mock sampler and spend nothing")
    args = parser.parse_args(argv)
    return cmd_manifest(args) if args.command == "manifest" else cmd_capture(args)


if __name__ == "__main__":
    raise SystemExit(main())
