"""The audited capture runner of Round 2: manifest, approval, dry run, capture.

Fakes only -- no live sampler is ever constructed here, and no real QPU
problem is ever submitted. ``--mock`` uses the recording double below; the
"real" path (``--mock`` absent) still runs against the same double, injected
through ``round2_capture.connect`` exactly as ``test_regime_capture.py`` does
for Round 1's ``regime_capture.py``, so the accounting and guard logic is
exercised without a network call.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave.ocean import SampleResult

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round2_capture.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round2_capture", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations via sys.modules[cls.__module__]; the
    # module must be registered there before exec_module runs the decorator.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


round2_capture = _load_module()

PROPERTIES = {
    "chip_id": "mock-chip",
    "problem_timing_data": {
        "version": "1.0.0",
        "typical_programming_time": 33631.59,
        "reverse_annealing_with_reinit_prog_time_delta": 0.0,
        "reverse_annealing_without_reinit_prog_time_delta": 31.33,
        "default_programming_thermalization": 1000.0,
        "default_annealing_time": 20.0,
        "readout_time_model": "pwl_log_log",
        "readout_time_model_parameters": [
            0.0, 0.7118156236873552, 1.622919864592155, 2.5460233863267376, 3.6610550848533783,
            1.3562242666114566, 1.6328679675785214, 1.7857781469507206, 1.99055000334358, 1.9905608299940198,
        ],
        "qpu_delay_time_per_sample": 60.57,
        "reverse_annealing_with_reinit_delay_time_delta": -44.5,
        "reverse_annealing_without_reinit_delay_time_delta": -41.5,
        "default_readout_thermalization": 0.0,
        "decorrelation_max_nominal_anneal_time": 2000.0,
        "decorrelation_time_range": [0.0, 0.0],
    },
}

VARIABLE_LABELS = [100, 101, 102]
BUNDLE_EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)


def _nonce(i: int) -> str:
    return format(i, "064x")


def _write_bundle(bundles_root, cell, nonce, h, j, offset=0.0):
    manifest = {
        "cell": cell,
        "nonce": nonce,
        "identity": {"topology_hash": "topo-fixture", "model_order_hash": "order-fixture"},
        "variable_labels": list(VARIABLE_LABELS),
        "offset": offset,
    }
    return round2_io.write_bundle(
        bundles_root / cell / nonce, manifest, {"h": h, "edges": BUNDLE_EDGES, "j": j}
    )


def _write_range_audit(range_audit_dir, cell, nonce, scales, ceiling=2.0):
    """``scales``: a list of ``(requested, submitted_scale, h_scaled, j_scaled)``."""
    model_dir = range_audit_dir / cell / nonce
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cell": cell, "nonce": nonce, "status": "ok",
        "j_range_used": "j_range", "ceiling": ceiling, "all_zero": False,
        "limiting_constraint": {"kind": "h", "label": 100, "value": 0.3, "bound": "upper"},
        "coefficient_stats": {}, "group_sums": {},
        "chain_strength": None, "chain_strength_ratio_to_logical": None,
        "scales": [{"requested": r, "submitted_scale": s} for r, s, _, _ in scales],
    }
    regime_io.atomic_write_json(model_dir / "manifest.json", manifest)
    arrays = {}
    for requested, _submitted, h_scaled, j_scaled in scales:
        suffix = round2_capture._scale_suffix(requested)
        arrays[f"h_{suffix}"] = h_scaled
        arrays[f"j_{suffix}"] = j_scaled
    regime_io.atomic_savez(model_dir / "arrays.npz", **arrays)


def _fixture(tmp_path, nonces=("aa",), cell="native-pm1", scale=0.5, submitted_scale=1.0):
    """One cell with the given nonces, each bundled and range-audited at ``scale``."""
    h = np.array([0.1, -0.2, 0.3])
    j = np.array([0.4, -0.5])
    for nonce in nonces:
        _write_bundle(tmp_path / "bundles", cell, nonce, h, j)
        _write_range_audit(
            tmp_path / "range-audit", cell, nonce,
            [(scale, submitted_scale, h * submitted_scale, j * submitted_scale)],
        )
    (tmp_path / "properties.json").write_text(json.dumps(PROPERTIES), encoding="utf-8")
    return tmp_path


def _job_spec(tmp_path, nonces=("aa",), cell="native-pm1", scale=0.5, anneal_us=(80,), reads=8,
              order_seed=1, margin=1.5):
    return {
        "cells": [cell],
        "nonces": {cell: list(nonces)},
        "requested_scales": [scale],
        "anneal_us": list(anneal_us),
        "reads": reads,
        "order_seed": order_seed,
        "bundles_root": str(tmp_path / "bundles"),
        "range_audit_dir": str(tmp_path / "range-audit"),
        "properties_path": str(tmp_path / "properties.json"),
        "spec_path": None,
        "embeddings_dir": None,
        "reservation_margin": margin,
    }


def _write_spec(path, nodes, edges):
    path.write_text(json.dumps({
        "nodes": nodes, "edges": [list(e) for e in edges],
        "allowed_h_milli": [0], "allowed_j_milli": [-1000, 1000],
    }), encoding="utf-8")


def _write_clique_fixture(tmp_path, nonce="aa", n=3, k=2, scale=0.5, submitted_scale=1.0):
    """A 3-asset clique-portfolio bundle, chain file, and matching range audit (I5).

    Single-qubit chains on a fully-connected 3-qubit triangle keep
    ``embed_clique`` trivial (nothing to route within a chain) while still
    exercising the real logical-to-physical mapping, ``resolve_chains``'s
    majority vote, and ``regimes.score``.
    """
    from quip_solver_dwave import regimes as regimes_module
    from quip_solver_dwave.embeddings import edge_set, embed_clique

    spec_path = tmp_path / "spec.json"
    nodes = [100, 101, 102]
    edges = [[100, 101], [100, 102], [101, 102]]
    _write_spec(spec_path, nodes, edges)
    spec = round2_capture.replay.load_spec(str(spec_path))
    topology_hash = round2_capture.regime_io.spec_topology_hash(spec)

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    chains = [[100], [101], [102]]
    (embeddings_dir / "clique.json").write_text(
        json.dumps({"topology_hash": topology_hash, "chains": chains}), encoding="utf-8"
    )

    h, edges_dense, j = regimes_module.clique_model(nonce, n, k)
    manifest = {"cell": "clique-portfolio", "nonce": nonce, "identity": {}}
    round2_io.write_bundle(
        tmp_path / "bundles" / "clique-portfolio" / nonce, manifest,
        {"h": h, "edges": edges_dense, "j": j},
    )

    working = edge_set(spec.edges.tolist())
    phys_nodes, phys_h, phys_edges, phys_j, _strength = embed_clique(h, edges_dense, j, chains, working, prefactor=3.0)
    _write_range_audit(
        tmp_path / "range-audit", "clique-portfolio", nonce,
        [(scale, submitted_scale, phys_h * submitted_scale, phys_j * submitted_scale)],
    )
    (tmp_path / "properties.json").write_text(json.dumps(PROPERTIES), encoding="utf-8")
    return spec_path, embeddings_dir, chains, phys_nodes


def _clique_job_spec(tmp_path, spec_path, embeddings_dir, nonce="aa", scale=0.5, anneal_us=(80,), reads=8,
                      order_seed=1, margin=1.5):
    return {
        "cells": ["clique-portfolio"],
        "nonces": {"clique-portfolio": [nonce]},
        "requested_scales": [scale],
        "anneal_us": list(anneal_us),
        "reads": reads,
        "order_seed": order_seed,
        "bundles_root": str(tmp_path / "bundles"),
        "range_audit_dir": str(tmp_path / "range-audit"),
        "properties_path": str(tmp_path / "properties.json"),
        "spec_path": str(spec_path),
        "embeddings_dir": str(embeddings_dir),
        "reservation_margin": margin,
    }


def _write_clique_manifest(tmp_path, spec_path, embeddings_dir) -> pathlib.Path:
    job_spec = _clique_job_spec(tmp_path, spec_path, embeddings_dir)
    job_spec_path = tmp_path / "job-spec.json"
    job_spec_path.write_text(json.dumps(job_spec), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    assert _run(["manifest", "--job-spec", str(job_spec_path), "--out", str(manifest_path)]) == 0
    return manifest_path


def _write_chain_fixture(tmp_path, n=40, cell="native-pm1", nonce="aa", scale=0.5, submitted_scale=1.0):
    """A realistic-size (default 40-qubit) chain bundle and range audit (I4).

    ``dimod.ExactSolver`` (the default ``MockSampler`` backend) enumerates
    every one of ``2**n`` states, so this is exactly the size the review's
    probe used to reproduce the out-of-memory failure against the default
    ``--mock`` path.
    """
    rng = np.random.default_rng(0)
    h = rng.uniform(-1.0, 1.0, n)
    j = rng.uniform(-1.0, 1.0, n - 1)
    labels = list(range(1000, 1000 + n))
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    manifest = {
        "cell": cell, "nonce": nonce,
        "identity": {"topology_hash": "topo-fixture", "model_order_hash": "order-fixture"},
        "variable_labels": labels,
    }
    round2_io.write_bundle(tmp_path / "bundles" / cell / nonce, manifest, {"h": h, "edges": edges, "j": j})
    _write_range_audit(
        tmp_path / "range-audit", cell, nonce, [(scale, submitted_scale, h * submitted_scale, j * submitted_scale)],
    )
    (tmp_path / "properties.json").write_text(json.dumps(PROPERTIES), encoding="utf-8")
    return labels, [(int(a), int(b)) for a, b in zip(labels, labels[1:])]


def _write_job_spec(tmp_path, **kwargs) -> pathlib.Path:
    path = tmp_path / "job-spec.json"
    path.write_text(json.dumps(_job_spec(tmp_path, **kwargs)), encoding="utf-8")
    return path


class _Sampler:
    """A double of ``OceanSampler``: answers each job with +1 spins at a fixed access time."""

    def __init__(self, access_us=40_000, live_nodes=VARIABLE_LABELS, live_edges=((100, 101), (101, 102)),
                 rows=None, during_job=None, submitted_on_s=None, solved_on_s=None, sapi_ms=None):
        self.access_us = access_us
        self._live_nodes = list(live_nodes)
        self._live_edges = list(live_edges)
        self.rows = rows
        self.during_job = during_job
        self.submitted_on_s = submitted_on_s
        self.solved_on_s = solved_on_s
        self.sapi_ms = sapi_ms
        self.calls = []
        self.closed = False

    @property
    def live_nodes(self):
        return list(self._live_nodes)

    @property
    def live_edges(self):
        return list(self._live_edges)

    def ensure_connected(self):
        pass

    def set_session_topology(self, nodes, edges):
        pass

    def sample(self, nodes, h, edges, j, *, num_reads, anneal_time_us, nonce_seed, label, auto_scale=None):
        assert auto_scale is False, "Round 2 must always submit with auto_scale=False"
        self.calls.append({"nodes": list(nodes), "nonce_seed": nonce_seed.hex(), "anneal_time_us": anneal_time_us})
        if self.during_job is not None:
            self.during_job()
        rows = num_reads if self.rows is None else self.rows
        extra = {} if self.sapi_ms is None else {"sapi_ms": str(self.sapi_ms)}
        return SampleResult(
            spins=np.ones((rows, len(nodes)), dtype=np.int8),
            variables=[int(q) for q in nodes],
            energies=[0.0] * rows,
            device_access_time_us=self.access_us,
            num_reads=rows,
            submitted_on_s=self.submitted_on_s,
            solved_on_s=self.solved_on_s,
            extra=extra,
        )

    def close(self):
        self.closed = True


def _run(argv):
    return round2_capture.main(argv)


@pytest.fixture(autouse=True)
def _no_real_sampler(monkeypatch):
    """Structural guarantee (M6): a test that forgets to patch ``connect`` for a
    real run must fail loudly here, never reach ``ensure_connected`` on a live
    ``OceanSampler``. A test-specific ``monkeypatch.setattr(round2_capture,
    "connect", ...)`` still wins, since it runs after this fixture.
    """
    real_connect = round2_capture.connect

    def guarded(mock):
        if not mock:
            raise AssertionError("a test tried to connect a real (non-mock) sampler; this must never happen")
        return real_connect(mock)

    monkeypatch.setattr(round2_capture, "connect", guarded)


# ------------------------------------------------------------- pure helpers


def test_estimate_access_us_matches_the_sdks_own_formula():
    from dwave.cloud.solver import StructuredSolver

    class _PropertyHolder:
        def __init__(self, properties):
            self.properties = properties

    stand_in = _PropertyHolder(PROPERTIES)
    expected = StructuredSolver.estimate_qpu_access_time(
        stand_in,  # type: ignore[arg-type]  # duck-typed stand-in, offline on purpose
        num_qubits=3, num_reads=8, annealing_time=80.0,
    )

    assert round2_capture.estimate_access_us(PROPERTIES, 3, 8, 80) == pytest.approx(expected)


def test_estimate_access_us_reports_the_limit_before_approval_when_it_cannot_bound_one():
    with pytest.raises(ValueError, match="cannot bound QPU access time"):
        round2_capture.estimate_access_us({}, 3, 8, 80)


def test_coefficient_hash_changes_with_either_array():
    h, j = np.array([1.0, 2.0]), np.array([0.5])
    base = round2_capture.coefficient_hash(h, j)
    assert round2_capture.coefficient_hash(h * 2, j) != base
    assert round2_capture.coefficient_hash(h, j * 2) != base


# ---------------------------------------------------------------- manifest


def test_build_manifest_writes_every_job_with_a_seeded_order_and_its_own_hash(tmp_path):
    _fixture(tmp_path, nonces=(_nonce(1), _nonce(2)))
    job_spec = _job_spec(tmp_path, nonces=(_nonce(1), _nonce(2)), anneal_us=(80, 400))

    manifest = round2_capture.build_manifest(job_spec)

    assert len(manifest["jobs"]) == 4  # 2 nonces x 2 anneal times x 1 scale
    assert sorted(manifest["job_order"]) == sorted(job["capture_key"] for job in manifest["jobs"])
    assert manifest["job_order_hash"] == round2_capture.hashlib.sha256(
        json.dumps(manifest["job_order"]).encode()
    ).hexdigest()
    job = manifest["jobs"][0]
    assert job["embedding_hash"] is None  # native-pm1 has no embedding
    assert job["requested_scale"] == 0.5
    assert job["submitted_scale"] == 1.0
    assert job["reservation_us"] == pytest.approx(job["estimate_us"] * job_spec["reservation_margin"])


def test_job_order_randomizes_whole_model_blocks_without_splitting_them():
    jobs = [
        {"cell": "a", "nonce": "1", "capture_key": "a1-x"},
        {"cell": "a", "nonce": "1", "capture_key": "a1-y"},
        {"cell": "a", "nonce": "2", "capture_key": "a2-x"},
        {"cell": "a", "nonce": "2", "capture_key": "a2-y"},
        {"cell": "b", "nonce": "1", "capture_key": "b1-x"},
        {"cell": "b", "nonce": "1", "capture_key": "b1-y"},
    ]

    assert round2_capture._randomized_job_order(jobs, 7) == [
        "b1-x", "b1-y", "a1-y", "a1-x", "a2-y", "a2-x",
    ]


def test_the_manifest_file_hash_is_its_own_raw_bytes(tmp_path):
    _fixture(tmp_path)
    job_spec = _write_job_spec(tmp_path)
    out = tmp_path / "manifest.json"

    assert _run(["manifest", "--job-spec", str(job_spec), "--out", str(out)]) == 0
    assert round2_capture.hashlib.sha256(out.read_bytes()).hexdigest()  # just must not raise


def test_two_jobs_that_collide_on_capture_key_refuse_to_build_a_manifest(tmp_path, monkeypatch):
    _fixture(tmp_path, nonces=(_nonce(1),))
    job_spec = _job_spec(tmp_path, nonces=(_nonce(1),))
    job_spec["anneal_us"] = [80, 400]

    # Force every job to carry the same capture key regardless of its fields,
    # without disturbing build_job's own hash-checked reads.
    real_build_job = round2_capture.build_job
    monkeypatch.setattr(
        round2_capture, "build_job",
        lambda *a, **k: {**real_build_job(*a, **k), "capture_key": "same-key"},
    )
    with pytest.raises(ValueError, match="same capture key"):
        round2_capture.build_manifest(job_spec)


# ----------------------------------------------------------------- approval


def _manifest_and_hash(tmp_path):
    _fixture(tmp_path)
    job_spec = _write_job_spec(tmp_path)
    out = tmp_path / "manifest.json"
    _run(["manifest", "--job-spec", str(job_spec), "--out", str(out)])
    return out, round2_capture.hashlib.sha256(out.read_bytes()).hexdigest()


def test_approve_refuses_a_manifest_hash_that_does_not_match_the_file(tmp_path):
    manifest_path, _real_hash = _manifest_and_hash(tmp_path)
    with pytest.raises(SystemExit, match="does not match"):
        _run([
            "approve", "--manifest", str(manifest_path), "--manifest-hash", "0" * 64,
            "--cap-seconds", "10", "--approver", "alice", "--out", str(tmp_path / "approval.json"),
        ])


@pytest.mark.parametrize("cap", ["nan", "inf", "0", "-5"])
def test_approve_refuses_a_cap_that_is_not_a_finite_positive_number(tmp_path, cap):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    with pytest.raises(SystemExit, match="finite positive"):
        _run([
            "approve", "--manifest", str(manifest_path), "--manifest-hash", manifest_hash,
            "--cap-seconds", cap, "--approver", "alice", "--out", str(tmp_path / "approval.json"),
        ])


def test_approve_requires_manifest_hash_and_cap_with_no_default(tmp_path, capsys):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    with pytest.raises(SystemExit):
        _run(["approve", "--manifest", str(manifest_path), "--approver", "alice", "--out", str(tmp_path / "a.json")])
    assert "required" in capsys.readouterr().err


def _approve(tmp_path, manifest_path, manifest_hash, cap_seconds=10.0, approver="alice"):
    approval_path = tmp_path / "approval.json"
    _run([
        "approve", "--manifest", str(manifest_path), "--manifest-hash", manifest_hash,
        "--cap-seconds", str(cap_seconds), "--approver", approver, "--out", str(approval_path),
    ])
    return approval_path


# ------------------------------------------------------------------ dry-run


def test_dry_run_reports_counts_and_reservations_without_submitting(tmp_path, capsys):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    round_dir = tmp_path / "round"

    assert _run(["dry-run", "--manifest", str(manifest_path), "--round-dir", str(round_dir)]) == 0
    out = capsys.readouterr().out
    assert "native-pm1: 1 jobs" in out
    assert "not a guaranteed hard bound" in out
    assert regime_io.read_spend(round_dir) == (0, 0)


# ------------------------------------------------------------------ capture


def _capture_argv(tmp_path, manifest_path, approval_path, round_dir, *extra):
    return [
        "capture", "--manifest", str(manifest_path), "--approval", str(approval_path),
        "--round-dir", str(round_dir), *extra,
    ]


def test_capture_refuses_without_an_approval_file(tmp_path, monkeypatch):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="no approval"):
        _run(_capture_argv(tmp_path, manifest_path, tmp_path / "missing.json", tmp_path / "round", "--mock"))


def test_capture_refuses_an_approval_naming_a_different_manifest_hash(tmp_path, monkeypatch):
    manifest_path, _hash = _manifest_and_hash(tmp_path)
    # approve validates the hash against the manifest file itself, so a
    # mismatched approval has to be written directly.
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"manifest_hash": "f" * 64, "cap_seconds": 10.0}), encoding="utf-8")
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="different manifest"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, tmp_path / "round", "--mock"))


def test_capture_refuses_an_approval_with_no_cap(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"manifest_hash": manifest_hash}), encoding="utf-8")
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="names no cap"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, tmp_path / "round", "--mock"))


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), 0.0, -5.0])
def test_capture_refuses_a_hand_written_approval_with_a_non_finite_or_non_positive_cap(tmp_path, monkeypatch, cap):
    # approve's own validation can't be bypassed through the CLI, but nothing
    # stops an approval file from being edited or written by hand (M2):
    # json.dumps/loads round-trips NaN and Infinity even though they are not
    # valid JSON per the spec, so check_approval must not trust the cap as-is.
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps({"manifest_hash": manifest_hash, "cap_seconds": cap}), encoding="utf-8")
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="finite positive"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, tmp_path / "round", "--mock"))


def test_a_mock_capture_runs_the_full_path_and_saves_every_field(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0
    assert sampler.closed
    assert regime_io.read_spend(round_dir) == (0, 0)  # mock spends nothing

    files = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as capture:
        assert bool(capture["mock"]) is True
        assert bool(capture["auto_scale"]) is False
        assert float(capture["requested_scale"]) == 0.5
        assert float(capture["submitted_scale"]) == 1.0
        assert capture["spins"].shape == (8, 3)
        assert str(capture["coefficient_hash"])
        assert str(capture["request_bytes_hash"])
    mode_marker = json.loads((round_dir / round2_capture.MODE_MARKER_FILE).read_text())
    assert mode_marker == {"mock": True}


def test_capture_times_through_scoring_and_discloses_cached_setup(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    ticks = iter((1_000_000_000, 1_200_000_000, 2_000_000_000,
                  2_100_000_000, 2_600_000_000, 2_900_000_000))
    score_finished = False

    def monotonic_ns():
        value = next(ticks)
        if value == 2_900_000_000:
            assert score_finished, "usable-answer time was recorded before scoring finished"
        return value

    real_score = round2_capture.regimes.score

    def score(model, spins):
        nonlocal score_finished
        energies = real_score(model, spins)
        score_finished = True
        return energies

    monkeypatch.setattr(round2_capture.time, "monotonic_ns", monotonic_ns)
    monkeypatch.setattr(round2_capture.regimes, "score", score)

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        assert float(capture["capture_setup_s"]) == pytest.approx(0.2)
        assert float(capture["job_setup_s"]) == pytest.approx(0.1)
        assert float(capture["sampler_call_s"]) == pytest.approx(0.5)
        assert float(capture["decode_score_s"]) == pytest.approx(0.3)
        assert float(capture["end_to_end_s"]) == pytest.approx(0.9)
        assert float(capture["device_access_s"]) == 0.0
        assert float(capture["job_started_monotonic_s"]) == 2.0
        assert float(capture["submitted_monotonic_s"]) == 2.1
        assert float(capture["scored_usable_monotonic_s"]) == 2.9
        assert str(capture["submitted_at_utc"])
        assert str(capture["model_setup_cache_status"]) == "cached"
        assert np.isnan(capture["cold_model_setup_s"])
        assert "prepared before capture" in str(capture["cold_model_setup_unavailable_reason"])


def test_capture_refuses_an_invalid_monotonic_timestamp_order(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    ticks = iter((1_000_000_000, 1_200_000_000, 2_000_000_000,
                  2_100_000_000, 2_050_000_000, 2_900_000_000))
    monkeypatch.setattr(round2_capture.time, "monotonic_ns", lambda: next(ticks))

    with pytest.raises(ValueError, match="monotonic timestamp order"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock"))

    assert list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz")) == []


def test_resume_does_not_trust_an_output_file_with_a_different_capture_key(tmp_path, monkeypatch):
    # M1: the output path is cell/scale/anneal/nonce only, so a stale file at
    # that path (for example from before a properties snapshot changed the
    # capture key) must not be counted as this job's completed capture.
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    manifest = json.loads(manifest_path.read_text())
    job = manifest["jobs"][0]
    out_path = round_dir / "native-pm1" / "scale-050" / "qpu-80" / "aa.npz"
    out_path.parent.mkdir(parents=True)
    np.savez_compressed(out_path, spins=np.zeros((1, 1)), capture_key="stale-key-from-a-different-manifest")

    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    assert len(sampler.calls) == 1  # re-captured rather than skipped
    with np.load(out_path) as capture:
        assert str(capture["capture_key"]) == job["capture_key"]


def test_a_mock_capture_is_idempotent_and_skips_a_finished_job(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    second = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: second)
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0
    assert second.calls == []


def test_the_unpatched_mock_connect_runs_a_realistic_sized_model(tmp_path):
    # I4: connect(True) must use an offline sampler that scales (Round 1's
    # OfflineSampler pattern), not MockSampler's default ExactSolver, which
    # cannot run a problem this size (the review's probe: an out-of-memory
    # error on a 40-qubit chain). connect() itself is not patched here.
    labels, edges = _write_chain_fixture(tmp_path, n=40)
    job_spec = _write_job_spec(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    assert _run(["manifest", "--job-spec", str(job_spec), "--out", str(manifest_path)]) == 0
    manifest_hash = round2_capture.hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        assert capture["spins"].shape == (8, len(labels))
        assert bool(capture["mock"]) is True


def test_a_real_run_without_yes_submits_nothing(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir)) == 2
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)


def test_a_capture_stops_before_the_job_that_would_pass_the_approved_cap(tmp_path, monkeypatch):
    nonces = (_nonce(1), _nonce(2), _nonce(3))
    _fixture(tmp_path, nonces=nonces)
    job_spec = _job_spec(tmp_path, nonces=nonces)
    manifest = round2_capture.build_manifest(job_spec)
    manifest_path = tmp_path / "manifest.json"
    regime_io.atomic_write_json(manifest_path, manifest)
    manifest_hash = round2_capture.hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    reservation_us = manifest["jobs"][0]["reservation_us"]
    # Two jobs charge 40,000 us each (below their reservation); the third
    # job's reservation alone would push the round over a cap set just above
    # what two charges plus one more reservation requires.
    cap_seconds = (2 * 40_000 + reservation_us) / 1e6 - 0.001
    approval_path = _approve(tmp_path, manifest_path, manifest_hash, cap_seconds=cap_seconds)
    sampler = _Sampler(access_us=40_000)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    round_dir = tmp_path / "round"

    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes")) == 3
    assert len(sampler.calls) == 2
    assert sampler.closed


def test_a_failed_submit_is_recorded_and_never_retried_automatically(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    def dropped():
        raise RuntimeError("the connection dropped")

    sampler = _Sampler(during_job=dropped)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(RuntimeError, match="the connection dropped"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1
    reason = next(iter(failed.values()))["reason"]
    assert "the connection dropped" in reason
    # An open reservation with no charge counts at its estimate.
    _spent, open_jobs = regime_io.read_spend(round_dir)
    assert open_jobs == 1

    # A rerun refuses the same capture key rather than resubmitting it.
    sampler2 = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler2)
    with pytest.raises(SystemExit, match="never retries an uncertain submission"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler2.calls == []


def test_a_short_answer_stops_the_run_after_the_charge_is_recorded(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler(rows=2)  # the job asks for 8 reads
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    with pytest.raises(SystemExit, match="holds 2 rows"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    spent, open_jobs = regime_io.read_spend(round_dir)
    assert spent == sampler.access_us and open_jobs == 0  # the charge is recorded before the decode fails
    assert list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz")) == []
    # I1(b): a post-charge failure is not a silent retry candidate either.
    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1 and "holds 2 rows" in next(iter(failed.values()))["reason"]


def test_a_decode_failure_still_leaves_the_charge_recorded(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    def _broken_decode(model, spins, variables):
        raise ValueError("the reads lack qubits of the model")

    monkeypatch.setattr(round2_capture.regimes, "decode_reads", _broken_decode)
    with pytest.raises(ValueError, match="the reads lack qubits"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    spent, open_jobs = regime_io.read_spend(round_dir)
    assert spent == sampler.access_us and open_jobs == 0
    assert list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz")) == []
    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1 and "the reads lack qubits" in next(iter(failed.values()))["reason"]


def test_a_save_failure_still_records_the_key_as_failed(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())

    def _broken_save(*_args, **_kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(round2_capture.regime_io, "atomic_savez", _broken_save)
    with pytest.raises(OSError, match="no space left"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1 and "no space left" in next(iter(failed.values()))["reason"]


def test_an_open_reservation_from_a_killed_run_is_never_retried_automatically(tmp_path, monkeypatch):
    # I1(a): a process killed between the reservation and the charge (SIGKILL,
    # OOM, power loss) leaves an open ``submit`` line with no matching
    # ``charge`` and no failed-capture record. The next run must not treat
    # that as safe to resubmit.
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    manifest = json.loads(manifest_path.read_text())
    job = manifest["jobs"][0]
    regime_io.append_spend(
        round_dir, regime_io.SUBMIT, job["cell"], job["capture_key"], job["anneal_us"], int(job["reservation_us"]),
    )

    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(SystemExit, match="open reservation"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []


def test_a_charge_with_no_matching_output_is_never_retried_automatically(tmp_path, monkeypatch):
    # Fix round 3: a kill (or lost power) between the normal success charge
    # and atomic_savez finishing leaves the journal with a matched
    # submit+charge (not open, per I1a) and no output file and no failed
    # record. Unlike an open reservation, D-Wave already ran this and it was
    # already paid for; the next run must refuse it rather than pay again.
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    manifest = json.loads(manifest_path.read_text())
    job = manifest["jobs"][0]
    regime_io.append_spend(
        round_dir, regime_io.SUBMIT, job["cell"], job["capture_key"], job["anneal_us"], int(job["reservation_us"]),
    )
    regime_io.append_spend(round_dir, regime_io.CHARGE, job["cell"], job["capture_key"], job["anneal_us"], 40_000)

    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(SystemExit, match="no output file"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []


def test_an_unknown_charge_is_charged_at_the_reservation_and_marks_the_key_failed(tmp_path, monkeypatch):
    # I2: answer._access_time_us returns 0 when SAPI's timing is missing. A
    # 0 charge on a real run would close the reservation and undercount the
    # round's spend from then on, so a zero access time must be treated as
    # unknown, not as "this job cost nothing".
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    manifest = json.loads(manifest_path.read_text())
    job = manifest["jobs"][0]
    sampler = _Sampler(access_us=0)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    with pytest.raises(SystemExit, match="no QPU access time"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    spent, open_jobs = regime_io.read_spend(round_dir)
    assert spent == int(job["reservation_us"]) and open_jobs == 0  # charged the reservation, not zero
    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1 and "unknown charge" in next(iter(failed.values()))["reason"]
    assert list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz")) == []


def test_an_interruption_after_the_unknown_charge_record_still_refuses_the_key(tmp_path, monkeypatch):
    # Re-review round 1: the unknown-charge branch wrote its closing `charge`
    # line before record_failed, with no wrapping around either write. A
    # kill (or a record_failed write failure) between the two left the key
    # looking "closed" (matched submit+charge) but not failed, so the next
    # run resubmitted and paid again -- the exact hazard I1 was built to
    # close, reintroduced in the one branch that manufactures its own charge.
    # record_failed must land durably before the closing charge line, so an
    # interruption between them leaves the key refused either way (I1a's
    # open check, or the failed-capture record itself).
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler(access_us=0)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    real_append_spend = round2_capture.regime_io.append_spend

    def flaky_append_spend(round_dir_arg, event, cell, nonce, anneal_us, micros):
        if event == round2_capture.regime_io.CHARGE:
            raise OSError("interrupted between record_failed and the charge write")
        return real_append_spend(round_dir_arg, event, cell, nonce, anneal_us, micros)

    monkeypatch.setattr(round2_capture.regime_io, "append_spend", flaky_append_spend)
    with pytest.raises(OSError, match="interrupted between"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))

    # record_failed must already be durable, even though the charge write
    # that was meant to follow it never landed.
    failed = round2_capture.load_failed(round_dir)
    assert len(failed) == 1 and "unknown charge" in next(iter(failed.values()))["reason"]

    # A second run refuses the same key rather than resubmitting and paying
    # for it again.
    monkeypatch.setattr(round2_capture.regime_io, "append_spend", real_append_spend)
    sampler2 = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler2)
    with pytest.raises(SystemExit, match="never retries an uncertain submission"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler2.calls == []


def test_mismatched_physical_array_lengths_are_refused_when_building_the_manifest(tmp_path):
    _fixture(tmp_path)
    # Corrupt the range audit's scaled coefficients to disagree in length
    # with the bundle's physical qubit/coupler count (I5).
    h = np.array([0.1, -0.2, 0.3])
    j_one_edge_short = np.array([0.4])
    _write_range_audit(tmp_path / "range-audit", "native-pm1", "aa", [(0.5, 1.0, h, j_one_edge_short)])
    job_spec = _job_spec(tmp_path)

    with pytest.raises(ValueError, match="physical arrays do not pair"):
        round2_capture.build_manifest(job_spec)


def test_mismatched_physical_array_lengths_are_refused_before_reservation(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    # Corrupt the range audit after the manifest was built, so the stored
    # coefficient_hash also no longer matches -- pin the array-length
    # message specifically by disabling that earlier hash check.
    h = np.array([0.1, -0.2, 0.3])
    j_one_edge_short = np.array([0.4])
    _write_range_audit(tmp_path / "range-audit", "native-pm1", "aa", [(0.5, 1.0, h, j_one_edge_short)])
    monkeypatch.setattr(round2_capture, "coefficient_hash", lambda h, j: json.loads(manifest_path.read_text())["jobs"][0]["coefficient_hash"])

    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(SystemExit, match="physical arrays do not pair"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)


def test_embedding_hash_is_recomputed_from_the_chain_file_not_the_bundles_stored_copy(tmp_path, monkeypatch):
    # I5: the old check compared the bundle's stored identity.embedding_hash
    # against a copy of that same field, which can never fire on its own.
    # Changing the chain file without touching the bundle must now be
    # caught, since embedding_hash is recomputed from the file each time.
    spec_path, embeddings_dir, chains, phys_nodes = _write_clique_fixture(tmp_path)
    manifest_path = _write_clique_manifest(tmp_path, spec_path, embeddings_dir)
    manifest_hash = round2_capture.hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    # Change the chain file's content (still the same topology hash) without
    # touching the bundle at all.
    spec = round2_capture.replay.load_spec(str(spec_path))
    topology_hash = round2_capture.regime_io.spec_topology_hash(spec)
    (embeddings_dir / "clique.json").write_text(
        json.dumps({"topology_hash": topology_hash, "chains": [[101], [100], [102]]}), encoding="utf-8",
    )

    # The full triangle, so preflight (which runs after the embedding check)
    # can never be what stops this run -- only the embedding-hash guard can.
    sampler = _Sampler(live_nodes=phys_nodes.tolist(), live_edges=[(100, 101), (100, 102), (101, 102)])
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(SystemExit, match="embedding file has changed"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []


def test_a_mock_clique_capture_decodes_chains_and_scores_energies(tmp_path, monkeypatch):
    spec_path, embeddings_dir, chains, phys_nodes = _write_clique_fixture(tmp_path)
    manifest_path = _write_clique_manifest(tmp_path, spec_path, embeddings_dir)
    manifest_hash = round2_capture.hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    sampler = _Sampler(live_nodes=phys_nodes.tolist(), live_edges=[(100, 101), (100, 102), (101, 102)])
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "clique-portfolio").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        # One logical asset per chain; every chain here is a single physical
        # qubit, so a +1 physical read never breaks and always decodes to +1.
        assert capture["spins"].shape == (8, len(chains))
        assert (capture["spins"] == 1).all()
        assert float(capture["broken_chain_share"]) == 0.0
        assert capture["raw_spins"].shape == (8, len(phys_nodes))
        assert np.isfinite(capture["energies"]).all()


def test_a_changed_bundle_is_refused_before_any_reservation(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"

    # Replace the bundle after the manifest was built, as a rebuilt export
    # would (write_bundle itself refuses to mutate one in place).
    import shutil

    shutil.rmtree(tmp_path / "bundles" / "native-pm1" / "aa")
    _write_bundle(tmp_path / "bundles", "native-pm1", "aa", np.array([9.0, 9.0, 9.0]), np.array([9.0, 9.0]))

    sampler = _Sampler()
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    with pytest.raises(SystemExit, match="bundle has changed"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)


def test_preflight_stops_before_any_reservation_when_the_live_chip_lacks_a_coupler(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler(live_edges=((100, 101),))  # missing (101, 102)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)

    with pytest.raises(SystemExit, match="lacks 0 of the qubits and 1 of the couplers"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))
    assert sampler.calls == []
    assert regime_io.read_spend(round_dir) == (0, 0)
    assert sampler.closed


def test_a_mock_and_a_real_capture_must_not_share_a_round_directory(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    with pytest.raises(SystemExit, match="mock is True"):
        _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--yes"))


def test_a_capture_refuses_a_round_locked_by_another_process(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    round_dir.mkdir()

    code = (
        "import fcntl,sys; "
        'f=open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX); '
        'print("locked", flush=True); sys.stdin.readline()'
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(round_dir / ".capture.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
        with pytest.raises(SystemExit, match="another capture"):
            _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock"))
    finally:
        holder.communicate("\n", timeout=10)


def test_the_output_records_the_scale_never_promises_returned_scaling_metadata(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        assert bool(capture["returned_scaling_present"]) is False
        assert "auto_scale=False" in str(capture["returned_scaling_reason"])


def test_the_output_saves_raw_reads_and_null_plus_reason_timing_when_absent(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    monkeypatch.setattr(round2_capture, "connect", lambda mock: _Sampler())  # no timing (I3)
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        assert capture["raw_spins"].shape == (8, 3)
        assert capture["raw_variables"].tolist() == VARIABLE_LABELS
        assert np.isnan(capture["submitted_on_s"]) and np.isnan(capture["solved_on_s"]) and np.isnan(capture["sapi_ms"])
        assert "cloud Future" in str(capture["timing_unavailable_reason"])


def test_the_output_saves_sapi_timing_when_the_result_exposes_it(tmp_path, monkeypatch):
    manifest_path, manifest_hash = _manifest_and_hash(tmp_path)
    approval_path = _approve(tmp_path, manifest_path, manifest_hash)
    round_dir = tmp_path / "round"
    sampler = _Sampler(submitted_on_s=1000.0, solved_on_s=1000.25, sapi_ms=250)
    monkeypatch.setattr(round2_capture, "connect", lambda mock: sampler)
    assert _run(_capture_argv(tmp_path, manifest_path, approval_path, round_dir, "--mock")) == 0

    [path] = list((round_dir / "native-pm1").glob("scale-*/qpu-*/*.npz"))
    with np.load(path) as capture:
        assert float(capture["submitted_on_s"]) == 1000.0
        assert float(capture["solved_on_s"]) == 1000.25
        assert float(capture["sapi_ms"]) == 250.0
        assert str(capture["timing_unavailable_reason"]) == ""
