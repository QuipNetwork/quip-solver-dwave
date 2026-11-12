"""Tests for scripts/portfolio_qpu_capture.py, offline on a small Zephyr graph."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
from typing import Any

import numpy as np
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("portfolio_qpu_capture", SCRIPTS / "portfolio_qpu_capture.py")
assert spec is not None and spec.loader is not None
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class SmallCliqueSampler:
    """The mock sampler's interface on a small graph, so the test needs no topology file."""

    def __init__(self, *_: Any) -> None:
        from dwave.graphs import zephyr_graph
        from dwave.samplers import SimulatedAnnealingSampler

        self.target_graph = zephyr_graph(2)
        self.solver_name = "mock"
        self.calls = 0
        self._sampler = SimulatedAnnealingSampler()

    def sample_qubo(self, qdict: Any, **kwargs: Any):
        self.calls += 1
        sampleset = self._sampler.sample_qubo(qdict, num_reads=int(kwargs["num_reads"]), num_sweeps=10, seed=1)
        return sampleset, {"timing": {}, "embedding_context": {}}

    def close(self) -> None:
        pass


def test_the_qubo_dict_matches_the_backend_to_dict():
    q = np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 0.5], [0.0, -0.5, 0.0]])
    assert capture.production_qubo_dict(q) == {(0, 0): 1.0, (0, 1): 5.0, (1, 1): -1.0, (2, 2): 0.0}
    assert list(capture.production_qubo_dict(q)) == [(0, 0), (0, 1), (1, 1), (2, 2)]


def test_chain_sets_ignore_order_within_and_between_chains():
    assert capture.chain_set([[1, 2], [3]]) == capture.chain_set([[3], [2, 1]])
    assert capture.chain_set([[1, 2], [3]]) != capture.chain_set([[1], [2, 3]])


def test_the_embedding_refuses_a_variable_order_production_never_used():
    with pytest.raises(SystemExit, match="0..n-1"):
        capture.embedding_for(SmallCliqueSampler(), {(1, 1): 1.0, (0, 0): 1.0})


def _round(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    families = tmp_path / "families"
    families.mkdir()
    rng = np.random.default_rng(0)
    models = []
    for variant in ("real", "iid"):
        name = f"draw000-{variant}"
        np.savez(families / f"{name}.npz", Q=np.triu(rng.normal(size=(4, 4))))
        models.append({"name": name, "arrays_file": f"{name}.npz", "n_assets": 4, "cardinality_k": 2})
    (families / "index.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    cliques = tmp_path / "cliques"
    cliques.mkdir()
    from minorminer.busclique import find_clique_embedding

    found = find_clique_embedding(4, SmallCliqueSampler().target_graph)
    chains = [[int(q) for q in found[i]] for i in range(4)]
    (cliques / "clique-4.json").write_text(json.dumps({"n": 4, "chains": chains}), encoding="utf-8")
    properties = tmp_path / "properties.json"
    properties.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(capture.capture_tools, "estimate_access_us", lambda *_: 1000.0)
    monkeypatch.setattr(capture, "MockCliqueSampler", SmallCliqueSampler)
    monkeypatch.setattr(capture.replay, "load_spec", lambda _: None)
    manifest_path = tmp_path / "manifest.json"
    capture.main([
        "manifest", "--families", str(families), "--cliques", str(cliques), "--spec", "unused",
        "--properties", str(properties), "--draws", "draw000", "--variants", "real", "iid",
        "--anneal-us", "20", "80", "--reads", "8", "--out", str(manifest_path),
    ])
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "cap_seconds": 1.0,
    }), encoding="utf-8")
    return argparse.Namespace(
        manifest=manifest_path, approval=approval, round_dir=tmp_path / "round", yes=False, mock=True,
        cliques=cliques,
    )


def test_a_mock_capture_saves_every_job_and_a_rerun_skips_them(tmp_path, monkeypatch):
    args = _round(tmp_path, monkeypatch)
    assert capture.cmd_capture(args) == 0
    saved = sorted(p.name for p in (args.round_dir / "jobs").iterdir())
    assert saved == sorted(
        f"draw000-{v}-{a}us.{ext}" for v in ("real", "iid") for a in (20, 80) for ext in ("npz", "json")
    )
    with np.load(args.round_dir / "jobs" / "draw000-real-20us.npz") as arrays:
        assert int(arrays["num_occurrences"].sum()) == 8
        assert arrays["bits"].shape[1] == 4
    calls = []
    monkeypatch.setattr(SmallCliqueSampler, "sample_qubo", lambda *_, **__: calls.append(1))
    assert capture.cmd_capture(args) == 0
    assert calls == []


def test_a_changed_clique_stops_the_capture_before_any_sample(tmp_path, monkeypatch):
    args = _round(tmp_path, monkeypatch)
    clique = args.cliques / "clique-4.json"
    chains = json.loads(clique.read_text(encoding="utf-8"))["chains"]
    chains[0], chains[1] = chains[0] + chains[1][:1], chains[1][1:]
    clique.write_text(json.dumps({"n": 4, "chains": chains}), encoding="utf-8")
    with pytest.raises(SystemExit, match="clique-4.json changed after the manifest was built"):
        capture.cmd_capture(args)
    assert not (args.round_dir / "jobs").exists()


def test_a_changed_capture_script_stops_the_capture(tmp_path, monkeypatch):
    args = _round(tmp_path, monkeypatch)
    edited = tmp_path / "edited.py"
    edited.write_text("# a later revision\n", encoding="utf-8")
    monkeypatch.setattr(capture, "SCRIPT_PATH", edited)
    with pytest.raises(SystemExit, match="capture script changed"):
        capture.cmd_capture(args)
    assert not (args.round_dir / "jobs").exists()


def test_a_changed_properties_file_stops_the_capture(tmp_path, monkeypatch):
    args = _round(tmp_path, monkeypatch)
    (tmp_path / "properties.json").write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(SystemExit, match="properties file changed"):
        capture.cmd_capture(args)
    assert not (args.round_dir / "jobs").exists()


class FakeLiveSampler(SmallCliqueSampler):
    """A live sampler stand-in that reports a fixed access time, so the paid path runs with no QPU."""

    access_us = 700

    def __init__(self, solver: str) -> None:
        super().__init__()
        self.solver_name = solver

    def sample_qubo(self, qdict: Any, **kwargs: Any):
        sampleset, info = super().sample_qubo(qdict, **kwargs)
        return sampleset, {**info, "timing": {"qpu_access_time": self.access_us}}


def _paid(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, cap_seconds: float = 1.0) -> argparse.Namespace:
    args = _round(tmp_path, monkeypatch)
    monkeypatch.setattr(capture, "LiveCliqueSampler", FakeLiveSampler)
    args.approval.write_text(json.dumps({
        "manifest_hash": hashlib.sha256(args.manifest.read_bytes()).hexdigest(), "cap_seconds": cap_seconds,
    }), encoding="utf-8")
    args.mock, args.yes = False, True
    return args


def _journal(round_dir: pathlib.Path) -> list[tuple[str, int]]:
    lines = (round_dir / capture.regime_io.SPEND_JOURNAL).read_text(encoding="utf-8").splitlines()
    return [(entry["event"], entry["us"]) for entry in map(json.loads, lines)]


def test_a_paid_capture_needs_yes_and_journals_a_reservation_and_a_charge_per_job(tmp_path, monkeypatch):
    args = _paid(tmp_path, monkeypatch)
    args.yes = False
    assert capture.cmd_capture(args) == 2
    assert not (args.round_dir / capture.regime_io.SPEND_JOURNAL).exists()
    args.yes = True
    assert capture.cmd_capture(args) == 0
    # Each reservation is the 1000 us estimate times the 1.1 margin.
    assert _journal(args.round_dir) == [("submit", 1100), ("charge", 700)] * 4
    meta = json.loads((args.round_dir / "jobs" / "draw000-real-20us.json").read_text(encoding="utf-8"))
    assert meta["charge_known"] is True
    assert meta["capture_script_sha256"] == capture.sha256_file(capture.SCRIPT_PATH)


def test_a_paid_capture_stops_before_a_reservation_would_pass_the_cap(tmp_path, monkeypatch):
    # 2000 us: two jobs fit (0 + 1100, then 700 + 1100), and the third would reach 2500.
    args = _paid(tmp_path, monkeypatch, cap_seconds=0.002)
    assert capture.cmd_capture(args) == 3
    assert _journal(args.round_dir) == [("submit", 1100), ("charge", 700)] * 2
    assert len(list((args.round_dir / "jobs").glob("*.npz"))) == 2


def test_an_unknown_charge_keeps_the_paid_reads_and_the_key_never_reruns(tmp_path, monkeypatch):
    args = _paid(tmp_path, monkeypatch)
    monkeypatch.setattr(FakeLiveSampler, "access_us", 0)
    with pytest.raises(SystemExit, match="no QPU access time reported"):
        capture.cmd_capture(args)
    assert _journal(args.round_dir) == [("submit", 1100), ("charge", 1100)]
    [saved] = (args.round_dir / "jobs").glob("*.json")
    assert json.loads(saved.read_text(encoding="utf-8"))["charge_known"] is False
    assert saved.with_suffix(".npz").exists()
    monkeypatch.setattr(FakeLiveSampler, "access_us", 700)
    with pytest.raises(SystemExit, match="failed before"):
        capture.cmd_capture(args)
    assert len(_journal(args.round_dir)) == 2


def test_an_open_reservation_stops_the_capture_before_it_spends_again(tmp_path, monkeypatch):
    args = _paid(tmp_path, monkeypatch)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    first = next(job for job in manifest["jobs"] if job["capture_key"] == manifest["job_order"][0])
    capture.regime_io.append_spend(args.round_dir, capture.regime_io.SUBMIT, first["name"], first["capture_key"],
                                   first["anneal_us"], 1100)
    with pytest.raises(SystemExit, match="open reservation with no charge"):
        capture.cmd_capture(args)
    assert _journal(args.round_dir) == [("submit", 1100)]


def test_the_recorded_chain_strength_is_the_chain_coupling_the_child_sampler_receives():
    """dwave-system 1.35 reports ``chain_strength`` as None, so read it off the embedded model."""
    from functools import partial

    import dimod
    from dwave.embedding.chain_strength import uniform_torque_compensation
    from dwave.system import FixedEmbeddingComposite

    graph = SmallCliqueSampler().target_graph
    q = np.triu(np.random.default_rng(5).normal(size=(12, 12)))
    qdict = capture.production_qubo_dict(q)
    embedding = capture.embedding_for(SmallCliqueSampler(), qdict)
    received = []

    class Spy(dimod.Sampler):
        @property
        def parameters(self) -> dict[str, Any]:
            return {}

        @property
        def properties(self) -> dict[str, Any]:
            return {}

        def sample(self, bqm, **_):
            received.append(bqm)
            return dimod.SampleSet.from_samples_bqm({v: 1 for v in bqm.variables}, bqm)

    child = dimod.StructureComposite(Spy(), list(graph.nodes), list(graph.edges))
    spin = dimod.BinaryQuadraticModel.from_qubo(qdict).change_vartype(dimod.SPIN, inplace=False)
    FixedEmbeddingComposite(child, embedding).sample(
        spin, chain_strength=partial(uniform_torque_compensation, prefactor=capture.CHAIN_STRENGTH_PREFACTOR),
    ).resolve()
    chain = next(c for c in embedding.values() if len(c) > 1)
    assert capture.chain_strength_for(qdict, embedding) == pytest.approx(-received[0].quadratic[chain[0], chain[1]])


def test_a_job_without_a_prefactor_keeps_production_s_and_its_file_names(tmp_path):
    job = {"name": "draw000-real", "anneal_us": 80}
    assert capture.job_prefactor(job) == capture.CHAIN_STRENGTH_PREFACTOR
    default = {**job, "chain_strength_prefactor": capture.CHAIN_STRENGTH_PREFACTOR}
    assert capture._output_paths(tmp_path, job) == capture._output_paths(tmp_path, default)
    assert capture._output_paths(tmp_path, job)[0].name == "draw000-real-80us.npz"
    assert capture._output_paths(tmp_path, {**job, "chain_strength_prefactor": 4.5})[0].name == (
        "draw000-real-80us-p4.5.npz")


def test_the_default_prefactor_keeps_the_capture_keys_of_a_manifest_built_before_the_option(tmp_path, monkeypatch):
    args = _round(tmp_path, monkeypatch)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for job in manifest["jobs"]:
        assert job["chain_strength_prefactor"] == capture.CHAIN_STRENGTH_PREFACTOR
        assert job["capture_key"] == hashlib.sha256(json.dumps({
            "name": job["name"], "model_hash": job["model_hash"], "anneal_us": job["anneal_us"],
            "reads": job["reads"], "clique_hash": job["clique_hash"], "solver": "Advantage2_system1",
            "prefactor": 3.0,
        }, sort_keys=True).encode()).hexdigest()


def test_every_prefactor_adds_a_job_and_reaches_the_sampler(tmp_path, monkeypatch):
    args = _round(tmp_path, monkeypatch)
    base = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "manifest-chains.json"
    capture.main([
        "manifest", "--families", str(tmp_path / "families"), "--cliques", str(args.cliques), "--spec", "unused",
        "--properties", str(tmp_path / "properties.json"), "--draws", "draw000", "--variants", "real", "iid",
        "--anneal-us", "20", "80", "--reads", "8", "--prefactor", "4.5", "6", "--out", str(manifest_path),
    ])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chain_strength_prefactors"] == [4.5, 6.0]
    assert len(manifest["jobs"]) == 2 * len(base["jobs"])
    assert len({job["capture_key"] for job in manifest["jobs"]} | {job["capture_key"] for job in base["jobs"]}) == (
        3 * len(base["jobs"]))
    approval = tmp_path / "approval-chains.json"
    approval.write_text(json.dumps({
        "manifest_hash": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "cap_seconds": 1.0,
    }), encoding="utf-8")
    received = []
    original = FakeLiveSampler.sample_qubo

    def spy(self, qdict, **kwargs):
        received.append(kwargs["prefactor"])
        return original(self, qdict, **kwargs)

    monkeypatch.setattr(FakeLiveSampler, "sample_qubo", spy)
    monkeypatch.setattr(capture, "LiveCliqueSampler", FakeLiveSampler)
    paid = argparse.Namespace(manifest=manifest_path, approval=approval, round_dir=tmp_path / "chains",
                              yes=True, mock=False)
    assert capture.cmd_capture(paid) == 0
    assert sorted(received) == sorted(job["chain_strength_prefactor"] for job in manifest["jobs"])
    for job in manifest["jobs"]:
        meta = json.loads(capture._output_paths(paid.round_dir, job)[1].read_text(encoding="utf-8"))
        assert meta["job"]["chain_strength_prefactor"] == job["chain_strength_prefactor"]


def test_the_recorded_chain_strength_scales_with_the_prefactor():
    q = np.triu(np.random.default_rng(6).normal(size=(12, 12)))
    qdict = capture.production_qubo_dict(q)
    embedding = capture.embedding_for(SmallCliqueSampler(), qdict)
    assert capture.chain_strength_for(qdict, embedding, 6.0) == pytest.approx(
        2.0 * capture.chain_strength_for(qdict, embedding))
