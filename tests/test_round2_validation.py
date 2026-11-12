"""Tests for read-only validation of saved Round 2 CPU campaign records."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave import round2_runner as runner
from quip_solver_dwave import round2_validation as validation

IDENTITY = {"package": "test", "build_hash": "abc"}
BETA = (0.25, 3.5)


def _bundle(root: Path, cell: str, *, fractional: bool = False) -> dict:
    h = np.array([0.125, -0.25]) if fractional else np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([-0.375] if fractional else [-1.0])
    model_hash = round2_io.write_bundle(
        root / cell / "01",
        {"cell": cell, "nonce": "01", "offset": 0.5},
        {"h": h, "edges": edges, "j": j},
    )
    return {"cell": cell, "nonce": "01", "model_hash": model_hash}


def _fake_msa(refuse: bool = False):
    class FakeMsa:
        def sample_research(self, h, edges, j, **kwargs):
            if refuse:
                raise ValueError("unit kernel requires unit coefficients")
            spins = np.array([[1, 1], [-1, 1]], dtype=np.int8)
            return spins, runner.score_original(spins, h, edges, j), {
                "observed_kernel": kwargs["kernel"], "representation": "test",
                "rng_scheme": "test-rng", "seeded_reads": 0, "workspace_bytes": 17,
            }

    return SimpleNamespace(Msa=FakeMsa)


def _saved(tmp_path, monkeypatch, *, cell="native-pm1", kernel="cpu-sa", fractional=False):
    row = _bundle(tmp_path, cell, fractional=fractional)
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa(refuse=kernel == "cpu-msa-unit" and fractional))
    job = runner.job_for_row(row, kernel, 8, 2, variant="quality-0")
    policy = runner.ComparisonPolicy.calibration()
    record, spins, energies = runner.execute_cpu_job(job, tmp_path, IDENTITY, BETA)
    record["caller_end_to_end_s"] = 1.0
    record["host"] = {"contamination": {"contaminated": False, "reasons": []}}
    record["comparison"] = runner.comparison_observation(record, policy)
    run_dir = tmp_path / "runs" / record["run_key"]
    runner.write_result(run_dir, record, spins, energies)
    _, arrays = round2_io.read_bundle(tmp_path / cell / "01")
    return job, run_dir, arrays, policy


def _check(job, run_dir, arrays, policy):
    record = json.loads((run_dir / "result.json").read_text())
    return validation.validate_campaign_record(
        job, record, run_dir, arrays, identity=IDENTITY, policy=policy, beta_range=BETA,
    )


def _edit(run_dir: Path, **changes) -> None:
    path = run_dir / "result.json"
    record = json.loads(path.read_text())
    record.update(changes)
    regime_io.atomic_write_json(path, record)


def test_an_untouched_completed_record_is_valid(tmp_path, monkeypatch):
    assert _check(*_saved(tmp_path, monkeypatch)) == []


def test_rehashed_samples_with_wrong_energies_fail_independent_rescoring(tmp_path, monkeypatch):
    job, run_dir, arrays, policy = _saved(tmp_path, monkeypatch)
    spins = np.array([[1, 1], [-1, 1]], dtype=np.int8)
    runner.write_result(
        run_dir, json.loads((run_dir / "result.json").read_text()), spins, np.array([-9.0, 0.0])
    )
    assert any("rescored" in problem for problem in _check(job, run_dir, arrays, policy))


def test_summary_energies_must_match_the_saved_reads(tmp_path, monkeypatch):
    job, run_dir, arrays, policy = _saved(tmp_path, monkeypatch)
    _edit(run_dir, best_energy=-100.0)
    assert any("best_energy" in problem for problem in _check(job, run_dir, arrays, policy))


def test_record_identity_must_match_the_frozen_job(tmp_path, monkeypatch):
    job, run_dir, arrays, policy = _saved(tmp_path, monkeypatch)
    _edit(run_dir, sweeps=16, solver_identity={"package": "other"})
    problems = _check(job, run_dir, arrays, policy)
    assert any("sweeps" in problem for problem in problems)
    assert any("solver_identity" in problem for problem in problems)


def test_a_stale_comparison_block_is_reported(tmp_path, monkeypatch):
    job, run_dir, arrays, policy = _saved(tmp_path, monkeypatch)
    _edit(run_dir, comparison={"available": True})
    assert any("comparison" in problem for problem in _check(job, run_dir, arrays, policy))


def test_missing_caller_timing_must_be_explicitly_withheld(tmp_path, monkeypatch):
    job, run_dir, arrays, policy = _saved(tmp_path, monkeypatch)
    _edit(run_dir, caller_end_to_end_s=None)
    record = json.loads((run_dir / "result.json").read_text())
    _edit(run_dir, comparison=runner.comparison_observation(record, policy))
    assert any("timing" in problem for problem in _check(job, run_dir, arrays, policy))
    _edit(run_dir, timing_evidence={"available": False, "reason": "parallel quality-only execution"})
    assert _check(job, run_dir, arrays, policy) == []


def test_unit_kernel_refusal_of_a_non_unit_model_is_expected(tmp_path, monkeypatch):
    checked = _saved(tmp_path, monkeypatch, cell="native-125", kernel="cpu-msa-unit", fractional=True)
    assert _check(*checked) == []


def test_unit_models_and_other_kernels_may_not_be_unsupported(tmp_path, monkeypatch):
    job, run_dir, arrays, policy = _saved(tmp_path, monkeypatch, kernel="cpu-msa-unit")
    _edit(run_dir, status="unsupported")
    assert any("unsupported" in problem for problem in _check(job, run_dir, arrays, policy))
    edges = np.array([[0, 1]], dtype=np.int64)
    h, j = np.array([0.125, -0.25]), np.array([-0.375])
    assert not validation.unsupported_is_expected("cpu-msa-f64", h, edges, j)
    assert validation.unsupported_is_expected("cpu-msa-unit", h, edges, j)
    assert not validation.unsupported_is_expected("cpu-msa-unit", np.zeros(2), edges, np.array([-1.0]))


BINARY = "b" * 64


def _tree_arrays():
    return {
        "h": np.zeros(2), "edges": np.array([[0, 1]], dtype=np.int64),
        "j": np.array([-1.0]), "offset": np.float64(0.0),
    }


def _tree_record(tmp_path: Path, **changes):
    spins = np.array([[1, 1], [-1, 1]], dtype=np.int8)
    energies = np.array([-1.0, 1.0])
    samples = tmp_path / "run.npz"
    np.savez(samples, spins=spins, energies=energies)
    record = {
        "cell": "native-pm1", "nonce": "01", "model_hash": "m" * 64,
        "requested_kernel": "cpu-sa", "observed_kernel": "cpu-sa",
        "sweeps": 8, "reads": 2, "repetition_id": 0, "repetition_kind": "timing",
        "variant": "timing", "seed": 1, "seed_input_hash": "s" * 64,
        "beta_range": [0.25, 3.5], "exit_ok": True, "error": None,
        "unsupported": False, "unsupported_reason": None,
        "solver_identity": {"binary_sha256": BINARY},
        "best_energy": -1.0, "mean_energy": 0.0, "unique_reads": 2,
    }
    record.update(changes)
    return record, samples


def test_saved_reads_must_have_shape_signed_spins_and_rescored_energies():
    arrays = _tree_arrays()
    spins = np.array([[1, 1], [-1, 1]], dtype=np.int8)
    assert validation.saved_read_problems(spins, np.array([-1.0, 1.0]), arrays, reads=2) == []
    assert "shape" in validation.saved_read_problems(spins, np.array([-1.0, 1.0]), arrays, reads=3)[0]
    assert "signed" in validation.saved_read_problems(
        np.array([[0, 1], [1, 1]], dtype=np.int8), np.array([0.0, -1.0]), arrays, reads=2
    )[0]
    assert "rescored" in validation.saved_read_problems(spins, np.array([-1.0, 0.0]), arrays, reads=2)[0]


def test_a_consistent_tree_record_is_valid(tmp_path):
    record, samples = _tree_record(tmp_path)
    assert validation.validate_tree_record(
        record, samples, _tree_arrays(), model_hash="m" * 64, binary_sha256=BINARY,
    ) == []


def test_tree_records_must_use_the_frozen_model_binary_and_summaries(tmp_path):
    record, samples = _tree_record(tmp_path, best_energy=-5.0, solver_identity={"binary_sha256": "x"})
    problems = validation.validate_tree_record(
        record, samples, _tree_arrays(), model_hash="other", binary_sha256=BINARY,
    )
    assert any("model_hash" in problem for problem in problems)
    assert any("binary" in problem for problem in problems)
    assert any("best_energy" in problem for problem in problems)


def test_tree_unit_refusal_must_match_the_unit_rule(tmp_path):
    record, samples = _tree_record(
        tmp_path, requested_kernel="cpu-msa-unit", observed_kernel=None, unsupported=True,
        unsupported_reason="unit kernel", best_energy=None, mean_energy=None, unique_reads=None,
    )
    samples.unlink()
    problems = validation.validate_tree_record(
        record, samples, _tree_arrays(), model_hash="m" * 64, binary_sha256=BINARY,
    )
    assert any("unsupported" in problem for problem in problems)


def test_seeded_records_split_best_energy_by_seeded_then_cold_lanes(tmp_path):
    record, samples = _tree_record(
        tmp_path, seed_lanes=1, cold_lanes=1, observed_kernel="cpu-msa-f64",
        best_seeded_energy=-1.0, best_cold_energy=1.0,
    )
    arguments = {"model_hash": "m" * 64, "binary_sha256": BINARY}
    assert validation.validate_seeded_record(record, samples, _tree_arrays(), **arguments) == []
    record["best_cold_energy"] = -1.0
    problems = validation.validate_seeded_record(record, samples, _tree_arrays(), **arguments)
    assert any("best_cold_energy" in problem for problem in problems)
