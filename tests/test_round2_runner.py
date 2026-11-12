"""Round 2 CPU runner contracts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quip_solver_dwave import round2_io
from quip_solver_dwave import round2_runner as runner


def _bundle(root: Path, cell: str, nonce: str, *, fractional: bool = False) -> dict:
    h = np.array([0.125, -0.25]) if fractional else np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([-0.375] if fractional else [-1.0])
    model_hash = round2_io.write_bundle(
        root / cell / nonce,
        {"cell": cell, "nonce": nonce, "offset": 0.0},
        {"h": h, "edges": edges, "j": j},
    )
    return {"cell": cell, "nonce": nonce, "model_hash": model_hash}


def _fake_msa(*, observed: str, energies: np.ndarray | None = None):
    class FakeMsa:
        def sample_research(self, h, edges, j, **kwargs):
            spins = np.array([[1, 1], [-1, 1]], dtype=np.int8)
            scored = runner.score_original(spins, h, edges, j)
            return spins, scored if energies is None else energies, {
                "requested_kernel": kwargs["kernel"],
                "observed_kernel": observed,
                "representation": "test",
                "rng_scheme": "test-rng",
                "seeded_reads": 0,
                "workspace_bytes": 17,
            }

    return SimpleNamespace(
        Msa=FakeMsa,
        default_beta_range=lambda h, edges, j: (0.25, 3.5),
        __file__="/wheel/quip_msa/__init__.py",
    )


def test_late_good_answers_are_timeouts():
    assert runner.deadline_status(10.001, 10.0, True) == "timeout"
    assert runner.deadline_status(9.999, 10.0, True) == "completed"


def test_failed_answers_are_failed_before_lateness_is_considered():
    assert runner.deadline_status(20.0, 10.0, False) == "failed"


def test_pilot_is_the_first_five_sorted_nonces_with_fixed_repeated_seeds():
    rows = [
        {"cell": "native-pm1", "nonce": nonce, "model_hash": f"hash-{nonce}"}
        for nonce in ("09", "02", "11", "01", "07", "03")
    ]
    jobs = runner.build_pilot_jobs({"rows": rows}, ["native-pm1"])
    assert len(jobs) == 5 * 2 * 3 * 3
    assert {job.nonce for job in jobs} == {"01", "02", "03", "07", "09"}
    grouped: dict[tuple, list[runner.CpuJob]] = {}
    for job in jobs:
        grouped.setdefault((job.model_hash, job.kernel, job.sweeps), []).append(job)
    assert all(len(group) == 3 for group in grouped.values())
    assert all(len({job.seed for job in group}) == 1 for group in grouped.values())
    assert all({job.repetition_id for job in group} == {0, 1, 2} for group in grouped.values())
    assert jobs == runner.build_pilot_jobs({"rows": rows}, ["native-pm1"])


def test_quality_repetitions_get_distinct_seeds_and_labels():
    jobs = runner.build_quality_jobs(
        {"rows": [{"cell": "native-pm1", "nonce": "01", "model_hash": "abc"}]},
        ["native-pm1"], repetitions=3, sweeps=(512,),
    )
    assert [job.repetition_kind for job in jobs] == ["quality"] * 9
    assert len({job.seed for job in jobs if job.kernel == "cpu-sa"}) == 3


def test_neal_seed_is_derived_from_the_same_seed_hash_within_its_api_limit():
    seed, seed_hash = runner.seed_for("model", "dwave-neal-historical", 500, 500, "deadline")
    assert runner.neal_seed(seed) == seed % (1 << 31)
    assert seed_hash == runner.seed_for(
        "model", "dwave-neal-historical", 500, 500, "deadline"
    )[1]


def test_resume_identity_includes_model_solver_schedule_and_seed():
    job = runner.CpuJob(
        cell="native-pm1", nonce="01", model_hash="model-a", kernel="cpu-sa",
        sweeps=512, reads=64, repetition_id=0, repetition_kind="timing",
        variant="timing", seed=7, seed_input_hash="seed-hash",
    )
    identity = {"package": "quip-msa", "version": "0.3.4", "build_hash": "cpu-head"}
    calibration = runner.ComparisonPolicy.calibration()
    a = runner.run_key(job, identity, (0.2, 4.0), calibration)
    assert a != runner.run_key(job, {**identity, "build_hash": "other"}, (0.2, 4.0), calibration)
    assert a != runner.run_key(job, identity, (0.2, 5.0), calibration)
    assert a != runner.run_key(
        runner.replace_job(job, model_hash="model-b"), identity, (0.2, 4.0), calibration
    )
    evaluation = runner.ComparisonPolicy.from_dict({
        "phase": "evaluation", "quality_target": -1.0,
        "target_source_hash": "a" * 64, "time_budget_s": 10.0,
        "quality_tolerance": 0.0, "runtime_tolerance_s": 0.0,
        "timing_boundary": "frozen-model-to-scored-answer",
        "setup_cache_policy": "fresh-process-cold-setup",
    })
    assert a != runner.run_key(job, identity, (0.2, 4.0), evaluation)
    assert a != runner.run_key(
        job, identity, (0.2, 4.0), calibration, seed_source_hash="b" * 64
    )


def test_frozen_comparison_is_unavailable_for_calibration_and_required_for_evaluation():
    calibration = runner.ComparisonPolicy.calibration()
    assert calibration.quality_target is None
    assert calibration.target_source_hash is None
    with pytest.raises(ValueError, match="frozen"):
        runner.ComparisonPolicy.from_dict({"phase": "evaluation"})
    frozen = runner.ComparisonPolicy.from_dict({
        "phase": "evaluation", "quality_target": -12.5,
        "target_source_hash": "a" * 64, "time_budget_s": 10.0,
        "quality_tolerance": 1e-9, "runtime_tolerance_s": 0.001,
        "timing_boundary": "frozen-model-to-scored-answer",
        "setup_cache_policy": "fresh-process-cold-setup",
    })
    assert frozen.to_dict()["target_source_hash"] == "a" * 64
    with pytest.raises(ValueError, match="timing_boundary"):
        runner.ComparisonPolicy.from_dict({**frozen.to_dict(), "timing_boundary": "sampler-only"})
    with pytest.raises(ValueError, match="setup_cache_policy"):
        runner.ComparisonPolicy.from_dict({**frozen.to_dict(), "setup_cache_policy": "warm"})


def test_frozen_per_model_targets_produce_censored_observations():
    policy = runner.ComparisonPolicy.from_dict({
        "phase": "evaluation", "quality_target": None,
        "targets": {"model-a": -10.0}, "target_source_hash": "b" * 64,
        "time_budget_s": 1.0, "quality_tolerance": 0.1,
        "runtime_tolerance_s": 0.01,
        "timing_boundary": "frozen-model-to-scored-answer",
        "setup_cache_policy": "fresh-process-cold-setup",
    })
    hit = runner.comparison_observation(
        {"model_hash": "model-a", "status": "completed", "best_energy": -10.05,
         "end_to_end_s": 0.8, "caller_end_to_end_s": 0.8}, policy,
    )
    assert hit["quality_target"] == -10.0
    assert hit["target_reached"] is True
    assert hit["time_to_target_s"] == 0.8
    assert hit["quality_within_time_budget"] == -10.05
    missed = runner.comparison_observation(
        {"model_hash": "model-a", "status": "completed", "best_energy": -9.0,
         "end_to_end_s": 1.2, "caller_end_to_end_s": 1.2}, policy,
    )
    assert missed["target_reached"] is False
    assert missed["budget_censored"] is True
    assert missed["quality_within_time_budget"] is None
    parent_timed = runner.comparison_observation(
        {"model_hash": "model-a", "status": "completed", "best_energy": -11.0,
         "end_to_end_s": 0.5, "caller_end_to_end_s": 1.2}, policy,
    )
    assert parent_timed["quality_within_time_budget"] is None


def test_original_float_model_scores_every_returned_state():
    spins = np.array([[1, -1], [-1, -1]], dtype=np.int8)
    h = np.array([0.125, -0.2])
    edges = np.array([[0, 1]])
    j = np.array([-0.375])
    np.testing.assert_allclose(runner.score_original(spins, h, edges, j), [0.7, -0.3])
    np.testing.assert_allclose(runner.score_original(spins, h, edges, j, offset=1.25), [1.95, 0.95])


@pytest.mark.parametrize(("requested", "observed"), [
    ("cpu-sa", "cpu-sa"), ("cpu-msa-unit", "cpu-msa-unit"),
    ("cpu-msa-f64", "cpu-msa-f64"), ("cpu-msa", "cpu-msa-unit"),
    ("cpu-msa", "cpu-msa-f64"),
])
def test_requested_and_observed_dispatch_identity_is_preserved(
    tmp_path, monkeypatch, requested, observed
):
    row = _bundle(tmp_path, "native-pm1", "01")
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa(observed=observed))
    job = runner.job_for_row(row, requested, 8, 2, variant="quality-0")
    record, spins, energies = runner.execute_cpu_job(
        job, tmp_path, {"package": "test", "build_hash": "abc"}, (0.25, 3.5)
    )
    assert record["requested_kernel"] == requested
    assert record["observed_kernel"] == observed
    assert record["dispatch_identity"] == f"{requested}->{observed}"
    assert spins is not None and energies is not None


def test_wrong_observed_kernel_is_rejected(tmp_path, monkeypatch):
    row = _bundle(tmp_path, "native-pm1", "01")
    monkeypatch.setattr(runner, "_msa", lambda: _fake_msa(observed="cpu-msa-f64"))
    job = runner.job_for_row(row, "cpu-sa", 8, 2, variant="quality-0")
    with pytest.raises(runner.RunnerError, match="observed"):
        runner.execute_cpu_job(job, tmp_path, {"build_hash": "abc"}, (0.25, 3.5))


def test_unsupported_unit_kernel_is_explicit_and_never_substituted(tmp_path, monkeypatch):
    row = _bundle(tmp_path, "native-125", "01", fractional=True)
    class RefusingMsa:
        def sample_research(self, *args, **kwargs):
            raise ValueError("unit kernel requires unit coefficients")
    monkeypatch.setattr(runner, "_msa", lambda: SimpleNamespace(Msa=RefusingMsa))
    job = runner.job_for_row(row, "cpu-msa-unit", 8, 2, variant="quality-0")
    record, spins, energies = runner.execute_cpu_job(
        job, tmp_path, {"build_hash": "abc"}, (0.25, 3.5)
    )
    assert record["status"] == "unsupported"
    assert record["observed_kernel"] is None
    assert spins is None and energies is None


def test_nonfinite_kernel_or_original_score_is_rejected(tmp_path, monkeypatch):
    row = _bundle(tmp_path, "native-pm1", "01")
    monkeypatch.setattr(
        runner, "_msa", lambda: _fake_msa(observed="cpu-sa", energies=np.array([np.nan, 1.0]))
    )
    job = runner.job_for_row(row, "cpu-sa", 8, 2, variant="quality-0")
    with pytest.raises(runner.RunnerError, match="nonfinite"):
        runner.execute_cpu_job(job, tmp_path, {"build_hash": "abc"}, (0.25, 3.5))


def test_result_write_is_atomic_and_resume_checks_exact_identity(tmp_path):
    run_dir = tmp_path / "run"
    record = {"run_key": "abc", "status": "completed", "best_energy": -1.0}
    runner.write_result(run_dir, record, np.array([[1, -1]], dtype=np.int8), np.array([-1.0]))
    assert not list(tmp_path.rglob("*.tmp*"))
    completed = runner.completed_result(run_dir, "abc")
    assert completed is not None and completed["status"] == "completed"
    with pytest.raises(runner.RunnerError, match="identity"):
        runner.completed_result(run_dir, "different")
    (run_dir / "samples.npz").unlink()
    with pytest.raises(runner.RunnerError, match="samples"):
        runner.completed_result(run_dir, "abc")


def test_resume_rejects_replaced_or_malformed_samples(tmp_path):
    run_dir = tmp_path / "run"
    record = {"run_key": "abc", "status": "completed", "reads": 2}
    spins = np.array([[1, -1], [-1, 1]], dtype=np.int8)
    runner.write_result(run_dir, record, spins, np.array([-1.0, -1.0]))
    with np.load(run_dir / "samples.npz") as saved:
        assert saved["spins"].shape == (2, 2)
    runner.write_result(run_dir, record, np.array([[0, 1]], dtype=np.int8), np.array([np.nan]))
    with pytest.raises(runner.RunnerError, match="samples"):
        runner.completed_result(run_dir, "abc")


def test_fresh_subprocess_timeout_is_killed_and_recorded(tmp_path):
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    outcome = runner.run_with_deadline(command, timeout_s=0.05, cwd=tmp_path)
    assert outcome["status"] == "timeout"
    assert outcome["elapsed_s"] >= 0.05
    assert outcome["returncode"] is None


def test_high_host_load_is_labeled_once_without_requesting_an_automatic_retry():
    verdict = runner.contamination_verdict(
        before_load=(112.0, 113.0, 95.0), after_load=(110.0, 111.0, 94.0),
        physical_cores=16, sibling_busy_fraction=0.9,
    )
    assert verdict["contaminated"] is True
    assert verdict["repeat_required"] is True
    assert verdict["automatic_repeat"] is False
    assert {reason["kind"] for reason in verdict["reasons"]} == {"loadavg", "sibling_busy"}


def test_cubic_unit_adapter_preserves_the_logical_annealing_schedule():
    job = runner.CpuJob(
        cell="cubic-dimer-pm1", nonce="01", model_hash="abc",
        kernel="cpu-msa-unit", sweeps=8, reads=2, repetition_id=0,
        repetition_kind="quality", variant="quality-0", seed=1,
        seed_input_hash="seed-hash",
    )
    h = np.zeros(3)
    edges = np.array([[0, 1], [1, 2]])
    j = np.array([-0.5, 1.0])
    _, adapted_edges, adapted_j, beta, scale, _ = runner.kernel_problem(
        job, h, edges, j, (0.2, 4.0)
    )
    assert beta == (0.1, 2.0)
    assert scale == 0.5
    np.testing.assert_array_equal(adapted_edges, [[0, 1], [1, 2], [1, 2]])
    np.testing.assert_array_equal(adapted_j, [-1.0, 1.0, 1.0])


def test_seed_source_order_is_exact_and_duplicate_lanes_are_reported():
    manifest = {"variable_labels": [7, 3, 9]}
    seeds = np.array([[1, -1, 1], [1, -1, 1], [-1, 1, -1]], dtype=np.int8)
    runner.validate_seed_source(manifest, seeds, np.array([7, 3, 9]), expected_lanes=3)
    with pytest.raises(runner.RunnerError, match="variable order"):
        runner.validate_seed_source(manifest, seeds, np.array([3, 7, 9]), expected_lanes=3)
    assert runner.lane_diagnostics(seeds) == {
        "lanes": 3, "unique_lanes": 2, "duplicate_lanes": 1,
    }


def test_seed_artifact_rejects_fractional_values_before_cast(tmp_path):
    path = tmp_path / "seeds.npz"
    np.savez(path, spins=np.array([[0.5, -1.0]]), variable_labels=np.array([7, 3]))
    with pytest.raises(runner.RunnerError, match="signed"):
        runner.load_seed_artifact(path)


def test_qpu_deadline_keeps_answer_status_separate_from_later_charge():
    record = runner.build_qpu_deadline_record(
        elapsed_s=10.1, deadline_s=10.0, exit_ok=True, charge_known=False
    )
    assert record["answer"]["status"] == "timeout"
    assert record["charge"] == {"known": False, "access_us": None, "reconciled_at": None}
    with pytest.raises(ValueError, match="access_us"):
        runner.build_qpu_deadline_record(
            elapsed_s=10.1, deadline_s=10.0, exit_ok=True,
            charge_known=False, access_us=123,
        )


def test_portfolio_diagnostics_keep_unknown_weighting_as_unknown():
    diagnostics = {"records": [{"weighting_failed": None}, {"weighting_failed": True}]}
    assert runner.portfolio_weighting_summary(diagnostics) == {"unknown": 1, "failed": 1, "ok": 0}


def test_portfolio_identity_covers_every_resume_input():
    base = {
        "model_hash": "m", "arm": "dwave-neal-historical", "reads": 500,
        "sweeps": 500, "deadline_s": 10.0, "seed": 1,
        "seed_input_hash": "s", "runtime_identity": {"python": "p", "neal_hash": "h"},
    }
    key = runner.portfolio_run_key(**base)
    for name, value in (
        ("reads", 501), ("sweeps", 501), ("deadline_s", 11.0),
        ("seed", 2), ("runtime_identity", {"python": "other", "neal_hash": "h"}),
    ):
        assert runner.portfolio_run_key(**{**base, name: value}) != key


def test_late_portfolio_answer_remains_evidence_but_deadline_status_is_timeout():
    saved = {"status": "completed", "objective": 1.25, "run_key": "k"}
    merged = runner.portfolio_deadline_result(
        saved, {"status": "timeout", "elapsed_s": 10.1, "returncode": None}, 10.0,
        caller_elapsed_s=10.2,
    )
    assert merged["status"] == "timeout"
    assert merged["deadline_status"] == "timeout"
    assert merged["answer"] == {"available": True, "status": "completed_late"}
    assert merged["objective"] == 1.25


def test_unsupported_portfolio_answer_stays_explicit():
    saved = {"status": "unsupported", "unsupported_reason": "fractional", "run_key": "k"}
    merged = runner.portfolio_deadline_result(
        saved, {"status": "completed", "elapsed_s": 0.1, "returncode": 0}, 10.0,
        caller_elapsed_s=0.2,
    )
    assert merged["status"] == "unsupported"
    assert merged["deadline_status"] == "completed"
    assert merged["answer"] == {"available": False, "status": "unsupported"}


def test_portfolio_deadline_uses_caller_visible_result_delivery_time():
    saved = {"status": "completed", "objective": 1.25, "run_key": "k"}
    merged = runner.portfolio_deadline_result(
        saved, {"status": "completed", "elapsed_s": 9.9, "returncode": 0}, 10.0,
        caller_elapsed_s=10.1,
    )
    assert merged["status"] == "timeout"
    assert merged["deadline_status"] == "timeout"
    assert merged["answer"] == {"available": True, "status": "completed_late"}


def test_a_failed_child_stays_failed_when_parent_bookkeeping_passes_the_deadline():
    merged = runner.portfolio_deadline_result(
        None, {"status": "failed", "elapsed_s": 9.0, "returncode": 1}, 10.0,
        caller_elapsed_s=10.5,
    )
    assert merged["status"] == "failed"
    assert merged["deadline_status"] == "timeout"
    assert merged["answer"] == {"available": False, "status": "unavailable"}


def test_portfolio_interrupted_parent_timing_remains_unknown():
    saved = {"status": "completed", "objective": 1.25, "run_key": "k"}
    merged = runner.portfolio_deadline_result(
        saved, {"status": "unknown", "elapsed_s": None, "returncode": None}, 10.0,
        caller_elapsed_s=None,
    )
    assert merged["status"] == "completed"
    assert merged["deadline_status"] == "unknown"
    assert merged["answer"] == {"available": True, "status": "completed_deadline_unknown"}


def test_cli_dry_run_writes_manifest_without_running_jobs(tmp_path):
    row = _bundle(tmp_path / "bundles", "native-pm1", "01")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    out = tmp_path / "out"
    script = Path(__file__).parents[1] / "scripts" / "round2_cpu.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    proc = runner.run_with_deadline(
        [sys.executable, str(script), "pilot", "--index", str(index),
         "--bundles-root", str(tmp_path / "bundles"), "--out", str(out),
         "--cells", "native-pm1", "--dry-run"],
        timeout_s=10.0, cwd=Path(__file__).parents[1], env=env,
    )
    assert proc["status"] == "completed", proc
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "calibration"
    assert manifest["comparison_policy"]["quality_target"] is None
    assert manifest["job_count"] == 18
    assert not (out / "runs").exists()


def test_campaign_refuses_a_calibration_policy(tmp_path):
    row = _bundle(tmp_path / "bundles", "native-pm1", "01")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"rows": [row]}), encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps(runner.ComparisonPolicy.calibration().to_dict()), encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "round2_cpu.py"
    proc = runner.run_with_deadline(
        [sys.executable, str(script), "campaign", "--index", str(index),
         "--bundles-root", str(tmp_path / "bundles"), "--out", str(tmp_path / "out"),
         "--cells", "native-pm1", "--comparison-policy", str(policy), "--dry-run"],
        timeout_s=10.0, cwd=Path(__file__).parents[1], env=os.environ,
    )
    assert proc["status"] == "failed"
    assert "evaluation" in proc["stderr"]


def test_campaign_estimate_scales_sampling_by_cell_kernel_and_sweeps():
    records = []
    for sweeps, sampling in ((512, 1.0), (2048, 4.0)):
        records.append({
            "cell": "native-pm1", "requested_kernel": "cpu-sa", "sweeps": sweeps,
            "status": "completed", "sampling_s": sampling, "fresh_setup_s": 0.1,
            "caller_end_to_end_s": sampling + 0.3,
        })
    estimate = runner.estimate_campaign(records, [
        runner.CpuJob(
            cell="native-pm1", nonce="n", model_hash="m", kernel="cpu-sa",
            sweeps=131072, reads=64, repetition_id=0, repetition_kind="quality",
            variant="evaluation", seed=1, seed_input_hash="s",
        )
    ])
    cell = estimate["cells"]["native-pm1"]["cpu-sa"]
    assert cell["estimated_sampling_s"] == pytest.approx(256.0)
    assert cell["estimated_sequential_s"] == pytest.approx(256.3)
    assert estimate["classification"] == "cost-planning extrapolation, not scientific runtime"


def test_campaign_estimate_reports_and_excludes_child_only_interrupted_timing():
    records = []
    for sweeps, sampling in ((512, 1.0), (2048, 4.0)):
        records.append({
            "cell": "native-pm1", "requested_kernel": "cpu-sa", "sweeps": sweeps,
            "status": "completed", "sampling_s": sampling,
            "caller_end_to_end_s": sampling + 0.3,
        })
    records.append({
        "cell": "native-pm1", "requested_kernel": "cpu-sa", "sweeps": 512,
        "status": "completed", "sampling_s": 0.9, "end_to_end_s": 1.0,
        "caller_end_to_end_s": None,
    })
    job = runner.CpuJob(
        cell="native-pm1", nonce="n", model_hash="m", kernel="cpu-sa",
        sweeps=8192, reads=64, repetition_id=0, repetition_kind="quality",
        variant="evaluation", seed=1, seed_input_hash="s",
    )
    estimate = runner.estimate_campaign(records, [job])
    cell = estimate["cells"]["native-pm1"]["cpu-sa"]
    assert cell["excluded_missing_caller_timing"] == 1
    assert cell["estimated_sequential_s"] == pytest.approx(16.3)


def test_campaign_estimate_skips_an_explicitly_unsupported_unit_kernel():
    records = [
        {
            "cell": "native-125", "requested_kernel": "cpu-msa-unit",
            "sweeps": sweeps, "status": "unsupported",
            "unsupported_reason": "fractional coefficients",
        }
        for sweeps in (512, 2048)
    ]
    job = runner.CpuJob(
        cell="native-125", nonce="n", model_hash="m", kernel="cpu-msa-unit",
        sweeps=8192, reads=64, repetition_id=0, repetition_kind="quality",
        variant="evaluation", seed=1, seed_input_hash="s",
    )
    estimate = runner.estimate_campaign(records, [job])
    cell = estimate["cells"]["native-125"]["cpu-msa-unit"]
    assert cell == {
        "jobs": 1, "status": "unsupported", "estimated_sequential_s": 0.0,
        "unsupported_reason": "fractional coefficients",
    }


def test_portfolio_cli_runs_a_controlled_arm_then_excluded_diagnostics(tmp_path):
    root = Path("/home/carback1/quip-data/regimes/round2")
    pinned_python = root / "reference/qportfolio/.venv/bin/python"
    fixtures = sorted(
        (root / "bundles-portfolio-fixtures/n4-k2").glob("*/manifest.json")
    )
    if not pinned_python.exists() or not fixtures:
        pytest.skip("pinned portfolio reference environment is unavailable")
    script = Path(__file__).parents[1] / "scripts" / "round2_cpu.py"
    fixture = fixtures[0].parent
    cpu = min(os.sched_getaffinity(0))
    proc = runner.run_with_deadline(
        [
            sys.executable, str(script), "portfolio", "--fixtures", str(fixture),
            "--out", str(tmp_path), "--portfolio-python", str(pinned_python),
            "--arms", "cpu-sa", "--reads", "2", "--sweeps", "8",
            "--deadline-s", "10", "--cpu", str(cpu),
        ],
        timeout_s=30.0, cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert proc["status"] == "completed", proc
    results = list(tmp_path.glob("*/cpu-sa/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["answer"] == {"available": True, "status": "completed"}
    assert result["beta_schedule_type"] == "geometric"
    assert isinstance(result["selected_bits"], list)
    assert isinstance(result["weights"], list)
    assert result["diagnostics"]["excluded_from_application_deadline"] is True
    assert result["deadline_evidence"]["eligible_for_solver_decision"] is (
        not result["host"]["contamination"]["contaminated"]
    )
    assert "round2_cpu_sha256" in result["runtime_identity"]
    diagnostics = json.loads(
        (results[0].parent / "diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["excluded_from_application_deadline"] is True

    samples = results[0].parent / "samples.npz"
    sample_mtime = samples.stat().st_mtime_ns
    original_caller_elapsed = result["caller_end_to_end_s"]
    (results[0].parent / "diagnostics.json").unlink()
    resumed_diagnostics = runner.run_with_deadline(
        [
            sys.executable, str(script), "portfolio", "--fixtures", str(fixture),
            "--out", str(tmp_path), "--portfolio-python", str(pinned_python),
            "--arms", "cpu-sa", "--reads", "2", "--sweeps", "8",
            "--deadline-s", "10", "--cpu", str(cpu),
        ],
        timeout_s=30.0, cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert resumed_diagnostics["status"] == "completed", resumed_diagnostics
    after_diagnostics = json.loads(results[0].read_text(encoding="utf-8"))
    assert samples.stat().st_mtime_ns == sample_mtime
    assert after_diagnostics["caller_end_to_end_s"] == original_caller_elapsed
    assert (results[0].parent / "diagnostics.json").exists()

    for field in (
        "caller_end_to_end_s", "host", "deadline_evidence", "answer",
        "deadline_status", "subprocess", "diagnostics",
    ):
        after_diagnostics.pop(field, None)
    results[0].write_text(json.dumps(after_diagnostics), encoding="utf-8")
    (results[0].parent / "diagnostics.json").unlink()
    resumed = runner.run_with_deadline(
        [
            sys.executable, str(script), "portfolio", "--fixtures", str(fixture),
            "--out", str(tmp_path), "--portfolio-python", str(pinned_python),
            "--arms", "cpu-sa", "--reads", "2", "--sweeps", "8",
            "--deadline-s", "10", "--cpu", str(cpu),
        ],
        timeout_s=30.0, cwd=Path(__file__).parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    assert resumed["status"] == "completed", resumed
    repaired = json.loads(results[0].read_text(encoding="utf-8"))
    assert samples.stat().st_mtime_ns == sample_mtime
    assert repaired["caller_end_to_end_s"] is None
    assert repaired["deadline_status"] == "unknown"
    assert repaired["answer"]["status"] == "completed_deadline_unknown"
    assert repaired["diagnostics"]["excluded_from_application_deadline"] is True
