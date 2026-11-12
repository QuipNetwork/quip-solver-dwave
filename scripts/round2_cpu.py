#!/usr/bin/env python3
"""Run the Round 2 CPU pilot or frozen evaluation campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave import round2_runner as runner

DEFAULT_BUNDLES = Path("/home/carback1/quip-data/regimes/round2/bundles")
DEFAULT_INDEX = DEFAULT_BUNDLES / "index.json"
DEFAULT_CELLS = (
    "native-pm1", "native-125", "diamond-pm1",
    "cubic-dimer-pm1", "clique-portfolio",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            low, high = part.split("-", 1)
            result.extend(range(int(low), int(high) + 1))
        else:
            result.append(int(part))
    return result


def _sibling(cpu: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    if not path.exists():
        return None
    return next((item for item in _cpu_list(path.read_text().strip()) if item != cpu), None)


def _governor(cpu: int) -> str | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    return path.read_text().strip() if path.exists() else None


def _cpu_times(cpu: int | None) -> tuple[int, int] | None:
    if cpu is None:
        return None
    prefix = f"cpu{cpu} "
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith(prefix):
            fields = [int(value) for value in line[len(prefix):].split()]
            idle = fields[3] + fields[4]
            return sum(fields) - idle, sum(fields)
    return None


def _busy(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    if before is None or after is None or after[1] <= before[1]:
        return None
    return (after[0] - before[0]) / (after[1] - before[1])


def _physical_cores() -> int:
    pairs = set()
    for cpu in os.sched_getaffinity(0):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            pairs.add((
                int((topology / "physical_package_id").read_text()),
                int((topology / "core_id").read_text()),
            ))
        except OSError:
            return max(1, len(os.sched_getaffinity(0)) // 2)
    return len(pairs)


def _choose_cpu(requested: int | None) -> int:
    allowed = sorted(os.sched_getaffinity(0))
    cpu = allowed[0] if requested is None else requested
    if cpu not in allowed:
        raise ValueError(f"CPU {cpu} is outside the allowed affinity {allowed}")
    return cpu


def _beta(job: runner.CpuJob, bundles_root: Path) -> tuple[float, float]:
    manifest, arrays = round2_io.read_bundle(bundles_root / job.cell / job.nonce)
    if manifest["hash"] != job.model_hash:
        raise runner.RunnerError("bundle hash changed after job planning")
    return runner.logical_beta_range(arrays["h"], arrays["edges"], arrays["j"])


def _run_one(args: argparse.Namespace) -> int:
    payload = _load(args.job)
    job = runner.CpuJob.from_dict(payload["job"])
    record, spins, energies = runner.execute_cpu_job(
        job, args.bundles_root, payload["solver_identity"],
        tuple(payload["logical_beta_range"]),
        comparison_policy=runner.ComparisonPolicy.from_dict(payload["comparison_policy"]),
        seed_source=payload.get("seed_source"),
    )
    runner.write_result(args.out, record, spins, energies)
    return 0


def _policy(args: argparse.Namespace) -> runner.ComparisonPolicy:
    if args.command == "pilot":
        if args.comparison_policy is not None:
            raise ValueError("pilot is calibration; do not pass frozen evaluation targets")
        return runner.ComparisonPolicy.calibration()
    if args.comparison_policy is None:
        raise ValueError("campaign requires --comparison-policy from separate calibration runs")
    policy = runner.ComparisonPolicy.from_dict(_load(args.comparison_policy))
    if policy.phase != "evaluation":
        raise ValueError("campaign requires an evaluation policy, not calibration mode")
    return policy


def _manifest(
    args: argparse.Namespace, jobs: Sequence[runner.CpuJob], identity: dict[str, Any],
    policy: runner.ComparisonPolicy, cpu: int,
) -> dict[str, Any]:
    sibling = _sibling(cpu)
    return {
        "schema": "round2-cpu-manifest-v1",
        "phase": policy.phase,
        "bundles_root": str(args.bundles_root.resolve()),
        "index": str(args.index.resolve()),
        "cells": args.cells,
        "job_count": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
        "solver_identity": identity,
        "comparison_policy": policy.to_dict(),
        "execution": {
            "sequential": True, "fresh_subprocess_per_run": True,
            "automatic_contamination_repeat": False,
            "cpu": cpu, "allowed_affinity": sorted(os.sched_getaffinity(0)),
            "sibling": sibling, "physical_cores": _physical_cores(),
            "governor": _governor(cpu),
        },
    }


def _run_campaign(args: argparse.Namespace) -> int:
    index = _load(args.index)
    jobs = (
        runner.build_pilot_jobs(index, args.cells)
        if args.command == "pilot"
        else runner.build_campaign_jobs(index, args.cells)
    )
    identity = runner.solver_identity()
    policy = _policy(args)
    cpu = _choose_cpu(args.cpu)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(args, jobs, identity, policy, cpu)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and _load(manifest_path) != manifest:
        raise runner.RunnerError(f"{manifest_path} differs from this invocation")
    regime_io.atomic_write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({"manifest": str(manifest_path), "jobs": len(jobs)}))
        return 0

    script = Path(__file__).resolve()
    sibling = _sibling(cpu)
    physical = manifest["execution"]["physical_cores"]
    for position, job in enumerate(jobs):
        caller_started = time.monotonic()
        beta = _beta(job, args.bundles_root)
        key = runner.run_key(job, identity, beta, policy)
        run_dir = args.out / "runs" / key
        existing = runner.completed_result(run_dir, key)
        if existing is not None:
            if "caller_end_to_end_s" not in existing:
                existing.setdefault("child_end_to_end_s", existing.get("end_to_end_s"))
                existing["caller_end_to_end_s"] = None
                existing["end_to_end_s"] = None
                existing["timing_evidence"] = {
                    "available": False,
                    "reason": "parent finalization was interrupted",
                }
            if "host" not in existing:
                existing["host"] = {
                    "contamination": {
                        "contaminated": True,
                        "reasons": [{"kind": "interrupted-before-host-record"}],
                        "repeat_required": True,
                        "automatic_repeat": False,
                    }
                }
            existing.setdefault("comparison", runner.comparison_observation(existing, policy))
            regime_io.atomic_write_json(run_dir / "result.json", existing)
            continue
        payload = {
            "job": job.to_dict(), "solver_identity": identity,
            "logical_beta_range": list(beta), "comparison_policy": policy.to_dict(),
        }
        payload_path = args.out / "dispatch" / f"{key}.json"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        regime_io.atomic_write_json(payload_path, payload)
        before_load = os.getloadavg()
        before_sibling = _cpu_times(sibling)
        setup_elapsed = time.monotonic() - caller_started
        outcome = runner.run_with_deadline(
            ["taskset", "-c", str(cpu), sys.executable, str(script), "run-one",
             "--job", str(payload_path), "--bundles-root", str(args.bundles_root),
             "--out", str(run_dir)],
            timeout_s=max(0.001, args.timeout_s - setup_elapsed),
            cwd=Path(__file__).parents[1], env=os.environ,
        )
        after_load = os.getloadavg()
        contamination = runner.contamination_verdict(
            before_load=before_load, after_load=after_load,
            physical_cores=physical,
            sibling_busy_fraction=_busy(before_sibling, _cpu_times(sibling)),
        )
        if outcome["status"] != "completed":
            record = {
                "schema": "round2-cpu-run-v1", "run_key": key, **job.to_dict(),
                "requested_kernel": job.kernel, "observed_kernel": None,
                "status": outcome["status"], "subprocess": outcome,
            }
            runner.write_result(run_dir, record, None, None)
        result_path = run_dir / "result.json"
        result = _load(result_path)
        result["child_end_to_end_s"] = result.get("end_to_end_s")
        result["caller_setup_before_child_s"] = setup_elapsed
        result["caller_end_to_end_s"] = time.monotonic() - caller_started
        result["end_to_end_s"] = result["caller_end_to_end_s"]
        result["host"] = {
            "loadavg_before": list(before_load), "loadavg_after": list(after_load),
            "cpu": cpu, "sibling": sibling, "governor": _governor(cpu),
            "affinity": [cpu], "contamination": contamination,
        }
        result["sequence_position"] = position
        result["comparison"] = runner.comparison_observation(result, policy)
        regime_io.atomic_write_json(result_path, result)

    records = [_load(path) for path in (args.out / "runs").glob("*/result.json")]
    if args.command == "pilot":
        campaign_jobs = runner.build_campaign_jobs(index, args.cells)
        estimate = runner.estimate_campaign(records, campaign_jobs)
        regime_io.atomic_write_json(args.out / "campaign-estimate.json", estimate)
    return 0


def _portfolio_identity() -> int:
    import importlib.metadata
    import inspect
    import neal  # pyright: ignore[reportMissingImports]
    import qpo  # pyright: ignore[reportMissingImports]

    if qpo.__file__ is None:
        raise runner.RunnerError("qpo package has no filesystem identity")
    qpo_file = Path(qpo.__file__).resolve()
    identity = {
        "python": sys.executable, "prefix": sys.prefix, "python_version": sys.version,
        "qpo_version": importlib.metadata.version("qpo"),
        "qpo_file": str(qpo_file),
        "qpo_sha256": runner.sha256_file(qpo_file),
        "qpo_source_tree_sha256": runner.sha256_tree(qpo_file.parent),
        "neal_version": importlib.metadata.version("dwave-neal"),
        "neal_sampler_file": str(Path(inspect.getfile(neal.SimulatedAnnealingSampler)).resolve()),
        "neal_sampler_sha256": runner.sha256_file(inspect.getfile(neal.SimulatedAnnealingSampler)),
        "quip_msa": runner.solver_identity(),
        "round2_cpu_sha256": runner.sha256_file(Path(__file__).resolve()),
    }
    print(json.dumps(identity, sort_keys=True))
    return 0


def _portfolio_one(args: argparse.Namespace) -> int:
    """Sample one arm and run the shared portfolio finisher once."""
    import numpy as np

    from quip_solver_dwave import portfolio_replication as portfolio

    started = time.perf_counter()
    manifest, arrays = round2_io.read_bundle(args.fixture)
    problem = portfolio.load_problem(manifest, arrays)
    qubo, _, _, _, _ = portfolio.encode_to_ising(problem)
    runtime_identity = _load(args.runtime_identity)
    sample_started = time.perf_counter()
    if args.arm.startswith("dwave-neal-"):
        import dimod
        import neal  # pyright: ignore[reportMissingImports]

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
        beta = tuple(float(value) for value in neal.default_beta_range(bqm))
        response = neal.SimulatedAnnealingSampler().sample(
            bqm, beta_range=beta, beta_schedule_type="geometric",
            num_reads=args.reads, num_sweeps=args.sweeps, seed=args.seed,
        )
        variables = list(response.variables)
        bits = response.record.sample[:, [variables.index(i) for i in range(qubo.n)]]
        spins = (2 * bits - 1).astype(np.int8)
        observed = args.arm
        representation = "dwave-neal-qubo"
        schedule = {"beta_range": list(beta), "beta_schedule_type": "geometric"}
        workspace_bytes = None
    else:
        import quip_msa  # pyright: ignore[reportMissingModuleSource]

        beta_values = quip_msa.default_beta_range(arrays["h"], arrays["edges"], arrays["j"])
        beta = (float(beta_values[0]), float(beta_values[1]))
        try:
            spins, _, metadata = quip_msa.Msa().sample_research(
                arrays["h"], arrays["edges"], arrays["j"], kernel=args.arm,
                num_reads=args.reads, num_sweeps=args.sweeps, seed=args.seed,
                beta_range=beta,
            )
        except ValueError as exc:
            if args.arm != "cpu-msa-unit":
                raise
            runner.write_result(args.out, {
                "schema": "round2-portfolio-run-v2", "run_key": args.run_key,
                "model_hash": manifest["hash"], "requested_kernel": args.arm,
                "observed_kernel": None, "status": "unsupported",
                "unsupported_reason": str(exc), "reads": args.reads,
                "sweeps": args.sweeps, "seed": args.seed,
                "seed_input_hash": args.seed_input_hash,
                "runtime_identity": runtime_identity,
            }, None, None)
            return 0
        observed = str(metadata["observed_kernel"])
        allowed = {args.arm} if args.arm != "cpu-msa" else {"cpu-msa-unit", "cpu-msa-f64"}
        if observed not in allowed:
            raise runner.RunnerError(f"requested {args.arm!r}, observed {observed!r}")
        representation = str(metadata["representation"])
        schedule = {"beta_range": list(beta), "beta_schedule_type": "geometric"}
        workspace_bytes = metadata["workspace_bytes"]
    sample_ended = time.perf_counter()
    score = portfolio.score_reads(problem, qubo, spins)
    ended = time.perf_counter()
    original_energies = runner.score_original(
        spins, arrays["h"], arrays["edges"], arrays["j"], offset=float(arrays["offset"])
    )
    record = {
        "schema": "round2-portfolio-run-v2", "run_key": args.run_key,
        "model_hash": manifest["hash"], "requested_kernel": args.arm,
        "observed_kernel": observed, "representation": representation,
        "runtime_identity": runtime_identity,
        "reads": args.reads, "sweeps": args.sweeps, "seed": args.seed,
        "seed_input_hash": args.seed_input_hash, **schedule,
        "kernel_workspace_bytes": workspace_bytes,
        "sampling_s": sample_ended - sample_started,
        "finish_s": ended - sample_ended, "child_end_to_end_s": ended - started,
        "monotonic_start_s": started, "scored_available_monotonic_s": ended,
        "monotonic_end_s": ended, "objective": float(score["objective"]),
        "feasible": bool(score["feasible"]),
        "selected_bits": _jsonable(score.get("selected_bits")),
        "weights": _jsonable(score.get("weights")),
        "status": "completed",
    }
    runner.write_result(args.out, record, spins, original_energies)
    return 0


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _portfolio_diagnostics(args: argparse.Namespace) -> int:
    import numpy as np

    from quip_solver_dwave import portfolio_replication as portfolio

    manifest, arrays = round2_io.read_bundle(args.fixture)
    problem = portfolio.load_problem(manifest, arrays)
    with np.load(args.samples) as saved:
        spins = np.asarray(saved["spins"], dtype=np.int8)
    diagnostics = portfolio.per_read_diagnostics(problem, spins)
    payload = {
        "schema": "round2-portfolio-diagnostics-v1",
        "excluded_from_application_deadline": True,
        "model_hash": manifest["hash"], "diagnostics": _jsonable(diagnostics),
        "weighting": runner.portfolio_weighting_summary(diagnostics),
    }
    regime_io.atomic_write_json(args.out, payload)
    return 0


def _portfolio_parent_finalized(result: dict[str, Any]) -> bool:
    return all(
        name in result
        for name in (
            "caller_end_to_end_s", "host", "deadline_evidence", "answer",
            "deadline_status", "subprocess",
        )
    )


def _portfolio_diagnostics_complete(result: dict[str, Any]) -> bool:
    metadata = result.get("diagnostics")
    return (
        isinstance(metadata, dict)
        and metadata.get("excluded_from_application_deadline") is True
        and Path(str(metadata.get("path", ""))).is_file()
    )


def _finish_portfolio_diagnostics(
    *, result: dict[str, Any], result_path: Path, run_dir: Path, fixture: Path,
    portfolio_python: Path, script: Path, environment: dict[str, str],
) -> None:
    if not result["answer"]["available"] or _portfolio_diagnostics_complete(result):
        return
    diagnostics_path = run_dir / "diagnostics.json"
    diagnostic = subprocess.run(
        [str(portfolio_python), str(script), "portfolio-diagnostics",
         "--fixture", str(fixture), "--samples", str(run_dir / "samples.npz"),
         "--out", str(diagnostics_path)],
        cwd=Path(__file__).parents[1], env=environment,
        capture_output=True, text=True,
    )
    if diagnostic.returncode != 0:
        raise runner.RunnerError(f"portfolio diagnostics failed: {diagnostic.stderr}")
    result["diagnostics"] = {
        "path": str(diagnostics_path),
        "excluded_from_application_deadline": True,
    }
    regime_io.atomic_write_json(result_path, result)


def _seeded_one(args: argparse.Namespace) -> int:
    """Run 32 seeded plus 32 cold weighted-MSA lanes from an explicit seed artifact."""
    manifest, arrays = round2_io.read_bundle(args.bundles_root / args.cell / args.nonce)
    initial_spins, variable_labels = runner.load_seed_artifact(args.seed_source)
    runner.validate_seed_source(manifest, initial_spins, variable_labels)
    row = {"cell": args.cell, "nonce": args.nonce, "model_hash": manifest["hash"]}
    job = runner.job_for_row(
        row, "cpu-msa-f64", runner.SEEDED_SWEEPS,
        runner.SEEDED_LANES + runner.COLD_LANES,
        variant=f"seeded-{args.source_kind}", repetition_kind="quality",
    )
    identity = runner.solver_identity()
    beta = runner.logical_beta_range(arrays["h"], arrays["edges"], arrays["j"])
    source_identity = {
        "kind": args.source_kind, "path": str(args.seed_source.resolve()),
        "sha256": runner.sha256_file(args.seed_source),
        "variable_order": "bundle-variable_labels-exact",
    }
    policy = runner.ComparisonPolicy.calibration()
    key = runner.run_key(
        job, identity, beta, policy, seed_source_hash=source_identity["sha256"]
    )
    existing = runner.completed_result(args.out, key)
    if existing is not None:
        return 0
    record, spins, energies = runner.execute_cpu_job(
        job, args.bundles_root, identity, beta,
        initial_spins=initial_spins, seed_source=source_identity,
        comparison_policy=policy,
    )
    runner.write_result(args.out, record, spins, energies)
    return 0


def _portfolio(args: argparse.Namespace) -> int:
    """Enforce one application deadline for every portfolio solver arm."""
    cpu = _choose_cpu(args.cpu)
    sibling = _sibling(cpu)
    physical = _physical_cores()
    args.out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1])
    identity_text = subprocess.check_output(
        [str(args.portfolio_python), str(script), "portfolio-identity"],
        cwd=Path(__file__).parents[1], env=environment, text=True,
    )
    runtime_identity = json.loads(identity_text)
    identity_path = args.out / "runtime-identity.json"
    regime_io.atomic_write_json(identity_path, runtime_identity)
    for fixture in args.fixtures:
        manifest, _ = round2_io.read_bundle(fixture)
        for arm in args.arms:
            arm_reads, arm_sweeps = _portfolio_arm_settings(arm, args.reads, args.sweeps)
            seed, seed_hash = runner.seed_for(
                manifest["hash"], arm, arm_sweeps, arm_reads, "portfolio-deadline",
            )
            seed = runner.neal_seed(seed)
            key = runner.portfolio_run_key(
                model_hash=manifest["hash"], arm=arm, reads=arm_reads,
                sweeps=arm_sweeps, deadline_s=args.deadline_s, seed=seed,
                seed_input_hash=seed_hash, runtime_identity=runtime_identity,
            )
            run_dir = args.out / manifest["hash"] / arm
            existing = runner.completed_result(run_dir, key)
            if existing is not None:
                result_path = run_dir / "result.json"
                if _portfolio_parent_finalized(existing):
                    _finish_portfolio_diagnostics(
                        result=existing, result_path=result_path, run_dir=run_dir,
                        fixture=fixture, portfolio_python=args.portfolio_python,
                        script=script, environment=environment,
                    )
                    if (
                        not existing["answer"]["available"]
                        or _portfolio_diagnostics_complete(existing)
                    ):
                        continue
                    raise runner.RunnerError(f"portfolio diagnostics incomplete in {run_dir}")
                existing.setdefault("child_end_to_end_s", existing.get("end_to_end_s"))
                result = runner.portfolio_deadline_result(
                    existing,
                    {"status": "unknown", "elapsed_s": None, "returncode": None,
                     "stdout": "", "stderr": "parent finalization was interrupted"},
                    args.deadline_s, caller_elapsed_s=None,
                )
                result.update({
                    "caller_end_to_end_s": None, "end_to_end_s": None,
                    "host": {
                        "cpu": cpu, "sibling": sibling, "governor": _governor(cpu),
                        "contamination": {
                            "contaminated": True,
                            "reasons": [{"kind": "interrupted-before-parent-finalization"}],
                            "repeat_required": True, "automatic_repeat": False,
                        },
                    },
                    "deadline_evidence": {
                        "eligible_for_solver_decision": False,
                        "reason": "caller timing and host evidence were interrupted",
                    },
                })
                regime_io.atomic_write_json(result_path, result)
                _finish_portfolio_diagnostics(
                    result=result, result_path=result_path, run_dir=run_dir,
                    fixture=fixture, portfolio_python=args.portfolio_python,
                    script=script, environment=environment,
                )
                continue
            result_path = run_dir / "result.json"
            caller_started = time.monotonic()
            before_load = os.getloadavg()
            before_sibling = _cpu_times(sibling)
            remaining_s = args.deadline_s - (time.monotonic() - caller_started)
            if remaining_s <= 0:
                outcome = {
                    "status": "timeout", "elapsed_s": 0.0, "returncode": None,
                    "stdout": "", "stderr": "deadline elapsed before child dispatch",
                }
            else:
                outcome = runner.run_with_deadline(
                    ["taskset", "-c", str(cpu), str(args.portfolio_python), str(script),
                     "portfolio-one", "--fixture", str(fixture), "--out", str(run_dir),
                     "--arm", arm, "--reads", str(arm_reads), "--sweeps", str(arm_sweeps),
                     "--seed", str(seed), "--seed-input-hash", seed_hash,
                     "--run-key", key, "--runtime-identity", str(identity_path)],
                    timeout_s=remaining_s, cwd=Path(__file__).parents[1], env=environment,
                )
            after_load = os.getloadavg()
            contamination = runner.contamination_verdict(
                before_load=before_load, after_load=after_load,
                physical_cores=physical,
                sibling_busy_fraction=_busy(before_sibling, _cpu_times(sibling)),
            )
            saved = None
            if result_path.exists():
                candidate = _load(result_path)
                if candidate.get("status") == "completed":
                    saved = runner.completed_result(run_dir, key)
                elif candidate.get("run_key") != key:
                    raise runner.RunnerError(f"portfolio resume identity mismatch in {result_path}")
                else:
                    saved = candidate
            caller_elapsed = time.monotonic() - caller_started
            result = runner.portfolio_deadline_result(
                saved, outcome, args.deadline_s, caller_elapsed_s=caller_elapsed,
            )
            result.update({
                "schema": "round2-portfolio-run-v2", "run_key": key,
                "model_hash": manifest["hash"], "requested_kernel": arm,
                "reads": arm_reads, "sweeps": arm_sweeps, "seed": seed,
                "seed_input_hash": seed_hash, "runtime_identity": runtime_identity,
                "caller_end_to_end_s": caller_elapsed, "end_to_end_s": caller_elapsed,
                "host": {
                    "loadavg_before": list(before_load), "loadavg_after": list(after_load),
                    "cpu": cpu, "sibling": sibling, "governor": _governor(cpu),
                    "contamination": contamination,
                },
                "deadline_evidence": {
                    "eligible_for_solver_decision": not contamination["contaminated"],
                    "reason": (
                        None if not contamination["contaminated"]
                        else "loaded-host deadline evidence is excluded; fixed-sweep quality remains usable"
                    ),
                },
            })
            if saved is None:
                runner.write_result(run_dir, result, None, None)
            else:
                regime_io.atomic_write_json(result_path, result)
            _finish_portfolio_diagnostics(
                result=result, result_path=result_path, run_dir=run_dir,
                fixture=fixture, portfolio_python=args.portfolio_python,
                script=script, environment=environment,
            )
    return 0


def _portfolio_arm_settings(arm: str, reads: int, sweeps: int) -> tuple[int, int]:
    if arm == "dwave-neal-reference":
        return 500, 500
    if arm == "dwave-neal-historical-overnight":
        return 500, 1000
    return reads, sweeps


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("pilot", "campaign"):
        command = commands.add_parser(name)
        command.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        command.add_argument("--bundles-root", type=Path, default=DEFAULT_BUNDLES)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
        command.add_argument("--cpu", type=int)
        command.add_argument("--timeout-s", type=float, default=3600.0)
        command.add_argument("--comparison-policy", type=Path)
        command.add_argument("--dry-run", action="store_true")
    one = commands.add_parser("run-one")
    one.add_argument("--job", type=Path, required=True)
    one.add_argument("--bundles-root", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    portfolio = commands.add_parser("portfolio-one")
    portfolio.add_argument("--fixture", type=Path, required=True)
    portfolio.add_argument("--out", type=Path, required=True)
    portfolio.add_argument("--reads", type=int, default=500)
    portfolio.add_argument("--sweeps", type=int, default=500)
    portfolio.add_argument("--seed", type=int, required=True)
    portfolio.add_argument("--seed-input-hash", required=True)
    portfolio.add_argument("--arm", required=True)
    portfolio.add_argument("--run-key", required=True)
    portfolio.add_argument("--runtime-identity", type=Path, required=True)
    commands.add_parser("portfolio-identity")
    diagnostics = commands.add_parser("portfolio-diagnostics")
    diagnostics.add_argument("--fixture", type=Path, required=True)
    diagnostics.add_argument("--samples", type=Path, required=True)
    diagnostics.add_argument("--out", type=Path, required=True)
    portfolio_run = commands.add_parser("portfolio")
    portfolio_run.add_argument("--fixtures", type=Path, nargs="+", required=True)
    portfolio_run.add_argument("--out", type=Path, required=True)
    portfolio_run.add_argument("--portfolio-python", type=Path, required=True)
    portfolio_run.add_argument("--reads", type=int, default=500)
    portfolio_run.add_argument("--sweeps", type=int, default=500)
    portfolio_run.add_argument("--deadline-s", type=float, default=10.0)
    portfolio_run.add_argument("--cpu", type=int)
    portfolio_run.add_argument(
        "--arms", nargs="+",
        default=[
            "dwave-neal-reference", "dwave-neal-historical-overnight",
            "cpu-sa", "cpu-msa-unit", "cpu-msa-f64", "cpu-msa",
        ],
        choices=[
            "dwave-neal-reference", "dwave-neal-historical-overnight",
            "cpu-sa", "cpu-msa-unit", "cpu-msa-f64", "cpu-msa",
        ],
    )
    seeded = commands.add_parser("seeded-one")
    seeded.add_argument("--bundles-root", type=Path, default=DEFAULT_BUNDLES)
    seeded.add_argument("--cell", required=True)
    seeded.add_argument("--nonce", required=True)
    seeded.add_argument("--seed-source", type=Path, required=True)
    seeded.add_argument("--source-kind", choices=("qpu", "cpu-lite"), required=True)
    seeded.add_argument("--out", type=Path, required=True)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run-one":
        return _run_one(args)
    if args.command == "portfolio-one":
        return _portfolio_one(args)
    if args.command == "portfolio-identity":
        return _portfolio_identity()
    if args.command == "portfolio-diagnostics":
        return _portfolio_diagnostics(args)
    if args.command == "portfolio":
        return _portfolio(args)
    if args.command == "seeded-one":
        return _seeded_one(args)
    return _run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
