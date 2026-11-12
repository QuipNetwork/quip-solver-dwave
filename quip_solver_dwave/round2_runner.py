"""Controlled, resumable CPU measurements for the Round 2 regime search."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from quip_solver_dwave import regime_io, round2_io

SWEEP_DEPTHS = (512, 2048, 8192, 32768, 131072)
PILOT_SWEEP_DEPTHS = (512, 2048)
CONTROLLED_KERNELS = ("cpu-sa", "cpu-msa-f64", "cpu-msa-unit")
DISPATCH_KERNELS = (*CONTROLLED_KERNELS, "cpu-msa")
PILOT_MODELS_PER_CELL = 5
PILOT_READS = 64
PILOT_TIMING_REPS = 3
CAMPAIGN_MODELS_PER_CELL = 100
CAMPAIGN_READS = 64
SEEDED_SWEEPS = 32768
SEEDED_LANES = 32
COLD_LANES = 32
PILOT_ORDER_SEED = 20260922
CAMPAIGN_ORDER_SEED = 20260923


class RunnerError(RuntimeError):
    """A benchmark identity, score, or durable-result invariant failed."""


def deadline_status(elapsed_s: float, deadline_s: float, exit_ok: bool) -> str:
    if not exit_ok:
        return "failed"
    if elapsed_s > deadline_s:
        return "timeout"
    return "completed"


@dataclass(frozen=True)
class ComparisonPolicy:
    """Frozen matched-comparison inputs, or explicit calibration unavailability."""

    phase: str
    quality_target: Optional[float]
    targets: Optional[dict[str, float]]
    target_source_hash: Optional[str]
    time_budget_s: Optional[float]
    quality_tolerance: Optional[float]
    runtime_tolerance_s: Optional[float]
    timing_boundary: str
    setup_cache_policy: str

    @classmethod
    def calibration(cls) -> "ComparisonPolicy":
        return cls(
            phase="calibration",
            quality_target=None,
            targets=None,
            target_source_hash=None,
            time_budget_s=None,
            quality_tolerance=None,
            runtime_tolerance_s=None,
            timing_boundary="frozen-model-to-scored-answer",
            setup_cache_policy="fresh-process-cold-setup",
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComparisonPolicy":
        if payload.get("phase") == "calibration":
            return cls.calibration()
        required = (
            "target_source_hash", "time_budget_s",
            "quality_tolerance", "runtime_tolerance_s", "timing_boundary",
            "setup_cache_policy",
        )
        missing = [name for name in required if name not in payload]
        if payload.get("phase") != "evaluation" or missing:
            raise ValueError(
                "evaluation requires a frozen comparison policy; missing " + ", ".join(missing)
            )
        policy = cls(
            phase=payload["phase"], quality_target=payload.get("quality_target"),
            targets=payload.get("targets"),
            **{name: payload[name] for name in required},
        )
        numeric = (
            policy.time_budget_s, policy.quality_tolerance, policy.runtime_tolerance_s,
        )
        if not all(value is not None and np.isfinite(value) for value in numeric):
            raise ValueError("frozen comparison values must be finite")
        assert policy.time_budget_s is not None
        assert policy.quality_tolerance is not None
        assert policy.runtime_tolerance_s is not None
        if policy.time_budget_s <= 0 or policy.quality_tolerance < 0 or policy.runtime_tolerance_s < 0:
            raise ValueError("frozen budgets must be positive and tolerances nonnegative")
        if len(policy.target_source_hash or "") != 64:
            raise ValueError("target_source_hash must be a SHA-256 hex digest")
        if policy.timing_boundary != "frozen-model-to-scored-answer":
            raise ValueError("timing_boundary must be frozen-model-to-scored-answer")
        if policy.setup_cache_policy != "fresh-process-cold-setup":
            raise ValueError("setup_cache_policy must be fresh-process-cold-setup")
        if (policy.quality_target is None) == (policy.targets is None):
            raise ValueError("freeze exactly one scalar quality_target or per-model targets map")
        if policy.quality_target is not None and not np.isfinite(policy.quality_target):
            raise ValueError("quality_target must be finite")
        if policy.targets is not None and (
            not policy.targets or not all(np.isfinite(value) for value in policy.targets.values())
        ):
            raise ValueError("every frozen per-model target must be finite")
        return policy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CpuJob:
    cell: str
    nonce: str
    model_hash: str
    kernel: str
    sweeps: int
    reads: int
    repetition_id: int
    repetition_kind: str
    variant: str
    seed: int
    seed_input_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CpuJob":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})


def replace_job(job: CpuJob, **changes: Any) -> CpuJob:
    return replace(job, **changes)


def seed_for(model_hash: str, kernel: str, sweeps: int, reads: int, variant: str) -> tuple[int, str]:
    seed_hash = round2_io.run_id({
        "model_hash": model_hash, "kernel": kernel, "sweeps": sweeps,
        "reads": reads, "variant": variant,
    })
    return int(seed_hash[:15], 16), seed_hash


def neal_seed(seed: int) -> int:
    """Map the common deterministic seed into dwave-neal 0.6's accepted range."""
    return seed % (1 << 31)


def job_for_row(
    row: Mapping[str, Any], kernel: str, sweeps: int, reads: int, *,
    variant: str, repetition_id: int = 0, repetition_kind: str = "quality",
) -> CpuJob:
    seed, seed_hash = seed_for(row["model_hash"], kernel, sweeps, reads, variant)
    return CpuJob(
        cell=row["cell"], nonce=row["nonce"], model_hash=row["model_hash"],
        kernel=kernel, sweeps=sweeps, reads=reads, repetition_id=repetition_id,
        repetition_kind=repetition_kind, variant=variant, seed=seed,
        seed_input_hash=seed_hash,
    )


def _rows(index: Mapping[str, Any], cell: str, count: int) -> list[dict[str, Any]]:
    rows = sorted((row for row in index["rows"] if row["cell"] == cell), key=lambda row: row["nonce"])
    return rows[:count]


def build_pilot_jobs(index: Mapping[str, Any], cells: Sequence[str]) -> list[CpuJob]:
    jobs: list[CpuJob] = []
    for cell in cells:
        for row in _rows(index, cell, PILOT_MODELS_PER_CELL):
            for sweeps in PILOT_SWEEP_DEPTHS:
                for kernel in CONTROLLED_KERNELS:
                    base = job_for_row(
                        row, kernel, sweeps, PILOT_READS, variant="timing",
                        repetition_kind="timing",
                    )
                    jobs.extend(replace(base, repetition_id=rep) for rep in range(PILOT_TIMING_REPS))
    random.Random(PILOT_ORDER_SEED).shuffle(jobs)
    return jobs


def build_campaign_jobs(index: Mapping[str, Any], cells: Sequence[str]) -> list[CpuJob]:
    jobs = [
        job_for_row(row, kernel, sweeps, CAMPAIGN_READS, variant="evaluation", repetition_kind="quality")
        for cell in cells
        for row in _rows(index, cell, CAMPAIGN_MODELS_PER_CELL)
        for sweeps in SWEEP_DEPTHS
        for kernel in CONTROLLED_KERNELS
    ]
    random.Random(CAMPAIGN_ORDER_SEED).shuffle(jobs)
    return jobs


def build_quality_jobs(
    index: Mapping[str, Any], cells: Sequence[str], *, repetitions: int,
    sweeps: Sequence[int] = SWEEP_DEPTHS, reads: int = CAMPAIGN_READS,
) -> list[CpuJob]:
    return [
        job_for_row(
            row, kernel, depth, reads, variant=f"quality-{rep}",
            repetition_id=rep, repetition_kind="quality",
        )
        for cell in cells
        for row in _rows(index, cell, CAMPAIGN_MODELS_PER_CELL)
        for depth in sweeps
        for kernel in CONTROLLED_KERNELS
        for rep in range(repetitions)
    ]


def _msa():
    import quip_msa  # pyright: ignore[reportMissingModuleSource]

    return quip_msa


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: os.PathLike[str] | str, pattern: str = "*.py") -> str:
    base = Path(root)
    digest = hashlib.sha256()
    for path in sorted(base.rglob(pattern)):
        digest.update(str(path.relative_to(base)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def solver_identity() -> dict[str, Any]:
    module = _msa()
    package_dir = Path(module.__file__).resolve().parent
    binaries = sorted(package_dir.glob("*.so"))
    if len(binaries) != 1:
        raise RunnerError(f"expected one quip_msa extension in {package_dir}; found {binaries}")
    binary = binaries[0]
    return {
        "package": "quip-msa",
        "version": importlib.metadata.version("quip-msa"),
        "module_file": str(Path(module.__file__).resolve()),
        "binary_file": str(binary),
        "binary_sha256": sha256_file(binary),
        "runner_module_sha256": sha256_file(Path(__file__).resolve()),
        "python": os.path.realpath(sys.executable),
    }


def logical_beta_range(h: np.ndarray, edges: np.ndarray, j: np.ndarray) -> tuple[float, float]:
    low, high = _msa().default_beta_range(h, edges, j)
    return float(low), float(high)


def kernel_problem(
    job: CpuJob, h: np.ndarray, edges: np.ndarray, j: np.ndarray,
    beta_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], float, str]:
    """Return kernel input while preserving one logical annealing schedule.

    The cubic dimer unit arm represents each canonical half-unit bond with a
    unit bond. Its Hamiltonian is twice the logical Hamiltonian, so its beta
    endpoints are half the logical endpoints. All scoring still uses the
    canonical arrays.
    """
    if job.cell != "cubic-dimer-pm1" or job.kernel not in ("cpu-msa-unit", "cpu-msa"):
        input_hash = round2_io.run_id({
            "h": round2_io.array_hash(h), "edges": round2_io.array_hash(edges),
            "j": round2_io.array_hash(j),
        })
        return h, edges, j, beta_range, 1.0, input_hash
    unit = 0.5
    ratios = np.abs(j) / unit
    rounded = np.rint(ratios).astype(np.int64)
    if not np.allclose(ratios, rounded, rtol=0.0, atol=1e-12):
        raise RunnerError("cubic-dimer unit adapter found a non-half-unit coupling")
    kernel_edges = np.repeat(edges, rounded, axis=0)
    kernel_j = np.repeat(np.sign(j), rounded).astype(np.float64)
    kernel_beta = (beta_range[0] * unit, beta_range[1] * unit)
    input_hash = round2_io.run_id({
        "h": round2_io.array_hash(h), "edges": round2_io.array_hash(kernel_edges),
        "j": round2_io.array_hash(kernel_j),
    })
    return h, kernel_edges, kernel_j, kernel_beta, unit, input_hash


def run_key(
    job: CpuJob, identity: Mapping[str, Any], beta_range: Sequence[float],
    comparison_policy: ComparisonPolicy, *, seed_source_hash: Optional[str] = None,
) -> str:
    return round2_io.run_id({
        "job": job.to_dict(), "solver_identity": dict(identity),
        "logical_beta_range": [float(beta_range[0]), float(beta_range[1])],
        "comparison_policy": comparison_policy.to_dict(),
        "seed_source_hash": seed_source_hash,
    })


def score_original(
    spins: np.ndarray, h: np.ndarray, edges: np.ndarray, j: np.ndarray, *, offset: float = 0.0,
) -> np.ndarray:
    states = np.asarray(spins, dtype=np.float64)
    return states @ h + (states[:, edges[:, 0]] * states[:, edges[:, 1]]) @ j + offset


def validate_seed_source(
    manifest: Mapping[str, Any], spins: np.ndarray, variable_labels: np.ndarray, *,
    expected_lanes: int = SEEDED_LANES,
) -> None:
    states = np.asarray(spins)
    labels = np.asarray(variable_labels, dtype=np.int64)
    expected_labels = np.asarray(manifest["variable_labels"], dtype=np.int64)
    if states.shape != (expected_lanes, expected_labels.size):
        raise RunnerError(
            f"seed spins must have shape ({expected_lanes}, {expected_labels.size}); got {states.shape}"
        )
    if not np.array_equal(labels, expected_labels):
        raise RunnerError("seed variable order does not match the bundle variable_labels exactly")
    if not np.isin(states, (-1, 1)).all():
        raise RunnerError("seed spins must be signed +/-1 values")


def load_seed_artifact(path: os.PathLike[str] | str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as source:
        raw_spins = np.asarray(source["spins"])
        raw_labels = np.asarray(source["variable_labels"])
    if not np.isin(raw_spins, (-1, 1)).all():
        raise RunnerError("seed spins must be signed +/-1 values before integer conversion")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise RunnerError("seed variable labels must be integers")
    return raw_spins.astype(np.int8), raw_labels.astype(np.int64)


def lane_diagnostics(spins: np.ndarray) -> dict[str, int]:
    states = np.asarray(spins)
    unique = int(np.unique(states, axis=0).shape[0])
    return {
        "lanes": int(states.shape[0]), "unique_lanes": unique,
        "duplicate_lanes": int(states.shape[0]) - unique,
    }


def execute_cpu_job(
    job: CpuJob, bundles_root: os.PathLike[str] | str,
    identity: Mapping[str, Any], beta_range: tuple[float, float],
    *, initial_spins: Optional[np.ndarray] = None,
    seed_source: Optional[Mapping[str, Any]] = None,
    comparison_policy: Optional[ComparisonPolicy] = None,
) -> tuple[dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    started = time.perf_counter()
    manifest, arrays = round2_io.read_bundle(Path(bundles_root) / job.cell / job.nonce)
    if manifest["hash"] != job.model_hash:
        raise RunnerError(f"bundle hash {manifest['hash']} does not match planned {job.model_hash}")
    h, edges, j = arrays["h"], arrays["edges"], arrays["j"]
    kh, kedges, kj, submitted_beta, energy_scale, input_hash = kernel_problem(
        job, h, edges, j, beta_range
    )
    policy = comparison_policy or ComparisonPolicy.calibration()
    source_hash = None if seed_source is None else str(seed_source["sha256"])
    key = run_key(job, identity, beta_range, policy, seed_source_hash=source_hash)
    record: dict[str, Any] = {
        "schema": "round2-cpu-run-v1", "run_key": key, **job.to_dict(),
        "requested_kernel": job.kernel,
        "solver_identity": dict(identity),
        "comparison_policy": policy.to_dict(),
        "logical_beta_range": list(beta_range),
        "submitted_beta_range": list(submitted_beta),
        "kernel_energy_unit": energy_scale,
        "kernel_input_hash": input_hash,
        "setup_cache_state": "fresh-process",
        "cached_setup_s": None,
        "cached_setup_unavailable_reason": "each measured run uses a fresh process",
        "monotonic_start_s": started,
    }
    setup_started = time.perf_counter()
    sampler = _msa().Msa()
    setup_ended = time.perf_counter()
    record["fresh_setup_s"] = setup_ended - setup_started
    try:
        sample_started = time.perf_counter()
        sample_kwargs: dict[str, Any] = {}
        if initial_spins is not None:
            sample_kwargs = {
                "initial_spins": initial_spins,
                "start_beta": submitted_beta[0],
            }
            record["seed_source"] = dict(seed_source or {})
            record["input_seed_lanes"] = lane_diagnostics(initial_spins)
            record["lane_sources"] = (
                ["seeded"] * int(initial_spins.shape[0])
                + ["cold"] * (job.reads - int(initial_spins.shape[0]))
            )
            record["logical_start_beta"] = beta_range[0]
            record["submitted_start_beta"] = submitted_beta[0]
        spins, reported, metadata = sampler.sample_research(
            kh, kedges, kj, kernel=job.kernel, num_sweeps=job.sweeps,
            num_reads=job.reads, seed=job.seed, beta_range=submitted_beta,
            **sample_kwargs,
        )
        sample_ended = time.perf_counter()
    except ValueError as exc:
        if job.kernel != "cpu-msa-unit":
            raise
        ended = time.perf_counter()
        record.update({
            "status": "unsupported", "unsupported_reason": str(exc),
            "observed_kernel": None, "dispatch_identity": "cpu-msa-unit->unsupported",
            "sampling_s": None, "scored_available_monotonic_s": ended,
            "monotonic_end_s": ended, "end_to_end_s": ended - started,
            "peak_process_rss_kb": _peak_rss_kb(), "kernel_workspace_bytes": None,
        })
        return record, None, None

    observed = metadata["observed_kernel"]
    allowed = {job.kernel} if job.kernel != "cpu-msa" else {"cpu-msa-unit", "cpu-msa-f64"}
    if observed not in allowed:
        raise RunnerError(f"requested {job.kernel!r}, observed {observed!r}")
    hamiltonian = score_original(spins, h, edges, j)
    rescored = hamiltonian + float(arrays["offset"])
    logical_reported = np.asarray(reported, dtype=np.float64) * energy_scale
    if not np.isfinite(rescored).all() or not np.isfinite(logical_reported).all():
        raise RunnerError("nonfinite kernel or original-model score")
    if not np.allclose(hamiltonian, logical_reported, rtol=1e-10, atol=1e-12):
        raise RunnerError("kernel energies disagree with original float64 model scoring")
    scored_at = time.perf_counter()
    record.update({
        "status": "completed", "observed_kernel": observed,
        "dispatch_identity": f"{job.kernel}->{observed}",
        "representation": metadata["representation"],
        "rng_scheme": metadata["rng_scheme"],
        "seeded_reads": metadata["seeded_reads"],
        "kernel_workspace_bytes": metadata["workspace_bytes"],
        "sampling_s": sample_ended - sample_started,
        "scored_available_monotonic_s": scored_at,
        "monotonic_end_s": scored_at, "end_to_end_s": scored_at - started,
        "best_energy": float(rescored.min()), "mean_energy": float(rescored.mean()),
        "unique_reads": int(np.unique(spins, axis=0).shape[0]),
        "peak_process_rss_kb": _peak_rss_kb(),
    })
    if initial_spins is not None:
        record["output_seeded_lanes"] = lane_diagnostics(spins[: initial_spins.shape[0]])
        record["output_cold_lanes"] = lane_diagnostics(spins[initial_spins.shape[0] :])
    return record, np.asarray(spins, dtype=np.int8), rescored


def _peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def write_result(
    run_dir: os.PathLike[str] | str, record: Mapping[str, Any],
    spins: Optional[np.ndarray], energies: Optional[np.ndarray],
) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    saved = dict(record)
    if spins is not None and energies is not None:
        sample_path = path / "samples.npz"
        regime_io.atomic_savez(sample_path, spins=spins, energies=energies)
        saved["sample_artifact"] = {
            "sha256": sha256_file(sample_path),
            "spins_shape": list(spins.shape), "spins_dtype": spins.dtype.str,
            "energies_shape": list(energies.shape), "energies_dtype": energies.dtype.str,
        }
    regime_io.atomic_write_json(path / "result.json", saved)


def completed_result(run_dir: os.PathLike[str] | str, expected_key: str) -> Optional[dict[str, Any]]:
    path = Path(run_dir) / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("run_key") != expected_key:
        raise RunnerError(f"resume identity mismatch in {path}")
    if result.get("status") == "completed" or result.get("answer", {}).get("available"):
        _validate_sample_artifact(Path(run_dir), result)
    return result


def _validate_sample_artifact(run_dir: Path, result: Mapping[str, Any]) -> None:
    sample_path = run_dir / "samples.npz"
    metadata = result.get("sample_artifact")
    if not sample_path.exists() or not isinstance(metadata, dict):
        raise RunnerError(f"samples artifact is missing from {run_dir}")
    if sha256_file(sample_path) != metadata.get("sha256"):
        raise RunnerError(f"samples artifact hash mismatch in {run_dir}")
    try:
        with np.load(sample_path) as saved:
            spins = np.asarray(saved["spins"])
            energies = np.asarray(saved["energies"])
    except Exception as exc:
        raise RunnerError(f"samples artifact cannot be read in {run_dir}: {exc}") from exc
    expected_reads = int(result.get("reads", spins.shape[0]))
    valid = (
        list(spins.shape) == metadata.get("spins_shape")
        and list(energies.shape) == metadata.get("energies_shape")
        and spins.dtype.str == metadata.get("spins_dtype")
        and energies.dtype.str == metadata.get("energies_dtype")
        and spins.ndim == 2 and energies.shape == (expected_reads,)
        and spins.shape[0] == expected_reads and np.isin(spins, (-1, 1)).all()
        and np.isfinite(energies).all()
    )
    if not valid:
        raise RunnerError(f"samples artifact metadata or values are invalid in {run_dir}")


def run_with_deadline(
    command: Sequence[str], *, timeout_s: float, cwd: os.PathLike[str] | str,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        list(command), cwd=cwd, env=None if env is None else dict(env),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        return {
            "status": "timeout", "elapsed_s": time.monotonic() - started,
            "returncode": None, "stdout": stdout, "stderr": stderr,
        }
    elapsed = time.monotonic() - started
    return {
        "status": deadline_status(elapsed, timeout_s, process.returncode == 0),
        "elapsed_s": elapsed, "returncode": process.returncode,
        "stdout": stdout, "stderr": stderr,
    }


def contamination_verdict(
    *, before_load: Sequence[float], after_load: Sequence[float],
    physical_cores: int, sibling_busy_fraction: Optional[float],
) -> dict[str, Any]:
    """Label competing load once. The caller decides when to repeat the run."""
    reasons: list[dict[str, Any]] = []
    threshold = physical_cores / 2.0
    if before_load[0] > threshold or after_load[0] > threshold:
        reasons.append({
            "kind": "loadavg", "threshold": threshold,
            "before_1m": before_load[0], "after_1m": after_load[0],
        })
    if sibling_busy_fraction is not None and sibling_busy_fraction > 0.2:
        reasons.append({
            "kind": "sibling_busy", "threshold": 0.2,
            "observed": sibling_busy_fraction,
        })
    return {
        "contaminated": bool(reasons), "reasons": reasons,
        "repeat_required": bool(reasons), "automatic_repeat": False,
    }


def build_qpu_deadline_record(
    *, elapsed_s: Optional[float], deadline_s: float, exit_ok: bool,
    charge_known: bool, access_us: Optional[int] = None,
    reconciled_at: Optional[str] = None,
) -> dict[str, Any]:
    if elapsed_s is None and exit_ok:
        raise ValueError("a successful answer requires elapsed_s")
    if charge_known != (access_us is not None):
        raise ValueError("access_us is required exactly when charge is known")
    if charge_known and reconciled_at is None:
        raise ValueError("known charge requires reconciled_at")
    elapsed = float("inf") if elapsed_s is None else elapsed_s
    return {
        "schema": "round2-qpu-deadline-v1",
        "answer": {
            "elapsed_s": elapsed_s, "deadline_s": deadline_s,
            "exit_ok": exit_ok, "status": deadline_status(elapsed, deadline_s, exit_ok),
        },
        "charge": {
            "known": charge_known, "access_us": access_us,
            "reconciled_at": reconciled_at,
        },
    }


def portfolio_weighting_summary(diagnostics: Mapping[str, Any]) -> dict[str, int]:
    summary = {"unknown": 0, "failed": 0, "ok": 0}
    for record in diagnostics["records"]:
        value = record["weighting_failed"]
        if value is None:
            summary["unknown"] += 1
        elif value is True:
            summary["failed"] += 1
        else:
            summary["ok"] += 1
    return summary


def portfolio_run_key(
    *, model_hash: str, arm: str, reads: int, sweeps: int, deadline_s: float,
    seed: int, seed_input_hash: str, runtime_identity: Mapping[str, Any],
) -> str:
    return round2_io.run_id({
        "model_hash": model_hash, "arm": arm, "reads": reads, "sweeps": sweeps,
        "deadline_s": deadline_s, "seed": seed, "seed_input_hash": seed_input_hash,
        "runtime_identity": dict(runtime_identity),
    })


def portfolio_deadline_result(
    saved: Optional[Mapping[str, Any]], outcome: Mapping[str, Any], deadline_s: float,
    *, caller_elapsed_s: Optional[float],
) -> dict[str, Any]:
    result = dict(saved or {})
    caller_timing_known = caller_elapsed_s is not None
    timed_out = outcome["status"] == "timeout" or (
        caller_elapsed_s is not None and caller_elapsed_s > deadline_s
    )
    answer_available = saved is not None and saved.get("status") == "completed"
    answer_unsupported = saved is not None and saved.get("status") == "unsupported"
    # A failed child stays failed; a late parent shows only in deadline_status.
    result["status"] = (
        "failed" if outcome["status"] == "failed" else
        "timeout" if timed_out else
        "unsupported" if answer_unsupported else
        "completed" if answer_available and outcome["status"] == "unknown" else
        str(outcome["status"])
    )
    result["deadline_status"] = (
        "timeout" if timed_out else
        str(outcome["status"]) if caller_timing_known else "unknown"
    )
    result["deadline_s"] = deadline_s
    result["subprocess"] = dict(outcome)
    result["answer"] = {
        "available": answer_available,
        "status": "completed_late" if timed_out and answer_available else (
            "completed" if answer_available and caller_timing_known else
            "completed_deadline_unknown" if answer_available else
            "unsupported" if answer_unsupported else "unavailable"
        ),
    }
    return result


def comparison_observation(
    record: Mapping[str, Any], policy: ComparisonPolicy,
) -> dict[str, Any]:
    """Record target attainment at the only observed batch boundary."""
    if policy.phase != "evaluation":
        return {
            "available": False,
            "reason": "calibration run has no frozen evaluation target",
        }
    target = (
        policy.quality_target
        if policy.targets is None
        else policy.targets.get(str(record["model_hash"]))
    )
    if target is None:
        return {"available": False, "reason": "model has no frozen target"}
    assert policy.quality_tolerance is not None
    assert policy.time_budget_s is not None
    completed = record.get("status") == "completed"
    caller_elapsed = record.get("caller_end_to_end_s")
    elapsed = float(caller_elapsed) if completed and caller_elapsed is not None else None
    score = float(record["best_energy"]) if completed else None
    reached = bool(completed and score is not None and score <= target + policy.quality_tolerance)
    within_budget = bool(completed and elapsed is not None and elapsed <= policy.time_budget_s)
    return {
        "available": True, "quality_target": target,
        "target_source_hash": policy.target_source_hash,
        "quality_tolerance": policy.quality_tolerance,
        "time_budget_s": policy.time_budget_s,
        "timing_boundary": policy.timing_boundary,
        "timing_available": elapsed is not None,
        "timing_unavailable_reason": (
            None if elapsed is not None else "caller-visible timing was interrupted"
        ),
        "target_reached": reached,
        "time_to_target_s": elapsed if reached else None,
        "budget_censored": not reached or not within_budget,
        "quality_within_time_budget": score if within_budget else None,
    }


def estimate_campaign(
    pilot_records: Sequence[Mapping[str, Any]], campaign_jobs: Sequence[CpuJob],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    unsupported: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in pilot_records:
        if row.get("status") == "completed" and row.get("caller_end_to_end_s") is not None:
            groups.setdefault((str(row["cell"]), str(row["requested_kernel"])), []).append(row)
        elif row.get("status") == "unsupported":
            unsupported.setdefault(
                (str(row["cell"]), str(row["requested_kernel"])), []
            ).append(row)
    cells: dict[str, dict[str, Any]] = {}
    total = 0.0
    total_available = True
    for (cell, kernel), jobs in _group_campaign_jobs(campaign_jobs).items():
        rows = groups.get((cell, kernel), [])
        missing_caller_timing = sum(
            1 for row in pilot_records
            if row.get("status") == "completed"
            and str(row["cell"]) == cell
            and str(row["requested_kernel"]) == kernel
            and row.get("caller_end_to_end_s") is None
        )
        unsupported_rows = unsupported.get((cell, kernel), [])
        unsupported_depths = {int(row["sweeps"]) for row in unsupported_rows}
        if unsupported_depths.issuperset(PILOT_SWEEP_DEPTHS):
            reasons = {str(row.get("unsupported_reason", "unsupported")) for row in unsupported_rows}
            cells.setdefault(cell, {})[kernel] = {
                "jobs": len(jobs), "status": "unsupported",
                "estimated_sequential_s": 0.0,
                "unsupported_reason": "; ".join(sorted(reasons)),
            }
            continue
        by_depth = {
            depth: [row for row in rows if int(row["sweeps"]) == depth]
            for depth in PILOT_SWEEP_DEPTHS
        }
        if any(not by_depth[depth] for depth in PILOT_SWEEP_DEPTHS):
            cells.setdefault(cell, {})[kernel] = {
                "jobs": len(jobs), "status": "unavailable",
                "reason": "missing caller-visible timing at one or both pilot depths",
                "estimated_sequential_s": None,
                "excluded_missing_caller_timing": missing_caller_timing,
            }
            total_available = False
            continue
        low, high = PILOT_SWEEP_DEPTHS
        low_sample = float(np.median([row["sampling_s"] for row in by_depth[low]]))
        high_sample = float(np.median([row["sampling_s"] for row in by_depth[high]]))
        slope = max(0.0, (high_sample - low_sample) / (high - low))
        intercept = max(0.0, low_sample - slope * low)
        overhead = float(np.median([
            float(row["caller_end_to_end_s"]) - float(row["sampling_s"])
            for row in rows
        ]))
        sampling = sum(intercept + slope * job.sweeps for job in jobs)
        sequential = sampling + overhead * len(jobs)
        cells.setdefault(cell, {})[kernel] = {
            "jobs": len(jobs), "pilot_depths": list(PILOT_SWEEP_DEPTHS),
            "sampling_s_per_sweep": slope, "sampling_intercept_s": intercept,
            "measured_process_and_setup_overhead_s": overhead,
            "estimated_sampling_s": sampling,
            "estimated_sequential_s": sequential,
            "excluded_missing_caller_timing": missing_caller_timing,
        }
        total += sequential
    return {
        "classification": "cost-planning extrapolation, not scientific runtime",
        "method": "per-cell and per-kernel two-depth sampling fit plus measured process/setup overhead",
        "cells": cells, "estimated_sequential_s": total if total_available else None,
    }


def _group_campaign_jobs(jobs: Sequence[CpuJob]) -> dict[tuple[str, str], list[CpuJob]]:
    groups: dict[tuple[str, str], list[CpuJob]] = {}
    for job in jobs:
        groups.setdefault((job.cell, job.kernel), []).append(job)
    return groups
