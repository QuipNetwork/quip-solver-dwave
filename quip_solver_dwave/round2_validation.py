"""Read-only validation of saved Round 2 CPU campaign records.

This module lives outside ``round2_runner`` on purpose: the runner's own file
hash is part of the frozen solver identity, so validation code must not
change it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from quip_solver_dwave import round2_runner as runner

ENERGY_TOLERANCE = 1e-6
UNIT_MAX_FIELD = 100


def unsupported_is_expected(
    kernel: str, h: np.ndarray, edges: np.ndarray, j: np.ndarray,
) -> bool:
    """Return True when a kernel may legitimately refuse this kernel input.

    ``h``, ``edges``, and ``j`` are the arrays submitted to the kernel, after
    the cubic dimer unit adapter, not the canonical bundle arrays. The unit
    kernel (``quip-miner-cpu/src/sa_int.rs``) accepts couplings of exactly
    -1 or +1, finite whole-number fields, and ``max_i(|h_i| + deg_i) <= 100``.
    """
    if kernel != "cpu-msa-unit":
        return False
    if h.size == 0 or not np.isfinite(h).all() or (h != np.rint(h)).any():
        return True
    # Self-loops and zero couplings are skipped by the kernel, so they neither
    # disqualify the model nor add degree.
    live = (edges[:, 0] != edges[:, 1]) & (j != 0.0)
    if (np.abs(j[live]) != 1.0).any():
        return True
    degree = np.bincount(edges[live].ravel(), minlength=h.size)
    return bool((np.abs(h) + degree).max() > UNIT_MAX_FIELD)


def rescore(spins: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Score spins with dimod, independently of ``runner.score_original``."""
    import dimod

    h, edges, j = arrays["h"], arrays["edges"], arrays["j"]
    bqm = dimod.BinaryQuadraticModel(
        dict(enumerate(h.tolist())), {}, float(arrays["offset"]), dimod.SPIN,
    )
    bqm.add_quadratic_from(zip(edges[:, 0].tolist(), edges[:, 1].tolist(), j.tolist()))
    return np.asarray(bqm.energies((spins, range(h.size))), dtype=np.float64)


def saved_read_problems(
    spins: np.ndarray, energies: np.ndarray, arrays: Mapping[str, np.ndarray], *, reads: int,
) -> list[str]:
    """Check saved reads against the frozen model. Empty means valid."""
    if spins.shape != (reads, arrays["h"].size) or energies.shape != (reads,):
        return [f"spins shape {spins.shape} differs from ({reads}, {arrays['h'].size})"]
    if not np.isin(spins, (-1, 1)).all():
        return ["spins are not signed +/-1 values"]
    if not np.allclose(energies, rescore(spins, arrays), rtol=0.0, atol=ENERGY_TOLERANCE):
        return ["saved energies differ from rescored original-unit energies"]
    return []


def _close(value: Any, expected: float) -> bool:
    return value is not None and bool(np.isclose(float(value), expected, rtol=0.0, atol=ENERGY_TOLERANCE))


def _load_reads(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as saved:
        return np.asarray(saved["spins"]), np.asarray(saved["energies"])


def _read_and_summary_problems(
    record: Mapping[str, Any], samples_path: Path, arrays: Mapping[str, np.ndarray], *, reads: int,
) -> list[str]:
    spins, energies = _load_reads(samples_path)
    found = saved_read_problems(spins, energies, arrays, reads=reads)
    if found:
        return found
    if not _close(record.get("best_energy"), float(energies.min())):
        found.append("best_energy differs from the rescored reads")
    if not _close(record.get("mean_energy"), float(energies.mean())):
        found.append("mean_energy differs from the rescored reads")
    if record.get("unique_reads") != int(np.unique(spins, axis=0).shape[0]):
        found.append("unique_reads differs from the saved spins")
    return found


def _tree_identity_problems(
    record: Mapping[str, Any], *, model_hash: str, binary_sha256: str,
) -> list[str]:
    problems = []
    if record.get("model_hash") != model_hash:
        problems.append("model_hash differs from the frozen bundle")
    if record.get("solver_identity", {}).get("binary_sha256") != binary_sha256:
        problems.append("solver binary differs from the frozen binary")
    if record.get("exit_ok") is not True or record.get("error"):
        problems.append("run did not exit cleanly")
    return problems


def validate_tree_record(
    record: Mapping[str, Any], samples_path: Path, arrays: Mapping[str, np.ndarray], *,
    model_hash: str, binary_sha256: str,
) -> list[str]:
    """Validate one quip-miner-cpu campaign record. Empty means valid."""
    problems = _tree_identity_problems(record, model_hash=model_hash, binary_sha256=binary_sha256)
    kernel = str(record.get("requested_kernel"))
    if record.get("unsupported"):
        job = runner.CpuJob.from_dict({**record, "kernel": kernel})
        beta = tuple(record["beta_range"])
        kh, kedges, kj, *_ = runner.kernel_problem(job, arrays["h"], arrays["edges"], arrays["j"], beta)
        if not unsupported_is_expected(kernel, kh, kedges, kj):
            problems.append(f"{kernel} is unsupported on a model it accepts")
        if samples_path.exists():
            problems.append("unsupported record has a samples artifact")
        return problems
    if record.get("observed_kernel") != kernel:
        problems.append("observed kernel differs from the requested kernel")
    if not samples_path.exists():
        return problems + ["samples artifact is missing"]
    return problems + _read_and_summary_problems(record, samples_path, arrays, reads=int(record["reads"]))


def validate_seeded_record(
    record: Mapping[str, Any], samples_path: Path, arrays: Mapping[str, np.ndarray], *,
    model_hash: str, binary_sha256: str,
) -> list[str]:
    """Validate one quip-miner-cpu seeded sweep. Seeded lanes precede cold lanes."""
    problems = _tree_identity_problems(record, model_hash=model_hash, binary_sha256=binary_sha256)
    if record.get("observed_kernel") != "cpu-msa-f64":
        problems.append("seeded sweep did not observe cpu-msa-f64")
    if not samples_path.exists():
        return problems + ["samples artifact is missing"]
    seeded = int(record["seed_lanes"])
    reads = seeded + int(record["cold_lanes"])
    spins, energies = _load_reads(samples_path)
    found = saved_read_problems(spins, energies, arrays, reads=reads)
    if found:
        return problems + found
    if not _close(record.get("best_seeded_energy"), float(energies[:seeded].min())):
        problems.append("best_seeded_energy differs from the rescored seeded lanes")
    if not _close(record.get("best_cold_energy"), float(energies[seeded:].min())):
        problems.append("best_cold_energy differs from the rescored cold lanes")
    if record.get("unique_reads") != int(np.unique(spins, axis=0).shape[0]):
        problems.append("unique_reads differs from the saved spins")
    return problems


def validate_campaign_record(
    job: runner.CpuJob, record: Mapping[str, Any], run_dir: Path,
    arrays: Mapping[str, np.ndarray], *, identity: Mapping[str, Any],
    policy: runner.ComparisonPolicy, beta_range: tuple[float, float],
) -> list[str]:
    """Return every way a saved record departs from its frozen job. Empty means valid."""
    problems: list[str] = []
    h, edges, j = arrays["h"], arrays["edges"], arrays["j"]
    kh, kedges, kj, submitted_beta, energy_unit, input_hash = runner.kernel_problem(
        job, h, edges, j, beta_range
    )
    expected = {
        "schema": "round2-cpu-run-v1",
        "run_key": runner.run_key(job, identity, beta_range, policy),
        **job.to_dict(),
        "requested_kernel": job.kernel,
        "solver_identity": dict(identity),
        "comparison_policy": policy.to_dict(),
        "logical_beta_range": list(beta_range),
        "submitted_beta_range": list(submitted_beta),
        "kernel_energy_unit": energy_unit,
        "kernel_input_hash": input_hash,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            problems.append(f"{name} differs from the frozen job")
    if record.get("comparison") != runner.comparison_observation(record, policy):
        problems.append("comparison differs from a recomputed observation")
    if "contamination" not in record.get("host", {}):
        problems.append("host contamination record is missing")
    timing = record.get("timing_evidence", {})
    if record.get("caller_end_to_end_s") is None and timing.get("available") is not False:
        problems.append("missing caller timing is not explicitly withheld")

    status = record.get("status")
    if status == "unsupported":
        if not unsupported_is_expected(job.kernel, kh, kedges, kj):
            problems.append(f"{job.kernel} is unsupported on a model it accepts")
        if (run_dir / "samples.npz").exists():
            problems.append("unsupported record has a samples artifact")
        return problems
    if status != "completed":
        return problems + [f"status is {status!r}"]
    if record.get("dispatch_identity") != f"{job.kernel}->{job.kernel}":
        problems.append("observed kernel differs from the requested kernel")
    try:
        runner.completed_result(run_dir, str(expected["run_key"]))
    except runner.RunnerError as exc:
        return problems + [str(exc)]
    return problems + _read_and_summary_problems(record, run_dir / "samples.npz", arrays, reads=job.reads)
