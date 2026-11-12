"""Validate every saved Round 2 CPU campaign record against its frozen manifest.

Reads only. Writes a single summary, ``validation.json``, beside the manifest,
and exits nonzero when any record, directory, or dispatch state is invalid.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Sequence

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave import round2_runner as runner
from quip_solver_dwave import round2_validation as validation


def _timing_class(record: dict[str, Any]) -> str:
    kinds = sorted(
        reason["kind"] for reason in record.get("host", {}).get("contamination", {}).get("reasons", [])
    )
    return "+".join(kinds) or "uncontaminated"


def validate(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["solver_identity"]
    policy = runner.ComparisonPolicy.from_dict(manifest["comparison_policy"])
    bundles = Path(manifest["bundles_root"])
    models: dict[tuple[str, str], tuple[dict[str, Any], tuple[float, float]]] = {}
    problems: list[dict[str, Any]] = []
    counts: collections.Counter[tuple[str, str, int, str]] = collections.Counter()
    timing: collections.Counter[tuple[str, str]] = collections.Counter()
    unsupported: collections.Counter[tuple[str, str, str]] = collections.Counter()
    keys: set[str] = set()
    for position, row in enumerate(manifest["jobs"]):
        job = runner.CpuJob.from_dict(row)
        if (job.cell, job.nonce) not in models:
            bundle, arrays = round2_io.read_bundle(bundles / job.cell / job.nonce)
            if bundle["hash"] != job.model_hash:
                raise runner.RunnerError(f"frozen model changed: {job.cell}/{job.nonce}")
            beta = runner.logical_beta_range(arrays["h"], arrays["edges"], arrays["j"])
            models[(job.cell, job.nonce)] = (arrays, beta)
        arrays, beta = models[(job.cell, job.nonce)]
        key = runner.run_key(job, identity, beta, policy)
        if key in keys:
            problems.append({"position": position, "problems": ["duplicate run key"]})
            continue
        keys.add(key)
        run_dir = root / "runs" / key
        result_path = run_dir / "result.json"
        if not result_path.exists():
            problems.append({"position": position, "run_key": key, "problems": ["result is missing"]})
            continue
        record = json.loads(result_path.read_text(encoding="utf-8"))
        found = validation.validate_campaign_record(
            job, record, run_dir, arrays, identity=identity, policy=policy, beta_range=beta,
        )
        if (run_dir / "dispatch-failure.json").exists():
            found.append("unresolved dispatch failure")
        if found:
            problems.append({"position": position, "run_key": key, "problems": found})
        counts[(job.cell, job.kernel, job.sweeps, str(record.get("status")))] += 1
        timing[(str(record.get("status")), _timing_class(record))] += 1
        if record.get("status") == "unsupported":
            unsupported[(job.cell, job.kernel, str(record.get("unsupported_reason")))] += 1
    orphans = sorted(path.name for path in (root / "runs").iterdir() if path.name not in keys)
    return {
        "schema": "round2-cpu-validation-v1",
        "manifest_sha256": runner.sha256_file(root / "manifest.json"),
        "live_solver_identity_matches": runner.solver_identity() == identity,
        "planned_jobs": len(manifest["jobs"]),
        "validated_records": len(keys) - sum(1 for item in problems if "run_key" in item),
        "rescored_records": sum(
            count for (_, _, _, status), count in counts.items() if status == "completed"
        ),
        "original_energy_tolerance": validation.ENERGY_TOLERANCE,
        "orphan_run_dirs": orphans,
        "invalid_records": problems,
        "counts": [
            {"cell": cell, "kernel": kernel, "sweeps": sweeps, "status": status, "count": count}
            for (cell, kernel, sweeps, status), count in sorted(counts.items())
        ],
        "timing_classes": [
            {"status": status, "contamination": kind, "count": count}
            for (status, kind), count in sorted(timing.items())
        ],
        "unsupported_reasons": [
            {"cell": cell, "kernel": kernel, "reason": reason, "count": count}
            for (cell, kernel, reason), count in sorted(unsupported.items())
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="campaign directory with manifest.json")
    args = parser.parse_args(argv)
    summary = validate(args.root)
    regime_io.atomic_write_json(args.root / "validation.json", summary)
    valid = not summary["invalid_records"] and not summary["orphan_run_dirs"]
    print(json.dumps({
        "valid": valid, "validated_records": summary["validated_records"],
        "rescored_records": summary["rescored_records"],
        "invalid_records": len(summary["invalid_records"]),
        "orphan_run_dirs": len(summary["orphan_run_dirs"]),
        "live_solver_identity_matches": summary["live_solver_identity_matches"],
    }))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
