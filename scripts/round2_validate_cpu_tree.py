"""Validate the quip-miner-cpu Round 2 CPU tree against the frozen campaign models.

Reads only. The frozen model set and solver binary come from the validated
``cpu-campaign`` manifest. Writes one summary, ``cpu-tree-validation.json``,
and exits nonzero when any record is invalid.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave import round2_runner as runner
from quip_solver_dwave import round2_validation as validation


def _records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob("*/*.json"))
    ]


def _agreement(
    tree_best: dict[tuple[str, str, int, str], float],
    campaign_best: dict[tuple[str, str, int, str], float],
) -> list[dict[str, Any]]:
    """Per-cell paired best-energy differences, tree minus campaign, by model."""
    cells: dict[tuple[str, str, int], list[float]] = collections.defaultdict(list)
    for key, energy in campaign_best.items():
        if key in tree_best:
            cell, kernel, sweeps, _ = key
            cells[(cell, kernel, sweeps)].append(tree_best[key] - energy)
    rows = []
    for (cell, kernel, sweeps), gaps in sorted(cells.items()):
        ties = sum(abs(gap) <= validation.ENERGY_TOLERANCE for gap in gaps)
        lower = sum(gap < -validation.ENERGY_TOLERANCE for gap in gaps)
        rows.append({
            "cell": cell, "kernel": kernel, "sweeps": sweeps, "paired_models": len(gaps),
            "median_gap": statistics.median(gaps), "ties": ties,
            "tree_lower": lower, "campaign_lower": len(gaps) - ties - lower,
        })
    return rows


def validate(round2: Path) -> dict[str, Any]:
    manifest = json.loads((round2 / "cpu-campaign" / "manifest.json").read_text(encoding="utf-8"))
    binary = manifest["solver_identity"]["binary_sha256"]
    bundles = Path(manifest["bundles_root"])
    frozen = {(job["cell"], job["nonce"]): job["model_hash"] for job in manifest["jobs"]}
    arrays = {key: round2_io.read_bundle(bundles / key[0] / key[1])[1] for key in frozen}
    offsets = {float(model["offset"]) for model in arrays.values()}
    tree = round2 / "cpu"
    invalid: list[dict[str, Any]] = []

    tree_best: dict[tuple[str, str, int, str], float] = {}
    counts: collections.Counter[tuple[str, str, int, str]] = collections.Counter()
    for path, record in _records(tree / "campaign"):
        key = (record["cell"], record["nonce"])
        problems = (
            ["model is not in the frozen campaign"] if key not in frozen
            else validation.validate_tree_record(
                record, path.with_suffix(".npz"), arrays[key],
                model_hash=frozen[key], binary_sha256=binary,
            )
        )
        if problems:
            invalid.append({"path": str(path), "problems": problems})
        status = "unsupported" if record.get("unsupported") else "completed"
        counts[(record["cell"], record["requested_kernel"], record["sweeps"], status)] += 1
        if status == "completed" and not problems:
            tree_best[(record["cell"], record["requested_kernel"], record["sweeps"], record["nonce"])] = float(
                record["best_energy"]
            )

    seeded: collections.Counter[tuple[str, str]] = collections.Counter()
    for path, record in _records(tree / "seeded-sweep-v2"):
        key = (record["cell"], record["nonce"])
        problems = (
            ["model is not in the frozen campaign"] if key not in frozen
            else validation.validate_seeded_record(
                record, path.with_suffix(".npz"), arrays[key],
                model_hash=frozen[key], binary_sha256=binary,
            )
        )
        if record.get("sweeps") != 32768 or (record.get("seed_lanes"), record.get("cold_lanes")) != (32, 32):
            problems.append("seeded sweep is not 32,768 sweeps with 32 seeded and 32 cold lanes")
        if problems:
            invalid.append({"path": str(path), "problems": problems})
        seeded[(record["cell"], record["seed_source"])] += 1

    campaign_best: dict[tuple[str, str, int, str], float] = {}
    for path in (round2 / "cpu-campaign" / "runs").glob("*/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] == "completed":
            campaign_best[(record["cell"], record["kernel"], record["sweeps"], record["nonce"])] = float(
                record["best_energy"]
            )

    portfolio = json.loads((tree / "portfolio-deadline" / "results.json").read_text(encoding="utf-8"))
    return {
        "schema": "round2-cpu-tree-validation-v1",
        "frozen_manifest_sha256": runner.sha256_file(round2 / "cpu-campaign" / "manifest.json"),
        "frozen_binary_sha256": binary,
        "bundle_offsets": sorted(offsets),
        "original_energy_tolerance": validation.ENERGY_TOLERANCE,
        "campaign_records": sum(counts.values()),
        "seeded_records": sum(seeded.values()),
        "invalid_records": invalid,
        "campaign_counts": [
            {"cell": cell, "kernel": kernel, "sweeps": sweeps, "status": status, "count": count}
            for (cell, kernel, sweeps, status), count in sorted(counts.items())
        ],
        "seeded_counts": [
            {"cell": cell, "seed_source": source, "count": count}
            for (cell, source), count in sorted(seeded.items())
        ],
        "seed_restriction": (
            "QPU seed lanes come from round1/<cell>/qpu-80 captures, which are also the frozen "
            "calibration target sources. Seeded-versus-cold comparisons stay valid; QPU-seeded "
            "results cannot support target-based win claims."
        ),
        "portfolio_deadline": {
            "arm": "dwave-neal historical settings, 500 reads and 500 sweeps",
            "rescored": False,
            "reason": "the tree saved objectives only, with no samples",
            "records": portfolio["records"],
        },
        "cross_runner_agreement": _agreement(tree_best, campaign_best),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2", type=Path, required=True, help="Round 2 data root")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args(argv)
    summary = validate(args.round2)
    args.out.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_write_json(args.out / "cpu-tree-validation.json", summary)
    print(json.dumps({
        "valid": not summary["invalid_records"],
        "campaign_records": summary["campaign_records"],
        "seeded_records": summary["seeded_records"],
        "invalid_records": len(summary["invalid_records"]),
        "bundle_offsets": summary["bundle_offsets"],
    }))
    return 0 if not summary["invalid_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
