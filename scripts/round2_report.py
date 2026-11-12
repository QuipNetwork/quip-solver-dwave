"""Build the Round 2 regime report from validated evidence.

Reads only validated inputs: the frozen CPU campaign, the quiet-host timing
repeat, the quip-miner-cpu seeded sweeps, the reconciled physical QPU pilot,
and the portfolio deadline arms. Refuses to run when any validation summary
reports a problem. Writes ``REPORT.md`` and SVG figures to ``--out``.

Quality comes from the campaign records. Timing comes only from sequential,
uncontaminated records. The routed ``cpu-msa`` arm is derived from measured
unit and f64 rows, never measured on its own.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quip_solver_dwave import round2_io
from quip_solver_dwave import round2_metrics as metrics

REGIMES = ("native-pm1", "native-125", "diamond-pm1", "cubic-dimer-pm1")
PILOT_REGIMES = ("diamond-pm1",)
KERNELS = ("cpu-sa", "cpu-msa-unit", "cpu-msa-f64")
ROUTED = "cpu-msa"
DEPTHS = (512, 2048, 8192, 32768, 131072)
DEEPEST = DEPTHS[-1]
MODELS_PER_REGIME = 100
PORTFOLIO_REFERENCE = "dwave-neal-reference"
#: Enough digits to separate depths whose energies differ in the fifth figure.
ENERGY_DIGITS = 7

#: What each regime's tables establish, what stays open, and the next control. Written after reading the tables.
NEXT_TESTS: Dict[str, Tuple[str, str, str]] = {
    "native-pm1": (
        "`cpu-msa-unit` reaches the frozen target on 100 of 100 models at 32,768 sweeps, "
        "with a median clean sampling time under 5 s.",
        "The only QPU evidence is the Round 1 capture set that also supplied the targets. "
        "No independent physical-range QPU capture exists for this regime.",
        "Capture 12 models at the physical range with the pilot protocol. "
        "Compare them with `cpu-msa-unit` at 8,192 and 32,768 sweeps.",
    ),
    "native-125": (
        "The unit kernel refuses all 100 models because the couplings are not unit values. "
        "`cpu-msa-f64` reaches the frozen target on 28 of 100 models at 8,192 sweeps and 93 of 100 at 32,768.",
        "This regime needs the most CPU time to match the historical QPU result. "
        "No independent physical-range QPU capture exists to show how often a fresh QPU run matches that result.",
        "Capture 12 models at the physical range with the pilot protocol. "
        "Count how many fresh QPU captures reach the frozen target.",
    ),
    "diamond-pm1": (
        "The QPU is strictly worse than every CPU arm at 131,072 sweeps on 12 of 12 models, at every scale and anneal. "
        "`cpu-msa-unit` at 2,048 sweeps is within 0.5% of the QPU on all 12 models and faster than the QPU cached-job time.",
        "The median gap is smallest at 400 µs, near 0.2%. Longer anneals were not tested.",
        "Sweep the anneal time to 1,000 and 2,000 µs at 50% scale on the same 12 models. "
        "Compare with `cpu-msa-unit` at 2,048 sweeps.",
    ),
    "cubic-dimer-pm1": (
        "At 131,072 sweeps, `cpu-sa` reaches the frozen target on 68 of 100 models. "
        "Both `cpu-msa` kernels reach it on 53 of 100.",
        "The CPU depth that reaches the target on every model is not known. "
        "No independent physical-range QPU capture exists for this regime.",
        "Capture 12 models at the physical range with the pilot protocol. "
        "Run `cpu-msa-unit` at 524,288 sweeps on the models that miss the target.",
    ),
}

Record = Dict[str, Any]


# ---------------------------------------------------------------- formatting


def fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}g}"


def outcome_text(counts: Mapping[str, int], first: str = "QPU", second: str = "CPU") -> str:
    total = sum(counts.values())
    return f"{first} {counts['qpu']} / tie {counts['tie']} / {second} {counts['cpu']} (of {total})"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("--" for _ in header) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def relative_gap(qpu: float, cpu: float) -> float:
    """Signed gap on the same scale as :func:`metrics.quality_outcome`; negative favors QPU."""
    return (qpu - cpu) / max(abs(qpu), abs(cpu), 1e-12)


# ------------------------------------------------------------------- loading


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_validated(round2: Path) -> Dict[str, Any]:
    """Return the validation summaries, or raise when any of them reports a problem."""
    campaign = _load_json(round2 / "cpu-campaign" / "validation.json")
    tree = _load_json(round2 / "combined" / "cpu-tree-validation.json")
    pilot = _load_json(round2 / "physical-pilot-analysis" / "reconciliation.json")
    problems = []
    if campaign["invalid_records"] or campaign["orphan_run_dirs"]:
        problems.append("cpu-campaign validation reports invalid records")
    if tree["invalid_records"]:
        problems.append("cpu tree validation reports invalid records")
    if not pilot.get("all_checks_passed"):
        problems.append("physical pilot reconciliation did not pass")
    if problems:
        raise SystemExit("; ".join(problems))
    return {"campaign": campaign, "tree": tree, "pilot": pilot}


def _run_records(root: Path) -> List[Record]:
    keep = (
        "cell", "nonce", "model_hash", "kernel", "sweeps", "status", "best_energy", "sampling_s",
        "caller_end_to_end_s", "host", "comparison", "unsupported_reason", "reads",
    )
    records = []
    for path in sorted((root / "runs").glob("*/result.json")):
        record = _load_json(path)
        slim = {name: record.get(name) for name in keep}
        slim["samples"] = path.with_name("samples.npz")
        records.append(slim)
    return records


def derive_routed(records: Sequence[Record]) -> List[Record]:
    """The routed ``cpu-msa`` arm: the unit row where it completed, else the f64 row."""
    by_key: Dict[Tuple[str, str, int], Dict[str, Record]] = defaultdict(dict)
    for record in records:
        if record["kernel"] in ("cpu-msa-unit", "cpu-msa-f64"):
            by_key[(record["cell"], record["nonce"], record["sweeps"])][record["kernel"]] = record
    routed = []
    for sources in by_key.values():
        unit, f64 = sources.get("cpu-msa-unit"), sources.get("cpu-msa-f64")
        source = unit if unit is not None and unit["status"] == "completed" else f64
        if source is None:
            continue
        routed.append({**source, "kernel": ROUTED, "routed_kernel": source["kernel"]})
    return routed


def _capture(round2: Path, row: Mapping[str, Any]) -> Record:
    scale = f"scale-{round(row['scale'] * 100):03d}"
    path = round2 / "qpu-physical-pilot" / row["cell"] / scale / f"qpu-{row['anneal_us']}" / f"{row['nonce']}.npz"
    with np.load(path) as saved:
        spins = np.asarray(saved["spins"])
        meta = {
            name: saved[name].item() if saved[name].shape == () else None
            for name in (
                "end_to_end_s", "device_access_s", "requested_scale", "submitted_scale", "ceiling",
                "limiting_constraint", "chain_strength", "broken_chain_share",
            )
            if name in saved
        }
    constraint = meta.get("limiting_constraint")
    if isinstance(constraint, str):
        meta["limiting_constraint"] = json.loads(constraint)
    return {**row, **meta, "spins": spins}


def load_evidence(round2: Path) -> Dict[str, Any]:
    validation = require_validated(round2)
    manifest = _load_json(round2 / "cpu-campaign" / "manifest.json")
    campaign = _run_records(round2 / "cpu-campaign")
    clean = _run_records(round2 / "cpu-timing-clean") if (round2 / "cpu-timing-clean" / "runs").exists() else []
    bundles = Path(manifest["bundles_root"])
    models: Dict[str, List[Record]] = defaultdict(list)
    for cell, nonce in sorted({(job["cell"], job["nonce"]) for job in manifest["jobs"]}):
        bundle, _ = round2_io.read_bundle(bundles / cell / nonce)
        models[cell].append({
            "nonce": nonce, "hash": bundle["hash"], "provenance": bundle.get("provenance"),
            "n_spins": bundle["n_spins"], "n_edges": bundle["n_edges"],
        })
    pilot = [_capture(round2, row) for row in _load_json(round2 / "physical-pilot-analysis" / "observations.json")]
    seeded = [_load_json(path) for path in sorted((round2 / "cpu" / "seeded-sweep-v2").glob("*/*.json"))]
    portfolio = []
    for path in sorted((round2 / "portfolio-deadline").glob("*/*/result.json")):
        record = _load_json(path)
        diagnostics = path.with_name("diagnostics.json")
        record["diagnostics_records"] = (
            _load_json(diagnostics)["diagnostics"]["records"] if diagnostics.exists() else []
        )
        portfolio.append(record)
    fixtures = {
        _load_json(path)["hash"]: {"fixture": f"{path.parents[1].name}/{path.parent.name}", **{
            key: _load_json(path)[key] for key in ("provenance", "n_assets", "cardinality_k")
        }}
        for path in sorted((round2 / "bundles-portfolio-fixtures").glob("*/*/manifest.json"))
    }
    return {
        "validation": validation, "manifest": manifest, "campaign": campaign + derive_routed(campaign),
        "clean": clean + derive_routed(clean), "models": models, "pilot": pilot, "seeded": seeded,
        "portfolio": portfolio, "fixtures": fixtures, "families": load_families(round2),
    }


#: Family model sets: (label, families dir, CPU arm dir, fixture dir).
FAMILY_SETS = (
    ("small", "portfolio-families", "portfolio-family-cpu", "portfolio-family-fixtures"),
    ("large", "portfolio-families-large", "portfolio-family-cpu-large", "portfolio-family-fixtures-large"),
)
FAMILY_VARIANTS = ("real", "iid", "shuffle", "rotate")
FAMILY_TUNED = ("neal-final-1x", "neal-final-3x", "neal-final-10x", "neal-final-30x")
FAMILY_QPU = ("qpu-20us", "qpu-80us", "qpu-400us")
#: Relative tolerance for "at the best objective", the report's numeric tolerance.


def _family_set(round2: Path, families: str, cpu: str, fixtures: str) -> Dict[str, Any]:
    """Every arm's final objective and timing per model, and the proven optimum where the oracle has one."""
    names = {r["model_hash"]: r["name"] for r in _load_json(round2 / fixtures / "fixtures.json")["fixtures"]}
    arms: Dict[str, Dict[str, Record]] = defaultdict(dict)
    for path in sorted((round2 / cpu).glob("*/*/result.json")):
        result = _load_json(path)
        arms[names[result["model_hash"]]][result["requested_kernel"]] = {
            "objective": result["objective"], "sampling_s": result.get("sampling_s"), "finish_s": result.get("finish_s"),
        }
    for model in _load_json(round2 / families / "diversity-controls.json")["models"]:
        arms[model["name"]].update(model["arms"])
    oracle = {m["name"]: m["best_objective"] for m in _load_json(round2 / families / "exact-oracle.json")["models"]}
    return {"arms": arms, "oracle": oracle}


def load_families(round2: Path) -> Dict[str, Any]:
    """The portfolio family sets and the QPU capture, with the capture checked against its manifest."""
    sets = {label: _family_set(round2, *dirs) for label, *dirs in FAMILY_SETS}
    capture = round2 / "portfolio-qpu"
    manifest = _load_json(capture / "manifest.json")
    jobs = [_load_json(path) for path in sorted((capture / "round" / "jobs").glob("*.json"))]
    expected = {job["capture_key"] for job in manifest["jobs"]}
    if {job["capture_key"] for job in jobs} != expected:
        raise SystemExit("the portfolio QPU capture does not hold exactly the manifest's jobs")
    spend = [json.loads(line) for line in (capture / "round" / "spend.jsonl").read_text(encoding="utf-8").splitlines()]
    analysis = _load_json(capture / "round" / "analysis.json")
    scored = {(j["name"], j["anneal_us"]): j for j in analysis["jobs"]}
    for job in jobs:
        name, anneal = job["job"]["name"], job["job"]["anneal_us"]
        sets["large"]["arms"][name][f"qpu-{anneal}us"] = {
            **scored[(name, anneal)], "wall_s": job["wall_s"],
        }
    return {
        **sets, "qpu_models": sorted({job["job"]["name"] for job in jobs}), "jobs": [scored[k] for k in sorted(scored)],
        "charged_us": sum(e["us"] for e in spend if e["event"] == "charge"),
        "failed": len(_load_json(failed)) if (failed := capture / "round" / "failed_captures.json").exists() else 0,
        "manifest": manifest,
        "chain_strength": _load_json(capture / "round" / "chain-strengths.json")["chain_strength"],
    }


# -------------------------------------------------------------------- tables


def arm(records: Sequence[Record], cell: str, kernel: str, sweeps: int) -> List[Record]:
    return [r for r in records if r["cell"] == cell and r["kernel"] == kernel and r["sweeps"] == sweeps]


def clean_sampling(records: Sequence[Record]) -> List[float]:
    """Sampling times from sequential, uncontaminated, completed records only."""
    return [
        float(r["sampling_s"]) for r in records
        if r["status"] == "completed" and r.get("sampling_s") is not None
        and r.get("caller_end_to_end_s") is not None
        and not ((r.get("host") or {}).get("contamination") or {}).get("contaminated", True)
    ]


def model_identity_table(models: Sequence[Record]) -> List[str]:
    provenance = Counter(str(m["provenance"]) for m in models)
    spins = [m["n_spins"] for m in models]
    edges = [m["n_edges"] for m in models]
    return table(
        ("Models", "Distinct model hashes", "Provenance", "Spins", "Edges"),
        [(
            len(models), len({m["hash"] for m in models}),
            ", ".join(f"{name} {count}" for name, count in sorted(provenance.items())),
            f"{min(spins)} to {max(spins)}" if spins else "n/a",
            f"{min(edges)} to {max(edges)}" if edges else "n/a",
        )],
    )


def depth_quality_time_table(cell: str, campaign: Sequence[Record], clean: Sequence[Record]) -> List[str]:
    rows = []
    for kernel in (*KERNELS, ROUTED):
        for sweeps in DEPTHS:
            records = arm(campaign, cell, kernel, sweeps)
            quality = metrics.summarize_arm(records, expected=MODELS_PER_REGIME)
            timing = metrics.summarize_arm(records + arm(clean, cell, kernel, sweeps))
            compared = [r for r in records if (r.get("comparison") or {}).get("available")]
            reached = sum(bool(r["comparison"]["target_reached"]) for r in compared)
            clean_label = timing["timing_by_label"].get(metrics.CLEAN_SERIAL)
            loaded = timing["timing_by_label"].get(metrics.CONTAMINATED)
            sampling = clean_sampling(records + arm(clean, cell, kernel, sweeps))
            rows.append((
                f"`{kernel}`" + (" (derived)" if kernel == ROUTED else ""), sweeps, quality["observed"],
                quality["missing"], quality["completed"], quality["unsupported"], quality["failed"],
                quality["nonfinite"], fmt(quality["median_best_energy"], ENERGY_DIGITS),
                metrics.denominator_text(reached, len(compared)) if compared else "n/a",
                f"{fmt(float(np.median(sampling)))} ({len(sampling)})" if sampling else "n/a (0)",
                f"{fmt(clean_label['median_wall_s'])} ({clean_label['n']})" if clean_label else "n/a (0)",
                f"{loaded['n']}" if loaded else "0", quality["timing_withheld"],
            ))
    return table(
        (
            "Kernel", "Sweeps", "Observed", "Missing", "Completed", "Unsupported", "Failed", "Nonfinite",
            "Median best energy", "Reached frozen target", "Clean sampling s (n)", "Clean end-to-end s (n)",
            "Loaded-host timings", "Timing withheld",
        ),
        rows,
    )


def cpu_best(campaign: Sequence[Record]) -> Dict[Tuple[str, str, int, str], float]:
    return {
        (r["cell"], r["kernel"], r["sweeps"], r["nonce"]): float(r["best_energy"])
        for r in campaign
        if r["status"] == "completed" and r.get("best_energy") is not None and math.isfinite(r["best_energy"])
    }


def _pilot_groups(pilot: Sequence[Record], cell: str) -> Dict[Tuple[float, int], List[Record]]:
    groups: Dict[Tuple[float, int], List[Record]] = defaultdict(list)
    for capture in pilot:
        if capture["cell"] == cell:
            groups[(float(capture["scale"]), int(capture["anneal_us"]))].append(capture)
    return dict(sorted(groups.items()))


def qpu_quality_table(cell: str, pilot: Sequence[Record], best: Mapping[Tuple[str, str, int, str], float]) -> List[str]:
    rows = []
    for (scale, anneal), captures in _pilot_groups(pilot, cell).items():
        for kernel in (*KERNELS, ROUTED):
            pairs = [
                (float(c["best_energy"]), best[(cell, kernel, DEEPEST, c["nonce"])])
                for c in captures if (cell, kernel, DEEPEST, c["nonce"]) in best
            ]
            if not pairs:
                rows.append((f"{scale:.0%}", anneal, f"`{kernel}`", "no comparable CPU result", "n/a", "n/a", "n/a", "n/a"))
                continue
            columns = [metrics.quality_columns(q, c) for q, c in pairs]
            gaps = [relative_gap(q, c) for q, c in pairs]
            nonces = [c["nonce"] for c in captures if (cell, kernel, DEEPEST, c["nonce"]) in best]
            interval = metrics.bootstrap_paired_gap(gaps, nonces)
            descriptive = " (descriptive)" if interval["descriptive"] else ""
            rows.append((
                f"{scale:.0%}", anneal, f"`{kernel}`",
                *(outcome_text(metrics.tally_quality_outcomes([col[name] for col in columns]))
                  for name in ("strict", "numeric_tolerance", "material")),
                f"{fmt(interval['point'], 3)}",
                f"{fmt(interval['low'], 3)} to {fmt(interval['high'], 3)}, {interval['group_count']} models{descriptive}",
            ))
    if not rows:
        return ["No physical-range QPU captures exist for this regime."]
    return table(
        (
            "Scale", "Anneal µs", f"CPU arm at {DEEPEST} sweeps", "Strict", "Numeric tolerance", "0.5% material",
            "Median relative gap (negative favors QPU)", "Bootstrap 95% interval",
        ),
        rows,
    )


def joint_table(
    cell: str, pilot: Sequence[Record], best: Mapping[Tuple[str, str, int, str], float],
    campaign: Sequence[Record], clean: Sequence[Record],
) -> List[str]:
    """Material quality crossed with speed, at full physical scale, against every CPU depth."""
    rows = []
    for (scale, anneal), captures in _pilot_groups(pilot, cell).items():
        if scale != 1.0:
            continue
        for kernel in (*KERNELS, ROUTED):
            for sweeps in DEPTHS:
                sampling = clean_sampling(arm(campaign, cell, kernel, sweeps) + arm(clean, cell, kernel, sweeps))
                pairs = [
                    (c, best[(cell, kernel, sweeps, c["nonce"])])
                    for c in captures if (cell, kernel, sweeps, c["nonce"]) in best
                ]
                if not pairs:
                    continue
                if not sampling:
                    rows.append((anneal, f"`{kernel}`", sweeps, len(pairs), "n/a (0)", "", "", "", "timing unavailable"))
                    continue
                cpu_time = float(np.median(sampling))
                joint: Counter[str] = Counter()
                for capture, energy in pairs:
                    quality = metrics.quality_outcome(float(capture["best_energy"]), energy, metrics.MATERIALITY_TOLERANCE)
                    faster = capture.get("end_to_end_s") is not None and float(capture["end_to_end_s"]) < cpu_time
                    joint[f"{quality}-{'qpu' if faster else 'cpu'}"] += 1
                rows.append((
                    anneal, f"`{kernel}`", sweeps, len(pairs), f"{fmt(cpu_time)} ({len(sampling)})",
                    joint["qpu-qpu"], joint["tie-qpu"], joint["qpu-cpu"] + joint["tie-cpu"],
                    joint["cpu-qpu"] + joint["cpu-cpu"],
                ))
    if not rows:
        return ["No full-scale QPU captures exist for this regime."]
    median_qpu = float(np.median([float(c["end_to_end_s"]) for c in pilot if c["cell"] == cell and c["scale"] == 1.0]))
    return [
        f"QPU cached-job time (job setup, sampler call, decode and score) has median {fmt(median_qpu)} s at 100% scale. "
        "CPU time is the arm's median clean sampling time, the smallest CPU boundary, so a QPU speed result here is conservative.",
        "",
        *table(
            (
                "Anneal µs", "CPU arm", "Sweeps", "Paired models", "CPU clean sampling s (n)",
                "QPU materially better and faster", "Within 0.5% and QPU faster (runtime win)", "QPU not materially worse but slower",
                "CPU materially better",
            ),
            rows,
        ),
    ]


def physical_scale_table(cell: str, pilot: Sequence[Record], best: Mapping[Tuple[str, str, int, str], float]) -> List[str]:
    rows = []
    for (scale, anneal), captures in _pilot_groups(pilot, cell).items():
        ceilings = [c["ceiling"] for c in captures if c.get("ceiling") is not None]
        submitted = [c["submitted_scale"] for c in captures if c.get("submitted_scale") is not None]
        broken = [c["broken_chain_share"] for c in captures if c.get("broken_chain_share") is not None]
        limits = Counter(
            f"{c['limiting_constraint']['kind']} {c['limiting_constraint']['bound']}"
            for c in captures if isinstance(c.get("limiting_constraint"), dict)
        )
        gaps = []
        for capture in captures:
            candidates = [best[(cell, k, DEEPEST, capture["nonce"])] for k in KERNELS if (cell, k, DEEPEST, capture["nonce"]) in best]
            if candidates:
                gaps.append(relative_gap(float(capture["best_energy"]), min(candidates)))
        rows.append((
            f"{scale:.0%}", anneal, len(captures),
            f"{fmt(min(ceilings))} to {fmt(max(ceilings))}" if ceilings else "unavailable",
            f"{fmt(float(np.median(submitted)))}" if submitted else "unavailable",
            ", ".join(f"{name} {count}" for name, count in sorted(limits.items())) or "unavailable",
            fmt(float(np.median(broken)), 3) if broken else "unavailable",
            fmt(float(np.median([c["best_energy"] for c in captures])), ENERGY_DIGITS),
            f"{fmt(float(np.median(gaps)), 3)} ({len(gaps)})" if gaps else "n/a",
        ))
    if not rows:
        return ["No physical-range QPU captures exist for this regime."]
    return table(
        (
            "Scale", "Anneal µs", "Captures", "Legal ceiling", "Median submitted scale", "Limiting constraints",
            "Median broken-chain share", "Median QPU best energy", f"Median gap to best CPU at {DEEPEST} (n)",
        ),
        rows,
    )


def seeds_table(cell: str, seeded: Sequence[Record]) -> List[str]:
    rows = []
    for source in ("qpu", "cpu-lite"):
        records = [r for r in seeded if r["cell"] == cell and r["seed_source"] == source]
        usable = [
            r for r in records
            if r.get("exit_ok") and all(
                r.get(name) is not None and math.isfinite(r[name]) for name in ("best_seeded_energy", "best_cold_energy")
            )
        ]
        if not usable:
            rows.append((source, len(records), len(records), "n/a", "n/a", "n/a", "n/a", "n/a"))
            continue
        columns = [metrics.quality_columns(r["best_seeded_energy"], r["best_cold_energy"]) for r in usable]
        gaps = [relative_gap(r["best_seeded_energy"], r["best_cold_energy"]) for r in usable]
        rows.append((
            source, len(records), len(records) - len(usable),
            *(outcome_text(metrics.tally_quality_outcomes([col[name] for col in columns]), "seeded", "cold")
              for name in ("strict", "numeric_tolerance", "material")),
            fmt(float(np.median(gaps)), 3), sum(int(r.get("duplicate_seed_lanes") or 0) for r in usable),
        ))
    return table(
        (
            "Seed source", "Models", "Excluded", "Strict", "Numeric tolerance", "0.5% material",
            "Median relative gap (negative favors seeded)", "Duplicate seed lanes",
        ),
        rows,
    )


def portfolio_table(portfolio: Sequence[Record], fixtures: Mapping[str, Mapping[str, Any]]) -> List[str]:
    reference = {
        r["model_hash"]: r for r in portfolio
        if r["requested_kernel"] == PORTFOLIO_REFERENCE and r["status"] == "completed"
    }
    rows = []
    for record in sorted(portfolio, key=lambda r: (fixtures[r["model_hash"]]["fixture"], r["requested_kernel"])):
        fixture = fixtures[record["model_hash"]]
        diagnostics = record["diagnostics_records"]
        repair = Counter(
            f"{d['raw_cardinality']} to {d['repaired_cardinality']}" for d in diagnostics
        )
        base = reference.get(record["model_hash"])
        if record is base:
            versus, speed = "reference", "reference"
        elif record["status"] != "completed" or record.get("objective") is None or base is None:
            versus, speed = "n/a", "n/a"
        else:
            columns = metrics.quality_columns(float(record["objective"]), float(base["objective"]))
            versus = f"strict {columns['strict']}, material {columns['material']}"
            speed = f"{record['caller_end_to_end_s'] / base['caller_end_to_end_s']:.2f}x"
        rows.append((
            fixture["fixture"], f"{fixture['provenance']}, n={fixture['n_assets']}, k={fixture['cardinality_k']}",
            f"`{record['requested_kernel']}`", record["status"], record.get("deadline_status"),
            fmt(record.get("objective"), 6), "n/a" if record.get("feasible") is None else record["feasible"],
            ", ".join(f"{name} ({count} reads)" for name, count in sorted(repair.items())) or "n/a",
            fmt(record.get("caller_end_to_end_s"), 3), versus, speed,
        ))
    return table(
        (
            "Fixture", "Provenance", "Arm", "Status", "Deadline status", "Final objective", "Feasible",
            "Raw to repaired cardinality", "End-to-end s", f"Objective vs `{PORTFOLIO_REFERENCE}`",
            "Time vs reference (one run each)",
        ),
        rows,
    )


# ------------------------------------------------------------------- figures
# The SVG helpers follow quip-miner-cpu research/round2/round2_report.py.


_PALETTE = ("#217a69", "#b8792a", "#445a86", "#a33a3a", "#6b4c9a")


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif">'
        f'<rect width="100%" height="100%" fill="white"/>{body}</svg>'
    )


def _text_width_estimate(text: str, font_size: int = 12) -> float:
    return len(text) * font_size * 0.6


def _wrap_lines(text: str, max_chars: int) -> List[str]:
    lines: List[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _pad_range(low: float, high: float, fraction: float = 0.12) -> Tuple[float, float]:
    span = high - low
    pad = span * fraction if span > 0 else max(abs(high), 1.0) * fraction
    return low - pad, high + pad


def _axis_ticks(low: float, high: float, *, log: bool = False) -> List[float]:
    """Round tick positions inside [low, high]: powers of ten on a log axis, 1-2-5 steps otherwise."""
    if high <= low:
        return [low]
    if log:
        decades = [float(k) for k in range(math.ceil(low), math.floor(high) + 1)]
        if len(decades) >= 2:
            return decades
    raw_step = (high - low) / 5
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_step)
    first = math.ceil(low / step)
    return [k * step for k in range(first, math.floor(high / step) + 1)]


def _overlaps(box: Tuple[float, float, float, float], placed: Sequence[Tuple[float, float, float, float]]) -> bool:
    left, top, right, bottom = box
    return any(left < r2 and l2 < right and top < b2 and t2 < bottom for l2, t2, r2, b2 in placed)


def scatter_figure(
    title: str, caption: str, series: Mapping[str, Sequence[Tuple[float, float, str]]],
    *, x_label: str, y_label: str, count_unit: str, log_x: bool = False, incomplete: Sequence[str] = (),
    zero_line: bool = False,
) -> str:
    """A scatter panel. Point labels that would overlap are left to the point's hover title."""
    width, height, margin = 760, 500, 95
    max_chars = int((width - 2 * margin) / (12 * 0.6))
    body = [f'<text x="{margin}" y="24" font-size="16" font-weight="bold">{html.escape(title)}</text>']
    y_cursor = 42
    for line in _wrap_lines(caption, max_chars):
        body.append(f'<text x="{margin}" y="{y_cursor}" font-size="12">{html.escape(line)}</text>')
        y_cursor += 15
    if incomplete:
        for line in _wrap_lines("Incomplete: " + ", ".join(incomplete), max_chars):
            body.append(f'<text x="{margin}" y="{y_cursor}" font-size="12" fill="#a33a3a">{html.escape(line)}</text>')
            y_cursor += 15
    points = [(name, x, y, label) for name, values in series.items() for x, y, label in values]
    if not points:
        body.append(f'<text x="{margin}" y="{height / 2:.0f}" font-size="14">No data available.</text>')
        return _svg_document(width, height, "".join(body))
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(series)}
    legend_x, legend_y = margin, y_cursor + 4
    for name, color in colors.items():
        label = f"{name} ({len(series[name])} {count_unit})"
        if legend_x > margin and legend_x + 16 + _text_width_estimate(label) > width - 10:
            legend_x, legend_y = margin, legend_y + 18
        body.append(f'<rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        body.append(f'<text x="{legend_x + 16}" y="{legend_y + 11}" font-size="12">{html.escape(label)}</text>')
        legend_x += 36 + int(_text_width_estimate(label))
    plot_top = legend_y + 28
    xs = [math.log10(max(x, 1e-9)) if log_x else x for _, x, _, _ in points]
    ys = [y for _, _, y, _ in points] + ([0.0] if zero_line else [])
    x_min, x_max = _pad_range(min(xs), max(xs))
    y_min, y_max = _pad_range(min(ys), max(ys))
    plot_height = height - margin - plot_top

    def to_x(value: float) -> float:
        raw = math.log10(max(value, 1e-9)) if log_x else value
        return margin + (raw - x_min) / (x_max - x_min) * (width - 2 * margin)

    def to_y(value: float) -> float:
        return height - margin - (value - y_min) / (y_max - y_min) * plot_height

    if zero_line:
        body.append(
            f'<line x1="{margin}" y1="{to_y(0.0):.1f}" x2="{width - margin}" y2="{to_y(0.0):.1f}" '
            'stroke="#999999" stroke-dasharray="4 3"/>'
        )
    placed = [(to_x(x) - 5, to_y(y) - 5, to_x(x) + 5, to_y(y) + 5) for _, x, y, _ in points]
    for name, x, y, label in points:
        cx, cy = to_x(x), to_y(y)
        body.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{colors[name]}" fill-opacity="0.8">'
            f'<title>{html.escape(f"{name}: {label}, x={x:.6g}, y={y:.6g}")}</title></circle>'
        )
        if not label:
            continue
        label_width = _text_width_estimate(label, 9)
        for left, baseline in ((cx + 7, cy - 6), (cx + 7, cy + 13), (cx - 7 - label_width, cy - 6), (cx - 7 - label_width, cy + 13)):
            box = (left, baseline - 9, left + label_width, baseline + 2)
            if box[0] >= margin and box[2] <= width - 5 and not _overlaps(box, placed):
                placed.append(box)
                body.append(
                    f'<text x="{left:.1f}" y="{baseline:.1f}" font-size="9" fill="#333333">{html.escape(label)}</text>'
                )
                break
    for tick in _axis_ticks(x_min, x_max, log=log_x):
        value = 10 ** tick if log_x else tick
        position = margin + (tick - x_min) / (x_max - x_min) * (width - 2 * margin)
        body.append(
            f'<text x="{position:.1f}" y="{height - margin + 16}" font-size="10" text-anchor="middle">{value:g}</text>'
        )
    for tick in _axis_ticks(y_min, y_max):
        body.append(
            f'<text x="{margin - 6}" y="{to_y(tick) + 3:.1f}" font-size="10" text-anchor="end">{tick:g}</text>'
        )
    body.append(
        f'<text x="{width / 2:.0f}" y="{height - margin + 34}" font-size="11" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    middle = (plot_top + height - margin) / 2
    body.append(
        f'<text x="{margin - 70}" y="{middle:.0f}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 {margin - 70} {middle:.0f})">{html.escape(y_label)}</text>'
    )
    body.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
    body.append(f'<line x1="{margin}" y1="{plot_top}" x2="{margin}" y2="{height - margin}" stroke="black"/>')
    return _svg_document(width, height, "".join(body))


def quality_time_figure(cell: str, campaign: Sequence[Record], clean: Sequence[Record], binary: str) -> str:
    series: Dict[str, List[Tuple[float, float, str]]] = {}
    incomplete = []
    for kernel in KERNELS:
        gaps = []
        for sweeps in DEPTHS:
            quality = metrics.summarize_arm(arm(campaign, cell, kernel, sweeps))
            sampling = clean_sampling(arm(campaign, cell, kernel, sweeps) + arm(clean, cell, kernel, sweeps))
            if quality["median_best_energy"] is None or not sampling:
                reason = "unsupported" if quality["unsupported"] else "no clean timing"
                gaps.append((sweeps, reason))
                continue
            series.setdefault(kernel, []).append((
                float(np.median(sampling)), quality["median_best_energy"], f"{sweeps:,}",
            ))
        if len(gaps) == len(DEPTHS) and len({reason for _, reason in gaps}) == 1:
            incomplete.append(f"{kernel} at every depth ({gaps[0][1]})")
        else:
            incomplete.extend(f"{kernel} at {sweeps} ({reason})" for sweeps, reason in gaps)
    caption = (
        f"Original-unit energy, lower is better, median over completed models. Time is median clean sampling time "
        f"from sequential uncontaminated runs, frozen model to scored answer. quip_msa binary {binary[:12]}. "
        "Labels give sweeps. The depth table gives the sample counts."
    )
    return scatter_figure(
        f"{cell}: quality and time by CPU kernel and depth", caption, series,
        x_label="Median clean sampling time (s), log scale", y_label="Median best energy (original units)",
        count_unit="depths", log_x=True, incomplete=incomplete,
    )


def physical_scale_figure(cell: str, pilot: Sequence[Record], best: Mapping[Tuple[str, str, int, str], float]) -> str:
    series: Dict[str, List[Tuple[float, float, str]]] = {}
    missing = 0
    for (scale, anneal), captures in _pilot_groups(pilot, cell).items():
        offset = -0.02 if anneal == 80 else 0.02
        for capture in captures:
            candidates = [best[(cell, k, DEEPEST, capture["nonce"])] for k in KERNELS if (cell, k, DEEPEST, capture["nonce"]) in best]
            if not candidates:
                missing += 1
                continue
            series.setdefault(f"{anneal} µs anneal", []).append((
                scale + offset, relative_gap(float(capture["best_energy"]), min(candidates)), "",
            ))
    caption = (
        f"Each point is one model's QPU capture at a fraction of its legal physical ceiling. The y value is the signed "
        f"relative gap to the best CPU kernel at {DEEPEST} sweeps on the same model. Below zero favors the QPU. "
        "Anneal times are offset horizontally for legibility. 12-model pilot: descriptive only, no regime verdict."
    )
    return scatter_figure(
        f"{cell}: physical scale and QPU quality", caption, series,
        x_label="Fraction of legal physical ceiling", y_label="Relative gap, QPU minus best CPU",
        count_unit="captures", incomplete=[f"{missing} captures without a CPU pair"] if missing else (), zero_line=True,
    )


def portfolio_figure(portfolio: Sequence[Record], fixtures: Mapping[str, Mapping[str, Any]]) -> str:
    series: Dict[str, List[Tuple[float, float, str]]] = {}
    names = sorted({fixtures[r["model_hash"]]["fixture"] for r in portfolio})
    completed = [r for r in portfolio if r["status"] == "completed" and r.get("objective") is not None]
    arms = sorted({r["requested_kernel"] for r in completed})
    for record in completed:
        position = names.index(fixtures[record["model_hash"]]["fixture"]) + 1
        offset = (arms.index(record["requested_kernel"]) - (len(arms) - 1) / 2) * 0.08
        series.setdefault(record["requested_kernel"], []).append((
            position + offset, float(record["objective"]), "",
        ))
    objectives: Dict[str, set] = defaultdict(set)
    for record in completed:
        objectives[record["model_hash"]].add(float(record["objective"]))
    tied = bool(objectives) and all(len(values) == 1 for values in objectives.values())
    raw = sorted({d["raw_cardinality"] for r in portfolio for d in r["diagnostics_records"]})
    provenance = sorted({str(fixtures[r["model_hash"]]["provenance"]) for r in portfolio})
    caption = (
        "Final portfolio objective after repair and weighting, lower is better, one point per arm and fixture. "
        f"Fixture provenance: {', '.join(provenance)}. "
        + ", ".join(f"{i + 1} = {name}" for i, name in enumerate(names))
        + f". Raw answers selected {', '.join(str(n) for n in raw) or 'unknown'} assets before repair."
        + (" Every completed arm reaches the same objective on each fixture. Arms are offset horizontally." if tied else
           " Arms are offset horizontally.")
    )
    return scatter_figure(
        "Portfolio deadline arms: final objective", caption, series,
        x_label="Fixture number", y_label="Final objective", count_unit="fixtures",
    )


# ---------------------------------------------------------- portfolio families


def _family_reference(family: Mapping[str, Any], name: str, reference_arms: Sequence[str]) -> float:
    """The proven optimum, or else the best objective of the reference arms that ran on this model."""
    if name in family["oracle"]:
        return float(family["oracle"][name])
    arms = family["arms"][name]
    return min(float(arms[arm]["objective"]) for arm in reference_arms if arm in arms)


def _family_misses(value: float, reference: float, tolerance: float) -> bool:
    """True when the reference beats the value by more than ``tolerance``, by the report's pairwise rule."""
    return metrics.quality_outcome(value, reference, tolerance) == "cpu"


def _family_names(family: Mapping[str, Any], variant: str, names: Optional[Iterable[str]] = None) -> List[str]:
    pool = family["arms"] if names is None else names
    return sorted(name for name in pool if name.rsplit("-", 1)[1] == variant)


def family_quality_table(
    family: Mapping[str, Any], arms: Sequence[Tuple[str, str]], names: Optional[Iterable[str]] = None,
    *, reference_arms: Sequence[str], proven_only: bool = False,
) -> List[str]:
    """Per arm and variant: models at the best objective, and models within 0.5% of it."""
    chosen = list(names) if names is not None else None
    header = ["Method"]
    columns = []
    for variant in FAMILY_VARIANTS:
        models = _family_names(family, variant, chosen)
        if proven_only:
            models = [m for m in models if m in family["oracle"]]
        columns.append(models)
        header.append(f"{variant} ({len(models)})")
    rows = []
    for arm_name, label in arms:
        cells = [label]
        for models in columns:
            at = within = 0
            for name in models:
                reference = _family_reference(family, name, reference_arms)
                value = float(family["arms"][name][arm_name]["objective"])
                at += not _family_misses(value, reference, metrics.NUMERIC_TOLERANCE)
                within += not _family_misses(value, reference, metrics.MATERIALITY_TOLERANCE)
            cells.append(f"{at} / {within}")
        rows.append(cells)
    return table(header, rows)


def family_spread_table(
    family: Mapping[str, Any], names: Sequence[str], arms: Sequence[Tuple[str, str]], *, reference_arms: Sequence[str],
) -> List[str]:
    """Median of (arm - best) / (random - best), over models where random reads miss the best objective."""
    header = ["Method", *(f"{variant}" for variant in FAMILY_VARIANTS)]
    rows = []
    for arm_name, label in arms:
        cells = [label]
        for variant in FAMILY_VARIANTS:
            ratios = []
            for name in _family_names(family, variant, names):
                reference = _family_reference(family, name, reference_arms)
                random_objective = float(family["arms"][name]["random"]["objective"])
                if _family_misses(random_objective, reference, metrics.NUMERIC_TOLERANCE):
                    spread = random_objective - reference
                    ratios.append((float(family["arms"][name][arm_name]["objective"]) - reference) / spread)
            cells.append(f"{np.median(ratios):.2f} (n={len(ratios)})" if ratios else "n/a")
        rows.append(cells)
    return table(header, rows)


def family_capture_table(families: Mapping[str, Any]) -> List[str]:
    jobs = families["jobs"]
    broken = [j for j in jobs if j["broken_read_share"] > 0]
    strengths = sorted(families["chain_strength"].values())
    manifest = families["manifest"]
    return table(("Item", "Value"), [
        ("Jobs", f"{len(jobs)} of {len(manifest['jobs'])} in the manifest"),
        ("QPU access time charged", f"{families['charged_us'] / 1e6:.2f} s"),
        ("Failed or uncertain jobs", str(families["failed"])),
        ("Jobs with any broken chain", f"{len(broken)} of {len(jobs)}, highest broken-read share "
         f"{max((j['broken_read_share'] for j in jobs), default=0.0):.1%}"),
        ("Chain strength, recomputed", f"median {np.median(strengths):.2e}, range {strengths[0]:.2e} to {strengths[-1]:.2e}"),
    ])


def family_runtime_table(family: Mapping[str, Any], names: Sequence[str], arms: Sequence[Tuple[str, str]]) -> List[str]:
    """Median sampling and finish time, and distinct baskets, over the QPU models."""
    rows = []
    for arm_name, label in arms:
        records = [family["arms"][name][arm_name] for name in names]
        if arm_name.startswith("qpu-"):
            sampling = (f"{np.median([r['qpu_access_time_us'] for r in records]) / 1e3:.0f} ms billed, "
                        f"{np.median([r['wall_s'] for r in records]):.2f} s round trip")
            finish = "not timed"
        else:
            sampling = f"{np.median([r['sampling_s'] for r in records]) * 1e3:.0f} ms"
            finish_s = [r["finish_s"] for r in records]
            finish = f"{np.median(finish_s) * 1e3:.0f} / {max(finish_s) * 1e3:.0f} ms"
        distinct = [r["distinct_supports"] for r in records if r.get("distinct_supports") is not None]
        rows.append((label, sampling, finish, f"{np.median(distinct):.0f}" if distinct else "n/a"))
    return table(("Method", "Sampling, median", "Finish, median / worst", "Distinct baskets after the finish, median"), rows)


FAMILY_CPU_ARMS = (
    (PORTFOLIO_REFERENCE, "production neal"), ("cpu-msa", "`cpu-msa`"), ("random", "random reads"),
    *((arm, f"tuned neal {arm.split('-')[-1].replace('x', '×')}") for arm in FAMILY_TUNED),
)
#: What the family tables establish, what stays open, and the next control. Written after reading the tables.
FAMILY_CONCLUSIONS = (
    ("Established",
     "With the fixed clique, 2 of 144 QPU jobs had any broken chain. On the 48 QPU models, the QPU reaches the best "
     "objective more often than production neal on the `real`, `iid`, and `shuffle` variants, and as often on `rotate`. "
     "Tuned neal 30× reaches it on 10 to 12 of 12 models per variant, more often than any QPU anneal time, "
     "in about 0.33 s on one core."),
    ("Unresolved",
     "No variant shows a QPU advantage that depends on real market structure, but 12 models per variant cannot detect "
     "a small one. For 44 of the 48 QPU models the reference is the best objective found, not a proven optimum. "
     "The QPU finish time was not measured."),
    ("Next control",
     "The ranking of the QPU and tuned SA needs no further control. If a real-data effect needs a formal test, "
     "capture more draws of the `real` and `rotate` variants, and add family B."),
)
FAMILY_QPU_ARMS = tuple((arm, f"QPU {arm[4:-2]} µs") for arm in FAMILY_QPU)
#: Where no proven optimum exists, the reference is the best objective of these arms: every arm the tables show.
FAMILY_REFERENCE_ARMS = tuple(arm for arm, _ in (*FAMILY_CPU_ARMS, *FAMILY_QPU_ARMS))


def families_section(families: Mapping[str, Any]) -> List[str]:
    small, large, qpu_models = families["small"], families["large"], families["qpu_models"]
    compared = (*FAMILY_CPU_ARMS[:1], FAMILY_CPU_ARMS[2], *FAMILY_CPU_ARMS[5:], *FAMILY_QPU_ARMS)
    return [
        "## Portfolio families", "",
        "This section replaces the `clique-portfolio` regime, whose QPU reads had broken chains. "
        "Every family model is a complete graph on N selection variables, embedded with production's fixed Zephyr clique. "
        "Family A (`real`) resamples real hourly returns in 24-hour blocks. "
        "Family C removes one kind of structure: `iid`, `shuffle`, or `rotate`. "
        "Every method returns 500 reads, and every read goes through the qpo finish. "
        "Tuned neal starts at a tenth of neal's default hot end and stops at m times the hot end.", "",
        "Each cell counts models at the best objective, then models within 0.5% of it. "
        "The best objective is the proven optimum where the exact oracle has one. Otherwise it is the best objective "
        "of the methods in these tables, including the QPU on the QPU models. "
        "Both counts use the report's pairwise rule: the gap divided by the larger magnitude of the two objectives.", "",
        "### Small set, proven optima", "",
        *family_quality_table(small, FAMILY_CPU_ARMS, reference_arms=FAMILY_REFERENCE_ARMS, proven_only=True), "",
        "### Large set, CPU methods", "",
        *family_quality_table(large, FAMILY_CPU_ARMS, reference_arms=FAMILY_REFERENCE_ARMS), "",
        "### QPU capture", "",
        *family_capture_table(families), "",
        "### QPU models", "",
        *family_quality_table(large, compared, qpu_models, reference_arms=FAMILY_REFERENCE_ARMS), "",
        "Median of (method − best) / (random − best), over the QPU models where random reads miss the best objective. "
        "0 is the best objective, and 1 is no better than random reads.", "",
        *family_spread_table(large, qpu_models, compared, reference_arms=FAMILY_REFERENCE_ARMS), "",
        "### Runtime on the QPU models", "",
        "CPU times are one pinned core per run. QPU times are the billed access time and the client round trip.", "",
        *family_runtime_table(large, qpu_models, compared), "",
        "### Next test", "",
        *(f"- **{label}.** {text}" for label, text in FAMILY_CONCLUSIONS), "",
    ]


# -------------------------------------------------------------------- report


def build_report(evidence: Mapping[str, Any]) -> Tuple[str, Dict[str, str]]:
    campaign, clean, pilot = evidence["campaign"], evidence["clean"], evidence["pilot"]
    manifest, validation = evidence["manifest"], evidence["validation"]
    binary = manifest["solver_identity"]["binary_sha256"]
    best = cpu_best(campaign)
    figures: Dict[str, str] = {}
    lines = [
        "# Round 2 regime report",
        "",
        "Status: draft generated from validated evidence. QPU means the D-Wave quantum processing unit. "
        "Pilot results are descriptive, and this report rejects no regime.",
        "",
        "## Evidence",
        "",
        *table(("Source", "Validation"), [
            ("CPU quality campaign, `cpu-campaign/`", f"{validation['campaign']['validated_records']} records matched to manifest "
             f"`{validation['campaign']['manifest_sha256'][:12]}`, {validation['campaign']['rescored_records']} rescored"),
            ("Quiet-host timing repeat, `cpu-timing-clean/`", f"{sum(1 for r in clean if r['kernel'] != ROUTED)} records, "
             "same frozen jobs as the campaign"),
            ("quip-miner-cpu seeded sweeps, `cpu/seeded-sweep-v2/`", f"{validation['tree']['seeded_records']} records rescored"),
            ("Physical-range QPU pilot", f"{validation['pilot']['captures']} captures, {validation['pilot']['reads']} reads, "
             f"manifest `{validation['pilot']['manifest_hash'][:12]}`, including the retired `clique-portfolio` captures"),
            ("Portfolio deadline arms, `portfolio-deadline/`", f"{len(evidence['portfolio'])} records"),
        ]),
        "",
        "## Method",
        "",
        f"- Every CPU arm uses 64 reads, the model's `quip_msa.default_beta_range`, and binary `{binary[:12]}`.",
        "- Energies are original-unit energies rescored from the frozen bundle. Every bundle has a zero offset.",
        "- Quality tables use every completed record. Timing uses only sequential runs without competing load.",
        "- The report derives the routed `cpu-msa` arm from the unit row where the unit kernel accepts the model, "
        "and from the f64 row elsewhere.",
        "- Strict, numeric-tolerance (1e-9), and 0.5% material quality are separate columns. "
        "Each tolerance applies to the gap divided by the larger magnitude of the two scores.",
        "- Bootstrap intervals resample models with 10,000 resamples and seed 20260922.",
        "- The frozen targets come from historical QPU captures. They measure CPU attainment only. "
        "They are not independent QPU evaluation evidence.",
        "- This report does not pool the 1,930 and 2,088 windows of the historical production analysis, which has its own page.",
        "",
    ]
    for cell in REGIMES:
        lines += [f"## `{cell}`", "", "### Model identity", "", *model_identity_table(evidence["models"][cell]), ""]
        lines += ["### Depth, quality, and time", "", *depth_quality_time_table(cell, campaign, clean), ""]
        figures[f"quality-time-{cell}.svg"] = quality_time_figure(cell, campaign, clean, binary)
        lines += [f"![Quality and time for {cell}](quality-time-{cell}.svg)", ""]
        if cell in PILOT_REGIMES:
            lines += ["### QPU win, tie, and loss", "", *qpu_quality_table(cell, pilot, best), ""]
            lines += ["### Joint quality and time at full physical scale", "", *joint_table(cell, pilot, best, campaign, clean), ""]
            lines += ["### Physical scale", "", *physical_scale_table(cell, pilot, best), ""]
            figures[f"physical-scale-{cell}.svg"] = physical_scale_figure(cell, pilot, best)
            lines += [f"![Physical scale for {cell}](physical-scale-{cell}.svg)", ""]
        lines += ["### Seeded and cold lanes at 32,768 sweeps", "", *seeds_table(cell, evidence["seeded"]), ""]
        lines += [
            "QPU seed lanes come from Round 1 80 µs captures, which also supplied the frozen targets. "
            "Seeded and cold lanes from the same run are a valid comparison. QPU-seeded results cannot support target-based wins.",
            "",
        ]
        established, unresolved, control = NEXT_TESTS[cell]
        lines += [
            "### Next test", "",
            f"- **Established.** {established}", f"- **Unresolved.** {unresolved}", f"- **Next control.** {control}", "",
        ]
    lines += families_section(evidence["families"])
    lines += [
        "## Portfolio deadline arms", "",
        *portfolio_table(evidence["portfolio"], evidence["fixtures"]), "",
        "![Portfolio objectives](portfolio-objective.svg)", "",
    ]
    figures["portfolio-objective.svg"] = portfolio_figure(evidence["portfolio"], evidence["fixtures"])
    return "\n".join(lines) + "\n", figures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round2", type=Path, required=True, help="Round 2 data root")
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    args = parser.parse_args(argv)
    report, figures = build_report(load_evidence(args.round2))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "REPORT.md").write_text(report, encoding="utf-8")
    for name, svg in figures.items():
        (args.out / name).write_text(svg, encoding="utf-8")
    print(json.dumps({"report": str(args.out / "REPORT.md"), "figures": sorted(figures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
