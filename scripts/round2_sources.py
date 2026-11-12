#!/usr/bin/env python3
"""Round 2 source inventory and historical-provenance check.

Task 1 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: before
any later task touches a sampler, this module answers one question about
every candidate data source: which of the twelve fields a historical replica
needs are actually on file, versus assumed from a current default.

``historical_gaps`` is the pure check the plan specifies verbatim. The rest
of the module inventories files on disk (path, size, hash, mtime) without
opening anything that looks like a credential, and assembles that inventory
plus known fields into one record per source with an explicit provenance tag
(``historical``, ``prospective``, or ``synthetic-test``).

Running this file directly regenerates round2/sources.json and
round2/environment.json under the data root (default
``/home/carback1/quip-data/regimes/round2``) from the live state of the D,
C, and P checkouts and the local market-archive inventory. Nothing here
substitutes current data for a historical value that record-keeping never
captured; a missing field stays missing.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

PROVENANCE_KINDS = ("historical", "prospective", "synthetic-test")

REQUIRED_FIELDS = (
    "market_snapshot_sha256",
    "market_as_of",
    "basket",
    "sliders",
    "deployment_commit",
    "runtime_config",
    "effective_beta",
    "reads",
    "sa_sweeps",
    "qpu_anneal_us",
    "qpu_solver",
    "tie_rules",
)

# Substrings checked case-insensitively against a path's components. A file
# whose name or containing directory matches one of these is never opened or
# hashed by inventory_paths.
CREDENTIAL_MARKERS = ("secret", "token", "keyring", "pass", ".env")


def historical_gaps(record: dict) -> list:
    """Required historical fields a record does not have on file.

    Verbatim from the Task 1 brief: a key counts as missing when it is
    absent or explicitly ``None``. A record built from current defaults
    (for example just a ``runtime_config``) must not read as complete.
    """
    required = (
        "market_snapshot_sha256", "market_as_of", "basket", "sliders",
        "deployment_commit", "runtime_config", "effective_beta",
        "reads", "sa_sweeps", "qpu_anneal_us", "qpu_solver", "tie_rules",
    )
    return [key for key in required if key not in record or record[key] is None]


def is_credential_like(path: Path) -> bool:
    """True if any path component looks like a credential file or store."""
    lowered = [part.lower() for part in path.parts]
    for part in lowered:
        for marker in CREDENTIAL_MARKERS:
            if marker in part:
                return True
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_mtime(path: Path) -> str:
    stamp = path.stat().st_mtime
    return datetime.datetime.fromtimestamp(stamp, tz=datetime.timezone.utc).isoformat()


def inventory_paths(
    root: Path, *, exclude_dirs: FrozenSet[str] = frozenset()
) -> List[Dict[str, Any]]:
    """Path, size, sha256, and mtime of every non-credential file under root.

    Returns an empty list when root does not exist. A directory named in
    ``exclude_dirs`` (for example ``.git`` or ``.venv``) is pruned from the
    walk before it is entered, so nothing inside it is ever opened, hashed,
    or even stat'd. Never opens a path that ``is_credential_like`` flags
    either; its presence is not even recorded, so a credential file's name
    cannot leak into round2/sources.json.
    """
    root = Path(root)
    if not root.exists():
        return []
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in exclude_dirs)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if is_credential_like(path):
                continue
            entries.append(
                {
                    "path": path,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "mtime": _iso_mtime(path),
                }
            )
    entries.sort(key=lambda entry: str(entry["path"]))
    return entries


def build_source_record(
    *,
    name: str,
    provenance: str,
    root: Path,
    known_fields: Dict[str, Any],
    exclude_dirs: FrozenSet[str] = frozenset(),
) -> Dict[str, Any]:
    """One round2/sources.json entry: known fields, file inventory, gaps.

    ``known_fields`` may supply any subset of REQUIRED_FIELDS plus free-form
    extras (for example a note on why a field is unrecoverable); fields the
    caller does not supply are recorded as explicit ``None`` rather than
    omitted, so historical_gaps sees them and so does anyone reading the
    JSON directly. ``exclude_dirs`` is passed straight to ``inventory_paths``
    to prune large or irrelevant subtrees (``.git``, ``.venv``) before any
    file inside them is opened.
    """
    if provenance not in PROVENANCE_KINDS:
        raise ValueError(f"unknown provenance {provenance!r}; expected one of {PROVENANCE_KINDS}")
    record: Dict[str, Any] = {"name": name, "provenance": provenance}
    for field in REQUIRED_FIELDS:
        record[field] = known_fields.get(field)
    for key, value in known_fields.items():
        if key not in REQUIRED_FIELDS:
            record[key] = value
    record["inventory"] = inventory_paths(root, exclude_dirs=exclude_dirs)
    record["gaps"] = historical_gaps(record)
    return record


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {value!r}")


# ------------------------------------------------------- archive field extraction
_BUDGET_RE = re.compile(r"(\d+)\s+reads?\s*x\s*(\d+)\s+sweeps")


def extract_gutcheck_fields(root: Path) -> Dict[str, Any]:
    """reads / sa_sweeps read from portfolio-gutcheck's own JSON output, not hardcoded.

    The probe ran several budgets in the same run (kernel arms at 64 reads
    across several sweep counts, the shipped SA arm at 500 reads and 500
    sweeps), so no single ``(reads, sa_sweeps)`` pair is *the* historical
    setting. When the JSON output shows more than one distinct value,
    ``reads``/``sa_sweeps`` are left out of the returned dict entirely (so
    ``historical_gaps`` reports them missing); every distinct value seen is
    still recorded, under ``observed_candidates``, so the raw budgets are
    not lost.
    """
    root = Path(root)
    reads_seen: set = set()
    sweeps_seen: set = set()
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            for row in entry.get("rows", []):
                match = _BUDGET_RE.search(row.get("budget", ""))
                if match:
                    reads_seen.add(int(match.group(1)))
                    sweeps_seen.add(int(match.group(2)))
    fields: Dict[str, Any] = {}
    if len(reads_seen) == 1:
        fields["reads"] = next(iter(reads_seen))
    if len(sweeps_seen) == 1:
        fields["sa_sweeps"] = next(iter(sweeps_seen))
    if reads_seen or sweeps_seen:
        fields["observed_candidates"] = {
            "reads": sorted(reads_seen),
            "sa_sweeps": sorted(sweeps_seen),
        }
    return fields


def extract_round1_manifest_fields(root: Path) -> Dict[str, Any]:
    """reads / qpu_anneal_us read from round1's manifest.json files, only where every one agrees.

    A round1 manifest never carries a solver identity (the chip id lives in
    the separate regimes/solver-properties.json, outside this root), so this
    function never returns ``qpu_solver``; filling that field from a
    different file would let ``historical_gaps`` treat it as recovered
    when the inventoried record itself does not state it.
    """
    root = Path(root)
    manifests = []
    for path in sorted(root.glob("*/manifest.json")) if root.exists() else []:
        try:
            with open(path, encoding="utf-8") as handle:
                manifests.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue
    fields: Dict[str, Any] = {}
    if not manifests:
        return fields
    reads_values = {m.get("reads") for m in manifests}
    if len(reads_values) == 1 and next(iter(reads_values)) is not None:
        fields["reads"] = next(iter(reads_values))
    anneal_values = {tuple(m.get("anneal_us") or ()) for m in manifests}
    if len(anneal_values) == 1 and next(iter(anneal_values)):
        fields["qpu_anneal_us"] = list(next(iter(anneal_values)))
    return fields


# --------------------------------------------------------------- environment
def _run(cmd: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Run cmd, reporting success/failure explicitly rather than collapsing both to None.

    A caller that only wants best-effort text (a version string, an import
    path) can read ``["stdout"]`` and treat a failure as "unknown". A caller
    where "unknown" and "definitely false" are different facts -- such as a
    dirty-tree check -- must look at ``["ok"]`` and ``["error"]`` instead of
    inferring failure from an empty or ``None`` result.
    """
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": None, "error": str(exc)}
    except OSError as exc:
        return {"ok": False, "stdout": None, "error": str(exc)}
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or str(exc)).strip()
        return {"ok": False, "stdout": None, "error": message}
    return {"ok": True, "stdout": out.stdout.strip(), "error": None}


def _run_text(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Best-effort stdout of cmd, or None on failure, for callers where "unknown" reads safely."""
    result = _run(cmd, cwd=cwd)
    return result["stdout"] if result["ok"] else None


def _git_facts(repo: Path) -> Dict[str, Any]:
    """HEAD, branch, and dirty state of repo, with a failed probe recorded as a failure.

    ``dirty`` is ``True``/``False`` only when ``git status`` actually ran; a
    failed probe (repo missing, not a git checkout, git itself unavailable)
    sets ``dirty`` to ``None`` and lists the failure in ``probe_errors``, so
    it can never be misread as "the tree is clean".
    """
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    branch = _run(["git", "branch", "--show-current"], cwd=repo)
    probe_errors = [
        result["error"] for result in (head, status, branch) if not result["ok"]
    ]
    return {
        "path": str(repo),
        "head": head["stdout"] if head["ok"] else None,
        "branch": branch["stdout"] if branch["ok"] else None,
        "dirty": bool(status["stdout"]) if status["ok"] else None,
        "dirty_files": status["stdout"].splitlines() if status["ok"] and status["stdout"] else [],
        "probe_errors": probe_errors,
    }


def _python_facts(python: Path, imports: List[str]) -> Dict[str, Any]:
    version = _run_text([str(python), "--version"])
    modules: Dict[str, Optional[str]] = {}
    for module in imports:
        code = f"import {module}; print({module}.__file__)"
        out = _run_text([str(python), "-c", code])
        modules[module] = out if out else None
    return {"interpreter": str(python), "version": version, "modules": modules}


def collect_environment(
    *,
    d_repo: Path,
    c_repo: Path,
    p_repo: Path,
    plan_d_commit: str,
    plan_c_commit: str,
    plan_p_commit: str,
) -> Dict[str, Any]:
    """Environment facts for D, C, and P: HEAD, dirty state, interpreters, imports, host."""
    d_git = _git_facts(d_repo)
    c_git = _git_facts(c_repo)
    p_git = _git_facts(p_repo)
    d_git["plan_commit"] = plan_d_commit
    c_git["plan_commit"] = plan_c_commit
    p_git["plan_commit"] = plan_p_commit

    d_python = _python_facts(d_repo / ".venv" / "bin" / "python", ["quip_solver_dwave", "quip_msa"])
    p_python = _python_facts(
        p_repo / ".venv" / "bin" / "python",
        ["qpo", "qpo.encoders", "qpo.solvers.sampling", "qpo.solvers.feasibility", "neal", "scipy"],
    )
    d_lock = d_repo / "uv.lock"
    p_lock = p_repo / "uv.lock"
    c_cargo_lock = c_repo / "Cargo.lock"
    c_py_cargo_lock = c_repo / "py" / "Cargo.lock"

    return {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "host": {
            "cpu_model": _cpu_model(),
            "platform": platform.platform(),
        },
        "compilers": {
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "repositories": {
            "D": {**d_git, "lockfile_sha256": sha256_file(d_lock) if d_lock.exists() else None},
            "C": {
                **c_git,
                "lockfile_sha256": sha256_file(c_cargo_lock) if c_cargo_lock.exists() else None,
                "py_lockfile_sha256": (
                    sha256_file(c_py_cargo_lock) if c_py_cargo_lock.exists() else None
                ),
            },
            "P": {**p_git, "lockfile_sha256": sha256_file(p_lock) if p_lock.exists() else None},
        },
        "environments": {
            "D": d_python,
            "P": p_python,
        },
    }


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


# --------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/carback1/quip-data/regimes/round2"),
        help="Directory holding sources.json, environment.json, and reference/.",
    )
    parser.add_argument("--d-repo", type=Path, default=Path("/home/carback1/quip-miner-dwave"))
    parser.add_argument("--c-repo", type=Path, default=Path("/home/carback1/quip-miner-cpu"))
    parser.add_argument(
        "--p-repo",
        type=Path,
        default=Path("/home/carback1/quip-data/regimes/round2/reference/qportfolio"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/carback1/quip-data"),
        help="Root to search for market archives and reference configuration.",
    )
    args = parser.parse_args(argv)

    gutcheck_root = args.archive_root / "regimes" / "portfolio-gutcheck"
    round1_root = args.archive_root / "regimes" / "round1"
    gutcheck_fields = extract_gutcheck_fields(gutcheck_root)
    round1_fields = extract_round1_manifest_fields(round1_root)

    sources = {
        "generated_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "records": [
            build_source_record(
                name="portfolio-gutcheck",
                provenance="synthetic-test",
                root=gutcheck_root,
                known_fields={
                    **gutcheck_fields,
                    "note": (
                        "Classical-only probe (2026-09-20) against qpo.build_problem's "
                        "shipped default basket, not a captured historical market "
                        "snapshot. No QPU fields recorded; the script never ran on "
                        "hardware. reads/sa_sweeps are extracted from the probe's own "
                        "JSON output (extract_gutcheck_fields) and are left as gaps "
                        "here because the probe used several budgets in one run; see "
                        "observed_candidates for every distinct value seen."
                    ),
                },
            ),
            build_source_record(
                name="round1-regime-manifests",
                provenance="historical",
                root=round1_root,
                known_fields={
                    **round1_fields,
                    "qpu_solver_observed_elsewhere": (
                        "Advantage2_system1, from regimes/solver-properties.json -- "
                        "not part of any round1 manifest.json, so qpu_solver stays a gap here"
                    ),
                    "note": (
                        "QPU regime captures for logical-topology benchmarks "
                        "(native-125, clique-portfolio, native-pm1, diamond-pm1, "
                        "cubic-dimer-pm1), not the portfolio problem instances. reads "
                        "and qpu_anneal_us are extracted from the manifest.json files "
                        "(extract_round1_manifest_fields) and filled only because all "
                        "five agree. No market snapshot, basket, sliders, deployment "
                        "commit, effective beta, sa_sweeps, tie_rules, or solver "
                        "identity is recorded in any manifest."
                    ),
                },
            ),
            build_source_record(
                name="analysis-tables-2026-09-21",
                provenance="historical",
                root=args.archive_root / "regimes" / "analysis-tables-2026-09-21",
                known_fields={
                    "note": (
                        "Notion-exported analysis and figures for the round1 capture "
                        "above; text and SVG/PNG outputs, not raw market or run data."
                    ),
                },
            ),
            build_source_record(
                name="qportfolio-reference-checkout",
                provenance="prospective",
                root=args.p_repo,
                exclude_dirs=frozenset({".git", ".venv"}),
                known_fields={
                    "deployment_commit": "9136be4c4b57abdab7f0c46120c341ba94fae014",
                    "note": (
                        "Detached checkout of qtbench/Q-Portfolio-Optimization-Benchmark "
                        "at the plan-pinned commit; reference code, not a data capture. "
                        "Inventory excludes .git and .venv (large, not provenance-bearing; "
                        "pruned before either tree is walked, so nothing inside is opened)."
                    ),
                },
            ),
        ],
    }

    environment = collect_environment(
        d_repo=args.d_repo,
        c_repo=args.c_repo,
        p_repo=args.p_repo,
        plan_d_commit="c6831057daeb720ed16175e5d8ce04f39520a904",
        plan_c_commit="281c697d0785b5a14df199263b4a9af291325dc6",
        plan_p_commit="9136be4c4b57abdab7f0c46120c341ba94fae014",
    )

    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "sources.json").write_text(
        json.dumps(sources, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (args.data_root / "environment.json").write_text(
        json.dumps(environment, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
