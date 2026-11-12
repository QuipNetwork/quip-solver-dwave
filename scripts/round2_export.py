#!/usr/bin/env python3
"""Freeze the five Round 1 regimes into immutable Round 2 logical-model bundles.

Task 2 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: every
later solver, the physical-range audit, and the CPU runner read a model only
through the bundles this script writes under
``round2/bundles/<cell>/<nonce>/``. Rebuilds each of the 500 saved models (5
cells x 100 nonces) from its nonce, exports the original logical
coefficients in energy units (milli cells divide the drawn integer
coefficients by 1000 exactly once; clique-portfolio is already float64), then
rescores every saved ``qpu-80``/``qpu-400`` capture on the canonical bundle
and compares it with Round 1's saved energies. Any mismatch fails the
export. cubic-dimer-pm1 also gets a separate unit-kernel input hash: the
canonical bundle holds one weighted edge per pair (never the kernel's
repeated-unit bonds), and the repeated form is hashed and recorded, not
exported as the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quip_solver_dwave import __version__, regime_io, regimes, replay, round2_io

ROUND1 = Path("/home/carback1/quip-data/regimes/round1")
SPEC = "/home/carback1/quip-data/regimes/topology-live-2026-09-21.spec.json"
EMBEDDINGS = "/home/carback1/quip-data/regimes/embeddings"
DEFAULT_BUNDLES_ROOT = Path("/home/carback1/quip-data/regimes/round2/bundles")
HOLDINGS = 9
MODELS_PER_CELL = 100
CAPTURE_DIRS = ("qpu-80", "qpu-400")
MILLI_CELLS = frozenset({"native-pm1", "native-125", "diamond-pm1", "cubic-dimer-pm1"})
D_REPO = Path(__file__).resolve().parents[1]


class ExportError(RuntimeError):
    """A model failed identity verification or rescoring; the whole export must fail."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_facts(repo: Path) -> Tuple[str, bool]:
    """HEAD and dirty state of ``repo``. Raises if either git call fails; never returns None."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if head.returncode != 0:
        raise RuntimeError(f"could not determine {repo}'s HEAD commit: {head.stderr.strip()}")
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    if status.returncode != 0:
        raise RuntimeError(f"could not determine {repo}'s dirty state: {status.stderr.strip()}")
    return head.stdout.strip(), bool(status.stdout.strip())


def package_versions() -> Dict[str, str]:
    """Installed version of every package the export depends on. Raises if one cannot be found."""
    versions: Dict[str, str] = {"quip_solver_dwave": __version__}
    for name in ("numpy", "quip_msa", "quip-solver-core"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"could not determine the installed version of {name}") from exc
    return versions


def canonical_model(cell: str, model: regimes.Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(h, edges, j) of the canonical bundle, in energy units, never the repeated-unit kernel form.

    ``score_edges``/``score_j`` are the true problem (one weighted edge per
    pair; ``lattices.cubic_dimer_edges`` already splits a z bond into two
    distinct half-strength edges rather than repeating one). Only
    ``kernel_edges`` explodes a coupling into repeated unit bonds, and that
    stays out of the canonical model.
    """
    if model.score_h is not None:  # clique-portfolio: already float64 energy units
        return model.score_h, model.score_edges, model.score_j
    h = np.asarray(model.kernel_h, dtype=np.float64)  # zeros(n): milli cells carry no field
    j = np.asarray(model.score_j, dtype=np.int64).astype(np.float64) / 1000.0
    return h, model.score_edges, j


def variable_labels(cell: str, model: regimes.Model, n: int) -> List[int]:
    if cell == "clique-portfolio":
        return list(range(n))  # one asset per chain; no single physical qubit labels one asset
    return [int(q) for q in model.qpu_nodes]


def unit_kernel_input_hash(model: regimes.Model) -> str:
    """Hash of the repeated-unit kernel input (h, edges, j), kept separate from the model hash."""
    hashes = {
        "h": round2_io.array_hash(np.asarray(model.kernel_h, dtype=np.float64)),
        "edges": round2_io.array_hash(np.asarray(model.kernel_edges, dtype=np.int64)),
        "j": round2_io.array_hash(np.asarray(model.kernel_j, dtype=np.float64)),
    }
    return round2_io.run_id(hashes)


def rescore_check(
    h: np.ndarray, edges: np.ndarray, j: np.ndarray, cell: str, round1_dir: Path, nonce: str
) -> Tuple[bool, float]:
    """Rescore every saved capture on the canonical bundle; compare with Round 1's saved energies.

    Milli cells: bundle energy x 1000 must equal the saved integer energy
    exactly or within 1e-6 absolute. clique-portfolio already scores in
    float64 with no unit conversion.
    """
    ok = True
    max_abs_error = 0.0
    for folder in CAPTURE_DIRS:
        with np.load(round1_dir / folder / f"{nonce}.npz") as cap:
            spins = cap["spins"]
            saved = cap["energies"]
        bundle_energy = regimes.energy(spins, h, edges, j)
        if cell in MILLI_CELLS:
            scaled = bundle_energy * 1000.0
            match = bool(np.allclose(scaled, saved, rtol=0.0, atol=1e-6))
            error = float(np.max(np.abs(scaled - saved)))
        else:
            match = bool(np.allclose(bundle_energy, saved, rtol=1e-9, atol=1e-9))
            error = float(np.max(np.abs(bundle_energy - saved)))
        ok = ok and match
        max_abs_error = max(max_abs_error, error)
    return ok, max_abs_error


def build_manifest(
    *,
    cell: str,
    nonce: str,
    identity: Dict[str, Any],
    model: regimes.Model,
    h: np.ndarray,
    edges: np.ndarray,
    round1_dir: Path,
    d_head: str,
    dirty: bool,
    versions: Dict[str, str],
    interpreter: Dict[str, str],
) -> Dict[str, Any]:
    n = int(h.shape[0])
    manifest: Dict[str, Any] = {
        "cell": cell,
        "nonce": nonce,
        "provenance": "historical",
        "offset": 0.0,
        "n_spins": n,
        "n_edges": int(edges.shape[0]),
        "variable_labels": variable_labels(cell, model, n),
        "scoring_rule": {
            "formula": "sum(h_i * s_i) + sum(J_ij * s_i * s_j) over edges, s in {-1, +1}",
            "milli_source": cell in MILLI_CELLS,
            "milli_divisor": 1000 if cell in MILLI_CELLS else None,
        },
        "identity": identity,
        "source_commits": {"D": d_head, "dirty": dirty},
        "package_versions": versions,
        "interpreter": interpreter,
        "original_data": {
            "round1_manifest_sha256": sha256_file(round1_dir / "manifest.json"),
            "round1_nonces_sha256": sha256_file(round1_dir / "nonces.json"),
            "captures": {
                folder: sha256_file(round1_dir / folder / f"{nonce}.npz") for folder in CAPTURE_DIRS
            },
        },
    }
    if cell == "cubic-dimer-pm1":
        manifest["unit_kernel_input_hash"] = unit_kernel_input_hash(model)
    return manifest


def export_cell(
    cell: str,
    spec: replay.TopologySpec,
    bundles_root: Path,
    threads: int,
    d_head: str,
    dirty: bool,
    versions: Dict[str, str],
    interpreter: Dict[str, str],
) -> List[Dict[str, Any]]:
    round1_dir = ROUND1 / cell
    manifest_round1 = json.loads((round1_dir / "manifest.json").read_text(encoding="utf-8"))
    identity = regime_io.cell_identity(cell, spec, HOLDINGS, EMBEDDINGS)
    for key in identity:
        if key in manifest_round1 and identity[key] != manifest_round1[key]:
            raise ExportError(
                f"{cell}: {key} differs from Round 1: {identity[key]!r} vs {manifest_round1[key]!r}"
            )
    nonces = regime_io.cell_nonces(round1_dir, MODELS_PER_CELL, manifest_round1.get("rng_seed", 20260920))
    _, model_of = regime_io.model_source(cell, spec, EMBEDDINGS, HOLDINGS)

    rows: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def process_one(nonce: str) -> None:
        model = model_of(nonce)
        h, edges, j = canonical_model(cell, model)

        # Rescore the candidate arrays against Round 1's saved captures BEFORE
        # freezing anything: a mismatch must never leave a written bundle
        # directory behind for a later, corrected run to trip over.
        ok, max_abs_error = rescore_check(h, edges, j, cell, round1_dir, nonce)
        if not ok:
            raise ExportError(
                f"{cell}/{nonce}: rescored bundle energy does not match Round 1's saved energies "
                f"(max abs error {max_abs_error})"
            )

        manifest = build_manifest(
            cell=cell, nonce=nonce, identity=identity, model=model, h=h, edges=edges,
            round1_dir=round1_dir, d_head=d_head, dirty=dirty, versions=versions, interpreter=interpreter,
        )
        bundle_path = bundles_root / cell / nonce
        model_hash = round2_io.write_bundle(bundle_path, manifest, {"h": h, "edges": edges, "j": j})
        row = {
            "cell": cell,
            "nonce": nonce,
            "model_hash": model_hash,
            "unit_kernel_input_hash": manifest.get("unit_kernel_input_hash"),
            "n": int(h.shape[0]),
            "m": int(edges.shape[0]),
            "rescore_ok": ok,
            "rescore_max_abs_error": max_abs_error,
        }
        with lock:
            rows.append(row)

    # A worker's ExportError must fail the whole run: `with` waits for every
    # submitted nonce to finish, then list() re-raises the first exception
    # a future recorded, instead of letting it get lost in the pool.
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(process_one, nonces))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", default=list(regimes.CELL_NAMES), choices=regimes.CELL_NAMES)
    parser.add_argument("--bundles-root", type=Path, default=DEFAULT_BUNDLES_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help=(
            "Export from a D tree with uncommitted changes. For tests only: a real export "
            "must run from a clean, committed tree so source_commits identifies the code "
            "that wrote the bundles."
        ),
    )
    args = parser.parse_args(argv)

    try:
        d_head, dirty = git_facts(D_REPO)
        if dirty and not args.allow_dirty:
            print(
                f"{D_REPO} has uncommitted changes; refusing to export from a dirty tree "
                "(source_commits would not identify the code that wrote the bundles). "
                "Commit or stash first, or pass --allow-dirty.",
                file=sys.stderr,
            )
            return 1
        versions = package_versions()
        interpreter = {"path": sys.executable, "version": platform.python_version()}

        spec = replay.load_spec(SPEC)

        # Keep index rows for cells this run does not touch, so a --cells subset
        # run never drops the other cells' entries from the summary index.
        index_path = args.bundles_root / "index.json"
        all_rows: List[Dict[str, Any]] = []
        if index_path.exists():
            previous = json.loads(index_path.read_text(encoding="utf-8"))
            all_rows = [row for row in previous["rows"] if row["cell"] not in args.cells]

        for cell in args.cells:
            rows = export_cell(
                cell, spec, args.bundles_root, args.threads, d_head, dirty, versions, interpreter
            )
            all_rows.extend(rows)
            print(f"{cell}: wrote {len(rows)} bundles under {args.bundles_root / cell}")
    except (RuntimeError, ExportError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1

    all_rows.sort(key=lambda row: (row["cell"], row["nonce"]))
    index = {"bundles_root": str(args.bundles_root), "models": len(all_rows), "rows": all_rows}
    args.bundles_root.mkdir(parents=True, exist_ok=True)
    regime_io.atomic_write_json(args.bundles_root / "index.json", index)
    print(f"wrote {args.bundles_root / 'index.json'} ({len(all_rows)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
