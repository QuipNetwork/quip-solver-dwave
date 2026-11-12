"""Immutable logical-model bundles and run identities of Round 2.

A bundle is a directory holding ``manifest.json`` and ``model.npz``: the
logical coefficients of one model (``h``, ``edges``, ``j``, ``offset``) in
energy units, plus provenance. Every later solver, the physical-range audit,
and the CPU runner read a model only through :func:`write_bundle` and
:func:`read_bundle`, and identify one run of a model through :func:`run_id`.

The hash contract is fixed by the Round 2 plan (Task 2 brief): ``array_hash``
hashes dtype, shape, and canonical little-endian bytes; the bundle hash is
the same canonical-JSON SHA-256 over the manifest, excluding its own
``hash`` field, computed by :func:`run_id`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from quip_solver_dwave.regime_io import PathLike, atomic_savez, atomic_write_json

ARRAY_NAMES = ("h", "edges", "j", "offset")


def array_hash(array: np.ndarray) -> str:
    """SHA-256 of an already validated array: dtype, shape, canonical little-endian bytes."""
    value = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": value.shape},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def run_id(config: Dict[str, Any]) -> str:
    """SHA-256 of a run config's canonical JSON. Also the bundle/manifest hash primitive."""
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_bundle_arrays(h: np.ndarray, edges: np.ndarray, j: np.ndarray, offset: float) -> None:
    """Reject nonfinite coefficients, invalid endpoints, duplicate undirected edges, self-loops, bad shapes."""
    if h.ndim != 1:
        raise ValueError(f"h must be 1-D; got shape {h.shape}")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (m, 2); got {edges.shape}")
    if j.shape != (edges.shape[0],):
        raise ValueError(f"j must have shape ({edges.shape[0]},) to match edges; got {j.shape}")
    if not np.isfinite(h).all():
        raise ValueError("h has a nonfinite coefficient")
    if not np.isfinite(j).all():
        raise ValueError("j has a nonfinite coefficient")
    if not np.isfinite(offset):
        raise ValueError("offset is nonfinite")
    n = h.shape[0]
    if edges.shape[0] == 0:
        return
    if (edges < 0).any() or (edges >= n).any():
        raise ValueError(f"edges must index into [0, {n}); got min {edges.min()}, max {edges.max()}")
    if (edges[:, 0] == edges[:, 1]).any():
        raise ValueError("edges contain a self-loop")
    pairs = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}
    if len(pairs) != edges.shape[0]:
        raise ValueError("edges contain a duplicate undirected pair")


def _canonical(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "hash"}


def write_bundle(path: PathLike, manifest: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> str:
    """Validate, hash, and atomically write a bundle. Returns the model/bundle hash.

    Refuses to replace an existing bundle whose hash differs. Re-running with
    identical content is a no-op: the files are left untouched and the same
    hash is returned.
    """
    h = np.asarray(arrays["h"], dtype=np.float64)
    edges = np.asarray(arrays["edges"], dtype=np.int64)
    j = np.asarray(arrays["j"], dtype=np.float64)
    offset = float(manifest.get("offset", 0.0))
    _validate_bundle_arrays(h, edges, j, offset)
    offset_array = np.asarray(offset, dtype=np.float64)
    numeric = dict(zip(ARRAY_NAMES, (h, edges, j, offset_array)))

    canonical = dict(_canonical(manifest))
    canonical["offset"] = offset
    canonical["arrays"] = {
        name: {"dtype": arr.dtype.str, "shape": list(arr.shape), "hash": array_hash(arr)}
        for name, arr in numeric.items()
    }
    model_hash = run_id(canonical)
    full_manifest = {**canonical, "hash": model_hash}

    bundle_dir = Path(path)
    manifest_path = bundle_dir / "manifest.json"
    model_path = bundle_dir / "model.npz"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("hash") != model_hash:
            raise ValueError(
                f"{bundle_dir} already holds a bundle with hash {existing.get('hash')}; "
                f"refusing to replace it with {model_hash}"
            )
        if model_path.exists():
            # The manifest claims this is already the bundle we would write. Verify the
            # on-disk bytes actually match that claim before treating this as a no-op:
            # a stored hash is not proof the file on disk still has the hashed content.
            try:
                read_bundle(bundle_dir)
            except ValueError as exc:
                raise ValueError(
                    f"{bundle_dir} is recorded with hash {model_hash}, but its on-disk "
                    f"bytes do not verify against that hash: {exc}"
                ) from exc
            return model_hash

    bundle_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(model_path, **numeric)
    atomic_write_json(manifest_path, full_manifest)
    return model_hash


def read_bundle(path: PathLike) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Read a bundle, verifying the manifest hash and every stored array hash."""
    bundle_dir = Path(path)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    stored_hash = manifest.get("hash")
    if run_id(_canonical(manifest)) != stored_hash:
        raise ValueError(f"{bundle_dir}: manifest hash does not match its contents")

    with np.load(bundle_dir / "model.npz") as npz:
        arrays = {name: np.asarray(npz[name]) for name in ARRAY_NAMES}

    for name in ARRAY_NAMES:
        expected = manifest["arrays"][name]["hash"]
        actual = array_hash(arrays[name])
        if actual != expected:
            raise ValueError(f"{bundle_dir}: {name} array hash does not match the manifest")

    return manifest, arrays
