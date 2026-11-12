"""Round trip, hashing, and identity tests for quip_solver_dwave.round2_io.

Task 2 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: every
later solver, the physical-range audit, and the CPU runner read models only
through the bundles and run IDs this module creates. A bundle must round
trip exactly (byte-for-byte arrays, verified hashes) and must reject the
malformed inputs the bundle contract forbids.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quip_solver_dwave import round2_io

# ------------------------------------------------------------------ array_hash


def test_adjacent_float_coefficients_have_distinct_hashes():
    a = np.array([0.1], dtype=np.float64)
    b = np.nextafter(a, np.inf)
    assert round2_io.array_hash(a) != round2_io.array_hash(b)


def test_array_hash_is_stable_across_byte_order():
    little = np.array([1.5, -2.25, 0.0], dtype="<f8")
    big = little.astype(">f8")
    assert round2_io.array_hash(little) == round2_io.array_hash(big)


# ----------------------------------------------------------------------- run_id


def test_physical_scale_changes_the_run_but_not_the_model():
    base = {"model_hash": "a", "scale": 0.5, "seed": 7}
    changed = dict(base, scale=1.0)
    assert base["model_hash"] == changed["model_hash"]
    assert round2_io.run_id(base) != round2_io.run_id(changed)


def test_run_id_is_order_independent():
    a = {"model_hash": "a", "scale": 0.5, "seed": 7}
    b = {"seed": 7, "scale": 0.5, "model_hash": "a"}
    assert round2_io.run_id(a) == round2_io.run_id(b)


# -------------------------------------------------------------- write/read bundle


def _arrays(h, edges, j):
    return {
        "h": np.asarray(h, dtype=np.float64),
        "edges": np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        "j": np.asarray(j, dtype=np.float64),
    }


def _manifest(**extra):
    return {"cell": "test-cell", "nonce": "00", "offset": 0.0, **extra}


def test_round_trip_preserves_fractional_values_zero_fields_and_order(tmp_path):
    arrays = _arrays(
        h=[0.0, -1.25, 0.0, 3.5],
        edges=[[0, 1], [1, 2], [2, 3]],
        j=[0.1, -0.2, 0.3],
    )
    path = tmp_path / "bundle"
    model_hash = round2_io.write_bundle(path, _manifest(offset=2.5), arrays)
    manifest, read_arrays = round2_io.read_bundle(path)

    assert manifest["hash"] == model_hash
    np.testing.assert_array_equal(read_arrays["h"], arrays["h"])
    np.testing.assert_array_equal(read_arrays["edges"], arrays["edges"])
    np.testing.assert_array_equal(read_arrays["j"], arrays["j"])
    assert read_arrays["h"].dtype == np.float64
    assert read_arrays["edges"].dtype == np.int64
    assert float(read_arrays["offset"]) == 2.5


def test_round_trip_keeps_isolated_nodes(tmp_path):
    # Node 2 has no incident edge.
    arrays = _arrays(h=[1.0, 0.0, 0.0], edges=[[0, 1]], j=[0.5])
    path = tmp_path / "bundle"
    round2_io.write_bundle(path, _manifest(), arrays)
    _, read_arrays = round2_io.read_bundle(path)
    assert read_arrays["h"].shape == (3,)


def test_round_trip_allows_zero_couplers(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[0, 1]], j=[0.0])
    path = tmp_path / "bundle"
    round2_io.write_bundle(path, _manifest(), arrays)
    _, read_arrays = round2_io.read_bundle(path)
    assert read_arrays["j"][0] == 0.0


def test_write_bundle_rejects_nonfinite_h(tmp_path):
    arrays = _arrays(h=[float("nan"), 0.0], edges=[[0, 1]], j=[1.0])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rejects_nonfinite_j(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[0, 1]], j=[float("inf")])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rejects_wrong_shapes(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[0, 1]], j=[1.0, 2.0])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rejects_negative_endpoint(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[-1, 1]], j=[1.0])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rejects_out_of_range_endpoint(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[0, 2]], j=[1.0])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rejects_duplicate_undirected_edge(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[0, 1], [1, 0]], j=[1.0, 1.0])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rejects_self_loop(tmp_path):
    arrays = _arrays(h=[0.0, 0.0], edges=[[0, 0]], j=[1.0])
    with pytest.raises(ValueError):
        round2_io.write_bundle(tmp_path / "bundle", _manifest(), arrays)


def test_write_bundle_rerun_with_identical_content_is_a_no_op(tmp_path):
    arrays = _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.5])
    path = tmp_path / "bundle"
    first = round2_io.write_bundle(path, _manifest(), arrays)
    second = round2_io.write_bundle(path, _manifest(), arrays)
    assert first == second


def test_write_bundle_refuses_to_replace_a_different_bundle(tmp_path):
    path = tmp_path / "bundle"
    round2_io.write_bundle(path, _manifest(), _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.5]))
    with pytest.raises(ValueError):
        round2_io.write_bundle(path, _manifest(), _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.75]))


def test_read_bundle_detects_a_tampered_array(tmp_path):
    path = tmp_path / "bundle"
    arrays = _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.5])
    round2_io.write_bundle(path, _manifest(), arrays)

    with np.load(path / "model.npz") as npz:
        tampered = {name: npz[name] for name in npz.files}
    tampered["j"] = tampered["j"] + 1.0
    np.savez(path / "model.npz", **tampered)

    with pytest.raises(ValueError):
        round2_io.read_bundle(path)


def test_read_bundle_detects_a_tampered_manifest(tmp_path):
    path = tmp_path / "bundle"
    arrays = _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.5])
    round2_io.write_bundle(path, _manifest(), arrays)

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cell"] = "tampered-cell"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        round2_io.read_bundle(path)


def test_read_bundle_detects_a_tampered_offset(tmp_path):
    path = tmp_path / "bundle"
    arrays = _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.5])
    round2_io.write_bundle(path, _manifest(offset=0.0), arrays)

    with np.load(path / "model.npz") as npz:
        tampered = {name: npz[name] for name in npz.files}
    tampered["offset"] = np.asarray(5.0, dtype=np.float64)
    np.savez(path / "model.npz", **tampered)

    with pytest.raises(ValueError):
        round2_io.read_bundle(path)


def test_write_bundle_no_op_path_verifies_the_on_disk_bytes(tmp_path):
    """A stored hash field is not proof the file on disk still has the hashed content."""
    path = tmp_path / "bundle"
    arrays = _arrays(h=[1.0, -1.0], edges=[[0, 1]], j=[0.5])
    round2_io.write_bundle(path, _manifest(), arrays)

    with np.load(path / "model.npz") as npz:
        tampered = {name: npz[name] for name in npz.files}
    tampered["j"] = tampered["j"] + 1.0
    np.savez(path / "model.npz", **tampered)  # manifest.json (and its "hash") is untouched

    with pytest.raises(ValueError):
        round2_io.write_bundle(path, _manifest(), arrays)
