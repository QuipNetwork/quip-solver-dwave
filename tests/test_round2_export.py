"""Direct tests of scripts/round2_export.py's model math and freeze-only-after-rescore ordering.

Follow-up from the review at
.superpowers/sdd/2026-09-22-regime-search-round2/task-2-review.md: the
canonical-model math (the /1000 milli conversion, the cubic unit-kernel
hash) and the export's write-only-after-rescoring-passes behavior had no
direct test. These build tiny synthetic ``regimes.Model`` instances and
Round 1 capture files instead of going through quip_msa or the real chip
topology, so they run without external data or hardware.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

from quip_solver_dwave import regime_io, regimes

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round2_export.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round2_export", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


round2_export = _load_module()


def _milli_model(score_j_milli, kernel_j) -> regimes.Model:
    n = 3
    score_edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    kernel_edges = np.array([[0, 1], [0, 1], [1, 2]], dtype=np.int64)
    return regimes.Model(
        kernel_h=np.zeros(n),
        kernel_edges=kernel_edges,
        kernel_j=np.asarray(kernel_j, dtype=np.float64),
        qpu_nodes=np.array([10, 11, 12], dtype=np.int64),
        qpu_h=np.zeros(n),
        qpu_edges=score_edges,
        qpu_j=np.zeros(2),
        score_edges=score_edges,
        score_j=np.asarray(score_j_milli, dtype=np.int64),
    )


# --------------------------------------------------------------- canonical_model


def test_canonical_model_milli_divides_by_1000_exactly_once():
    model = _milli_model([1500, -500], kernel_j=[1.0, -1.0, 1.0])
    h, edges, j = round2_export.canonical_model("native-pm1", model)
    np.testing.assert_array_equal(h, np.zeros(3))
    np.testing.assert_array_equal(edges, model.score_edges)
    np.testing.assert_array_equal(j, np.array([1.5, -0.5]))
    assert j.dtype == np.float64


def test_canonical_model_clique_is_already_float64_energy_units():
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    h = np.array([0.25, -0.75, 0.1])
    j = np.array([0.3, -0.4])
    model = regimes.Model(
        kernel_h=h, kernel_edges=edges, kernel_j=j,
        qpu_nodes=np.arange(6), qpu_h=np.zeros(6), qpu_edges=edges, qpu_j=j,
        score_edges=edges, score_j=j, score_h=h,
    )
    out_h, out_edges, out_j = round2_export.canonical_model("clique-portfolio", model)
    np.testing.assert_array_equal(out_h, h)
    np.testing.assert_array_equal(out_edges, edges)
    np.testing.assert_array_equal(out_j, j)


# ------------------------------------------------------------ unit_kernel_input_hash


def test_unit_kernel_input_hash_is_sensitive_to_the_repeated_unit_bonds():
    a = _milli_model([1000, -1000], kernel_j=[1.0, -1.0, 1.0])
    b = _milli_model([1000, -1000], kernel_j=[1.0, 1.0, 1.0])
    c = _milli_model([1000, -1000], kernel_j=[1.0, -1.0, 1.0])
    assert round2_export.unit_kernel_input_hash(a) != round2_export.unit_kernel_input_hash(b)
    assert round2_export.unit_kernel_input_hash(a) == round2_export.unit_kernel_input_hash(c)


# ------------------------------------------------------------------- rescore_check


def _write_capture(path: pathlib.Path, spins: np.ndarray, energies: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, spins=spins, energies=energies)


def test_rescore_check_matches_milli_energies_after_the_unit_conversion(tmp_path):
    h = np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([0.5])  # energy units; the milli original was 500
    spins = np.array([[1, 1], [1, -1]], dtype=np.int8)
    energies_milli = np.array([500, -500], dtype=np.int64)  # bundle energy * 1000
    round1_dir = tmp_path / "round1"
    for folder in round2_export.CAPTURE_DIRS:
        _write_capture(round1_dir / folder / "nonce.npz", spins, energies_milli)

    ok, error = round2_export.rescore_check(h, edges, j, "native-pm1", round1_dir, "nonce")
    assert ok
    assert error == 0.0


def test_rescore_check_flags_a_mismatch(tmp_path):
    h = np.zeros(2)
    edges = np.array([[0, 1]], dtype=np.int64)
    j = np.array([0.5])
    spins = np.array([[1, 1]], dtype=np.int8)
    wrong_energies = np.array([999], dtype=np.int64)
    round1_dir = tmp_path / "round1"
    for folder in round2_export.CAPTURE_DIRS:
        _write_capture(round1_dir / folder / "nonce.npz", spins, wrong_energies)

    ok, error = round2_export.rescore_check(h, edges, j, "native-pm1", round1_dir, "nonce")
    assert not ok
    assert error > 0.0


# ------------------------------------------------------ export_cell write-after-rescore


def test_export_cell_writes_only_after_rescoring_passes_and_fails_the_run_cleanly(tmp_path, monkeypatch):
    """A mismatched nonce must never get a bundle directory, and the whole cell must fail.

    Two nonces share the same model math; only the saved capture for "bad"
    is wrong. That must raise (not a SystemExit swallowed inside the thread
    pool) and "bad" must never appear under bundles_root.
    """
    cell = "native-pm1"
    round1_dir = tmp_path / "round1" / cell
    bundles_root = tmp_path / "bundles"

    n = 2
    edges = np.array([[0, 1]], dtype=np.int64)
    j_milli = np.array([500], dtype=np.int64)  # -> 0.5 energy units
    spins = np.array([[1, -1]], dtype=np.int8)  # energy = -1 * 0.5 = -0.5 -> milli -500
    good_energies = np.array([-500], dtype=np.int64)
    bad_energies = np.array([12345], dtype=np.int64)

    def make_model(nonce: str) -> regimes.Model:
        return regimes.Model(
            kernel_h=np.zeros(n), kernel_edges=edges, kernel_j=np.array([-1.0]),
            qpu_nodes=np.array([100, 101]), qpu_h=np.zeros(n), qpu_edges=edges, qpu_j=np.zeros(1),
            score_edges=edges, score_j=j_milli,
        )

    monkeypatch.setattr(round2_export, "ROUND1", tmp_path / "round1")
    monkeypatch.setattr(regime_io, "cell_identity", lambda *a, **k: {})
    monkeypatch.setattr(regime_io, "cell_nonces", lambda *a, **k: ["good", "bad"])
    monkeypatch.setattr(regime_io, "model_source", lambda *a, **k: ("cpu-msa", make_model))

    round1_dir.mkdir(parents=True)
    (round1_dir / "manifest.json").write_text(json.dumps({"rng_seed": 1}))
    (round1_dir / "nonces.json").write_text(json.dumps({"nonces": ["good", "bad"]}))
    for nonce, energies in (("good", good_energies), ("bad", bad_energies)):
        for folder in round2_export.CAPTURE_DIRS:
            _write_capture(round1_dir / folder / f"{nonce}.npz", spins, energies)

    interpreter = {"path": "/usr/bin/python3", "version": "3.11.0"}
    with pytest.raises(round2_export.ExportError) as excinfo:
        round2_export.export_cell(
            cell, object(), bundles_root, 2, "deadbeef", False, {"numpy": "0"}, interpreter
        )

    assert "bad" in str(excinfo.value)
    assert not (bundles_root / cell / "bad").exists()
