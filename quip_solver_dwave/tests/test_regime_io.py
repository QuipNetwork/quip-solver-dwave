"""Files of the regime search."""

import json

import numpy as np
import pytest

from quip_solver_dwave import regime_io, regimes
from quip_solver_dwave.embeddings import Placement
from quip_solver_dwave.replay import TopologySpec

# Two chains of two qubits around the single qubit 2: 0-1, 3-4, and the couplers between the chains.
EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 3]], dtype=np.int64)
SPEC = TopologySpec(
    nodes=np.arange(5, dtype=np.int64),
    edges=EDGES,
    dense_edges=EDGES,
    allowed_h_milli=[0],
    allowed_j_milli=[-1000, 1000],
)


def test_atomic_savez_leaves_only_the_final_file(tmp_path):
    path = tmp_path / "aa.npz"
    regime_io.atomic_savez(path, spins=np.ones((2, 3), dtype=np.int8), access_us=np.int64(5))
    with np.load(path) as loaded:
        assert int(loaded["access_us"]) == 5
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_savez_removes_the_temp_file_when_the_write_fails(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(regime_io.np, "savez_compressed", boom)
    with pytest.raises(RuntimeError, match="boom"):
        regime_io.atomic_savez(tmp_path / "aa.npz", spins=np.ones(1))
    assert list(tmp_path.iterdir()) == []


PLACEMENT = Placement(
    (1, 0),
    np.array([10, 11, -1, 13], dtype=np.int64),
    np.array([True, True, False, True]),
    np.array([True, False, True]),
)


def test_a_placement_survives_the_round_trip_through_its_file(tmp_path):
    path = tmp_path / "diamond.json"
    regime_io.save_placement(path, "diamond", (4, 4, 8), PLACEMENT, "ab12")
    dims, loaded = regime_io.load_placement(path, "ab12")
    assert dims == (4, 4, 8)
    assert loaded.shift == (1, 0)
    assert loaded.labels.tolist() == [10, 11, -1, 13]
    assert loaded.alive.tolist() == [True, True, False, True]
    assert loaded.keep.tolist() == [True, False, True]
    assert json.loads(path.read_text())["topology_hash"] == "ab12"


def test_load_placement_refuses_a_placement_of_another_topology(tmp_path):
    path = tmp_path / "diamond.json"
    regime_io.save_placement(path, "diamond", (4, 4, 8), PLACEMENT, "ab12")
    with pytest.raises(ValueError, match="diamond.json is for the topology ab12, and --spec gives cd34"):
        regime_io.load_placement(path, "cd34")


def test_cell_nonces_writes_the_list_once_and_refuses_to_mix_runs(tmp_path):
    first = regime_io.cell_nonces(tmp_path / "native-pm1", 4, 7)
    assert regime_io.cell_nonces(tmp_path / "native-pm1", 4, 7) == first
    saved = json.loads((tmp_path / "native-pm1" / "nonces.json").read_text())
    assert saved == {"rng_seed": 7, "nonces": first}
    with pytest.raises(ValueError, match="refusing to mix runs"):
        regime_io.cell_nonces(tmp_path / "native-pm1", 4, 8)
    with pytest.raises(ValueError, match="refusing to mix runs"):
        regime_io.cell_nonces(tmp_path / "native-pm1", 5, 7)


def test_load_cell_needs_the_embeddings_directory_for_a_lattice_cell():
    with pytest.raises(ValueError, match="needs --embeddings-dir"):
        regime_io.load_cell("diamond-pm1", None, None)  # type: ignore[arg-type]


def test_model_source_builds_a_drawn_model_for_a_native_cell():
    pytest.importorskip("quip_msa")
    label, model_of = regime_io.model_source("native-125", SPEC, None, 1)
    assert label == regimes.CPU_SA
    model = model_of("5a" * 32)
    assert model.kernel_j.shape == (5,) and model.chains is None
    assert model.qpu_nodes.tolist() == [0, 1, 2, 3, 4]


def test_model_source_builds_a_clique_model_with_one_asset_for_each_chain(tmp_path):
    regime_io.atomic_write_json(
        tmp_path / regime_io.CLIQUE_FILE,
        {"topology_hash": regime_io.spec_topology_hash(SPEC), "chains": [[0, 1], [2], [3, 4]]},
    )
    label, model_of = regime_io.model_source("clique-portfolio", SPEC, tmp_path, 1)
    assert label == regimes.CPU_SA
    model = model_of("5a" * 32)
    assert model.chains == [[0, 1], [2], [3, 4]] and model.chain_strength > 0
    assert sorted(model.qpu_nodes.tolist()) == [0, 1, 2, 3, 4]
    assert len(model.kernel_h) == 3


def test_model_source_needs_the_embeddings_directory_for_the_clique_cell():
    with pytest.raises(ValueError, match="clique-portfolio needs --embeddings-dir"):
        regime_io.model_source("clique-portfolio", SPEC, None, 1)


def test_model_source_refuses_clique_chains_of_another_topology(tmp_path):
    regime_io.atomic_write_json(
        tmp_path / regime_io.CLIQUE_FILE, {"topology_hash": "00ff", "chains": [[0, 1], [2], [3, 4]]}
    )
    with pytest.raises(ValueError, match="clique.json is for the topology 00ff"):
        regime_io.model_source("clique-portfolio", SPEC, tmp_path, 1)


def test_cell_identity_names_the_holdings_for_the_clique_cell_only(tmp_path):
    assert set(regime_io.cell_identity("native-pm1", SPEC, 9)) == {"topology_hash", "model_order_hash"}
    regime_io.atomic_write_json(tmp_path / regime_io.CLIQUE_FILE, {
        "topology_hash": regime_io.spec_topology_hash(SPEC), "chains": [[0, 1], [2], [3, 4]],
    })
    assert regime_io.cell_identity("clique-portfolio", SPEC, 9, tmp_path)["clique_holdings"] == 9


def test_check_manifest_records_a_cell_on_first_use_and_accepts_the_same_run_again(tmp_path):
    regime_io.check_manifest(tmp_path, reads=64, anneal_us=[80, 400], mock=False)
    regime_io.check_manifest(tmp_path, reads=64, anneal_us=[80, 400], mock=False)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved == {"reads": 64, "anneal_us": [80, 400], "mock": False}


def test_check_manifest_refuses_a_run_that_differs_from_the_first_one(tmp_path):
    regime_io.check_manifest(tmp_path, reads=8, mock=False)
    with pytest.raises(ValueError, match="reads is 8 in .*manifest.json and 64 in this run"):
        regime_io.check_manifest(tmp_path, reads=64, mock=False)
    with pytest.raises(ValueError, match="mock is False in .*manifest.json and True in this run"):
        regime_io.check_manifest(tmp_path, reads=8, mock=True)


def test_check_manifest_adds_the_fields_that_a_second_script_brings(tmp_path):
    regime_io.check_manifest(tmp_path, topology_hash="ab12")                 # the ladder ran first
    regime_io.check_manifest(tmp_path, topology_hash="ab12", reads=64)       # then the capture
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"topology_hash": "ab12", "reads": 64}
    with pytest.raises(ValueError, match="topology_hash is ab12"):
        regime_io.check_manifest(tmp_path, topology_hash="cd34")


def test_read_spend_is_zero_for_a_round_that_has_not_started(tmp_path):
    assert regime_io.read_spend(tmp_path) == (0, 0)


def test_read_spend_adds_each_charge_and_counts_an_open_job_at_its_estimate(tmp_path):
    regime_io.append_spend(tmp_path, regime_io.SUBMIT, "native-pm1", "aa", 80, 52_393)
    regime_io.append_spend(tmp_path, regime_io.CHARGE, "native-pm1", "aa", 80, 50_000)
    regime_io.append_spend(tmp_path, regime_io.SUBMIT, "diamond-pm1", "aa", 400, 72_873)   # no answer came back
    assert regime_io.read_spend(tmp_path) == (122_873, 1)


def test_read_spend_counts_both_jobs_when_a_capture_ran_twice(tmp_path):
    # The save failed after the first job, so a later run bought the capture again.
    for charge in (50_000, 51_000):
        regime_io.append_spend(tmp_path, regime_io.SUBMIT, "native-pm1", "aa", 80, 52_393)
        regime_io.append_spend(tmp_path, regime_io.CHARGE, "native-pm1", "aa", 80, charge)
    assert regime_io.read_spend(tmp_path) == (101_000, 0)


def test_read_spend_keeps_an_open_job_open_when_a_later_attempt_gets_its_charge(tmp_path):
    regime_io.append_spend(tmp_path, regime_io.SUBMIT, "native-pm1", "aa", 80, 52_393)     # no answer came back
    regime_io.append_spend(tmp_path, regime_io.SUBMIT, "native-pm1", "aa", 80, 52_393)
    regime_io.append_spend(tmp_path, regime_io.CHARGE, "native-pm1", "aa", 80, 50_000)
    assert regime_io.read_spend(tmp_path) == (102_393, 1)


def test_read_spend_refuses_a_journal_line_that_it_does_not_know(tmp_path):
    regime_io.append_spend(tmp_path, "refund", "native-pm1", "aa", 80, 1)
    with pytest.raises(ValueError, match="refund"):
        regime_io.read_spend(tmp_path)


def test_cell_identity_refuses_a_different_embedding_on_the_same_graph(tmp_path):
    embeddings = tmp_path / 'embeddings'
    embeddings.mkdir()
    path = embeddings / regime_io.CLIQUE_FILE
    regime_io.atomic_write_json(path, {'topology_hash': regime_io.spec_topology_hash(SPEC),
                                      'chains': [[0, 1], [2], [3, 4]]})
    first = regime_io.cell_identity('clique-portfolio', SPEC, 1, embeddings)
    regime_io.check_manifest(tmp_path / 'cell', **first)
    regime_io.atomic_write_json(path, {'topology_hash': regime_io.spec_topology_hash(SPEC),
                                      'chains': [[0], [1], [2], [3, 4]]})
    with pytest.raises(ValueError, match='embedding_hash'):
        regime_io.check_manifest(tmp_path / 'cell', **regime_io.cell_identity('clique-portfolio', SPEC, 1, embeddings))


def test_cell_identity_preserves_the_order_that_assigns_drawn_couplings(tmp_path):
    reordered = TopologySpec(
        nodes=SPEC.nodes, edges=SPEC.edges[::-1], dense_edges=SPEC.dense_edges[::-1],
        allowed_h_milli=[0], allowed_j_milli=[-1000, 1000],
    )
    assert regime_io.spec_topology_hash(reordered) == regime_io.spec_topology_hash(SPEC)
    regime_io.check_manifest(tmp_path, **regime_io.cell_identity('native-pm1', SPEC, 9))
    with pytest.raises(ValueError, match='model_order_hash'):
        regime_io.check_manifest(tmp_path, **regime_io.cell_identity('native-pm1', reordered, 9))


def test_the_archive_topology_digest_keeps_its_recorded_format():
    # The regime manifests and embedding files under ~/quip-data/regimes record
    # this blake2b digest. It is not the FAM-19 wire hash: changing it would
    # make every archived cell refuse to resume. The vector is the output of
    # the blake2b function the miner advertised before quip-solver-core 0.0.11.
    digest = regime_io.archive_topology_digest([3, 0, 2, 1], [(1, 0), (1, 2), (3, 2)])
    assert digest.hex() == "f85996f82af85750180fe346f1ff8b34e360e47b0a9e96bfa7114fda4dc2900c"
