import json

import numpy as np
import pytest

from quip_solver_dwave import replay

quip_msa = pytest.importorskip("quip_msa")

SPEC = {
    "nodes": [10, 12, 30, 31],
    "edges": [[10, 12], [12, 30], [30, 31], [31, 10]],
    "allowed_h_milli": [0],
    "allowed_j_milli": [-1000, 1000],
}


def write_spec(tmp_path, spec=SPEC):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_load_spec_maps_edge_labels_to_dense_indices(tmp_path):
    spec = replay.load_spec(write_spec(tmp_path))
    assert spec.nodes.tolist() == [10, 12, 30, 31]
    assert spec.dense_edges.tolist() == [[0, 1], [1, 2], [2, 3], [3, 0]]


def test_load_spec_keeps_the_edge_order_of_the_file(tmp_path):
    shuffled = dict(SPEC, edges=[[31, 10], [10, 12], [30, 31], [12, 30]])
    spec = replay.load_spec(write_spec(tmp_path, shuffled))
    assert spec.edges.tolist() == shuffled["edges"]


def test_load_spec_dense_edges_follow_a_shuffled_nodes_order(tmp_path):
    # dense_edges indexes into h/j by position in `nodes`. If the shuffled
    # order were not honoured, every kernel call built on it would score the
    # wrong pair of variables.
    shuffled = dict(SPEC, nodes=[31, 10, 30, 12])
    spec = replay.load_spec(write_spec(tmp_path, shuffled))
    assert spec.nodes.tolist() == [31, 10, 30, 12]
    # edges are [[10, 12], [12, 30], [30, 31], [31, 10]]; position of each
    # label in the shuffled nodes list is 31->0, 10->1, 30->2, 12->3.
    assert spec.dense_edges.tolist() == [[1, 3], [3, 2], [2, 0], [0, 1]]


def test_load_spec_refuses_an_edge_that_names_an_unknown_node(tmp_path):
    broken = dict(SPEC, edges=[[10, 99]])
    with pytest.raises(ValueError, match="99"):
        replay.load_spec(write_spec(tmp_path, broken))


def test_model_from_nonce_matches_the_protocol_draw(tmp_path):
    spec = replay.load_spec(write_spec(tmp_path))
    nonce = "ab" * 32
    h, j = replay.model_from_nonce(spec, nonce)
    want_h, want_j = quip_msa.draw_ising(bytes.fromhex(nonce), 4, 4, [0], [-1000, 1000])
    assert np.array_equal(h, want_h) and np.array_equal(j, want_j)


def test_model_from_nonce_accepts_a_0x_prefix(tmp_path):
    spec = replay.load_spec(write_spec(tmp_path))
    plain = replay.model_from_nonce(spec, "ab" * 32)
    prefixed = replay.model_from_nonce(spec, "0x" + "ab" * 32)
    assert np.array_equal(plain[1], prefixed[1])


def test_load_attempts_keeps_the_lowest_energy_of_a_repeated_nonce(tmp_path):
    path = tmp_path / "attempts.csv"
    path.write_text("nonce,raw_best_energy_milli\naa,-100\nbb,-300\naa,-250\n")
    assert replay.load_attempts(path) == {"aa": -250, "bb": -300}
