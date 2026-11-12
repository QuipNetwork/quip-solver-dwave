"""Step 0 of the regime search: the placements and the clique, on a small Zephyr graph."""

import importlib.util
import json
import pathlib
import sys

import pytest

from quip_solver_dwave import embeddings, regime_io, replay

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "embed_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("embed_probe", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embed_probe = _load_module()


def _write_spec(path: pathlib.Path, dead: int | None = None) -> replay.TopologySpec:
    """The topology spec of an ideal Z(3, 4), less one dead qubit when asked."""
    graphs = pytest.importorskip("dwave.graphs")
    graph = graphs.zephyr_graph(3, 4)
    path.write_text(json.dumps({
        "nodes": sorted(n for n in graph.nodes if n != dead),
        "edges": [list(edge) for edge in graph.edges if dead not in edge],
        "allowed_h_milli": [0],
        "allowed_j_milli": [-1000, 1000],
    }))
    return replay.load_spec(path)


def test_the_probe_writes_a_placement_for_each_lattice_and_the_chains_of_the_largest_clique(
    tmp_path, monkeypatch, capsys
):
    topology = regime_io.spec_topology_hash(_write_spec(tmp_path / "spec.json"))
    out = tmp_path / "embeddings"
    monkeypatch.setattr(sys, "argv", [
        "embed_probe.py", "--spec", str(tmp_path / "spec.json"), "--out-dir", str(out),
        "--diamond", "4", "4", "8", "--cubic-dimer", "3", "3", "2", "--zephyr-m", "3",
    ])
    assert embed_probe.main() == 0

    # The loaders accept the files, so the probe wrote the hash of this topology into each one.
    dims, diamond = regime_io.load_placement(out / "diamond.json", topology)
    assert dims == (4, 4, 8) and len(diamond.labels) == 32
    assert (diamond.vacant_spins, diamond.dropped_edges) == (0, 0)
    dims, cubic = regime_io.load_placement(out / "cubic-dimer.json", topology)
    assert dims == (3, 3, 2) and len(cubic.labels) == 36
    assert (cubic.vacant_spins, cubic.dropped_edges) == (0, 0)
    # An ideal Zephyr graph with 4 wires holds a clique of 16 m - 8 variables, and the probe takes all of it.
    assert len(regime_io.load_chains(out / regime_io.CLIQUE_FILE, topology)) == 40
    # The solver properties need the network, so the probe reads them only when asked.
    assert not (tmp_path / "solver-properties.json").exists()

    printed = capsys.readouterr().out
    assert "diamond (4, 4, 8): shift" in printed
    assert "32 spins and 48 edges placed, 0 of the 32 spins inside the graph are vacant (0.0%)" in printed
    assert "clique 40: chains of 4 to 4 qubits, 160 qubits in total" in printed


def test_find_clique_takes_the_largest_clique_on_a_graph_with_a_dead_qubit(tmp_path):
    dead = embed_probe.find_clique(None, _write_spec(tmp_path / "ideal.json"), 3, 4)[0][0]
    spec = _write_spec(tmp_path / "spec.json", dead=dead)
    chains = embed_probe.find_clique(None, spec, 3, 4)
    embeddings.validate_clique_embedding(chains, embeddings.edge_set(spec.edges.tolist()))
    assert all(dead not in chain for chain in chains)
    # Largest means that a clique of one more variable does not fit.
    with pytest.raises(SystemExit, match="no clique of"):
        embed_probe.find_clique(len(chains) + 1, spec, 3, 4)


def test_largest_clique_keeps_the_best_clique_over_the_seeds(monkeypatch):
    by_seed = {
        0: {0: [1, 2, 3], 1: [4, 5, 6]},
        1: {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]},   # more variables win
        2: {0: [1, 2], 1: [4, 5], 2: [7, 8, 9]},         # then fewer qubits, at an equal longest chain
        3: {0: [1, 2], 1: [4, 5], 2: [7, 8]},            # a shorter longest chain wins before that
        4: {0: [2, 1], 1: [5, 4], 2: [8, 7]},            # an equal rank keeps the earlier seed
    }

    class _Cache:
        def __init__(self, graph, seed: int) -> None:
            self.seed = seed

        def largest_clique(self):
            return by_seed[self.seed]

    monkeypatch.setattr("minorminer.busclique.busgraph_cache", _Cache)
    assert embed_probe.largest_clique(None, 2) == by_seed[1]
    assert embed_probe.largest_clique(None, 3) == by_seed[2]
    assert embed_probe.largest_clique(None, 5) == by_seed[3]


def test_find_clique_keeps_clear_of_a_dead_qubit(tmp_path):
    # Kill a qubit that the clique uses on the ideal graph, so the search has to move.
    dead = embed_probe.find_clique(8, _write_spec(tmp_path / "ideal.json"), 3, 4)[0][0]
    spec = _write_spec(tmp_path / "spec.json", dead=dead)
    chains = embed_probe.find_clique(8, spec, 3, 4)
    embeddings.validate_clique_embedding(chains, embeddings.edge_set(spec.edges.tolist()))
    assert all(dead not in chain for chain in chains)


def test_find_clique_stops_when_the_clique_does_not_fit(tmp_path):
    spec = _write_spec(tmp_path / "spec.json")
    with pytest.raises(SystemExit, match="no clique of 200 variables"):
        embed_probe.find_clique(200, spec, 3, 4)


def test_solver_properties_keeps_the_named_properties_and_closes_the_sampler(monkeypatch):
    closed = []

    class _Solver:
        properties = {"chip_id": "test-chip", "j_range": [-1.0, 1.0], "qubits": [0, 1, 2],
                      "per_group_coupling_range": [-13.0, 10.0]}

    class _Sampler:
        def __init__(self, mock: bool) -> None:
            assert mock is False
            self.sampler = _Solver()

        def ensure_connected(self) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("quip_solver_dwave.ocean.OceanSampler", _Sampler)
    properties = embed_probe.solver_properties()
    assert properties["chip_id"] == "test-chip" and properties["j_range"] == [-1.0, 1.0]
    assert properties["per_group_coupling_range"] == [-13.0, 10.0]
    assert set(properties) == set(embed_probe.PROPERTY_NAMES)
    assert closed == [True]
