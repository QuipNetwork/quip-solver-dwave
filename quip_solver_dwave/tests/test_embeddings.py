"""Validators and the chain round trip of the regime-search embeddings."""

import numpy as np
import pytest

from quip_solver_dwave import embeddings, lattices

# A working graph of two triangles that share qubit 2: 0-1-2 and 2-3-4.
WORKING = embeddings.edge_set([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)])
PATH = np.array([[0, 1], [1, 2]], dtype=np.int64)


def test_lattice_embedding_accepts_an_injective_map_on_working_couplers():
    embeddings.validate_lattice_embedding(3, PATH, [0, 1, 2], WORKING)


def test_lattice_embedding_rejects_a_map_that_is_not_injective():
    with pytest.raises(ValueError, match="not injective"):
        embeddings.validate_lattice_embedding(3, PATH, [0, 1, 0], WORKING)


def test_lattice_embedding_rejects_a_bond_off_the_working_graph():
    with pytest.raises(ValueError, match="not a working coupler"):
        embeddings.validate_lattice_embedding(3, PATH, [0, 1, 4], WORKING)


def test_lattice_embedding_rejects_a_map_of_the_wrong_length():
    with pytest.raises(ValueError, match="covers 2 spins"):
        embeddings.validate_lattice_embedding(3, PATH, [0, 1], WORKING)


def test_clique_embedding_accepts_connected_disjoint_chains_that_all_touch():
    working = embeddings.edge_set([(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)])
    embeddings.validate_clique_embedding([[0, 1], [2], [3, 4]], working)


def test_clique_embedding_rejects_a_shared_qubit():
    with pytest.raises(ValueError, match="qubit 1 is in chain 0 and in chain 1"):
        embeddings.validate_clique_embedding([[0, 1], [1, 2]], WORKING)


def test_clique_embedding_rejects_a_chain_that_is_not_connected():
    with pytest.raises(ValueError, match="chain 0 is not connected"):
        embeddings.validate_clique_embedding([[0, 3], [2]], WORKING)


def test_clique_embedding_rejects_a_pair_without_a_coupler():
    # Qubit 4 touches neither qubit 0 nor qubit 1.
    with pytest.raises(ValueError, match=r"2 chain pairs share no working coupler; first: \(0, 2\)"):
        embeddings.validate_clique_embedding([[0], [1], [4]], WORKING)


def test_resolve_chains_takes_the_majority_and_counts_broken_chains():
    chains = [[10, 11, 12], [20, 21]]
    variables = [10, 11, 12, 20, 21]
    spins = np.array([[1, 1, -1, -1, -1], [-1, -1, -1, 1, -1]], dtype=np.int8)
    logical, broken = embeddings.resolve_chains(spins, variables, chains)
    # Read 0: chain 0 votes +1 and is broken. Read 1: chain 1 ties, and the first qubit decides.
    assert logical.tolist() == [[1, -1], [-1, 1]]
    assert broken == pytest.approx(2 / 4)


def test_clique_round_trip_returns_the_logical_state():
    chains = [[0, 1], [2], [3, 4]]
    working = embeddings.edge_set([(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)])
    h = np.array([0.5, -0.25, 0.1])
    edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    j = np.array([0.3, -0.2, 0.4])
    nodes, phys_h, phys_edges, phys_j, strength = embeddings.embed_clique(h, edges, j, chains, working)
    assert strength > 0
    assert sorted(nodes.tolist()) == [0, 1, 2, 3, 4]
    # Every embedded coupler is a working coupler.
    assert embeddings.edge_set(phys_edges.tolist()) <= working

    logical_state = np.array([[1, -1, 1], [-1, -1, 1]], dtype=np.int8)
    owner = {q: i for i, chain in enumerate(chains) for q in chain}
    physical = np.array([[row[owner[int(q)]] for q in nodes] for row in logical_state], dtype=np.int8)
    back, broken = embeddings.resolve_chains(physical, nodes.tolist(), chains)
    assert back.tolist() == logical_state.tolist()
    assert broken == 0.0

    # With intact chains, the embedded energy is the logical energy plus one constant.
    s = physical.astype(np.float64)
    embedded = s @ phys_h + (s[:, [list(nodes).index(u) for u in phys_edges[:, 0]]]
                             * s[:, [list(nodes).index(v) for v in phys_edges[:, 1]]]) @ phys_j
    t = logical_state.astype(np.float64)
    logical_energy = t @ h + (t[:, edges[:, 0]] * t[:, edges[:, 1]]) @ j
    assert np.ptp(embedded - logical_energy) == pytest.approx(0.0, abs=1e-12)


def _ideal_zephyr(m, t):
    graphs = pytest.importorskip("dwave.graphs")
    graph = graphs.zephyr_graph(m, t)
    return set(graph.nodes), embeddings.edge_set(graph.edges)


def _place_diamond(dims, nodes, couplers):
    """The best placement of a diamond lattice on a Z(4, 4) working graph."""
    edges = lattices.diamond_edges(dims)
    placement = embeddings.best_placement(
        lambda shift: embeddings.diamond_labels(dims, shift, m=4, t=4),
        edges, np.arange(len(edges)), nodes, couplers, max_shift=8,
    )
    return edges, placement


def test_zephyr_label_equals_the_ocean_numbering():
    graphs = pytest.importorskip("dwave.graphs")
    convert = graphs.zephyr_coordinates(3, 4)
    coordinates = list(graphs.zephyr_graph(3, 4, coordinates=True).nodes)
    assert len(coordinates) == 4 * 4 * 3 * 7
    for coordinate in coordinates:
        assert embeddings.zephyr_label(*coordinate, m=3, t=4) == convert.zephyr_to_linear(coordinate)
    assert embeddings.zephyr_label(0, 7, 0, 0, 0, m=3, t=4) == -1      # column 7 is outside Z(3, 4)


@pytest.mark.parametrize("dims", [(4, 4, 8), (6, 6, 16)])
def test_the_diamond_map_puts_every_bond_on_a_coupler_of_the_ideal_graph(dims):
    nodes, couplers = _ideal_zephyr(4, 4)
    edges, placement = _place_diamond(dims, nodes, couplers)
    assert placement.vacant_spins == 0 and placement.dropped_edges == 0
    embeddings.validate_lattice_embedding(len(placement.labels), edges, placement.labels.tolist(), couplers)


@pytest.mark.parametrize("dims", [(3, 3, 2), (4, 4, 8)])
def test_the_cubic_dimer_map_puts_every_edge_on_a_coupler_of_the_ideal_graph(dims):
    nodes, couplers = _ideal_zephyr(4, 4)
    bonds, _ = lattices.cubic_logical_bonds(dims)
    edges, _, made_by = lattices.cubic_dimer_edges(dims, np.full(len(bonds), 1000, dtype=np.int64))
    placement = embeddings.best_placement(
        lambda shift: embeddings.cubic_dimer_labels(dims, shift, m=4, t=4),
        edges, made_by, nodes, couplers, max_shift=8,
    )
    assert placement.vacant_spins == 0 and placement.dropped_edges == 0
    embeddings.validate_lattice_embedding(len(placement.labels), edges, placement.labels.tolist(), couplers)


def test_best_placement_moves_the_lattice_away_from_a_dead_qubit():
    nodes, couplers = _ideal_zephyr(4, 4)
    _, first = _place_diamond((4, 4, 8), nodes, couplers)
    dead = int(first.labels[0])
    live_couplers = {pair for pair in couplers if dead not in pair}
    _, placement = _place_diamond((4, 4, 8), nodes - {dead}, live_couplers)
    assert placement.shift != first.shift
    assert placement.vacant_spins == 0
    assert dead not in placement.labels.tolist()


def test_a_lattice_larger_than_the_graph_is_cut_to_the_shape_of_the_graph():
    nodes, couplers = _ideal_zephyr(4, 4)
    edges, placement = _place_diamond((20, 20, 16), nodes, couplers)
    placed = placement.labels[placement.alive]
    assert len(placement.labels) == 1600 and set(placed.tolist()) == nodes
    dense = np.cumsum(placement.alive) - 1
    embeddings.validate_lattice_embedding(len(placed), dense[edges[placement.keep]], placed.tolist(), couplers)


def test_a_spin_on_a_dead_qubit_is_vacant_and_loses_its_edges():
    labels = np.array([0, 1, 2], dtype=np.int64)
    placement = embeddings.place_lattice(labels, PATH, np.arange(2), {0, 1}, WORKING, (0, 0))
    assert placement.alive.tolist() == [True, True, False]
    assert placement.keep.tolist() == [True, False]
    assert (placement.vacant_spins, placement.dropped_edges) == (1, 1)


def test_a_spin_that_keeps_no_edge_is_vacant():
    # Qubits 1 and 4 share no coupler, so spin 2 loses its only bond.
    labels = np.array([0, 1, 4], dtype=np.int64)
    placement = embeddings.place_lattice(labels, PATH, np.arange(2), {0, 1, 4}, WORKING, (0, 0))
    assert placement.alive.tolist() == [True, True, False]
    assert placement.keep.tolist() == [True, False]


def test_a_dead_dimer_coupler_makes_the_whole_site_vacant_and_a_half_dead_z_bond_drops_both_halves():
    dims = (2, 2, 3)
    n = 12
    bonds, axis = lattices.cubic_logical_bonds(dims)
    edges, _, made_by = lattices.cubic_dimer_edges(dims, np.full(len(bonds), 1000, dtype=np.int64))
    labels = np.arange(2 * n)
    working_nodes = set(labels.tolist())
    working_edges = embeddings.edge_set(edges.tolist())
    working_edges.discard((5, 5 + n))                                     # the dimer of site 5
    z_bond = int(np.flatnonzero(axis == 2)[0])
    first_half = edges[np.flatnonzero(made_by == z_bond)[0]]
    working_edges.discard((int(first_half.min()), int(first_half.max())))
    placement = embeddings.place_lattice(labels, edges, made_by, working_nodes, working_edges, (0, 0))

    assert not placement.alive[5] and not placement.alive[5 + n]
    assert placement.vacant_spins == 2
    assert not placement.keep[made_by == z_bond].any()


def test_a_depth_beyond_the_wires_of_a_line_is_refused():
    with pytest.raises(ValueError, match="more than 4 wires"):
        embeddings.diamond_labels((4, 4, 20), (0, 0))
    with pytest.raises(ValueError, match="more than 4 wires"):
        embeddings.cubic_dimer_labels((3, 3, 9), (0, 0))
