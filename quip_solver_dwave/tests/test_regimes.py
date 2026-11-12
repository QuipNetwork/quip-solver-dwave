"""Regimes: alphabets, encodings, energies, and the clique model."""

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quip_solver_dwave import embeddings, lattices, regimes
from quip_solver_dwave.replay import TopologySpec

# A tiny working graph with shuffled labels, so a mix-up of labels and dense
# indices cannot pass.
SPEC = TopologySpec(
    nodes=np.array([30, 10, 20, 40], dtype=np.int64),
    edges=np.array([[30, 10], [10, 20], [20, 40], [30, 40]], dtype=np.int64),
    dense_edges=np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64),
    allowed_h_milli=[0],
    allowed_j_milli=[-1000, 1000],
)


def test_the_125_level_alphabet_has_250_values_no_zero_and_a_gcd_of_8():
    alphabet = regimes.LEVELS_125_ALPHABET
    assert len(alphabet) == len(set(alphabet)) == 250
    assert 0 not in alphabet
    assert int(np.gcd.reduce(np.abs(alphabet))) == 8
    assert (min(alphabet), max(alphabet)) == (-1000, 1000)


def test_cells_carry_the_kernel_label_of_the_specification():
    assert regimes.native_cell("native-pm1", SPEC).kernel_label == regimes.CPU_MSA
    assert regimes.native_cell("native-125", SPEC).kernel_label == regimes.CPU_SA
    assert regimes.diamond_cell((3, 3, 8)).kernel_label == regimes.CPU_MSA
    assert regimes.cubic_dimer_cell((3, 3, 2)).kernel_label == regimes.CPU_MSA


def test_make_nonces_is_reproducible_and_gives_32_byte_hex():
    first = regimes.make_nonces(5, 11)
    assert first == regimes.make_nonces(5, 11)
    assert len(set(first)) == 5
    assert all(len(bytes.fromhex(nonce)) == 32 for nonce in first)


def test_a_pm1_cell_becomes_one_unit_bond_for_each_edge():
    cell = regimes.diamond_cell((3, 3, 8))
    drawn = np.where(np.arange(len(cell.draw_edges)) % 2 == 0, 1000, -1000).astype(np.int64)
    edges, j_milli = regimes.true_problem(cell, drawn)
    h, k_edges, k_j = regimes.kernel_problem(cell, edges, j_milli)
    assert h.shape == (cell.n_spins,) and not h.any()
    assert k_edges.tolist() == edges.tolist()
    assert k_j.tolist() == (drawn / 1000).tolist()


def test_the_cubic_dimer_encoding_has_unit_500_and_at_most_10_bonds_for_each_spin():
    cell = regimes.cubic_dimer_cell((4, 4, 4))
    drawn = np.full(len(cell.draw_edges), 1000, dtype=np.int64)
    edges, j_milli = regimes.true_problem(cell, drawn)
    _, k_edges, k_j = regimes.kernel_problem(cell, edges, j_milli)
    assert set(np.abs(k_j).tolist()) == {1.0}
    assert len(k_edges) == int((np.abs(j_milli) // 500).sum())
    assert regimes.max_bonds_per_spin(cell.n_spins, k_edges) == 10


def test_the_cubic_dimer_unit_stays_500_for_a_model_with_no_half_bond():
    cell = regimes.cubic_dimer_cell((3, 3, 3))
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    _, k_edges, k_j = regimes.kernel_problem(cell, edges, np.array([1000, -2000], dtype=np.int64))
    assert k_edges.tolist() == [[0, 1]] * 2 + [[1, 2]] * 4
    assert k_j.tolist() == [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]


def test_a_coupling_that_is_not_a_multiple_of_the_unit_bond_is_refused():
    cell = regimes.diamond_cell((3, 3, 8))
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="not a multiple of 1000 milli"):
        regimes.kernel_problem(cell, edges, np.array([1000, 500], dtype=np.int64))


@settings(max_examples=50, deadline=None)
@given(data=st.data())
def test_the_cubic_dimer_kernel_encoding_scores_twice_the_true_energy(data):
    cell = regimes.cubic_dimer_cell((3, 3, 3))
    bonds = len(cell.draw_edges)
    signs = data.draw(st.lists(st.sampled_from([-1000, 1000]), min_size=bonds, max_size=bonds))
    one_read = st.lists(st.sampled_from([-1, 1]), min_size=cell.n_spins, max_size=cell.n_spins)
    spins = np.array(data.draw(st.lists(one_read, min_size=1, max_size=4)), dtype=np.int8)
    edges, j_milli = regimes.true_problem(cell, np.array(signs, dtype=np.int64))
    _, k_edges, k_j = regimes.kernel_problem(cell, edges, j_milli)
    true_energy = regimes.energy_milli(spins, edges, j_milli)
    kernel_energy = regimes.energy_milli(spins, k_edges, np.rint(k_j * 1000).astype(np.int64))
    # The unit is 500 milli, so one unit bond of 1.0 stands for half an energy unit.
    assert kernel_energy.tolist() == (2 * true_energy).tolist()


def test_a_cpu_sa_cell_passes_its_couplings_as_floats():
    cell = regimes.native_cell("native-125", SPEC)
    j_milli = np.array([8, -1000, 504, -16], dtype=np.int64)
    _, k_edges, k_j = regimes.kernel_problem(cell, cell.draw_edges, j_milli)
    assert k_edges.tolist() == SPEC.dense_edges.tolist()
    assert k_j.tolist() == [0.008, -1.0, 0.504, -0.016]


def test_the_qpu_problem_uses_physical_labels():
    j_milli = np.array([1000, -1000, 1000, -1000], dtype=np.int64)
    nodes, h, edges, j = regimes.qpu_problem(SPEC.dense_edges, j_milli, SPEC.nodes)
    assert nodes.tolist() == [30, 10, 20, 40]
    assert not h.any()
    assert edges.tolist() == SPEC.edges.tolist()
    assert j.tolist() == [1.0, -1.0, 1.0, -1.0]


def test_energy_milli_is_the_sum_over_bonds():
    spins = np.array([[1, 1, 1, 1], [1, -1, 1, -1]], dtype=np.int8)
    j_milli = np.array([1000, -1000, 500, 8], dtype=np.int64)
    assert regimes.energy_milli(spins, SPEC.dense_edges, j_milli).tolist() == [508, -508]


def test_the_clique_model_is_reproducible_and_normalized():
    h, edges, j = regimes.clique_model("ab" * 32, 108)
    again = regimes.clique_model("ab" * 32, 108)
    assert np.array_equal(h, again[0]) and np.array_equal(j, again[2])
    assert len(h) == 108 and len(edges) == 5778
    assert max(np.abs(h).max(), np.abs(j).max()) == pytest.approx(1.0)
    other = regimes.clique_model("cd" * 32, 108)
    assert not np.array_equal(h, other[0])


def test_the_lowest_clique_energy_belongs_to_the_best_portfolio_of_k_assets():
    # The portfolio cost of section 5.4 at equal weights, with no Ising form: gamma / 2 * w' Sigma w - mu' w.
    n, k, gamma = 12, 4, 5.0
    states = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    for nonce in regimes.make_nonces(5, 7):
        sigma, returns = regimes.clique_portfolio(nonce, n)
        costs = {}
        for subset in itertools.combinations(range(n), k):
            w = np.bincount(subset, minlength=n) / k
            costs[subset] = gamma / 2 * (w @ sigma @ w) - returns @ w
        h, edges, j = regimes.clique_model(nonce, n=n, k=k)
        lowest = states[int(regimes.energy(states, h, edges, j).argmin())]
        assert tuple(np.flatnonzero(lowest > 0).tolist()) == min(costs, key=lambda subset: costs[subset])


def test_the_kernel_takes_repeated_edges_and_refuses_float_couplings_when_seeded():
    quip_msa = pytest.importorskip("quip_msa")
    kernel = quip_msa.Msa()
    cell = regimes.cubic_dimer_cell((2, 2, 2))
    drawn = regimes.draw_couplings(cell, "11" * 32)
    assert set(np.abs(drawn).tolist()) == {1000}
    edges, j_milli = regimes.true_problem(cell, drawn)
    h, k_edges, k_j = regimes.kernel_problem(cell, edges, j_milli)
    assert regimes.observed_kernel_label(kernel, h, k_edges, k_j) == regimes.CPU_MSA
    spins, reported = kernel.sample(h, k_edges, k_j, num_sweeps=200, num_reads=8, seed=1)
    assert reported.tolist() == (2 * regimes.energy_milli(spins, edges, j_milli)).tolist()

    native = regimes.native_cell("native-125", SPEC)
    floats = regimes.kernel_problem(native, native.draw_edges, regimes.draw_couplings(native, "11" * 32))
    assert regimes.observed_kernel_label(kernel, *floats) == regimes.CPU_SA


def test_native_pm1_is_the_production_draw():
    pytest.importorskip("quip_msa")
    from quip_solver_dwave import replay

    cell = regimes.native_cell("native-pm1", SPEC)
    nonce = "5a" * 32
    _, j = replay.model_from_nonce(SPEC, nonce)
    assert regimes.draw_couplings(cell, nonce).tolist() == np.rint(j * 1000).astype(np.int64).tolist()


def test_a_placement_drops_vacancies_after_the_draw_and_renumbers_the_spins():
    dims = (3, 3, 8)
    edges = lattices.diamond_edges(dims)
    count = lattices.diamond_site_count(dims)
    labels = 100 + np.arange(count)
    working_nodes = set(labels.tolist()) - {100 + 4}                      # spin 4 lands on a dead qubit
    working_edges = embeddings.edge_set(labels[edges].tolist())
    dead_coupler = (int(labels[edges[0][0]]), int(labels[edges[0][1]]))   # and one coupler is dead
    working_edges.discard((min(dead_coupler), max(dead_coupler)))
    placement = embeddings.place_lattice(labels, edges, np.arange(len(edges)), working_nodes, working_edges, (0, 0))
    cell = regimes.diamond_cell(dims, placement)

    assert cell.draw_nodes == count and len(cell.draw_edges) == len(edges)
    assert cell.n_spins == count - 1
    assert 104 not in cell.physical_nodes.tolist()

    drawn = np.where(np.arange(len(edges)) % 2 == 0, 1000, -1000).astype(np.int64)
    kept_edges, kept_j = regimes.true_problem(cell, drawn)
    touches_vacancy = (edges == 4).any(axis=1)
    expected = ~touches_vacancy
    expected[0] = False
    assert len(kept_edges) == int(expected.sum())
    assert kept_j.tolist() == drawn[expected].tolist()
    assert kept_edges.max() == cell.n_spins - 1
    # Every kept edge is a working coupler between the physical nodes of its spins.
    physical = cell.physical_nodes[kept_edges]
    assert embeddings.edge_set(physical.tolist()) <= working_edges



def test_coupling_extremes_sums_the_couplings_on_each_qubit():
    # Qubit 10 carries +1.0 and +0.25. Qubit 40 carries -2.0 and -0.5.
    # The mean square of the four couplings is (1 + 0.0625 + 4 + 0.25) / 4 = 1.328125.
    j = np.array([1.0, 0.25, -2.0, -0.5])
    nodes, h, edges, _ = regimes.qpu_problem(SPEC.dense_edges, np.zeros(4, dtype=np.int64), SPEC.nodes)
    model = regimes.Model(h, SPEC.dense_edges, j, nodes, h, edges, j, SPEC.dense_edges, np.zeros(4, dtype=np.int64))
    extremes = regimes.coupling_extremes(model)
    assert extremes == {
        "j_min": -2.0, "j_max": 1.0, "sum_min": -2.5, "sum_max": 1.25, "h_abs_max": 0.0,
        "j_rms": pytest.approx(1.328125 ** 0.5), "j_abs_min": 0.25,
    }
