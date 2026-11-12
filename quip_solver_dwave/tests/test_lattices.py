"""The lattice ports against fixtures from the authors' own generator."""

import json
from pathlib import Path

import numpy as np
import pytest

from quip_solver_dwave import lattices

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lattices"


@pytest.mark.parametrize("dims", [(3, 3, 8), (4, 4, 8), (6, 6, 16)])
def test_diamond_edges_equal_the_authors_edge_list(dims):
    fixture = json.loads((FIXTURES / "diamond_{}_{}_{}.json".format(*dims)).read_text())
    assert lattices.diamond_edges(dims).tolist() == fixture["edges"]


@pytest.mark.parametrize("dims", [(3, 3, 8), (4, 4, 8), (6, 6, 16)])
def test_diamond_has_degree_four_or_less_and_the_stated_site_count(dims):
    edges = lattices.diamond_edges(dims)
    degree = np.bincount(edges.ravel(), minlength=lattices.diamond_site_count(dims))
    assert degree.max() <= 4
    assert len(degree) == dims[0] * dims[1] * dims[2] // 4
    assert degree.min() >= 1


def test_diamond_rejects_a_depth_that_is_not_a_multiple_of_four():
    with pytest.raises(ValueError, match="multiple of 4"):
        lattices.diamond_edges((4, 4, 6))


@pytest.mark.parametrize("dims", [(3, 2, 2), (3, 3, 3), (4, 3, 5)])
def test_cubic_logical_bonds_equal_the_authors_bond_list(dims):
    fixture = json.loads((FIXTURES / "cubic_dimer_{}_{}_{}.json".format(*dims)).read_text())
    bonds, axis = lattices.cubic_logical_bonds(dims)
    assert bonds.tolist() == fixture["logical_bonds"]
    assert len(axis) == len(bonds)


@pytest.mark.parametrize("dims", [(3, 2, 2), (3, 3, 3), (4, 3, 5)])
def test_cubic_dimer_edges_equal_the_authors_dimer_lattice(dims):
    fixture = json.loads((FIXTURES / "cubic_dimer_{}_{}_{}.json".format(*dims)).read_text())
    bonds, _ = lattices.cubic_logical_bonds(dims)
    # The fixture gave bond k the value 2 (k + 1) in energy units.
    drawn = 2000 * (np.arange(len(bonds), dtype=np.int64) + 1)
    edges, j_milli, made_by = lattices.cubic_dimer_edges(dims, drawn)
    assert (made_by[: dims[0] * dims[1] * dims[2]] == -1).all() and made_by.max() == len(bonds) - 1
    ours = sorted([min(u, v), max(u, v), j / 1000] for (u, v), j in zip(edges.tolist(), j_milli.tolist()))
    assert ours == fixture["physical"]


def test_cubic_dimer_refuses_an_odd_coupling():
    bonds, _ = lattices.cubic_logical_bonds((3, 3, 3))
    drawn = np.full(len(bonds), 1001, dtype=np.int64)
    with pytest.raises(ValueError, match="even"):
        lattices.cubic_dimer_edges((3, 3, 3), drawn)


def test_diamond_sites_give_one_site_for_each_column_in_each_block_of_four_levels():
    sites = lattices.diamond_sites((4, 4, 8))
    assert sites.shape == (32, 3)
    assert len({tuple(s) for s in sites.tolist()}) == 32
    # Level parity and column parity go together: see the module comment.
    for x, y, level in sites.tolist():
        assert (x % 2, y % 2) == [(0, 1), (0, 0), (1, 0), (1, 1)][level % 4]
