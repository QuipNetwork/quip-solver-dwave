import numpy as np
import pytest

from quip_solver_dwave import replay


def spec_of(nodes):
    return replay.TopologySpec(
        nodes=np.array(nodes, dtype=np.int64),
        edges=np.zeros((0, 2), dtype=np.int64),
        dense_edges=np.zeros((0, 2), dtype=np.int64),
        allowed_h_milli=[0],
        allowed_j_milli=[-1000, 1000],
    )


def test_spec_order_moves_each_qubit_column_to_its_place_in_the_spec():
    spins = np.array([[1, -1, 1], [-1, -1, 1]], dtype=np.int8)
    ordered = replay.spec_order(spins, [30, 10, 20], spec_of([10, 20, 30]))
    assert ordered.tolist() == [[-1, 1, 1], [-1, 1, -1]]
    assert ordered.dtype == np.int8


def test_spec_order_refuses_reads_that_lack_a_spec_node():
    spins = np.ones((1, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="30"):
        replay.spec_order(spins, [10, 20], spec_of([10, 20, 30]))
