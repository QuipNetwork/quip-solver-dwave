"""The advertised native topology hash is the SDK's FAM-19 digest."""
from quip_solver_core import session

from quip_solver_dwave.ocean import OceanSampler
from quip_solver_dwave.session_loop import capabilities_message

NODES = [0, 1, 2, 3]
EDGES = [(0, 1), (1, 2), (2, 3)]


def _mock_sampler_with_topology(nodes, edges) -> OceanSampler:
    sampler = OceanSampler(mock=True)
    sampler.set_session_topology(nodes, edges)
    return sampler


def test_advertised_hash_equals_the_sdk_digest_for_the_same_graph():
    advertised = _mock_sampler_with_topology(NODES, EDGES).native_topology_hash
    assert advertised is not None and len(advertised) == 32
    assert advertised == session.native_topology_hash(NODES, EDGES)


def test_node_order_and_edge_direction_do_not_change_the_advertised_hash():
    canonical = _mock_sampler_with_topology(NODES, EDGES)
    shuffled = _mock_sampler_with_topology([3, 1, 0, 2], [(3, 2), (1, 0), (2, 1)])
    assert shuffled.native_topology_hash == canonical.native_topology_hash


def test_a_different_graph_advertises_a_different_hash():
    full = _mock_sampler_with_topology(NODES, EDGES)
    smaller = _mock_sampler_with_topology(NODES[:3], EDGES[:2])
    assert smaller.native_topology_hash != full.native_topology_hash


def test_capabilities_carry_the_hash_only_when_the_session_has_one():
    sampler = _mock_sampler_with_topology(NODES, EDGES)
    assert capabilities_message(sampler.native_topology_hash).native_topology_hash == (
        session.native_topology_hash(NODES, EDGES)
    )
    # --capabilities runs with the device closed and advertises no graph.
    assert not capabilities_message().HasField("native_topology_hash")
