"""Tests for quip_solver_dwave.physical_range: legal physical scale from the live graph.

Task 6 of docs/superpowers/plans/2026-09-22-regime-search-round2.md.
``audit_scale`` finds the highest positive multiplier a physical model's
coefficients can carry without breaking the solver's ``h_range``, its J
range (the extended range when the solver documents one), or a Zephyr
graph's per-group coupling bound, then returns the coefficients scaled to
0.5, 0.75, and 1.0 of that ceiling. A missing solver property or a graph
outside the Zephyr family blocks the audit instead of guessing.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from quip_solver_dwave import physical_range as pr

PROPERTIES = {
    "chip_id": "Advantage2_system1",
    "h_range": [-6.0, 6.0],
    "j_range": [-1.0, 1.0],
    "extended_j_range": [-2.0, 1.0],
    "per_qubit_coupling_range": None,
    "per_group_coupling_range": [-13.0, 10.0],
}

# Real Zephyr(2, 4) coordinates (from dwave_networkx.zephyr_graph(2, t=4)):
# two coupling groups that share no edge. GROUP_POS is the 10-member
# external group of qubit 16; GROUP_NEG is the 9-member external group of
# qubit 17. Confirmed against dwave.system.coupling_groups.coupling_groups
# directly before writing this fixture.
GROUP_POS = [(17, 16), (18, 16), (96, 16), (98, 16), (100, 16),
             (102, 16), (104, 16), (106, 16), (108, 16), (110, 16)]
GROUP_NEG = [(19, 17), (128, 17), (130, 17), (132, 17),
             (134, 17), (136, 17), (138, 17), (140, 17), (142, 17)]


def _zephyr_2_graph(edges):
    nodes = sorted({q for edge in edges for q in edge})
    return pr.build_zephyr_graph(nodes, edges, m=2, t=4)


# ------------------------------------------------------- signed_scale_limit
# The two misreadings the brief calls out by name, verbatim.


def test_positive_limit_stops_diamond_amplification():
    assert pr.signed_scale_limit([-1.0, 1.0], -2.0, 1.0) == 1.0


def test_group_limit_can_bind_before_individual_edges():
    individual = pr.signed_scale_limit([-1.0] * 14, -2.0, 1.0)
    group = pr.signed_scale_limit([-14.0], -13.0, 10.0)
    assert min(individual, group) == 13.0 / 14.0


def test_signed_scale_limit_rejects_a_range_without_finite_bounds_around_zero():
    with pytest.raises(ValueError):
        pr.signed_scale_limit([1.0], 0.0, 1.0)
    with pytest.raises(ValueError):
        pr.signed_scale_limit([1.0], -1.0, math.inf)


def test_signed_scale_limit_rejects_a_nonfinite_coefficient():
    with pytest.raises(ValueError):
        pr.signed_scale_limit([math.inf], -1.0, 1.0)


def test_signed_scale_limit_is_infinite_with_no_signed_values():
    assert pr.signed_scale_limit([0.0, 0.0], -1.0, 1.0) == math.inf


# --------------------------------------------------- build_zephyr_graph / groups
# Compare group membership against the installed Ocean helper on a small,
# correctly attributed Zephyr graph.


def test_build_zephyr_graph_matches_ocean_on_a_real_small_graph():
    dnx = pytest.importorskip("dwave_networkx")
    from dwave.system.coupling_groups import coupling_groups

    reference = dnx.zephyr_graph(2, t=4)
    ours = pr.build_zephyr_graph(list(reference.nodes), list(reference.edges), m=2, t=4)

    def canonical(groups):
        return {frozenset((min(a, b), max(a, b)) for a, b in group) for group in groups}

    assert canonical(coupling_groups(ours)) == canonical(coupling_groups(reference))


def test_a_non_zephyr_graph_yields_no_groups():
    graph = nx.Graph([(0, 1)])
    graph.graph.update(family="pegasus", rows=2)
    assert list(pr._groups_for(graph)) == []


# -------------------------------------------------------------------- audit_scale
# Blocked audits: never a guessed default.


def _zephyr_graph_from_topology():
    edges = GROUP_POS + GROUP_NEG
    return _zephyr_2_graph(edges)


def test_unrecognized_topology_blocks_the_audit():
    graph = nx.Graph([(0, 1)])
    graph.graph.update(family="pegasus", rows=2)
    result = pr.audit_scale(
        np.array([0, 1]), np.array([0.0, 0.0]), np.array([[0, 1]]), np.array([0.5]), graph, PROPERTIES
    )
    assert result["status"] == "blocked"
    assert "topology" in result["reason"]


@pytest.mark.parametrize("missing_key", ["h_range", "j_range_and_extended", "per_group_coupling_range"])
def test_missing_required_property_blocks_the_audit(missing_key):
    graph = _zephyr_graph_from_topology()
    properties = dict(PROPERTIES)
    if missing_key == "j_range_and_extended":
        del properties["j_range"]
        del properties["extended_j_range"]
    else:
        del properties[missing_key]
    result = pr.audit_scale(
        np.array([16, 17]), np.array([0.0, 0.0]), np.array([[16, 17]]), np.array([0.1]), graph, properties
    )
    assert result["status"] == "blocked"
    assert missing_key.split("_and_")[0] in result["reason"] or "j_range" in result["reason"]


def test_a_placement_outside_the_live_graph_blocks_as_topology_drift():
    graph = _zephyr_graph_from_topology()
    result = pr.audit_scale(
        np.array([16, 9999]), np.array([0.1, 0.1]), np.array([[16, 9999]]), np.array([0.1]), graph, PROPERTIES
    )
    assert result["status"] == "blocked"
    assert "drift" in result["reason"]


# ------------------------------------------------------------ ceiling and limits


def test_h_ceiling_and_limiting_constraint():
    graph = _zephyr_graph_from_topology()
    nodes = np.array([16, 17])
    h = np.array([3.0, -6.5])  # -6.5 exceeds h_range's -6.0: the tightest constraint
    edges = np.zeros((0, 2), dtype=np.int64)
    j = np.zeros(0)
    result = pr.audit_scale(nodes, h, edges, j, graph, PROPERTIES)
    assert result["status"] == "ok"
    assert result["ceiling"] == pytest.approx(-6.0 / -6.5)
    assert result["limiting_constraint"]["kind"] == "h"
    assert result["limiting_constraint"]["label"] == 17
    assert result["limiting_constraint"]["bound"] == "lower"


def test_j_ceiling_prefers_extended_j_range():
    graph = _zephyr_graph_from_topology()
    nodes = np.array([16, 18])
    h = np.zeros(2)
    edges = np.array([[16, 18]])
    j = np.array([-1.8])  # within extended (-2, 1) but outside plain j_range (-1, 1)
    result = pr.audit_scale(nodes, h, edges, j, graph, PROPERTIES)
    assert result["status"] == "ok"
    assert result["j_range_used"] == "extended_j_range"
    assert result["limiting_constraint"]["kind"] == "j"
    assert result["ceiling"] == pytest.approx(-2.0 / -1.8)


def test_ordinary_j_range_is_used_and_recorded_when_extended_is_absent():
    graph = _zephyr_graph_from_topology()
    properties = dict(PROPERTIES)
    del properties["extended_j_range"]
    nodes = np.array([16, 18])
    h = np.zeros(2)
    edges = np.array([[16, 18]])
    j = np.array([0.5])
    result = pr.audit_scale(nodes, h, edges, j, graph, properties)
    assert result["status"] == "ok"
    assert result["j_range_used"] == "j_range"


# --------------------------------------------------- coupling-group fixtures
# Positive and negative groups, zero values, and nonconsecutive physical
# labels, all in one model. A per-qubit-style sum (bounded generously here)
# would pass; the tighter, correctly computed group sum must still bind.


def test_group_binds_even_when_a_per_qubit_style_sum_would_pass():
    graph = _zephyr_graph_from_topology()
    nodes = np.array(sorted({q for edge in GROUP_POS + GROUP_NEG for q in edge}))
    h = np.zeros(len(nodes))
    edges = np.array(GROUP_POS + GROUP_NEG + [(16, 17)])  # (16, 17) is a zero-value, non-member edge
    j = np.array([0.4] * len(GROUP_POS) + [-0.4] * len(GROUP_NEG) + [0.0])
    properties = dict(PROPERTIES)
    properties["j_range"] = [-5.0, 5.0]
    properties["extended_j_range"] = [-5.0, 5.0]
    properties["per_group_coupling_range"] = [-3.0, 3.5]
    properties["per_qubit_coupling_range"] = [-100.0, 100.0]  # generous: would not bind alone

    result = pr.audit_scale(nodes, h, edges, j, graph, properties)

    assert result["status"] == "ok"
    # GROUP_NEG sums to -3.6: limit -3.0 / -3.6 = 0.8333...; GROUP_POS sums to
    # 4.0: limit 3.5 / 4.0 = 0.875. The negative group is tighter and wins.
    assert result["limiting_constraint"]["kind"] == "group"
    assert result["limiting_constraint"]["bound"] == "lower"
    assert result["limiting_constraint"]["value"] == pytest.approx(-3.6)
    assert result["ceiling"] == pytest.approx(3.0 / 3.6)


def test_dropping_the_group_check_changes_the_ceiling():
    """Mutation guard: a per-qubit-sum stand-in must not reproduce this ceiling."""
    graph = _zephyr_graph_from_topology()
    nodes = np.array(sorted({q for edge in GROUP_POS + GROUP_NEG for q in edge}))
    h = np.zeros(len(nodes))
    edges = np.array(GROUP_POS + GROUP_NEG)
    j = np.array([0.4] * len(GROUP_POS) + [-0.4] * len(GROUP_NEG))
    properties = dict(PROPERTIES)
    properties["j_range"] = [-5.0, 5.0]
    properties["extended_j_range"] = [-5.0, 5.0]
    properties["per_group_coupling_range"] = [-3.0, 3.5]
    result = pr.audit_scale(nodes, h, edges, j, graph, properties)

    # A naive per-qubit sum (touching qubit 17 sums 0.4 - 3.6 = -3.2) is a
    # different number from the correct group sum (-3.6) and would give a
    # different, wrong ceiling if it were substituted for the group check.
    naive_qubit_17_sum = 0.4 - 0.4 * 9
    wrong_ceiling = pr.signed_scale_limit([naive_qubit_17_sum], -3.0, 3.5)
    assert result["ceiling"] != pytest.approx(wrong_ceiling)
    assert result["ceiling"] == pytest.approx(3.0 / 3.6)


# --------------------------------------------------------------- scale generation


def test_nonzero_model_gets_three_scales_at_the_documented_fractions():
    graph = _zephyr_graph_from_topology()
    nodes = np.array([16, 18])
    h = np.zeros(2)
    edges = np.array([[16, 18]])
    j = np.array([-1.8])
    result = pr.audit_scale(nodes, h, edges, j, graph, PROPERTIES)
    scales = result["scales"]
    assert [s["requested"] for s in scales] == list(pr.SCALE_FRACTIONS)
    ceiling = result["ceiling"]
    base = math.nextafter(ceiling, 0.0)
    for fraction, entry in zip(pr.SCALE_FRACTIONS, scales):
        assert entry["submitted_scale"] == pytest.approx(fraction * base, rel=1e-12)
        assert entry["j"][0] == pytest.approx(j[0] * entry["submitted_scale"])
        # Every scaled bound must hold, not just the individual J bound.
        assert PROPERTIES["extended_j_range"][0] <= entry["j"][0] <= PROPERTIES["extended_j_range"][1]


def test_all_zero_model_records_a_single_scale_of_one():
    graph = _zephyr_graph_from_topology()
    nodes = np.array([16, 18])
    h = np.zeros(2)
    edges = np.array([[16, 18]])
    j = np.zeros(1)
    result = pr.audit_scale(nodes, h, edges, j, graph, PROPERTIES)
    assert result["all_zero"] is True
    assert result["scales"] == [
        {"requested": 1.0, "submitted_scale": 1.0, "h": pytest.approx(h), "j": pytest.approx(j)}
    ]


def test_submitted_coefficients_divide_back_to_the_same_scale_including_chain_couplers():
    """Item 6: dividing each nonzero submitted coefficient by its original gives
    the same scale, for logical couplers and chain couplers alike."""
    graph = _zephyr_graph_from_topology()
    nodes = np.array([16, 17, 18])
    h = np.array([0.2, 0.0, -0.3])
    # (16, 17) plays the role of a chain coupler: a large, negative,
    # embedding-only coefficient that must scale uniformly with the rest.
    edges = np.array([[16, 18], [16, 17]])
    j = np.array([0.5, -3.0])
    result = pr.audit_scale(nodes, h, edges, j, graph, PROPERTIES)
    for entry in result["scales"]:
        scale = entry["submitted_scale"]
        for original, submitted in zip(h, entry["h"]):
            if original != 0.0:
                assert submitted / original == pytest.approx(scale, rel=1e-9)
        for original, submitted in zip(j, entry["j"]):
            if original != 0.0:
                assert submitted / original == pytest.approx(scale, rel=1e-9)


# ------------------------------------------------------------- _shrink_until_valid


def test_shrink_until_valid_moves_down_by_one_ulp_when_needed():
    start = 1.0

    def violates(scale):
        return scale >= start

    result = pr._shrink_until_valid(start, violates)
    assert result == math.nextafter(start, 0.0)


def test_shrink_until_valid_gives_up_eventually():
    with pytest.raises(ValueError):
        pr._shrink_until_valid(1.0, lambda scale: True, max_tries=10)
