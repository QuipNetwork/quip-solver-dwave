"""A solver whose graph is not the network's must say so."""

import logging

import pytest

pytest.importorskip("numpy")  # ocean imports numpy at module load

from quip_solver_dwave.ocean import OceanSampler  # noqa: E402


def _sampler_with_live(nodes, edges, caplog=None):
    """Build a sampler with a known live graph.

    Construction logs its own line ("real sampler deferred"), so the caller
    clears the capture first and every assertion below is about the fit line.
    """
    s = OceanSampler(mock=False)
    if caplog is not None:
        caplog.clear()
    s._live_nodes = list(nodes)
    s._live_edges = list(edges)
    return s


def test_exact_match_is_reported_once(caplog):
    caplog.set_level(logging.DEBUG)
    s = _sampler_with_live([0, 1, 2], [(0, 1), (1, 2)], caplog)
    s.set_session_topology([0, 1, 2], [(0, 1), (1, 2)])
    assert "matches the live graph" in caplog.text
    assert [r.levelname for r in caplog.records] == ["INFO"]


def test_a_few_dead_qubits_warn(caplog):
    """A production chip is missing a handful of qubits: worth a line, not alarm."""
    caplog.set_level(logging.DEBUG)
    live_nodes = list(range(100))
    live_edges = [(i, i + 1) for i in range(99)]
    s = _sampler_with_live(live_nodes[1:], live_edges[1:], caplog)
    s.set_session_topology(live_nodes, live_edges)
    assert [r.levelname for r in caplog.records] == ["WARNING"]
    assert "1/100 nodes" in caplog.text


def test_a_different_chip_is_an_error(caplog):
    """Two thirds of the couplers absent is a wrong-solver signature."""
    caplog.set_level(logging.DEBUG)
    session_nodes = list(range(30))
    session_edges = [(i, (i + 1) % 30) for i in range(30)]
    s = _sampler_with_live(session_nodes, session_edges[:10], caplog)
    s.set_session_topology(session_nodes, session_edges)
    assert [r.levelname for r in caplog.records] == ["ERROR"]
    assert "DWAVE_API_SOLVER" in caplog.text
    assert "20/30 couplers" in caplog.text


def test_missing_qubits_are_still_clamped(caplog):
    """The log is an addition, not a change of behaviour."""
    caplog.set_level(logging.ERROR)
    s = _sampler_with_live([0, 1], [(0, 1)])
    s.set_session_topology([0, 1, 2], [(0, 1), (1, 2)])
    assert s._defective_qubits == [2]
    assert s._defective_edges == set()
