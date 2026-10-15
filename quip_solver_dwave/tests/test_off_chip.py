"""A graph the chip cannot hold is refused as a topology mismatch, and the plan
cache never hands one graph another graph's plan."""

from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pytest
from quip_solver_core import miner_pb2, wire

from quip_solver_dwave.job import handle_job
from quip_solver_dwave.ocean import OceanSampler
from quip_solver_dwave.qp import OffChipError, QpEncoder

# A four-qubit ring: every qubit has degree two.
_RING_QUBITS = [0, 1, 2, 3]
_RING_COUPLERS = [(0, 1), (1, 2), (2, 3), (0, 3)]


class _Identity:
    def dict(self):
        return {"name": "FakeSolver", "version": {"graph_id": "ring"}}


class _Solver:
    def __init__(self):
        self._encoding_qubits = _RING_QUBITS
        self._encoding_couplers = _RING_COUPLERS
        self._params: dict = {}
        self.parameters = {"num_reads": None, "annealing_time": None, "label": None}
        self.identity = _Identity()

    def _format_params(self, type_, params):
        pass


def _ring_sampler():
    """A sampler on the real submit path against the ring, cloud stubbed out."""
    s = OceanSampler(mock=False)
    s._connected = True
    s._is_mock = False

    class _Sampler:
        solver = _Solver()

    s.sampler = _Sampler()

    def _submit_encoded(solver, body, cancel_key):
        raise AssertionError("an off-chip graph must not reach SAPI")

    s._submit_encoded = _submit_encoded  # type: ignore[method-assign]
    return s


def test_an_off_chip_coupler_is_an_off_chip_error():
    encoder = QpEncoder(_RING_QUBITS, _RING_COUPLERS)
    with pytest.raises(OffChipError, match="coupler"):
        encoder.plan(np.array(_RING_QUBITS), np.array([(0, 2)]))


def test_an_off_chip_qubit_is_an_off_chip_error():
    encoder = QpEncoder(_RING_QUBITS, _RING_COUPLERS)
    with pytest.raises(OffChipError, match="qubit 7"):
        encoder.plan(np.array([0, 7]), np.empty((0, 2), dtype=np.int64))


def test_an_off_chip_error_is_still_a_value_error():
    # Callers that caught the old ValueError keep working.
    assert issubclass(OffChipError, ValueError)


def _inline_job(n: int) -> miner_pb2.Job:
    edges = list(itertools.combinations(range(n), 2))
    return miner_pb2.Job(
        job_id=b"k-inline",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(
                u=[u for u, _ in edges], v=[v for _, v in edges]
            ),
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([0] * n),
            j=wire.encode_i32_le([1000] * len(edges)),
            num_reads=1,
        ),
    )


def test_an_inline_k28_is_refused_as_a_topology_mismatch_without_a_traceback(caplog):
    with caplog.at_level(logging.WARNING):
        msgs = handle_job(
            _inline_job(28), _ring_sampler(), session_nodes=[], session_edges=[]
        )

    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[0].reject.reason == miner_pb2.TOPOLOGY_MISMATCH
    assert msgs[1].job_request.credits == 1
    assert not any(r.exc_info for r in caplog.records), "expected input, not a crash"
    assert any("TOPOLOGY_MISMATCH" in r.getMessage() for r in caplog.records)


def test_a_graph_with_the_same_counts_but_other_edges_gets_its_own_plan():
    s = _ring_sampler()
    solver = _Solver()
    nodes = np.array(_RING_QUBITS)
    first = np.array([(0, 1), (1, 2)])
    second = np.array([(2, 3), (0, 3)])

    _, first_plan = s._graph_plan(solver, nodes, first)
    _, second_plan = s._graph_plan(solver, nodes, second)

    fresh = QpEncoder(_RING_QUBITS, _RING_COUPLERS).plan(nodes, second)
    assert second_plan is not first_plan
    assert second_plan.quad_slots.tolist() == fresh.quad_slots.tolist()


def test_the_same_graph_reuses_its_plan():
    s = _ring_sampler()
    solver = _Solver()
    nodes = np.array(_RING_QUBITS)
    edges = np.array([(0, 1), (1, 2)])

    _, first = s._graph_plan(solver, nodes, edges)
    _, again = s._graph_plan(solver, nodes.copy(), edges.copy())

    assert again is first
