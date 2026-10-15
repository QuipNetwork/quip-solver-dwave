"""The submission payload is built from numpy, byte for byte.

Submitting one job costs ~27.5 ms of GIL-bound CPU before anything reaches the
network: ~13 ms building h/J dicts from the wire arrays, then ~14.5 ms inside
the Ocean SDK turning those dicts straight back into two dense float64 arrays.
Both dicts exist only to be walked once in solver-encoding order.

The encoding order is fixed for the life of a session, so the mapping from the
job's edge order into it is a permutation that can be computed once. After
that, encoding a job is two scatters and a base64 of the raw buffer.

Every test here is the same assertion: the bytes match what
``dwave.cloud.coders.encode_problem_as_qp`` produces. That function is the
specification, and pinning against it is what makes bypassing it safe.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dwave.cloud.coders import encode_problem_as_qp

from quip_solver_dwave.qp import QpEncoder


class _Solver:
    """The only two attributes encode_problem_as_qp reads off a solver."""

    def __init__(self, qubits, couplers):
        self._encoding_qubits = list(qubits)
        self._encoding_couplers = [tuple(c) for c in couplers]


def _reference(solver, nodes, h, edges, j):
    linear = {int(n): float(b) for n, b in zip(nodes, h)}
    quadratic = {(int(u), int(v)): float(b) for (u, v), b in zip(edges, j)}
    return encode_problem_as_qp(solver, linear, quadratic, 0.0)


def _encoded(solver, nodes, h, edges, j):
    enc = QpEncoder(solver._encoding_qubits, solver._encoding_couplers)
    plan = enc.plan(
        np.asarray(nodes, dtype=np.int64),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )
    return enc.encode(
        plan, np.asarray(h, dtype=np.float64), np.asarray(j, dtype=np.float64)
    )


def _assert_same(solver, nodes, h, edges, j):
    ref = _reference(solver, nodes, h, edges, j)
    got = _encoded(solver, nodes, h, edges, j)
    assert got["format"] == ref["format"]
    assert got["offset"] == ref.get("offset")
    assert got["lin"] == ref["lin"]
    assert got["quad"] == ref["quad"]


def test_a_two_qubit_problem_encodes_identically():
    solver = _Solver([0, 1], [(0, 1)])
    _assert_same(solver, [0, 1], [1.0, -1.0], [(0, 1)], [0.5])


def test_the_coupler_order_is_the_solvers_not_ours():
    # _encoding_couplers is not sorted on a real solver, and the payload is
    # positional, so a coupler written at the wrong index is a wrong problem.
    solver = _Solver([0, 1, 2], [(1, 2), (0, 1), (0, 2)])
    _assert_same(solver, [0, 1, 2], [0.0, 0.0, 0.0], [(0, 1), (0, 2), (1, 2)], [1.0, 2.0, 3.0])


def test_a_coupler_given_in_the_opposite_orientation_lands_in_the_same_slot():
    # The reference sums both orientations; the chip has one undirected edge.
    solver = _Solver([0, 1], [(0, 1)])
    _assert_same(solver, [0, 1], [0.0, 0.0], [(1, 0)], [0.75])


def test_a_qubit_the_solver_has_but_the_problem_does_not_encodes_as_nan():
    # Inactive qubits are NaN, not zero: the reference distinguishes "not in
    # this problem" from "in the problem with zero bias".
    solver = _Solver([0, 1, 2], [(0, 1), (1, 2)])
    _assert_same(solver, [0, 1], [1.0, 2.0], [(0, 1)], [1.0])


def test_couplers_touching_an_inactive_qubit_are_dropped_not_zeroed():
    # The reference filters them out entirely, which shortens the array. An
    # encoder that wrote zeros instead would produce a longer payload.
    solver = _Solver([0, 1, 2], [(0, 1), (1, 2), (0, 2)])
    ref = _reference(solver, [0, 1], [1.0, 2.0], [(0, 1)], [1.0])
    got = _encoded(solver, [0, 1], [1.0, 2.0], [(0, 1)], [1.0])
    assert got["quad"] == ref["quad"]


def test_an_isolated_qubit_with_a_bias_stays_active():
    solver = _Solver([0, 1, 2], [(0, 1), (1, 2)])
    _assert_same(solver, [0, 1, 2], [1.0, 0.0, -3.0], [(0, 1)], [1.0])


def test_negative_and_fractional_biases_survive_the_round_trip():
    solver = _Solver([0, 1, 2], [(0, 1), (1, 2)])
    _assert_same(
        solver, [0, 1, 2], [-0.125, 0.0, 1e-9], [(0, 1), (1, 2)], [-1.5, 2.25]
    )


def test_an_empty_problem_encodes_identically():
    solver = _Solver([0, 1], [(0, 1)])
    _assert_same(solver, [], [], [], [])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_problems_match_the_reference(seed):
    rng = np.random.default_rng(seed)
    n = 60
    qubits = list(range(n))
    couplers = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2]
    rng.shuffle(couplers)
    solver = _Solver(qubits, couplers)

    keep = rng.random(len(couplers)) < 0.7
    edges = [c for c, k in zip(couplers, keep) if k]
    rng.shuffle(edges)
    nodes = sorted({q for e in edges for q in e})
    h = rng.normal(size=len(nodes))
    j = rng.normal(size=len(edges))

    _assert_same(solver, nodes, h, edges, j)


def test_the_permutation_is_reusable_across_jobs():
    # The whole point: build the mapping once, then encode many jobs. A second
    # call must not be polluted by the first one's values.
    solver = _Solver([0, 1, 2], [(0, 1), (1, 2)])
    enc = QpEncoder(solver._encoding_qubits, solver._encoding_couplers)
    nodes = np.array([0, 1, 2])
    edges = np.array([(0, 1), (1, 2)])
    plan = enc.plan(nodes, edges)

    first = enc.encode(plan, np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
    second = enc.encode(plan, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0]))

    assert first["lin"] == _reference(solver, [0, 1, 2], [1.0, 2.0, 3.0], [(0, 1), (1, 2)], [1.0, 2.0])["lin"]
    assert second["lin"] == _reference(solver, [0, 1, 2], [0.0, 0.0, 0.0], [(0, 1), (1, 2)], [0.0, 0.0])["lin"]
    assert first["quad"] != second["quad"]


def test_an_edge_the_solver_does_not_have_is_rejected():
    # Submitting it would make SAPI reject the whole problem. Failing here
    # names the coupler instead of returning a ProblemStructureError later.
    solver = _Solver([0, 1, 2], [(0, 1)])
    enc = QpEncoder(solver._encoding_qubits, solver._encoding_couplers)

    with pytest.raises(ValueError, match="not on the solver"):
        enc.plan(np.array([0, 1, 2]), np.array([(1, 2)]))


def test_nan_padding_is_real_nan_not_a_sentinel():
    # Guards against encoding 0.0 for an inactive qubit, which would silently
    # add a qubit to the problem.
    import base64
    import struct

    solver = _Solver([0, 1, 2], [(0, 1)])
    got = _encoded(solver, [0, 1], [1.0, 2.0], [(0, 1)], [1.0])
    vals = struct.unpack("<3d", base64.b64decode(str(got["lin"])))
    assert vals[0] == 1.0 and vals[1] == 2.0
    assert math.isnan(vals[2])


# --- the guard that matters: same model, at production scale ---------------


def _pegasus_like(n_qubits=4577, n_couplers=41514, seed=99):
    """A solver the size and shape of Advantage2_system1.

    Not a real chip graph, but it reproduces the two properties that break
    naive encoders: the coupler order is not sorted, and the coupler count is
    large enough that an off-by-one permutation is invisible except in the
    bytes.
    """
    rng = np.random.default_rng(seed)
    qubits = list(range(n_qubits))
    seen, couplers = set(), []
    while len(couplers) < n_couplers:
        u, v = int(rng.integers(n_qubits)), int(rng.integers(n_qubits))
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        couplers.append(key)
    rng.shuffle(couplers)
    return _Solver(qubits, couplers)


def test_production_scale_model_is_byte_identical_to_the_sdk():
    # The contract: the miner must submit exactly the model Ocean would have
    # submitted. 4577 nodes and 41514 couplers, the solver's own unsorted
    # order, and the job's edges shuffled against it.
    solver = _pegasus_like()
    rng = np.random.default_rng(4)
    edges = list(solver._encoding_couplers)
    rng.shuffle(edges)
    nodes = sorted({q for e in edges for q in e})
    h = rng.normal(size=len(nodes))
    j = rng.choice(np.array([-1.0, 1.0]), size=len(edges))

    ref = _reference(solver, nodes, h, edges, j)
    got = _encoded(solver, nodes, h, edges, j)

    assert got["lin"] == ref["lin"]
    assert got["quad"] == ref["quad"]
    assert len(str(ref["quad"])) > 400_000  # the payload really is production-sized


def test_production_scale_holds_on_a_partial_subgraph():
    # A job on half the chip. Qubits outside it are inactive, so they encode
    # as NaN and every coupler touching them drops out, shortening the
    # quadratic payload. Getting that filter wrong shifts every later value.
    solver = _pegasus_like()
    rng = np.random.default_rng(5)
    half = set(range(len(solver._encoding_qubits) // 2))
    edges = [e for e in solver._encoding_couplers if e[0] in half and e[1] in half]
    rng.shuffle(edges)
    nodes = sorted({q for e in edges for q in e})
    h = rng.normal(size=len(nodes))
    j = rng.normal(size=len(edges))

    ref = _reference(solver, nodes, h, edges, j)
    got = _encoded(solver, nodes, h, edges, j)

    assert got["lin"] == ref["lin"]
    assert got["quad"] == ref["quad"]
    # The filter really fired: fewer couplers encoded than the chip has.
    assert len(str(got["quad"])) < len(solver._encoding_couplers) * 8


# --- gapped qubit lists: real chips have dead qubits -----------------------


def test_a_qubit_the_solver_does_not_have_is_rejected():
    # Advantage2_system1 reports 4577 qubits with a maximum label of 4799, so
    # the label space has holes. A label in a hole has no slot, and writing it
    # anyway lands on lin[-1] and silently corrupts the last qubit's bias.
    solver = _Solver([0, 2], [(0, 2)])
    enc = QpEncoder(solver._encoding_qubits, solver._encoding_couplers)

    with pytest.raises(ValueError, match="not on the solver"):
        enc.plan(np.array([0, 1, 2]), np.array([(0, 2)]))


def test_a_qubit_label_past_the_end_is_rejected_too():
    solver = _Solver([0, 1, 2], [(0, 1)])
    enc = QpEncoder(solver._encoding_qubits, solver._encoding_couplers)

    with pytest.raises(ValueError, match="not on the solver"):
        enc.plan(np.array([0, 1, 99]), np.array([(0, 1)]))


def test_a_gapped_qubit_list_still_encodes_identically():
    # The labels are sparse but every one of them is real, so this must work
    # and must match the reference.
    solver = _Solver([0, 2, 5], [(0, 2), (2, 5)])
    _assert_same(solver, [0, 2, 5], [1.0, 2.0, 3.0], [(0, 2), (2, 5)], [1.0, 2.0])


def test_an_edge_naming_a_missing_qubit_is_rejected():
    solver = _Solver([0, 2], [(0, 2)])
    enc = QpEncoder(solver._encoding_qubits, solver._encoding_couplers)

    with pytest.raises(ValueError, match="not on the solver"):
        enc.plan(np.array([0, 2]), np.array([(0, 1)]))
