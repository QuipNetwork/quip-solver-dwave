"""Missing couplers must be reduced out, with or without missing qubits.

A chip whose qubits all match the session topology but whose coupler set does
not is the ordinary case for a correctly pinned solver. Submitting a coupler
the QPU does not have makes SAPI reject the entire problem with
ProblemStructureError, so every job fails.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")

from quip_solver_dwave.defects import (  # noqa: E402
    prepare_problem,
    reconstruct_sample,
)


def test_coupler_only_defects_are_removed_without_a_seed():
    h = {0: 1.0, 1: -1.0, 2: 0.5}
    j = {(0, 1): 1.0, (1, 2): -1.0}
    h_r, j_r, info = prepare_problem(
        h, j, defective_qubits=(), defective_edges={(1, 2)}, nonce_seed=None
    )
    assert j_r == {(0, 1): 1.0}
    assert h_r == h
    assert info is not None
    assert info.removed_edges == {(1, 2): -1.0}


def test_coupler_only_defects_are_removed_with_a_seed():
    j = {(0, 1): 1.0, (1, 2): -1.0}
    _, j_r, _ = prepare_problem(
        {}, j, defective_qubits=(), defective_edges={(1, 2)}, nonce_seed=b"\x01"
    )
    assert j_r == {(0, 1): 1.0}


def test_removed_coupler_energy_is_scored_back():
    """The QPU never annealed the removed coupler, but the reported energy
    must still describe the full topology."""
    j = {(0, 1): 1.0, (1, 2): -1.0}
    _, j_r, info = prepare_problem(
        {}, j, defective_qubits=(), defective_edges={(1, 2)}, nonce_seed=None
    )
    assert j_r == {(0, 1): 1.0}
    spins = {0: 1, 1: -1, 2: 1}
    full, energy = reconstruct_sample(spins, reduced_energy=-1.0, defect_info=info)
    # reduced energy plus J(1,2)*s1*s2 = -1.0 + (-1.0 * -1 * 1) = 0.0
    assert full == spins
    assert energy == pytest.approx(0.0)


def test_qubit_defects_still_require_a_seed():
    """Clamping picks spins from the seed; a missing seed is a caller error,
    not a silent pass-through of an unsubmittable problem."""
    with pytest.raises(ValueError, match="nonce seed"):
        prepare_problem(
            {0: 1.0}, {(0, 1): 1.0}, defective_qubits=(1,), nonce_seed=None
        )


def test_no_defects_is_still_a_pass_through():
    h, j = {0: 1.0}, {(0, 1): 1.0}
    h_r, j_r, info = prepare_problem(h, j, defective_qubits=(), defective_edges=set())
    assert (h_r, j_r, info) == (h, j, None)


def test_sample_submits_only_couplers_the_chip_has(monkeypatch):
    """The regression: with every qubit present and one coupler missing, the
    problem handed to the encoder must not carry the missing coupler."""
    import numpy as np

    from quip_solver_dwave.ocean import OceanSampler

    s = OceanSampler(mock=False)
    s._live_nodes = [0, 1, 2]
    s._live_edges = [(0, 1)]
    s.set_session_topology([0, 1, 2], [(0, 1), (1, 2)])
    submitted = []

    def submit_sync(nodes, h, edges, j, *rest):
        # The last step before the encoder and SAPI. The answer is one read
        # of the reduced problem, shaped like the SDK's future.
        submitted.append(([tuple(e) for e in np.asarray(edges).tolist()], np.asarray(j).tolist()))
        return SimpleNamespace(
            samples=np.array([[1, -1, 1]], dtype=np.int8),
            variables=[0, 1, 2],
            energies=[-1.0],
            num_occurrences=[1],
            timing={"qpu_programming_time": 1_000, "qpu_sampling_time": 100},
        )

    monkeypatch.setattr(s, "_submit_sync", submit_sync)
    result = s.sample(
        np.array([0, 1, 2]),
        np.array([1.0, 0.0, -1.0]),
        np.array([(0, 1), (1, 2)]),
        np.array([1.0, -1.0]),
        nonce_seed=b"\x07",
    )
    s.close()

    # (1, 2) is not on the chip. Submitting it makes SAPI reject the whole
    # problem, so it must be gone from what the encoder sees.
    assert submitted == [([(0, 1)], [1.0])]
    # The removed coupler is scored back: -1.0 + J(1,2) * s1 * s2 = 0.0.
    assert result.energies == pytest.approx([0.0])
