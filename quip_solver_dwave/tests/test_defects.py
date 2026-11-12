"""Defect clamping unit tests."""
import numpy as np

from quip_solver_dwave.defects import (
    DefectInfo,
    clamp_fixed_variables,
    prepare_problem,
    reconstruct_sample,
)


def test_no_defects_passthrough():
    h = {0: 1.0, 1: -1.0}
    j = {(0, 1): 0.5}
    h2, j2, info = prepare_problem(h, j)
    assert h2 == h
    assert j2 == j
    assert info is None


def test_clamp_and_reconstruct_energy():
    h = {0: 1.0, 1: -0.5, 2: 0.25}
    j = {(0, 1): 0.5, (1, 2): -0.3}
    h_r, j_r, fixed, offset, removed = clamp_fixed_variables(
        h, j, nonce_seed=42, defective_qubits=[1], defective_edges=set()
    )
    assert 1 not in h_r
    assert (0, 1) not in j_r
    assert (1, 2) not in j_r
    assert 1 in fixed
    # Reduced problem on remaining qubits
    reduced_sample = {0: 1, 2: -1}
    # Dummy reduced energy (QPU would return this)
    e_red = h_r.get(0, 0) * 1 + h_r.get(2, 0) * (-1)
    full, e_full = reconstruct_sample(
        reduced_sample,
        e_red,
        DefectInfo(fixed, offset, removed),
    )
    assert 1 in full
    # Full energy should match direct evaluation
    e_direct = sum(h[i] * full[i] for i in full)
    for (u, v), val in j.items():
        e_direct += val * full[u] * full[v]
    assert abs(e_full - e_direct) < 1e-9


def test_a_warm_start_clamps_a_missing_qubit_to_its_own_spin_not_the_seeded_draw():
    # J(1,2) = 1 and qubit 2 is missing. The reduced problem folds the clamped
    # spin into qubit 1's bias, so it has to be the spin the start state holds
    # there, or the QPU refines a state that disagrees with its own problem.
    h = {0: 0.0, 1: 0.0, 2: 0.0}
    j = {(0, 1): 1.0, (1, 2): 1.0}
    for spin in (-1, 1):
        h_r, j_r, info = prepare_problem(
            h,
            j,
            defective_qubits=[2],
            nonce_seed=b"\x07",
            start_state={0: 1, 1: -1, 2: spin},
        )
        assert info is not None
        assert info.fixed_spins == {2: spin}
        assert h_r == {0: 0.0, 1: float(spin)}
        assert j_r == {(0, 1): 1.0}


def test_a_warm_start_needs_no_seed_to_clamp_a_missing_qubit():
    _, _, info = prepare_problem(
        {0: 0.0, 1: 0.0},
        {(0, 1): 1.0},
        defective_qubits=[1],
        start_state={0: 1, 1: -1},
    )
    assert info is not None and info.fixed_spins == {1: -1}


def test_a_defective_qubit_absent_from_the_start_state_does_not_raise():
    # A session-wide defect outside this job's own graph (e.g. an inline
    # EdgeList subgraph) has no spin in start_state. The reduction never
    # reads it, so it must clamp deterministically rather than crash.
    h_r, j_r, info = prepare_problem(
        {0: 0.0, 1: 0.0},
        {(0, 1): 1.0},
        defective_qubits=[5],
        start_state={0: 1, 1: -1},
    )
    assert info is not None
    assert 5 in info.fixed_spins
    # The fallback matches the cold (seedless) path for the same qubit.
    rng = np.random.default_rng(0)
    expected = int(2 * rng.integers(2) - 1)
    assert info.fixed_spins[5] == expected
    # Qubit 5 has no couplings in this problem, so clamping it changes
    # nothing else; the reduced problem is passed through untouched.
    assert h_r == {0: 0.0, 1: 0.0}
    assert j_r == {(0, 1): 1.0}
