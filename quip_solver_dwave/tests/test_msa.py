"""The fill rule for a seeded MSA-heavy job, and the optional kernel import."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from quip_solver_dwave.msa import LANES, MsaUnavailable, load_kernel, pack_lanes


def _states(rng, count: int, width: int = 16) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(count, width))


def test_qpu_reads_go_first_best_first_and_msa_lite_fills_the_rest():
    rng = np.random.default_rng(1)
    qpu, lite = _states(rng, 3), _states(rng, 4)
    states, qpu_count = pack_lanes(
        qpu, np.array([-5.0, -9.0, -7.0]), lite, np.array([-1, -4, -2, -3]), lanes=5
    )

    assert qpu_count == 3
    assert states.dtype == np.int8 and states.shape == (5, 16)
    # QPU rows by energy: index 1 (-9), 2 (-7), 0 (-5).
    assert np.array_equal(states[:3], qpu[[1, 2, 0]])
    # Then the two best MSA-lite rows: index 1 (-4), 3 (-3).
    assert np.array_equal(states[3:], lite[[1, 3]])


def test_a_full_word_of_qpu_reads_leaves_no_room_for_fill():
    rng = np.random.default_rng(2)
    qpu, lite = _states(rng, LANES, 32), _states(rng, 8, 32)
    states, qpu_count = pack_lanes(qpu, np.arange(LANES), lite, np.arange(8))

    assert qpu_count == LANES
    assert states.shape == (LANES, 32)


def test_identical_states_take_one_lane_between_them():
    # Two lanes that start identical stay identical: every lane shares one
    # acceptance threshold. The duplicate would cost a lane and find nothing.
    rng = np.random.default_rng(3)
    qpu = _states(rng, 2)
    qpu = np.vstack([qpu, qpu[0]])  # the QPU returned one state twice
    lite = np.vstack([qpu[1], _states(rng, 1)])  # MSA-lite also found qpu[1]
    states, qpu_count = pack_lanes(
        qpu, np.array([-3.0, -2.0, -3.0]), lite, np.array([-2.0, -1.0])
    )

    assert qpu_count == 2
    assert states.shape[0] == 3
    assert len({row.tobytes() for row in states}) == 3


def test_a_job_with_no_msa_lite_reads_packs_the_qpu_reads_alone():
    rng = np.random.default_rng(4)
    qpu = _states(rng, 5)
    states, qpu_count = pack_lanes(qpu, np.arange(5))

    assert qpu_count == 5 and states.shape == (5, 16)


def test_an_energy_count_that_does_not_match_the_state_count_is_refused():
    rng = np.random.default_rng(5)
    qpu = _states(rng, 3)
    with pytest.raises(ValueError, match="one energy each"):
        pack_lanes(qpu, np.arange(2.0))


def test_a_one_dimensional_qpu_spins_array_is_refused():
    with pytest.raises(ValueError, match="one energy each"):
        pack_lanes(np.array([1, -1, 1], dtype=np.int8), np.array([-1.0]))


def test_an_empty_qpu_block_still_packs():
    states, qpu_count = pack_lanes(np.empty((0, 16), dtype=np.int8), np.array([]))

    assert qpu_count == 0
    assert states.shape == (0, 16)


def test_lite_spins_without_lite_energies_is_refused():
    rng = np.random.default_rng(6)
    qpu, lite = _states(rng, 2), _states(rng, 2)
    with pytest.raises(ValueError, match="lite_spins and lite_energies"):
        pack_lanes(qpu, np.array([-1.0, -2.0]), lite)


def test_lite_energies_without_lite_spins_is_refused():
    rng = np.random.default_rng(7)
    qpu = _states(rng, 2)
    with pytest.raises(ValueError, match="lite_spins and lite_energies"):
        pack_lanes(qpu, np.array([-1.0, -2.0]), lite_energies=np.array([-1.0, -2.0]))


def test_a_miner_without_the_binding_says_how_to_get_it(monkeypatch):
    real_import = builtins.__import__

    def no_quip_msa(name, *args, **kwargs):
        if name == "quip_msa":
            raise ImportError("No module named 'quip_msa'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_quip_msa)
    with pytest.raises(MsaUnavailable, match="maturin develop"):
        load_kernel()
