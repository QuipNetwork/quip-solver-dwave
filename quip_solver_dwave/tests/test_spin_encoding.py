"""Spins stay in numpy from the sampler to the wire.

The QPU returns a (reads x qubits) array. Turning each read into a dict keyed
by qubit label, only to turn it straight back into a dense vector in session
node order, costs ~220k Python-level operations per job in each direction.
That work is GIL-bound, so it does not spread across cores, and at production
size it was the reason a single process could not feed the chip at its own
rate.

These tests pin the two things that make the numpy path safe: the wire format
really is one signed byte per spin, and the reordering puts every spin exactly
where the dict path put it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quip_solver_core import wire

from quip_solver_dwave.defects import DefectInfo, reconstruct_samples
from quip_solver_dwave.job import _build_result, sample_dict_to_vector, spins_to_bytes
from quip_solver_dwave.ocean import SampleResult


def _rng():
    return np.random.default_rng(20260911)


# --- the assumption the whole optimisation rests on ------------------------


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 64, 4577])
def test_the_wire_format_is_one_signed_byte_per_spin(n):
    # encode_spins takes a Python list, which is the cost being removed. If the
    # SDK ever moves to real bit packing this fails loudly, which is the point:
    # the numpy path would silently produce a wrong-length payload otherwise.
    row = _rng().choice(np.array([-1, 1], dtype=np.int8), size=n)
    assert wire.encode_spins(row.tolist()) == row.astype(np.int8).tobytes()


def test_the_wire_format_holds_for_arbitrary_random_rows():
    rng = _rng()
    for _ in range(200):
        row = rng.choice(np.array([-1, 1], dtype=np.int8), size=int(rng.integers(0, 300)))
        assert wire.encode_spins(row.tolist()) == row.astype(np.int8).tobytes()


# --- the reorder must match what the dict path produced --------------------


def _legacy_solution_bytes(spins_row, variables, nodes):
    """What the dict round-trip produced, kept as the reference to match."""
    sample = {int(v): int(s) for v, s in zip(variables, spins_row)}
    return spins_to_bytes(sample_dict_to_vector(sample, nodes))


def _result(spins, variables, energies, defect_info=None):
    return SampleResult(
        spins=np.asarray(spins, dtype=np.int8),
        variables=[int(v) for v in variables],
        energies=[float(e) for e in energies],
        device_access_time_us=43_200,
        num_reads=len(energies),
        defect_info=defect_info,
    )


def test_spins_come_out_in_session_node_order_not_sampler_order():
    # The sampler orders columns however it likes; the coordinator reads the
    # payload positionally against the session node list.
    nodes = [10, 20, 30]
    variables = [30, 10, 20]
    spins = np.array([[1, -1, 1]], dtype=np.int8)  # 30:+1, 10:-1, 20:+1

    msgs = _build_result(b"j", nodes, _result(spins, variables, [-1.0]), 64)

    assert msgs[0].result.solutions[0].spins_bytes == bytes([0xFF, 0x01, 0x01])


def test_a_node_the_sampler_never_reported_defaults_to_plus_one():
    # Matches sample_dict_to_vector's `sample.get(n, 1)`. A clamped or absent
    # qubit must not shift every later spin by one position.
    nodes = [1, 2, 3]
    variables = [1, 3]
    spins = np.array([[-1, -1]], dtype=np.int8)

    msgs = _build_result(b"j", nodes, _result(spins, variables, [0.0]), 64)

    assert msgs[0].result.solutions[0].spins_bytes == bytes([0xFF, 0x01, 0xFF])


def test_every_read_becomes_its_own_solution():
    nodes = [1, 2]
    spins = np.array([[1, 1], [-1, -1], [1, -1]], dtype=np.int8)

    msgs = _build_result(b"j", nodes, _result(spins, [1, 2], [-3.0, -2.0, -1.0]), 64)

    sols = msgs[0].result.solutions
    assert len(sols) == 3
    assert [s.energy_milli for s in sols] == [-3000, -2000, -1000]


def test_the_numpy_path_matches_the_dict_path_at_production_size():
    # The regression that matters: same bytes as before, 4577 wide, 48 deep,
    # with the sampler's column order shuffled against session node order.
    rng = _rng()
    nodes = list(range(4577))
    variables = list(rng.permutation(4577))
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(48, 4577))

    msgs = _build_result(
        b"j", nodes, _result(spins, variables, [-1.0] * 48), 64
    )

    sols = msgs[0].result.solutions
    assert len(sols) == 48
    for i in range(48):
        assert sols[i].spins_bytes == _legacy_solution_bytes(
            spins[i], variables, nodes
        )


def test_energies_are_quantised_to_milli_the_same_way():
    nodes = [1]
    spins = np.array([[1], [1]], dtype=np.int8)

    msgs = _build_result(b"j", nodes, _result(spins, [1], [-14.2345, -0.0005]), 64)

    sols = msgs[0].result.solutions
    assert sols[0].energy_milli == -14234  # round(-14234.5) -> banker's rounding
    assert sols[1].energy_milli == 0


def test_a_result_still_carries_the_meta_and_the_credit_refill():
    nodes = [1]
    msgs = _build_result(
        b"j", nodes, _result(np.array([[1]], dtype=np.int8), [1], [-1.0]), 64
    )

    assert [m.WhichOneof("msg") for m in msgs] == ["result", "job_request"]
    assert msgs[0].result.meta.device_access_time_us == 43_200
    assert msgs[0].result.meta.sweeps == 64
    assert msgs[1].job_request.credits == 1


# --- defect reconstruction has to survive the rewrite ----------------------


def test_a_clamped_qubit_becomes_a_constant_column():
    # Qubit 2 is defective and clamped to -1, so the sampler never returns it.
    # It holds that spin for every read, which is one appended column.
    spins = np.array([[1, 1], [-1, 1]], dtype=np.int8)
    info = DefectInfo(fixed_spins={2: -1}, energy_offset=0.0, removed_edges={})

    out, variables, energies = reconstruct_samples(spins, [1, 3], [0.0, 0.0], info)

    assert variables == [1, 3, 2]
    assert out.tolist() == [[1, 1, -1], [-1, 1, -1]]


def test_a_clamped_qubit_reaches_the_wire_at_its_own_position():
    # The composition that matters: ocean reconstructs, then job packs. A
    # clamped qubit must land in session node order, not be appended.
    nodes = [1, 2, 3]
    spins = np.array([[1, 1]], dtype=np.int8)
    info = DefectInfo(fixed_spins={2: -1}, energy_offset=0.0, removed_edges={})
    out, variables, energies = reconstruct_samples(spins, [1, 3], [0.0], info)

    msgs = _build_result(b"j", nodes, _result(out, variables, energies), 64)

    assert msgs[0].result.solutions[0].spins_bytes == bytes([0x01, 0xFF, 0x01])


def test_the_clamped_energy_offset_is_applied_to_every_read():
    spins = np.array([[1], [-1]], dtype=np.int8)
    info = DefectInfo(fixed_spins={}, energy_offset=2.5, removed_edges={})

    _, _, energies = reconstruct_samples(spins, [1], [-10.0, -20.0], info)

    assert energies == [-7.5, -17.5]


def test_a_removed_coupler_corrects_energy_per_read():
    # Edge (1,2) exists in the problem but not on the chip, so its term was
    # dropped before submit and has to be added back from the spins.
    spins = np.array([[1, 1], [1, -1]], dtype=np.int8)
    info = DefectInfo(
        fixed_spins={}, energy_offset=0.0, removed_edges={(1, 2): 3.0}
    )

    _, _, energies = reconstruct_samples(spins, [1, 2], [0.0, 0.0], info)

    assert energies == [3.0, -3.0]


def test_no_defects_passes_the_batch_through_untouched():
    # The production path: the live graph matches, so nothing is clamped and
    # the array must not be copied or widened.
    spins = np.array([[1, -1]], dtype=np.int8)

    out, variables, energies = reconstruct_samples(spins, [1, 2], [-5.0], None)

    assert out is spins
    assert variables == [1, 2]
    assert energies == [-5.0]
