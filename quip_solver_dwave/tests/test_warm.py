"""Warm-start states off the wire: the bit packing and the parameter ladder."""

from __future__ import annotations

import numpy as np
import pytest

from quip_solver_core import miner_pb2

from quip_solver_dwave.config import SamplingDefaults
from quip_solver_dwave.warm import (
    MalformedWarmStart,
    pack_spins,
    unpack_spins,
    warm_start_from_ising,
)


def _reference_pack(spins) -> bytes:
    """SPEC section 3, written out longhand: node i is bit i % 8 of byte
    i // 8, least significant bit first, and a set bit is spin +1."""
    out = bytearray((len(spins) + 7) // 8)
    for i, s in enumerate(spins):
        if s > 0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 4577])
def test_unpacking_inverts_the_spec_packing_at_every_tail_length(n):
    rng = np.random.default_rng(n)
    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    packed = _reference_pack(spins.tolist())

    assert pack_spins(spins) == packed
    got = unpack_spins(packed, n)
    assert got.dtype == np.int8
    assert got.tolist() == spins.tolist()


def test_the_conformance_drivers_seed_decodes_to_minus_one_plus_one():
    # quip-solver-conformance sends encode_spins_packed(&[-1, 1]) == 0b10.
    assert unpack_spins(bytes([0b10]), 2).tolist() == [-1, 1]


def test_a_state_of_the_wrong_length_is_malformed():
    with pytest.raises(MalformedWarmStart, match="bytes"):
        unpack_spins(bytes(2), 2)
    with pytest.raises(MalformedWarmStart, match="bytes"):
        unpack_spins(b"", 2)


def test_a_set_padding_bit_is_malformed():
    # Two nodes use bits 0 and 1; bit 2 is padding and must be zero.
    with pytest.raises(MalformedWarmStart, match="padding"):
        unpack_spins(bytes([0b100]), 2)


def _ising(**fields) -> miner_pb2.IsingProblem:
    return miner_pb2.IsingProblem(**fields)


def test_a_job_with_no_state_is_cold_whatever_the_start_point_fields_hold():
    # Mirrors parse_warm_start in quip-solver-core: fields 10 to 12 mean
    # nothing without a state, so even an out-of-range one is not an error.
    assert warm_start_from_ising(_ising(reversal_s_milli=5000), 2) is None


def test_the_first_state_is_the_one_a_reverse_anneal_starts_from():
    best, second = pack_spins(np.array([-1, 1])), pack_spins(np.array([1, 1]))
    warm = warm_start_from_ising(_ising(initial_spins=[best, second]), 2)

    assert warm is not None
    assert warm.state.tolist() == [-1, 1]


def test_every_state_is_validated_even_though_only_the_first_is_used():
    good = pack_spins(np.array([-1, 1]))
    with pytest.raises(MalformedWarmStart):
        warm_start_from_ising(_ising(initial_spins=[good, bytes(3)]), 2)


def test_a_reversal_point_of_one_thousand_milli_or_more_is_malformed():
    state = pack_spins(np.array([-1, 1]))
    with pytest.raises(MalformedWarmStart, match="reversal_s_milli"):
        warm_start_from_ising(_ising(initial_spins=[state], reversal_s_milli=1000), 2)


def test_the_job_beats_the_operator_config_which_beats_the_default():
    state = pack_spins(np.array([-1, 1]))
    config = SamplingDefaults(reversal_s_milli=300, reversal_pause_us=50)

    from_job = warm_start_from_ising(
        _ising(initial_spins=[state], reversal_s_milli=900, reversal_pause_us=7),
        2,
        config,
    )
    from_config = warm_start_from_ising(_ising(initial_spins=[state]), 2, config)
    from_default = warm_start_from_ising(_ising(initial_spins=[state]), 2)

    assert from_job is not None and from_config is not None and from_default is not None
    assert (from_job.reversal_s, from_job.reversal_pause_us) == (0.9, 7)
    assert (from_config.reversal_s, from_config.reversal_pause_us) == (0.3, 50)
    assert (from_default.reversal_s, from_default.reversal_pause_us) == (0.5, 25)
