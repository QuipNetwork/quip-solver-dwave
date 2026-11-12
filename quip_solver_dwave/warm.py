"""Warm-start states off the wire (``IsingProblem`` fields 9 to 12).

quip-solver-core 0.0.2 lets a job carry start states. A reverse anneal starts
from one classical state, so this backend uses the first state, which the
protocol orders best first. ``start_beta_milli`` (field 10) is the seeded-SA
start point and means nothing to an annealer, so it is not read here.

The wire form is bit-packed in topology node order: node ``i`` is bit
``i % 8`` of byte ``i // 8``, least significant bit first, and a set bit is
spin +1. An entry is exactly ``ceil(num_nodes / 8)`` bytes and its padding bits
are zero. SPEC section 3 makes anything else ``MALFORMED``, and the Rust
session applies the same rule to every solver, so the two agree job for job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quip_solver_core import miner_pb2

from quip_solver_dwave.config import SamplingDefaults
from quip_solver_dwave.schedule import DEFAULT_REVERSAL_PAUSE_US, DEFAULT_REVERSAL_S

# The feature string SPEC section 3 names for a solver that uses the states.
INITIAL_SPINS_FEATURE = "initial-spins"


class MalformedWarmStart(ValueError):
    """A start state or reversal point the wire contract does not allow."""


@dataclass(frozen=True)
class WarmStart:
    """One resolved reverse-anneal start: the state and where to reverse to.

    ``state`` is int8 +1/-1, one entry per job node, in job node order.
    """

    state: np.ndarray
    reversal_s: float
    reversal_pause_us: int


def unpack_spins(packed: bytes, num_nodes: int) -> np.ndarray:
    """Decode one bit-packed state to int8 +1/-1.

    Raises:
        MalformedWarmStart: wrong length, or a padding bit set.
    """
    expected = (num_nodes + 7) // 8
    if len(packed) != expected:
        raise MalformedWarmStart(
            f"state is {len(packed)} bytes; {num_nodes} nodes need {expected}"
        )
    raw = np.frombuffer(packed, dtype=np.uint8)
    tail = num_nodes % 8
    if tail and int(raw[-1]) >> tail:
        raise MalformedWarmStart("padding bits are set")
    bits = np.unpackbits(raw, bitorder="little", count=num_nodes)
    return bits.astype(np.int8) * 2 - 1


def pack_spins(spins: np.ndarray) -> bytes:
    """Encode +1/-1 spins to the wire form. The inverse of :func:`unpack_spins`."""
    return np.packbits(np.asarray(spins) > 0, bitorder="little").tobytes()


def warm_start_from_ising(
    ising: miner_pb2.IsingProblem,
    num_nodes: int,
    session_defaults: SamplingDefaults = SamplingDefaults(),
) -> Optional[WarmStart]:
    """Validate and resolve a job's warm start, or ``None`` for a cold job.

    Mirrors ``parse_warm_start`` in quip-solver-core: a job with no state is
    cold whatever fields 10 to 12 hold, and every state is validated even
    though only the first is used.

    The reversal point and the pause follow the sampling ladder: the job, then
    ``Configure.backend_toml``, then the built-in default. Zero means unset at
    every rung.

    Raises:
        MalformedWarmStart: a bad state, or ``reversal_s_milli`` of 1000 or more.
    """
    if not ising.initial_spins:
        return None
    if ising.reversal_s_milli >= 1000:
        raise MalformedWarmStart(
            f"reversal_s_milli is {ising.reversal_s_milli}; it must be under 1000"
        )
    states = [unpack_spins(bytes(p), num_nodes) for p in ising.initial_spins]

    s_milli = int(ising.reversal_s_milli) or session_defaults.reversal_s_milli
    reversal_s = s_milli / 1000.0 if s_milli else DEFAULT_REVERSAL_S
    pause_us = (
        int(ising.reversal_pause_us)
        or session_defaults.reversal_pause_us
        or DEFAULT_REVERSAL_PAUSE_US
    )
    return WarmStart(state=states[0], reversal_s=reversal_s, reversal_pause_us=pause_us)
