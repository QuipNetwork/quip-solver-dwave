"""Spin layouts the cascade engine speaks.

The engine holds 64 lanes in one 64-bit word per spin. Lane ``r`` is bit
``r % 64`` of word ``r // 64``. A set bit is spin -1. Published reads use
one bit per spin, low bit first, with the same sign rule.
"""

from __future__ import annotations

import numpy as np

LANES_PER_WORD = 64


def lane_words(states: np.ndarray, words: int) -> np.ndarray:
    """``uint64[nodes, words]`` for ``states[reads, nodes]`` of -1 and +1."""
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(f"states must be 2-D, got shape {states.shape}")
    reads, nodes = states.shape
    if reads > LANES_PER_WORD * words:
        raise ValueError(f"{reads} states do not fit {LANES_PER_WORD * words} lanes")
    if not np.isin(states, (-1, 1)).all():
        raise ValueError("a state holds a value other than -1 and +1")
    lanes = np.zeros((nodes, words), dtype=np.uint64)
    for read in range(reads):
        bit = np.uint64(read % LANES_PER_WORD)
        lanes[:, read // LANES_PER_WORD] |= (states[read] < 0).astype(np.uint64) << bit
    return lanes


def encode_lanes(states: np.ndarray, words: int) -> str:
    """Hex of the little-endian lane words, node by node."""
    return lane_words(states, words).astype("<u8").tobytes().hex()


def decode_reads(samples_hex: str, reads: int, nodes: int) -> np.ndarray:
    """``int8[reads, nodes]`` of -1 and +1 from the engine's packed reads."""
    packed = np.frombuffer(bytes.fromhex(samples_hex), dtype=np.uint8)
    width = (nodes + 7) // 8
    if packed.size != reads * width:
        raise ValueError(f"{packed.size} bytes do not hold {reads} reads of {nodes} spins")
    bits = np.unpackbits(packed.reshape(reads, width), axis=1, bitorder="little")[:, :nodes]
    return (1 - 2 * bits.astype(np.int8)).astype(np.int8)


def encode_reads(states: np.ndarray) -> str:
    """Inverse of :func:`decode_reads`. Tests and the mock engine use it."""
    bits = (np.asarray(states) < 0).astype(np.uint8)
    return np.packbits(bits, axis=1, bitorder="little").tobytes().hex()
