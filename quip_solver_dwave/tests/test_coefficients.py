"""Coefficient decoding follows quip-solver-core's ``encoding.rs`` exactly.

``IsingProblem.h`` and ``.j`` carry little-endian elements of ``encoding``'s
type. An integer element ``x`` at ``scale`` is ``x / scale`` units; a float
element is units and needs ``scale == 0``. Every value must land on a whole
milli in ``i32`` range, because the coordinator scores in milli. Anything else
is ``MALFORMED``: a lossy conversion here would anneal a different problem
than the one the coordinator re-scores.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from quip_solver_core import miner_pb2

from quip_solver_dwave.job import CoefficientsInvalid, decode_coefficients_milli

E = miner_pb2


def _problem(encoding, scale, h=b"", j=b""):
    return miner_pb2.IsingProblem(encoding=encoding, scale=scale, h=h, j=j)


def _decode(encoding, scale, h=b"", j=b""):
    return decode_coefficients_milli(_problem(encoding, scale, h, j))


def test_i32_at_scale_1000_is_the_canonical_milli_form():
    h = struct.pack("<2i", 1000, -460)
    j = struct.pack("<i", 500)
    assert _decode(E.COEFFICIENT_ENCODING_I32, 1000, h, j) == ([1000, -460], [500])


@pytest.mark.parametrize(
    ("encoding", "fmt", "scale", "stored", "milli"),
    [
        (E.COEFFICIENT_ENCODING_I32, "<i", 1, -3, -3000),
        (E.COEFFICIENT_ENCODING_I16, "<h", 1, 2, 2000),
        (E.COEFFICIENT_ENCODING_I16, "<h", 4, -1, -250),
        (E.COEFFICIENT_ENCODING_I8, "<b", 2, 1, 500),
        (E.COEFFICIENT_ENCODING_I8, "<b", 1000, -128, -128),
    ],
)
def test_integer_elements_are_stored_over_scale(encoding, fmt, scale, stored, milli):
    assert _decode(encoding, scale, struct.pack(fmt, stored)) == ([milli], [])


@pytest.mark.parametrize(
    ("encoding", "raw", "milli"),
    [
        (E.COEFFICIENT_ENCODING_F16, np.float16(-0.5).tobytes(), -500),
        (E.COEFFICIENT_ENCODING_F32, struct.pack("<f", 0.25), 250),
        (E.COEFFICIENT_ENCODING_F64, struct.pack("<d", 1.5), 1500),
        # The multiply lands off the integer (536870910.00000006); the round
        # trip back to units is what decides exactness.
        (E.COEFFICIENT_ENCODING_F64, struct.pack("<d", 536870.91), 536870910),
    ],
)
def test_float_elements_are_units_at_scale_zero(encoding, raw, milli):
    assert _decode(encoding, 0, raw) == ([milli], [])


def test_h_and_j_decode_independently():
    h = struct.pack("<3h", 1, 0, -1)
    j = struct.pack("<2h", 2, -2)
    assert _decode(E.COEFFICIENT_ENCODING_I16, 1, h, j) == (
        [1000, 0, -1000],
        [2000, -2000],
    )


def test_empty_arrays_are_well_formed():
    assert _decode(E.COEFFICIENT_ENCODING_I32, 1000) == ([], [])


@pytest.mark.parametrize(
    ("encoding", "scale", "h"),
    [
        pytest.param(E.COEFFICIENT_ENCODING_UNSPECIFIED, 0, b"", id="unspecified"),
        pytest.param(99, 1000, b"", id="unknown-enum"),
        pytest.param(E.COEFFICIENT_ENCODING_I32, 0, struct.pack("<i", 1), id="int-scale-0"),
        pytest.param(E.COEFFICIENT_ENCODING_F32, 1000, struct.pack("<f", 1), id="float-scale"),
        pytest.param(E.COEFFICIENT_ENCODING_I32, 1000, b"\x01\x02\x03", id="partial-i32"),
        pytest.param(E.COEFFICIENT_ENCODING_I16, 1, b"\x01", id="partial-i16"),
        pytest.param(E.COEFFICIENT_ENCODING_F64, 0, b"\x00" * 7, id="partial-f64"),
        pytest.param(E.COEFFICIENT_ENCODING_I8, 3, struct.pack("<b", 1), id="sub-milli-int"),
        pytest.param(E.COEFFICIENT_ENCODING_I32, 1, struct.pack("<i", 2**31 - 1), id="i32-overflow"),
        pytest.param(E.COEFFICIENT_ENCODING_F64, 0, struct.pack("<d", 0.0005), id="sub-milli-float"),
        pytest.param(E.COEFFICIENT_ENCODING_F64, 0, struct.pack("<d", float("nan")), id="nan"),
        pytest.param(E.COEFFICIENT_ENCODING_F32, 0, struct.pack("<f", float("inf")), id="inf"),
        pytest.param(E.COEFFICIENT_ENCODING_F64, 0, struct.pack("<d", 3e6), id="f64-overflow"),
    ],
)
def test_invalid_coefficients_are_rejected(encoding, scale, h):
    with pytest.raises(CoefficientsInvalid):
        _decode(encoding, scale, h)


def test_a_bad_j_is_rejected_even_when_h_is_good():
    with pytest.raises(CoefficientsInvalid):
        _decode(E.COEFFICIENT_ENCODING_I32, 1000, struct.pack("<i", 1000), b"\x00")
