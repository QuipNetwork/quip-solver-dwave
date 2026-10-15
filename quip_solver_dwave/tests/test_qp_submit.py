"""The submission body must be what Ocean would have sent.

``QpEncoder`` produces the ``data`` block; this is the envelope around it.
Ocean builds the same envelope inside ``StructuredSolver._sample``, so these
tests pin the shape against that function's behaviour rather than against a
guess about SAPI.
"""

from __future__ import annotations

import json

import pytest

from quip_solver_dwave.qp import build_submission_body


class _Identity:
    def dict(self):
        return {"name": "FakeSolver", "version": {"graph_id": "abc123"}}


class _Solver:
    """The surface build_submission_body reads, mirroring StructuredSolver."""

    def __init__(self, parameters=None, defaults=None):
        self.identity = _Identity()
        self._params = dict(defaults or {})
        self.parameters = dict(
            parameters or {"num_reads": None, "annealing_time": None, "label": None}
        )
        self.formatted = []

    def _format_params(self, type_, params):
        # Ocean mutates the dict in place; record that we deferred to it.
        self.formatted.append((type_, dict(params)))


_DATA = {"format": "qp", "lin": "AAA=", "quad": "BBB=", "offset": 0.0}


def test_the_body_carries_the_solver_identity_and_the_encoded_problem():
    solver = _Solver()
    body = build_submission_body(solver, _DATA, {"num_reads": 48}, label=None)
    parsed = json.loads(body)

    assert parsed["solver"] == {"name": "FakeSolver", "version": {"graph_id": "abc123"}}
    assert parsed["data"] == _DATA
    assert parsed["type"] == "ising"


def test_solver_defaults_are_merged_under_the_caller_params():
    # Ocean does dict(self._params) then .update(params): the caller wins.
    solver = _Solver(defaults={"num_reads": 1, "annealing_time": 20})
    body = build_submission_body(solver, _DATA, {"num_reads": 48}, label=None)
    parsed = json.loads(body)

    assert parsed["params"]["num_reads"] == 48
    assert parsed["params"]["annealing_time"] == 20


def test_the_solver_formats_the_params_itself():
    # _format_params transforms some values in place. Reimplementing that
    # would be a second source of truth, so it stays Ocean's job.
    solver = _Solver()
    build_submission_body(solver, _DATA, {"num_reads": 48}, label=None)

    assert solver.formatted and solver.formatted[0][0] == "ising"


def test_an_unknown_parameter_is_refused_before_submission():
    # Ocean raises KeyError here rather than letting SAPI reject the job and
    # burn a coordinator credit on the round trip.
    solver = _Solver()

    with pytest.raises(KeyError, match="not a parameter"):
        build_submission_body(solver, _DATA, {"nonsense": 1}, label=None)


def test_an_x_prefixed_parameter_is_allowed_through():
    # Ocean's escape hatch for solver-specific extensions.
    solver = _Solver()
    body = build_submission_body(solver, _DATA, {"x_custom": 7}, label=None)

    assert json.loads(body)["params"]["x_custom"] == 7


def test_a_label_is_included_only_when_given():
    solver = _Solver()

    with_label = json.loads(
        build_submission_body(solver, _DATA, {}, label="quip-abc")
    )
    without = json.loads(build_submission_body(solver, _DATA, {}, label=None))

    assert with_label["label"] == "quip-abc"
    assert "label" not in without


def test_the_body_is_bytes_ready_for_the_transport():
    solver = _Solver()
    body = build_submission_body(solver, _DATA, {"num_reads": 1}, label=None)

    assert isinstance(body, (bytes, bytearray))
