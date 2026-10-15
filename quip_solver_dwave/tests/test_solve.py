"""``--solve``: one problem in, every read out, and the SPEC section 2 exit codes.

Every test runs on the mock sampler or a stub; none reaches a QPU.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
import pytest
from quip_solver_core import scoring

from quip_solver_dwave import solve as solve_mod
from quip_solver_dwave.cli import main
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.qp import OffChipError
from quip_solver_dwave.schedule import ScheduleError
from quip_solver_dwave.solve import Capacity, parse_problem, run_solve, solve

_RING = {
    "h": [1, 0, 0, 0],
    "j": [1, -1, 1, -1],
    "edges": [[0, 1], [1, 2], [2, 3], [3, 0]],
    "num_reads": 4,
    "num_sweeps": 64,
    "sweeps_per_beta": 1,
    "beta_range": None,
    "seed": 7,
}


def _run(problem, *, mock=True):
    raw = problem if isinstance(problem, bytes) else json.dumps(problem).encode()
    out = io.StringIO()
    code = run_solve(solver_name=None, mock=mock, stdin=io.BytesIO(raw), stdout=out)
    return code, out.getvalue()


_DROP = object()


def _with(**changes):
    problem = dict(_RING)
    for key, value in changes.items():
        if value is _DROP:
            problem.pop(key)
        else:
            problem[key] = value
    return problem


# --- the output -----------------------------------------------------------


def test_the_ring_solves_with_one_object_per_read_and_exact_energies():
    code, out = _run(_RING)

    assert code == 0
    assert out.endswith("\n")
    reads = json.loads(out)
    assert len(reads) == 4
    edges = [tuple(e) for e in _RING["edges"]]
    for read in reads:
        assert set(read) == {"spins", "energy_milli", "timing", "sampler"}
        assert read["energy_milli"] == scoring.energy_milli(
            read["spins"], _RING["h"], _RING["j"], edges
        )
        assert read["timing"]["sample_us"] <= read["timing"]["total_us"]
    assert reads[0]["sampler"] == {
        "backend": "dwave-qpu",
        "algorithm": "quantum-anneal",
        "num_reads": 4,
        "num_sweeps": 64,
        "sweeps_per_beta": 1,
        "beta_range": None,
        "beta_schedule_type": "geometric",
        "seed": 7,
    }


def test_the_same_problem_gives_the_same_reads_on_the_mock():
    first, second = _run(_RING)[1], _run(_RING)[1]
    assert [r["spins"] for r in json.loads(first)] == [
        r["spins"] for r in json.loads(second)
    ]


def test_an_anneal_time_is_echoed():
    code, out = _run(_with(anneal_time_us=20))
    assert code == 0
    assert json.loads(out)[0]["sampler"]["anneal_time_us"] == 20


def test_a_model_with_fewer_states_than_reads_still_returns_every_read():
    # One node has two states; the exact mock answers all five anneals.
    code, out = _run(
        {**_RING, "h": [1], "j": [], "edges": [], "num_reads": 5}
    )
    assert code == 0
    assert len(json.loads(out)) == 5


# --- refusals: exit code and nothing on stdout -----------------------------


@pytest.mark.parametrize(
    "problem",
    [
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"[]", id="not-an-object"),
        pytest.param(_with(seed=_DROP), id="missing-key"),
        pytest.param(_with(num_sweep=1), id="unknown-key"),
        pytest.param(_with(num_reads=0), id="zero-reads"),
        pytest.param(_with(sweeps_per_beta=0), id="zero-sweeps-per-beta"),
        pytest.param(_with(num_reads=True), id="bool-is-not-an-integer"),
        pytest.param(_with(h=[1, "x", 0, 0]), id="string-coefficient"),
        pytest.param(_with(j=[1]), id="couplings-mismatch"),
        pytest.param(_with(edges=[[0, 1], [1, 1], [2, 3], [3, 0]]), id="self-loop"),
        pytest.param(_with(edges=[[0, 4], [1, 2], [2, 3], [3, 0]]), id="endpoint-outside-h"),
        pytest.param(_with(edges=[[0, 1, 2], [1, 2], [2, 3], [3, 0]]), id="not-a-pair"),
        pytest.param(_with(beta_range=[1]), id="beta-range-not-a-pair"),
        pytest.param(_with(seed=2**64), id="seed-past-u64"),
        pytest.param(_with(beta_schedule_type="cosine"), id="unknown-schedule"),
        pytest.param(_with(beta_schedule_type="linear"), id="linear-schedule"),
        pytest.param(_with(anneal_time_us=0), id="zero-anneal-time"),
    ],
)
def test_a_malformed_problem_exits_64_and_writes_nothing(problem):
    assert _run(problem) == (64, "")


def test_a_model_past_the_node_limit_exits_65(monkeypatch):
    monkeypatch.setattr(solve_mod, "MAX_NODES", 3)
    assert _run(_RING) == (65, "")


def test_a_solver_that_does_not_open_exits_69(monkeypatch):
    def refuse(self):
        raise RuntimeError("no route to SAPI")

    monkeypatch.setattr(OceanSampler, "ensure_connected", refuse)
    assert _run(_RING) == (69, "")


def test_a_sampler_failure_exits_70(monkeypatch):
    def fail(self, *args, **kwargs):
        raise RuntimeError("SAPI said no")

    monkeypatch.setattr(OceanSampler, "sample", fail)
    assert _run(_RING) == (70, "")


# --- solve() against a stub sampler -----------------------------------------


class _Stub:
    """The sampler surface solve() uses, with a canned answer."""

    def __init__(self, result=None, error=None, max_reads=None):
        self.result = result
        self.error = error
        self.max_reads = max_reads
        self.calls = []

    def sample(self, nodes, h, edges, j, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _problem(**changes):
    return parse_problem(json.dumps(_with(**changes)).encode())


def _result(spins, variables, occurrences):
    return SampleResult(
        spins=np.asarray(spins, dtype=np.int8),
        variables=variables,
        energies=[0.0] * len(spins),
        device_access_time_us=10,
        num_reads=sum(occurrences),
        occurrences=occurrences,
    )


def test_folded_rows_repeat_once_per_anneal_in_node_order():
    # The cloud folds identical reads; the sampler's columns are not node order.
    stub = _Stub(_result([[1, 1, -1, -1], [-1, -1, -1, -1]], [3, 2, 1, 0], [3, 1]))

    reads = solve(_problem(), stub, started_ns=time.monotonic_ns())  # type: ignore[arg-type]

    assert [r["spins"] for r in reads] == [[-1, -1, 1, 1]] * 3 + [[-1, -1, -1, -1]]


def test_a_short_answer_is_a_sampler_failure():
    stub = _Stub(_result([[1, 1, 1, 1]], [0, 1, 2, 3], [3]))
    with pytest.raises(RuntimeError, match="3 reads for num_reads 4"):
        solve(_problem(), stub, started_ns=time.monotonic_ns())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "error",
    [OffChipError("coupler (0, 2) is not on the solver"), ScheduleError("too long")],
    ids=["off-chip", "anneal-time-out-of-range"],
)
def test_what_the_chip_cannot_run_is_capacity(error):
    with pytest.raises(Capacity):
        solve(_problem(), _Stub(error=error), started_ns=time.monotonic_ns())  # type: ignore[arg-type]


def test_more_reads_than_the_solver_allows_is_capacity_before_any_submit():
    stub = _Stub(max_reads=3)
    with pytest.raises(Capacity, match="maximum of 3"):
        solve(_problem(), stub, started_ns=time.monotonic_ns())  # type: ignore[arg-type]
    assert stub.calls == []


def test_the_anneal_time_reaches_the_sampler():
    stub = _Stub(_result([[1, 1, 1, 1]], [0, 1, 2, 3], [4]))
    solve(_problem(anneal_time_us=50), stub, started_ns=time.monotonic_ns())  # type: ignore[arg-type]
    assert stub.calls[0]["anneal_time_us"] == 50
    assert stub.calls[0]["num_reads"] == 4


# --- the CLI ----------------------------------------------------------------


def test_solve_is_reached_from_the_command_line(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(_RING).encode())))
    assert main(["--solve", "--mock", "--log-level", "warning"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 4


def test_capabilities_lists_anneal_time(capsys):
    assert main(["--capabilities"]) == 0
    assert "anneal-time" in json.loads(capsys.readouterr().out)["features"]
