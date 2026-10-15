"""``--embed``: a model off the working graph is embedded onto the chip through
Ocean, and refused as off-chip without the flag.

The chip is the offline mock chip (``QUIP_DWAVE_MOCK_BACKEND=chip``), so Ocean's
real EmbeddingComposite and minorminer run, and nothing reaches a QPU.
"""

from __future__ import annotations

import io
import itertools
import json
import time

import numpy as np
import pytest
from quip_solver_core import miner_pb2, wire

from quip_solver_dwave.cli import main
from quip_solver_dwave.job import handle_job
from quip_solver_dwave.ocean import OceanSampler
from quip_solver_dwave.qp import OffChipError
from quip_solver_dwave.schedule import ScheduleError
from quip_solver_dwave.solve import run_solve
from quip_solver_dwave.warm import WarmStart

# Degree 23 is off every chip in service, and off the mock chip's Pegasus.
_K = 24
_K_EDGES = list(itertools.combinations(range(_K), 2))


@pytest.fixture(autouse=True)
def _mock_chip(monkeypatch):
    monkeypatch.setenv("QUIP_DWAVE_MOCK", "1")
    monkeypatch.setenv("QUIP_DWAVE_MOCK_BACKEND", "chip")


def _sample_complete_graph(sampler):
    return sampler.sample(
        np.arange(_K),
        np.zeros(_K),
        np.array(_K_EDGES),
        np.ones(len(_K_EDGES)),
        num_reads=4,
    )


def test_an_off_graph_model_is_embedded_and_reports_its_chains():
    result = _sample_complete_graph(OceanSampler(mock=True, embed=True))

    assert sum(result.occurrences or []) == 4
    assert sorted(result.variables) == list(range(_K)), "logical nodes come back"
    assert result.embedding is not None
    assert len(result.embedding.chain_break_fraction) == len(result.spins)
    assert all(0.0 <= f <= 1.0 for f in result.embedding.chain_break_fraction)
    assert result.embedding.max_chain_length >= 2, "K24 cannot sit one qubit per node"


def test_without_embed_an_off_graph_model_is_off_chip():
    with pytest.raises(OffChipError):
        _sample_complete_graph(OceanSampler(mock=True))


def test_a_model_on_the_working_graph_needs_no_embedding():
    # The mock chip numbers a native 8-cycle 0 to 7.
    result = OceanSampler(mock=True).sample(
        np.arange(8),
        np.zeros(8),
        np.array([(i, (i + 1) % 8) for i in range(8)]),
        np.ones(8),
        num_reads=2,
    )
    assert result.embedding is None


def test_an_embedder_that_gives_up_is_off_chip():
    class _GivesUp:
        def sample_ising(self, h, j, **kwargs):
            raise ValueError("no embedding found")

    sampler = OceanSampler(sampler=_GivesUp(), mock=False, embed=True)
    with pytest.raises(OffChipError, match="no embedding"):
        _sample_complete_graph(sampler)


def test_a_warm_start_cannot_be_embedded():
    sampler = OceanSampler(mock=True, embed=True)
    warm = WarmStart(state=np.ones(_K, dtype=np.int8), reversal_s=0.5, reversal_pause_us=0)
    with pytest.raises(ScheduleError, match="embedded"):
        sampler.sample(
            np.arange(_K),
            np.zeros(_K),
            np.array(_K_EDGES),
            np.ones(len(_K_EDGES)),
            num_reads=1,
            warm_start=warm,
        )


def test_a_topology_free_mock_cannot_embed(monkeypatch):
    monkeypatch.setenv("QUIP_DWAVE_MOCK_BACKEND", "exact")
    with pytest.raises(ValueError, match="QUIP_DWAVE_MOCK_BACKEND=chip"):
        OceanSampler(mock=True, embed=True)


def test_an_embedding_session_clamps_nothing():
    # Under --embed the session graph is logical, so a node the chip lacks is
    # embedded rather than clamped as a defect.
    sampler = OceanSampler(mock=True, embed=True)
    sampler.set_session_topology(list(range(_K)), _K_EDGES)
    assert sampler._defective_qubits == []
    assert sampler._defective_edges == set()


# --- the two front doors ------------------------------------------------------


def _problem() -> bytes:
    return json.dumps(
        {
            "h": [0] * _K,
            "j": [1] * len(_K_EDGES),
            "edges": _K_EDGES,
            "num_reads": 4,
            "num_sweeps": 64,
            "sweeps_per_beta": 1,
            "beta_range": None,
            "seed": 1,
        }
    ).encode()


def _solve(embed: bool):
    out = io.StringIO()
    code = run_solve(
        solver_name=None, mock=True, embed=embed, stdin=io.BytesIO(_problem()), stdout=out
    )
    return code, out.getvalue()


def test_solve_embeds_under_the_flag_and_refuses_without_it():
    code, out = _solve(embed=True)
    assert code == 0
    reads = json.loads(out)
    assert len(reads) == 4
    for read in reads:
        assert read["embedding"]["embedded"] is True
        assert 0.0 <= read["embedding"]["chain_break_fraction"] <= 1.0
        assert read["embedding"]["max_chain_length"] >= 2

    assert _solve(embed=False) == (65, "")


def _inline_job() -> miner_pb2.Job:
    return miner_pb2.Job(
        job_id=b"k24",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(
                u=[u for u, _ in _K_EDGES], v=[v for _, v in _K_EDGES]
            ),
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([0] * _K),
            j=wire.encode_i32_le([1000] * len(_K_EDGES)),
            num_reads=2,
        ),
    )


def test_a_session_job_off_the_graph_is_refused_without_embed_and_solved_with_it():
    refused = handle_job(
        _inline_job(), OceanSampler(mock=True), session_nodes=[], session_edges=[]
    )
    assert refused[0].reject.reason == miner_pb2.TOPOLOGY_MISMATCH

    solved = handle_job(
        _inline_job(), OceanSampler(mock=True, embed=True), session_nodes=[], session_edges=[]
    )
    assert solved[0].WhichOneof("msg") == "result"
    assert len(solved[0].result.solutions) >= 1


def test_embed_with_a_topology_free_mock_is_a_configuration_error(monkeypatch, capsys):
    monkeypatch.setenv("QUIP_DWAVE_MOCK_BACKEND", "exact")
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(_problem())))
    assert main(["--solve", "--embed", "--mock"]) == 64
    assert capsys.readouterr().out == ""


def test_capabilities_lists_embed(capsys):
    assert main(["--capabilities"]) == 0
    assert "embed" in json.loads(capsys.readouterr().out)["features"]
