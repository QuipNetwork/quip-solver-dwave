"""Energy reporting + job reject paths (offline mock sampler)."""
import time

import numpy as np
import pytest
from dwave.cloud.exceptions import SolverOfflineError
from typing import Any, Optional

from quip_solver_core import miner_pb2, scoring, wire

from quip_solver_dwave.job import SolverUnavailable, handle_job
from quip_solver_dwave.ocean import OceanSampler, SampleResult


def test_energy_milli_matches_golden_shape():
    # Two-spin ferromagnetic: h=[1,-1], J=[0.5], spins=[+1,+1]
    spins = [1, 1]
    h = [1.0, -1.0]
    j = [0.5]
    edges = [(0, 1)]
    e = scoring.energy_milli(spins, h, j, edges)
    # E = 1*1 + (-1)*1 + 0.5*1*1 = 0.5 → 500 milli
    assert e == 500


def test_malformed_and_expired_rejects():
    sampler = OceanSampler(mock=True)
    # MALFORMED h
    bad = miner_pb2.Job(
        job_id=b"bad",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=b"\x01\x02\x03",
            j=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(bad, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED

    # EXPIRED
    expired = miner_pb2.Job(
        job_id=b"old",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) - 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(expired, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    assert msgs[0].reject.reason == miner_pb2.EXPIRED
    sampler.close()


def test_oversized_job_rejects_too_large():
    # The capability envelope (MAX_NODES) advertised in Hello/Capabilities is
    # enforced: a job above it must reject TOO_LARGE, never reach the sampler.
    from quip_solver_dwave import MAX_NODES

    sampler = OceanSampler(mock=True)
    n = MAX_NODES + 1
    big = miner_pb2.Job(
        job_id=b"huge",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([0] * n),
            j=b"",
            edges=miner_pb2.EdgeList(u=[], v=[]),
            num_reads=1,
        ),
    )
    msgs = handle_job(big, sampler, session_nodes=list(range(n)), session_edges=[])
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.TOO_LARGE
    sampler.close()


def test_malformed_j_rejects():
    # h decodes cleanly but j is not a whole number of i32s. Covers the second
    # decode guard, which the h-only malformed case never reaches.
    #
    # The topology deliberately has no edges. With edges present, a swallowed
    # decode error would leave j_vals empty and trip the j-length check
    # instead, so the test would still see MALFORMED and prove nothing about
    # the guard it is meant to cover.
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"bad-j",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=b"\x01\x02\x03",
            num_reads=1,
        ),
    )
    msgs = handle_job(job, sampler, session_nodes=[0, 1], session_edges=[])
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED
    sampler.close()


def test_valid_job_produces_result_with_access_time():
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"job-1",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(job, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    kinds = [m.WhichOneof("msg") for m in msgs]
    assert "result" in kinds
    assert "job_request" in kinds
    result = next(m.result for m in msgs if m.WhichOneof("msg") == "result")
    assert result.job_id == b"job-1"
    assert len(result.solutions) >= 1
    assert result.meta.device_access_time_us > 0
    # The sampler's reported energy must agree with the golden scorer on the
    # returned spins; a divergence means the problem handed to the sampler is
    # not the problem the job described.
    sol = result.solutions[0]
    spins = wire.decode_spins_packed(sol.spins, 2)
    expected = scoring.energy_milli(
        spins, [1.0, -1.0], [0.5], [(0, 1)]
    )
    assert sol.energy_milli == expected
    sampler.close()


def test_sparse_session_topology_scores_edges():
    # Session topology with non-dense qubit ids (10, 20): the J coupling must
    # reach the sampler keyed by qubit id and be reflected in the reported
    # energy, which is checked against the golden scorer on spin positions.
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"sparse",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 3_600_000,
        ising=miner_pb2.IsingProblem(
            # No inline EdgeList -> resolve against the session topology.
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            num_reads=1,
        ),
    )
    msgs = handle_job(
        job, sampler, session_nodes=[10, 20], session_edges=[(10, 20)]
    )
    result = next(m.result for m in msgs if m.WhichOneof("msg") == "result")
    sol = result.solutions[0]
    spins = wire.decode_spins_packed(sol.spins, 2)
    # Golden scorer indexes by spin position: edge (10,20) -> positions (0,1).
    expected = scoring.energy_milli(spins, [1.0, -1.0], [0.5], [(0, 1)])
    assert sol.energy_milli == expected
    # The coupling must actually be counted (ground state is -2500 milli, not
    # the -2000 an unmapped/dropped edge would yield).
    assert sol.energy_milli == -2500
    sampler.close()


def test_unsupported_kind():
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"gate",
        kind=miner_pb2.GATE_CIRCUIT,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000]),
        ),
    )
    msgs = handle_job(job, sampler, session_nodes=[], session_edges=[])
    assert msgs[0].reject.reason == miner_pb2.UNSUPPORTED_KIND
    sampler.close()


def _hash_job(job_id: bytes, topology_hash: bytes) -> "miner_pb2.Job":
    return miner_pb2.Job(
        job_id=job_id,
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            topology_hash=topology_hash,
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            num_reads=1,
        ),
    )


def test_topology_hash_job_without_cache_rejects_missing():
    sampler = OceanSampler(mock=True)
    job = _hash_job(b"hash-missing", b"\x11" * 32)
    msgs = handle_job(
        job, sampler, session_nodes=[], session_edges=[], session_hash=None
    )
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.TOPOLOGY_MISSING
    sampler.close()


def test_topology_hash_job_wrong_hash_rejects_mismatch():
    sampler = OceanSampler(mock=True)
    job = _hash_job(b"hash-wrong", b"\x22" * 32)
    msgs = handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_hash=b"\x11" * 32,
    )
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.TOPOLOGY_MISMATCH
    sampler.close()


def test_topology_hash_job_match_resolves_to_result():
    sampler = OceanSampler(mock=True)
    job = _hash_job(b"hash-ok", b"\x11" * 32)
    msgs = handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_hash=b"\x11" * 32,
    )
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


def test_set_target_num_reads_override_is_honored():
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"st",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            num_reads=0,  # unset -> falls through to SetTarget
        ),
    )
    target = miner_pb2.SetTarget(max_energy_milli=1000, min_solutions=1, num_reads=3)
    msgs = handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_target=target,
    )
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


class _RecordingSampler:
    """Duck-typed stand-in for OceanSampler that records ``sample()`` kwargs.

    Avoids exercising the real dimod/D-Wave path — only checks that job.py
    resolves and forwards ``anneal_time_us`` correctly (quip-w5p.3).
    """

    def __init__(self):
        self.calls: list = []

    def sample(
        self,
        nodes,
        h,
        edges,
        j,
        *,
        num_reads: int = 1,
        anneal_time_us: Optional[int] = None,
        nonce_seed: Optional[bytes] = None,
        label: str = "",
        cancel_key: Optional[bytes] = None,
    ) -> SampleResult:
        self.calls.append({"num_reads": num_reads, "anneal_time_us": anneal_time_us})
        return SampleResult(
            spins=np.array([[1, -1]], dtype=np.int8),
            variables=[0, 1],
            energies=[0.0],
            device_access_time_us=1,
            num_reads=1,
            extra={},
        )

    def close(self) -> None:
        pass


def _make_job(job_id: bytes, *, anneal_time_us: int = 0) -> "miner_pb2.Job":
    return miner_pb2.Job(
        job_id=job_id,
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            num_reads=1,
            anneal_time_us=anneal_time_us,
        ),
    )


def _topology_job(
    job_id: bytes, h_values: list, j_values: Optional[list] = None
) -> "miner_pb2.Job":
    """Job with no inline EdgeList, so the graph resolves to the session
    Topology and h/j are checked against its node and edge counts."""
    return miner_pb2.Job(
        job_id=job_id,
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le(h_values),
            j=wire.encode_i32_le(
                [500] if j_values is None else j_values
            ),
            num_reads=1,
        ),
    )


def test_h_longer_than_session_topology_rejects_malformed():
    # Three biases for a two-node topology: the extra bias used to be dropped
    # silently and the job solved as if it had never been sent.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"h-long", [1000, -1000, 250])
    msgs = handle_job(job, sampler, session_nodes=[10, 20], session_edges=[(10, 20)])
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED
    sampler.close()


def test_h_shorter_than_session_topology_rejects_malformed():
    # The mirror case: trailing nodes would silently get no bias at all.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"h-short", [1000])
    msgs = handle_job(job, sampler, session_nodes=[10, 20], session_edges=[(10, 20)])
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED
    sampler.close()


def test_h_matching_session_topology_still_solves():
    # The check must not reject well-formed topology jobs.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"h-exact", [1000, -1000])
    msgs = handle_job(job, sampler, session_nodes=[10, 20], session_edges=[(10, 20)])
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


_TWO_EDGE_NODES = [10, 20, 30]
_TWO_EDGES = [(10, 20), (20, 30)]


def test_j_longer_than_topology_edges_rejects_malformed():
    # Two couplings for a one-edge topology: the extra used to be ignored.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"j-long", [1000, -1000], [500, 250])
    msgs = handle_job(job, sampler, session_nodes=[10, 20], session_edges=[(10, 20)])
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED
    sampler.close()


def test_j_shorter_than_topology_edges_rejects_malformed():
    # One coupling for a two-edge topology: edge (20,30) used to be silently
    # un-coupled, so the QPU annealed a different problem than was sent.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"j-short", [1000, -1000, 250], [500])
    msgs = handle_job(
        job, sampler, session_nodes=_TWO_EDGE_NODES, session_edges=_TWO_EDGES
    )
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[1].job_request.credits == 1
    assert msgs[0].reject.reason == miner_pb2.MALFORMED
    sampler.close()


def test_sampler_exception_rejects_overloaded_with_refund():
    """A QPU/Leap exception must answer the job, not strand it: the reply is
    a transient OVERLOADED reject carrying the credit refund, instead of an
    exception dying inside a discarded pool Future."""

    class ExplodingSampler:
        def sample(self, *args, **kwargs):
            raise RuntimeError("leap: solver offline")

    job = miner_pb2.Job(
        job_id=b"boom",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    msgs = handle_job(
        job, ExplodingSampler(), session_nodes=[0, 1], session_edges=[(0, 1)]
    )
    assert [m.WhichOneof("msg") for m in msgs] == ["reject", "job_request"]
    assert msgs[0].reject.reason == miner_pb2.OVERLOADED
    assert msgs[1].job_request.credits == 1


def test_solver_offline_raises_solver_unavailable_instead_of_a_refund():
    """An offline solver is not a transient per-job fault: refunding the
    credit re-dispatches the next job at once and the miner spins against
    SAPI for the whole qblock. The session loop parks the credit instead, so
    handle_job hands the case up rather than answering it."""

    class OfflineSampler:
        def sample(self, *args, **kwargs):
            raise SolverOfflineError("Solver is offline.")

    job = miner_pb2.Job(
        job_id=b"down",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            num_reads=1,
        ),
    )
    with pytest.raises(SolverUnavailable) as info:
        handle_job(
            job, OfflineSampler(), session_nodes=[0, 1], session_edges=[(0, 1)]
        )
    assert info.value.job_id == b"down"
    assert isinstance(info.value.__cause__, SolverOfflineError)


def test_result_meta_echoes_the_resolved_sweep_budget():
    """``SamplerMeta.sweeps`` echoes the resolved budget (the pin is a budget,
    not work the QPU performs), with per-job > SetTarget > session-default
    precedence. The conformance driver's ``sweeps_honoured`` axis grades the
    echo verbatim; it was hardcoded 0 before."""
    from quip_solver_core.session import DEFAULT_NUM_SWEEPS

    sampler = OceanSampler(mock=True)

    def job(num_sweeps=0):
        return miner_pb2.Job(
            job_id=b"job-sweeps",
            kind=miner_pb2.ISING_SAMPLE,
            deadline_ms=int(time.time() * 1000) + 60_000,
            ising=miner_pb2.IsingProblem(
                encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
                scale=1000,
                h=wire.encode_i32_le([1000, -1000]),
                j=wire.encode_i32_le([500]),
                edges=miner_pb2.EdgeList(u=[0], v=[1]),
                num_reads=1,
                num_sweeps=num_sweeps,
            ),
        )

    def meta_sweeps(msgs):
        return next(
            m.result.meta.sweeps for m in msgs if m.WhichOneof("msg") == "result"
        )

    kw: dict[str, Any] = dict(session_nodes=[0, 1], session_edges=[(0, 1)])
    # Absolute default.
    assert meta_sweeps(handle_job(job(), sampler, **kw)) == DEFAULT_NUM_SWEEPS
    # Session default, as resolved from Configure.backend_toml.
    assert meta_sweeps(handle_job(job(), sampler, session_sweeps=512, **kw)) == 512
    # A SetTarget pin wins over the session default.
    target = miner_pb2.SetTarget(num_sweeps=256)
    assert (
        meta_sweeps(
            handle_job(
                job(), sampler, session_target=target, session_sweeps=512, **kw
            )
        )
        == 256
    )
    # A per-job override wins over everything.
    assert (
        meta_sweeps(
            handle_job(
                job(num_sweeps=128),
                sampler,
                session_target=target,
                session_sweeps=512,
                **kw,
            )
        )
        == 128
    )
    sampler.close()


def test_j_matching_topology_edges_still_solves():
    # The check must not reject well-formed multi-edge topology jobs.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"j-exact", [1000, -1000, 250], [500, -500])
    msgs = handle_job(
        job, sampler, session_nodes=_TWO_EDGE_NODES, session_edges=_TWO_EDGES
    )
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


def test_h_only_job_with_no_edges_is_well_formed():
    # No edges and no couplings is a legal degenerate problem, not a mismatch.
    sampler = OceanSampler(mock=True)
    job = _topology_job(b"h-only", [1000, -1000], [])
    msgs = handle_job(job, sampler, session_nodes=[10, 20], session_edges=[])
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


def test_inline_edgelist_job_never_trips_h_length_check():
    # Inline EdgeList derives nodes from len(h), so a session topology of a
    # different size is irrelevant and must not cause a reject.
    sampler = OceanSampler(mock=True)
    job = miner_pb2.Job(
        job_id=b"inline",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
            scale=1000,
            h=wire.encode_i32_le([1000, -1000]),
            j=wire.encode_i32_le([500]),
            num_reads=1,
        ),
    )
    msgs = handle_job(
        job, sampler, session_nodes=[10, 20, 30], session_edges=[(10, 20)]
    )
    assert any(m.HasField("result") for m in msgs)
    sampler.close()


class _AggregatingSampler:
    """Stub shaped like the D-Wave cloud client, which folds identical reads
    into a single record row carrying ``num_occurrences`` rather than repeating
    them. The offline mock samplers never aggregate, so no other test in this
    suite exercises that shape.
    """

    def __init__(self, occurrences: int):
        self._occurrences = occurrences

    def sample_ising(self, h, j, **kwargs):
        import dimod

        return dimod.SampleSet.from_samples(
            [{0: 1, 1: -1}],
            dimod.SPIN,
            energy=[-2.5],
            num_occurrences=[self._occurrences],
            info={"timing": {"qpu_programming_time": 100, "qpu_sampling_time": 50}},
        )


def test_reads_counts_aggregated_occurrences_not_record_rows():
    # Ten anneals that all land on the same state come back as one row; reads
    # must report the anneals performed, not the distinct solutions returned.
    sampler = OceanSampler(sampler=_AggregatingSampler(10), mock=False)
    result = sampler.sample(
        np.array([0, 1]),
        np.array([1.0, -1.0]),
        np.array([(0, 1)]),
        np.array([0.5]),
        num_reads=10,
    )
    assert result.spins.shape[0] == 1
    assert result.num_reads == 10
    sampler.close()


def test_anneal_time_us_unset_passes_none_to_sampler():
    sampler = _RecordingSampler()
    job = _make_job(b"anneal-unset")
    handle_job(job, sampler, session_nodes=[0, 1], session_edges=[(0, 1)])
    assert sampler.calls[0]["anneal_time_us"] is None


def test_anneal_time_us_per_job_override_is_honored():
    sampler = _RecordingSampler()
    job = _make_job(b"anneal-job", anneal_time_us=250)
    target = miner_pb2.SetTarget(anneal_time_us=999)
    handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_target=target,
    )
    # Per-job override wins over SetTarget.
    assert sampler.calls[0]["anneal_time_us"] == 250


def test_anneal_time_us_session_target_fallback_is_honored():
    sampler = _RecordingSampler()
    job = _make_job(b"anneal-target")  # unset -> falls through to SetTarget
    target = miner_pb2.SetTarget(anneal_time_us=777)
    handle_job(
        job,
        sampler,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
        session_target=target,
    )
    assert sampler.calls[0]["anneal_time_us"] == 777
