"""A job's warm start, from the wire to the sampler call."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from quip_solver_core import miner_pb2, wire

from quip_solver_dwave import FEATURES
from quip_solver_dwave.job import handle_job
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.schedule import ScheduleError
from quip_solver_dwave.warm import INITIAL_SPINS_FEATURE, pack_spins


class _Recording:
    """Records the keyword arguments of each ``sample`` call."""

    def __init__(self, raises: Optional[Exception] = None):
        self.calls: list = []
        self._raises = raises

    def sample(self, nodes, h, edges, j, **kwargs) -> SampleResult:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return SampleResult(
            spins=np.array([[-1, 1]], dtype=np.int8),
            variables=[0, 1],
            energies=[-2.5],
            device_access_time_us=1,
            num_reads=1,
            extra={},
        )


def _job(**ising_fields) -> miner_pb2.Job:
    return miner_pb2.Job(
        job_id=b"warm-job",
        kind=miner_pb2.ISING_SAMPLE,
        deadline_ms=int(time.time() * 1000) + 60_000,
        ising=miner_pb2.IsingProblem(
            edges=miner_pb2.EdgeList(u=[0], v=[1]),
            h_milli_le32=wire.encode_i32_le([1000, -1000]),
            j_milli_le32=wire.encode_i32_le([500]),
            num_reads=1,
            **ising_fields,
        ),
    )


def _handle(job, sampler):
    return handle_job(job, sampler, session_nodes=[], session_edges=[])


def _reject_reason(replies) -> Optional[int]:
    for reply in replies:
        if reply.WhichOneof("msg") == "reject":
            return reply.reject.reason
    return None


def test_the_miner_advertises_the_initial_spins_feature():
    assert INITIAL_SPINS_FEATURE in FEATURES


def test_a_cold_job_calls_the_sampler_without_a_warm_start_argument():
    # A sampler double written before warm starts existed has no such
    # parameter, and it must keep serving cold jobs.
    sampler = _Recording()
    _handle(_job(), sampler)

    assert "warm_start" not in sampler.calls[0]


def test_a_seeded_job_hands_the_decoded_state_and_start_point_to_the_sampler():
    sampler = _Recording()
    state = pack_spins(np.array([-1, 1]))
    _handle(
        _job(initial_spins=[state], reversal_s_milli=900, reversal_pause_us=40),
        sampler,
    )

    warm = sampler.calls[0]["warm_start"]
    assert warm.state.tolist() == [-1, 1]
    assert (warm.reversal_s, warm.reversal_pause_us) == (0.9, 40)


def test_a_state_of_the_wrong_length_is_rejected_malformed_with_its_refund():
    sampler = _Recording()
    replies = _handle(_job(initial_spins=[bytes(3)]), sampler)

    assert _reject_reason(replies) == miner_pb2.MALFORMED
    # The credit ledger: a reject is terminal, so the credit comes back.
    assert [r.WhichOneof("msg") for r in replies] == ["reject", "job_request"]
    assert sampler.calls == []


def test_a_reversal_point_of_one_thousand_milli_is_rejected_malformed():
    sampler = _Recording()
    state = pack_spins(np.array([-1, 1]))
    replies = _handle(_job(initial_spins=[state], reversal_s_milli=1000), sampler)

    assert _reject_reason(replies) == miner_pb2.MALFORMED
    assert sampler.calls == []


def test_an_anneal_the_chip_cannot_run_is_malformed_not_overloaded():
    # OVERLOADED tells the coordinator to try again later. The same job fails
    # the same way every time, so it must not be reported as transient.
    sampler = _Recording(raises=ScheduleError("a 5000 us schedule is outside the range"))
    replies = _handle(_job(anneal_time_us=5000), sampler)

    assert _reject_reason(replies) == miner_pb2.MALFORMED


def test_the_mock_miner_returns_the_seeds_energy_for_a_seeded_job():
    # End to end through the real sampler in mock mode: h = [1, -1], J = 0.5,
    # seeded with its ground state [-1, +1] at -2.5, which is what the
    # conformance driver's job-seeded sends.
    sampler = OceanSampler(mock=True)
    state = pack_spins(np.array([-1, 1]))
    replies = _handle(_job(initial_spins=[state]), sampler)

    result = next(r.result for r in replies if r.WhichOneof("msg") == "result")
    assert [s.energy_milli for s in result.solutions] == [-2500]
    assert wire.decode_spins(result.solutions[0].spins_bytes) == [-1, 1]
    sampler.close()
