"""End-to-end: a Cancel must not lose QPU access time from the ledger.

The ledger is the only durable record of what D-Wave has charged this period,
and the pacer hands out headroom against it. Every path a cancelled job can
take out of the session loop is checked here for what it bills, because an
under-count does not announce itself — it shows up a month later as a blown
quota.
"""

from __future__ import annotations

import threading

import numpy as np
import time
from typing import List, cast

import pytest

from quip_solver_core import miner_pb2, wire

from quip_solver_dwave import session_loop
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.usage import UsageLedger

ACCESS_US = 46_000
JOB_GENERATION = 5


class BlockingSampler:
    """Holds a job on the "QPU" until the test lets it go.

    Standing in for OceanSampler at the seam the session loop uses, so a
    Cancel can be delivered while a job is provably still in flight.
    """

    def __init__(self, *, outcome: str = "result"):
        self._outcome = outcome
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled: List[bytes] = []
        self._unobserved = 0
        self.native_topology_hash = None

    # -- the session loop's sampler surface --------------------------------
    def ensure_connected(self) -> None:
        pass

    def set_session_topology(self, nodes, edges) -> None:
        pass

    def close(self) -> None:
        pass

    def cancel_inflight(self, keys) -> int:
        self.cancelled.extend(keys)
        return len(list(keys))

    def drain_unobserved_access_us(self) -> int:
        owed, self._unobserved = self._unobserved, 0
        return owed

    def sample(self, nodes, h, edges, j, **kwargs) -> SampleResult:
        self.entered.set()
        self.release.wait(timeout=5)
        if self._outcome == "cancelled":
            # Mirrors OceanSampler: a problem SAPI accepted and then cancelled
            # raises with no timing attached, and books the estimated charge
            # on its way out rather than letting it go unbilled.
            self._unobserved += ACCESS_US
            raise RuntimeError("problem cancelled")
        return SampleResult(
            spins=np.array([[1, -1]], dtype=np.int8),
            variables=[0, 1],
            energies=[-1.0],
            device_access_time_us=ACCESS_US,
            num_reads=1,
        )


def _configure(usage_db: str) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        configure=miner_pb2.Configure(
            queue_depth=3,
            idle_timeout_s=300,
            # A real budget, so a pacer and a ledger exist to be checked.
            backend_toml=f'budget = "3000s"\nusage_db = "{usage_db}"\n',
        )
    )


def _job() -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        job=miner_pb2.Job(
            job_id=b"cancel-me",
            kind=miner_pb2.ISING_SAMPLE,
            generation=JOB_GENERATION,
            deadline_ms=int(time.time() * 1000) + 3_600_000,
            ising=miner_pb2.IsingProblem(
                h_milli_le32=wire.encode_i32_le([1000, -1000]),
                j_milli_le32=wire.encode_i32_le([500]),
                edges=miner_pb2.EdgeList(u=[0], v=[1]),
                num_reads=1,
            ),
        )
    )


def _cancel(generation: int) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        cancel=miner_pb2.Cancel(max_generation=generation)
    )


def _run(monkeypatch, sampler: BlockingSampler, usage_db: str) -> List:
    """Drive run_session against a scripted coordinator; return outbound msgs.

    The script joins a qblock first (credits only move on a boundary), then
    dispatches one job and cancels its generation while the sampler still has
    it, which is the race the whole feature exists for.
    """
    sent: List = []

    def responses():
        yield miner_pb2.CoordMsg(
            welcome=miner_pb2.Welcome(protocol_version=1)
        )
        yield _configure(usage_db)
        # A boundary the gate can join on, so the job is not refused on budget.
        yield _cancel(1)
        yield miner_pb2.CoordMsg(
            topology=miner_pb2.Topology(
                nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
            )
        )
        yield _job()
        # Do not cancel until the job is provably on the QPU.
        assert sampler.entered.wait(timeout=5), "job never reached the sampler"
        yield _cancel(JOB_GENERATION)
        sampler.release.set()
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            # Drain the outbound queue on a side thread; run_session's feeder
            # is otherwise never consumed and shutdown would block on it.
            def drain():
                for msg in request_iter:
                    sent.append(msg)

            t = threading.Thread(target=drain, daemon=True)
            t.start()
            return responses()

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(
        session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel()
    )
    monkeypatch.setattr(
        session_loop.grpc, "channel_ready_future", lambda ch: _Ready()
    )
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")

    session_loop.run_session(
        "unix:///tmp/nope.sock", "qpu-test", cast(OceanSampler, sampler)
    )
    return sent


def _spent_us(usage_db: str) -> float:
    return UsageLedger(usage_db).spent_us_since(0.0)


@pytest.fixture
def usage_db(tmp_path) -> str:
    # A real file, not ":memory:" — the pacer and the assertion open the
    # ledger separately, and an in-memory one would not be the same database.
    return str(tmp_path / "usage.db")


def test_an_abandoned_job_that_annealed_anyway_is_still_billed(
    monkeypatch, usage_db
):
    # The cancel lost the race: samples came back, so D-Wave charged for the
    # anneal. The coordinator discards the answer; the quota is spent either
    # way and the ledger has to say so.
    sampler = BlockingSampler(outcome="result")
    _run(monkeypatch, sampler, usage_db)

    assert _spent_us(usage_db) == ACCESS_US


def test_an_abandoned_job_sends_no_result_to_the_coordinator(
    monkeypatch, usage_db
):
    # SPEC section 5: no Result for a generation the coordinator reseeded past.
    sampler = BlockingSampler(outcome="result")
    sent = _run(monkeypatch, sampler, usage_db)

    kinds = [m.WhichOneof("msg") for m in sent]
    assert "result" not in kinds


def test_an_abandoned_job_sends_no_reject_either(monkeypatch, usage_db):
    # A cancelled submission surfaces as a Reject. Forwarding it answers a job
    # the coordinator has already thrown away, and logs it at WARNING.
    sampler = BlockingSampler(outcome="cancelled")
    sent = _run(monkeypatch, sampler, usage_db)

    kinds = [m.WhichOneof("msg") for m in sent]
    assert "reject" not in kinds


def test_a_cancelled_job_bills_the_conservative_estimate(monkeypatch, usage_db):
    # No samples came back, so there is no measured access time. Billing zero
    # here is the leak that cancellation would otherwise introduce.
    sampler = BlockingSampler(outcome="cancelled")
    _run(monkeypatch, sampler, usage_db)

    assert _spent_us(usage_db) == ACCESS_US


def test_the_cancel_reaches_the_job_that_is_in_flight(monkeypatch, usage_db):
    sampler = BlockingSampler(outcome="result")
    _run(monkeypatch, sampler, usage_db)

    assert sampler.cancelled == [b"cancel-me"]


def test_a_job_of_a_live_generation_is_not_cancelled(monkeypatch, usage_db):
    # Only generations at or below the watermark are abandoned; cancelling a
    # live one would throw away work the coordinator still wants.
    sampler = BlockingSampler(outcome="result")

    def responses():
        yield miner_pb2.CoordMsg(
            welcome=miner_pb2.Welcome(protocol_version=1)
        )
        yield _configure(usage_db)
        yield _cancel(1)
        yield miner_pb2.CoordMsg(
            topology=miner_pb2.Topology(
                nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
            )
        )
        yield _job()  # generation 5
        assert sampler.entered.wait(timeout=5)
        yield _cancel(JOB_GENERATION - 1)  # reseed stops short of this job
        sampler.release.set()
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent: List = []

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            threading.Thread(
                target=lambda: [sent.append(m) for m in request_iter], daemon=True
            ).start()
            return responses()

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(
        session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel()
    )
    monkeypatch.setattr(
        session_loop.grpc, "channel_ready_future", lambda ch: _Ready()
    )
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")
    session_loop.run_session(
        "unix:///tmp/nope.sock", "qpu-test", cast(OceanSampler, sampler)
    )

    assert sampler.cancelled == []
    assert "result" in [m.WhichOneof("msg") for m in sent]
    assert _spent_us(usage_db) == ACCESS_US
