"""End-to-end: an offline solver parks credits instead of spinning on rejects.

A reject normally carries a credit refund, and the coordinator dispatches the
next staged job the moment the refund lands. While the solver is offline that
loop runs at SAPI round-trip speed for the whole qblock: 7,623 rejects in one
11-minute round during the Advantage2_system1 outage. The miner now holds the
credit until the next qblock boundary, where one probe per round is enough to
notice the solver is back.
"""

from __future__ import annotations

import threading
import time
from typing import List, cast

import numpy as np
import pytest
from dwave.cloud.exceptions import SolverNotFoundError, SolverOfflineError

from quip_solver_core import miner_pb2, wire

from quip_solver_dwave import EXIT_INTERNAL_FATAL, session_loop
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.usage import UsageLedger

JOB_GENERATION = 5
QUEUE_DEPTH = 3


class FlakySampler:
    """Raises the offline error for the first ``offline_jobs`` jobs, then
    answers normally."""

    def __init__(self, offline_jobs: int):
        self._offline_left = offline_jobs
        self.native_topology_hash = None

    def ensure_connected(self) -> None:
        pass

    def set_session_topology(self, nodes, edges) -> None:
        pass

    def close(self) -> None:
        pass

    def cancel_inflight(self, keys) -> int:
        return 0

    def drain_unobserved_access_us(self) -> int:
        return 0

    def sample(self, nodes, h, edges, j, **kwargs) -> SampleResult:
        if self._offline_left > 0:
            self._offline_left -= 1
            raise SolverOfflineError("Solver is offline.")
        return SampleResult(
            spins=np.array([[1, -1]], dtype=np.int8),
            variables=[0, 1],
            energies=[-1.0],
            device_access_time_us=46_000,
            num_reads=1,
        )


def _configure(usage_db: str) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        configure=miner_pb2.Configure(
            queue_depth=QUEUE_DEPTH,
            idle_timeout_s=300,
            backend_toml=f'budget = "3000s"\nusage_db = "{usage_db}"\n',
        )
    )


def _job(job_id: bytes, generation: int) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        job=miner_pb2.Job(
            job_id=job_id,
            kind=miner_pb2.ISING_SAMPLE,
            generation=generation,
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
    return miner_pb2.CoordMsg(cancel=miner_pb2.Cancel(max_generation=generation))


def _topology() -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        topology=miner_pb2.Topology(
            nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
        )
    )


def _wait_for(sent: List, kind: str, count: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if sum(1 for m in sent if m.WhichOneof("msg") == kind) >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"never saw {count} {kind} message(s)")


def _run(monkeypatch, sampler, usage_db: str, script, codes=None) -> List:
    """Drive run_session against ``script``, a generator that yields CoordMsgs
    and may inspect ``sent`` between yields. Returns the outbound messages;
    the exit code is appended to ``codes`` when a list is given."""
    sent: List = []

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            threading.Thread(
                target=lambda: [sent.append(m) for m in request_iter], daemon=True
            ).start()
            return script(sent)

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
    code = session_loop.run_session(
        "unix:///tmp/nope.sock", "qpu-test", cast(OceanSampler, sampler)
    )
    if codes is not None:
        codes.append(code)
    return sent


def _credits(msgs) -> List[int]:
    return [m.job_request.credits for m in msgs if m.WhichOneof("msg") == "job_request"]


@pytest.fixture
def usage_db(tmp_path) -> str:
    return str(tmp_path / "usage.db")


def test_an_offline_solver_rejects_the_job_without_a_refund(monkeypatch, usage_db):
    boundary_at: List[int] = []

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        yield _cancel(1)  # the boundary the gate joins on: credits go out here
        yield _topology()
        yield _job(b"down", JOB_GENERATION)
        _wait_for(sent, "reject", 1)
        boundary_at.append(len(sent))
        yield _cancel(JOB_GENERATION + 1)
        _wait_for(sent, "job_request", 2)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent = _run(monkeypatch, FlakySampler(offline_jobs=1), usage_db, script)

    rejects = [m for m in sent if m.WhichOneof("msg") == "reject"]
    assert [r.reject.reason for r in rejects] == [miner_pb2.OVERLOADED]
    # The join granted the pipeline depth. The reject carried no refund, so
    # nothing else went out until the boundary re-granted the parked credit.
    assert _credits(sent[: boundary_at[0]]) == [QUEUE_DEPTH]
    assert _credits(sent[boundary_at[0] :]) == [1]


def test_an_offline_solver_bills_the_ledger_nothing(monkeypatch, usage_db):
    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        yield _cancel(1)
        yield _topology()
        yield _job(b"down", JOB_GENERATION)
        _wait_for(sent, "reject", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    _run(monkeypatch, FlakySampler(offline_jobs=1), usage_db, script)

    assert UsageLedger(usage_db).spent_us_since(0.0) == 0


# --- An offline solver at connect time ---------------------------------------
#
# The coordinator's supervisor restarts a miner that exits six times with a
# backoff of 0.5 s doubling to 8 s, then gives up. An offline solver at
# Configure used to be exactly that: the connect raised, the session failed
# with exit code 70, and the miner was dead until an operator restarted it.
# The session now survives the connect failure, grants nothing, and retries
# the connect once per qblock boundary.


class DownAtConnectSampler(FlakySampler):
    """Refuses to connect ``down_for`` times, then connects."""

    def __init__(self, error: Exception, down_for: int):
        super().__init__(offline_jobs=0)
        self._error = error
        self._down_for = down_for
        self.connect_attempts = 0
        self.topologies: List[List[int]] = []

    def ensure_connected(self) -> None:
        self.connect_attempts += 1
        if self.connect_attempts <= self._down_for:
            raise self._error

    def set_session_topology(self, nodes, edges) -> None:
        self.topologies.append(list(nodes))


def _kinds(msgs) -> List[str]:
    return [m.WhichOneof("msg") for m in msgs]


@pytest.mark.parametrize(
    "error",
    [SolverOfflineError("Solver is offline."), SolverNotFoundError("not available")],
    ids=["offline", "not-found"],
)
def test_a_solver_down_at_configure_is_probed_once_per_qblock(
    monkeypatch, usage_db, error
):
    sampler = DownAtConnectSampler(error, down_for=2)
    marks: List[int] = []

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)  # connect attempt 1 fails
        _wait_for(sent, "ready", 1)
        yield _topology()
        yield _cancel(1)  # attempt 2 fails: sat out
        yield _cancel(2)  # attempt 3 connects: credits go out
        _wait_for(sent, "job_request", 1)
        marks.append(len(sent))
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent = _run(monkeypatch, sampler, usage_db, script)

    assert "ready" in _kinds(sent)
    assert sampler.connect_attempts == 3
    assert _credits(sent) == [QUEUE_DEPTH]
    # The session topology arrived while the solver was down. It is bound
    # again once the live graph is known, so defects are computed against it.
    assert sampler.topologies == [[0, 1], [0, 1]]


def test_an_unbudgeted_miner_gets_its_credits_when_the_solver_returns(
    monkeypatch, usage_db
):
    sampler = DownAtConnectSampler(SolverOfflineError("Solver is offline."), down_for=1)

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        # No budget: an unmetered miner is granted credits at Configure, so
        # the grant is what has to be withheld here.
        yield miner_pb2.CoordMsg(
            configure=miner_pb2.Configure(queue_depth=QUEUE_DEPTH, idle_timeout_s=300)
        )
        _wait_for(sent, "ready", 1)
        assert "job_request" not in _kinds(sent)
        yield _cancel(1)  # attempt 2 connects
        _wait_for(sent, "job_request", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent = _run(monkeypatch, sampler, usage_db, script)

    assert _credits(sent) == [QUEUE_DEPTH]


def test_a_round_sat_out_for_the_solver_is_recorded_as_such(monkeypatch, usage_db):
    sampler = DownAtConnectSampler(SolverOfflineError("Solver is offline."), down_for=2)

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        _wait_for(sent, "ready", 1)
        yield _cancel(1)
        yield _cancel(2)
        _wait_for(sent, "job_request", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    _run(monkeypatch, sampler, usage_db, script)

    import sqlite3

    rows = sqlite3.connect(usage_db).execute(
        "SELECT generation, joined, reason FROM miner_rounds ORDER BY start_ts_s"
    ).fetchall()
    # A boundary starts the generation after the one it cancelled.
    assert rows[0] == (2, 0, "solver-offline")
    assert rows[1][0] == 3 and rows[1][1] == 1


def test_any_other_connect_error_is_still_fatal(monkeypatch, usage_db):
    sampler = DownAtConnectSampler(RuntimeError("bad token"), down_for=99)

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        time.sleep(0.2)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    codes: List[int] = []
    sent = _run(monkeypatch, sampler, usage_db, script, codes=codes)

    assert codes == [EXIT_INTERNAL_FATAL]
    assert sampler.connect_attempts == 1
    assert "ready" not in _kinds(sent)


def test_the_parked_credit_is_the_probe_that_finds_the_solver_back(
    monkeypatch, usage_db
):
    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        yield _cancel(1)
        yield _topology()
        yield _job(b"down", JOB_GENERATION)
        _wait_for(sent, "reject", 1)
        yield _cancel(JOB_GENERATION + 1)
        _wait_for(sent, "job_request", 2)
        # The solver is back: the re-granted credit's job completes normally.
        yield _job(b"back", JOB_GENERATION + 2)
        _wait_for(sent, "result", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    sent = _run(monkeypatch, FlakySampler(offline_jobs=1), usage_db, script)

    kinds = [m.WhichOneof("msg") for m in sent]
    assert kinds.count("result") == 1
    # A completed job refills its own credit as usual.
    assert _credits(sent)[-1] == 1
