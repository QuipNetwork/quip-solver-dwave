"""End to end: a completed job lands in the history and billing is unchanged.

Same scripted-coordinator harness as test_cancel_accounting.py. The history
is optional and the ledger is not, so the one property that must survive
any failure here is that the ledger bills exactly what it billed before.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import List, cast

import numpy as np
import pytest

from quip_solver_core import miner_pb2, wire

from quip_solver_dwave import cli, session_loop
from quip_solver_dwave.budget import DEFAULT_USAGE_DB, usage_db_from_backend_toml
from quip_solver_dwave.history import HistoryStore
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.usage import UsageLedger

ACCESS_US = 46_000


class QuickSampler:
    """Answers at once with the timing extras the cloud sampler would attach."""

    def __init__(self):
        self.entered = threading.Event()
        # Set already: sample() returns at once. A test can clear it to hold
        # an anneal open while it changes state (e.g. sends a Cancel) that
        # the result should see once the anneal is allowed to finish.
        self.hold = threading.Event()
        self.hold.set()
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
        self.entered.set()
        self.hold.wait(timeout=5)
        return SampleResult(
            spins=np.array([[1, -1]], dtype=np.int8),
            variables=[0, 1],
            energies=[-1.0],
            device_access_time_us=ACCESS_US,
            num_reads=1,
            extra={"mock": "0", "inflight": "2", "sapi_ms": "2900"},
        )


def _configure(usage_db: str) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        configure=miner_pb2.Configure(
            queue_depth=3,
            idle_timeout_s=300,
            backend_toml=f'budget = "3000s"\nusage_db = "{usage_db}"\n',
        )
    )


def _configure_without_budget(usage_db: str) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        configure=miner_pb2.Configure(
            queue_depth=3,
            idle_timeout_s=300,
            backend_toml=f'usage_db = "{usage_db}"\n',
        )
    )


def _job(generation: int) -> miner_pb2.CoordMsg:
    return miner_pb2.CoordMsg(
        job=miner_pb2.Job(
            job_id=b"history-1",
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


def _wait_until(condition, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


def _run(
    monkeypatch,
    sampler: QuickSampler,
    usage_db: str,
    attempts_dir: str,
    *,
    expect_job: bool = True,
) -> List:
    sent: List = []

    def responses():
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        # Cancel(1) opens the round whose jobs carry generation 2.
        yield _cancel(1)
        yield miner_pb2.CoordMsg(
            topology=miner_pb2.Topology(
                nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
            )
        )
        # A target of 0 milli: the -1.0 energy result clears it, so hits == 1.
        yield miner_pb2.CoordMsg(
            set_target=miner_pb2.SetTarget(max_energy_milli=0, min_solutions=1)
        )
        yield _job(generation=2)
        if expect_job:
            assert sampler.entered.wait(timeout=5), "job never reached the sampler"
            # Wait for the worker to fold the result in before the boundary,
            # instead of guessing how long that takes: its Result is what the
            # drain thread appends to `sent` once the reply is enqueued.
            assert _wait_until(
                lambda: any(m.WhichOneof("msg") == "result" for m in sent)
            ), "job result never reached the outbound queue"
        yield _cancel(2)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            def drain():
                for msg in request_iter:
                    sent.append(msg)

            threading.Thread(target=drain, daemon=True).start()
            return responses()

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel())
    monkeypatch.setattr(session_loop.grpc, "channel_ready_future", lambda ch: _Ready())
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")

    session_loop.run_session(
        "unix:///tmp/nope.sock",
        "qpu-test",
        cast(OceanSampler, sampler),
        attempts_dir=attempts_dir,
    )
    return sent


@pytest.fixture
def usage_db(tmp_path) -> str:
    return str(tmp_path / "usage.db")


def test_usage_db_from_backend_toml_defaults_and_reads_the_key():
    assert usage_db_from_backend_toml("") == DEFAULT_USAGE_DB
    assert usage_db_from_backend_toml("budget = 250m") == DEFAULT_USAGE_DB  # unparsable
    assert usage_db_from_backend_toml('budget = "250m"') == DEFAULT_USAGE_DB
    assert usage_db_from_backend_toml('usage_db = "/tmp/x.db"') == "/tmp/x.db"


def test_a_completed_job_lands_in_the_history_and_billing_is_unchanged(
    monkeypatch, usage_db, tmp_path
):
    sent = _run(monkeypatch, QuickSampler(), usage_db, str(tmp_path / "attempts"))

    assert UsageLedger(usage_db).spent_us_since(0.0) == ACCESS_US
    assert any(m.WhichOneof("msg") == "result" for m in sent)

    store = HistoryStore(usage_db)
    (hour,) = store.hourly_rows(0)
    assert hour["source"] == "live"
    assert (hour["jobs"], hour["reads"], hour["wasted_jobs"]) == (1, 1, 0)
    assert hour["access_us_sum"] == ACCESS_US
    assert hour["inflight_sum"] == 2
    assert (hour["sapi_ms_sum"], hour["sapi_jobs"]) == (2900, 1)
    assert hour["rtt_ms_sum"] >= 0 and hour["busy_ms"] >= 0

    rows = {r["generation"]: r for r in store.rounds(since_ts=0, limit=10)}
    assert set(rows) == {2, 3}
    # A recorder means a strategy is attached (Task 4), so the reason comes
    # from its verdict, not the old blanket "budget"; with no history yet
    # that verdict is "no-data" and it still joins every round.
    assert rows[2]["joined"] == 1 and rows[2]["reason"] == "no-data"
    # A no-data round makes no prediction, so the calibration column stays
    # NULL rather than recording a fabricated 0.0.
    assert rows[2]["expected_jobs"] is None
    assert rows[2]["target_milli"] == 0
    assert (rows[2]["jobs"], rows[2]["hits"], rows[2]["best_energy_milli"]) == (1, 1, -1000)
    assert rows[2]["end_ts_s"] is not None and rows[3]["end_ts_s"] is None
    assert store.margin_counts(0) == {-1: 1}


def test_a_history_failure_leaves_billing_and_the_reply_untouched(
    monkeypatch, usage_db, tmp_path
):
    def boom(self, sample, round_start_ts):
        raise RuntimeError("disk full")

    monkeypatch.setattr(HistoryStore, "record_job", boom)
    sent = _run(monkeypatch, QuickSampler(), usage_db, str(tmp_path / "attempts"))
    assert UsageLedger(usage_db).spent_us_since(0.0) == ACCESS_US
    assert any(m.WhichOneof("msg") == "result" for m in sent)


def test_an_abandoned_result_lands_in_wasted_jobs_not_jobs(monkeypatch, usage_db, tmp_path):
    # A job still in flight when its generation is cancelled comes back as a
    # "result" anyway (SAPI ran the anneal despite the cancel), and that
    # completed anneal must count against wasted_jobs, not jobs, or the
    # estimator's denominator for expected_jobs is wrong.
    sampler = QuickSampler()
    sampler.hold.clear()  # hold the anneal open until the cancel lands
    sent: List = []

    def responses():
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure(usage_db)
        # A qblock boundary before the job: budget pacing only grants
        # credits, and lets the gate's job check pass, once one has run.
        yield _cancel(1)
        yield miner_pb2.CoordMsg(
            topology=miner_pb2.Topology(
                nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b""
            )
        )
        yield _job(generation=5)
        assert sampler.entered.wait(timeout=5), "job never reached the sampler"
        yield _cancel(5)  # abandons generation 5 while the anneal is still held
        sampler.hold.set()  # only now let it finish, already abandoned
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            def drain():
                for msg in request_iter:
                    sent.append(msg)

            threading.Thread(target=drain, daemon=True).start()
            return responses()

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel())
    monkeypatch.setattr(session_loop.grpc, "channel_ready_future", lambda ch: _Ready())
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")

    session_loop.run_session(
        "unix:///tmp/nope.sock",
        "qpu-test",
        cast(OceanSampler, sampler),
        attempts_dir=str(tmp_path / "attempts"),
    )

    # SPEC section 5: no Result for an abandoned generation.
    assert not any(m.WhichOneof("msg") == "result" for m in sent)
    (hour,) = HistoryStore(usage_db).hourly_rows(0)
    assert (hour["jobs"], hour["wasted_jobs"]) == (0, 1)


def test_a_budget_named_after_the_recorder_exists_still_gets_a_strategy(
    monkeypatch, usage_db, tmp_path, caplog
):
    # First Configure names no budget: the recorder and refresher get built,
    # but there is no gate for a strategy to attach to. Second Configure
    # names a budget: the gate is created only now, after the refresher
    # already exists, so the attach has to happen at the gate too, not only
    # at the recorder.
    sampler = QuickSampler()
    sent: List = []

    def responses():
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=1))
        yield _configure_without_budget(usage_db)
        yield _configure(usage_db)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    class _Stub:
        def __init__(self, channel):
            pass

        def Session(self, request_iter):
            def drain():
                for msg in request_iter:
                    sent.append(msg)

            threading.Thread(target=drain, daemon=True).start()
            return responses()

    class _Channel:
        def close(self):
            pass

    class _Ready:
        def result(self, timeout=None):
            return True

    monkeypatch.setattr(session_loop.grpc, "insecure_channel", lambda *a, **k: _Channel())
    monkeypatch.setattr(session_loop.grpc, "channel_ready_future", lambda ch: _Ready())
    monkeypatch.setattr(session_loop.miner_pb2_grpc, "MinerServiceStub", _Stub)
    monkeypatch.setenv("QUIP_SESSION_TOKEN", "test-token")

    with caplog.at_level(logging.INFO):
        session_loop.run_session(
            "unix:///tmp/nope.sock",
            "qpu-test",
            cast(OceanSampler, sampler),
            attempts_dir=str(tmp_path / "attempts"),
        )

    assert any("round strategy attached" in r.getMessage() for r in caplog.records)


def test_an_unreadable_history_path_does_not_stop_mining(monkeypatch, tmp_path):
    # The ledger opens fine; only the history is broken. Mining continues.
    usage_db = str(tmp_path / "usage.db")
    monkeypatch.setattr(
        session_loop.HistoryRecorder, "open", classmethod(lambda cls, path: None)
    )
    sent = _run(monkeypatch, QuickSampler(), usage_db, str(tmp_path / "attempts"))
    assert any(m.WhichOneof("msg") == "result" for m in sent)


def test_the_clean_shutdown_path_signals_end_of_outbound_before_closing_history(
    monkeypatch, usage_db, tmp_path
):
    # Results already queued keep flowing while the history joins run, but
    # only until the coordinator's grace window expires and closes the
    # stream. _STOP must reach out_q before the joins and the close spend
    # any of that window, or billed anneals still in the queue are lost.
    events: List[str] = []

    orig_put = queue.Queue.put

    def tracking_put(self, item, *a, **kw):
        if item is session_loop._STOP:
            events.append("stop")
        return orig_put(self, item, *a, **kw)

    orig_close = session_loop.HistoryRecorder.close

    def tracking_close(self):
        events.append("close")
        return orig_close(self)

    monkeypatch.setattr(queue.Queue, "put", tracking_put)
    monkeypatch.setattr(session_loop.HistoryRecorder, "close", tracking_close)

    _run(monkeypatch, QuickSampler(), usage_db, str(tmp_path / "attempts"))

    assert "stop" in events and "close" in events
    assert events.index("stop") < events.index("close")


def test_attempts_dir_flag_reaches_run_session(monkeypatch):
    seen = {}

    def fake(coordinator, miner_id, sampler, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_session_sync", fake)
    assert cli.main(["--quip-coordinator", "unix:///x", "--mock", "--attempts-dir", "/tmp/a"]) == 0
    assert seen["attempts_dir"] == "/tmp/a"
    assert cli.main(["--quip-coordinator", "unix:///x", "--mock"]) == 0
    assert seen["attempts_dir"] is None
