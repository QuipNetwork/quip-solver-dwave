"""In-flight SAPI cancellation: the sampler side.

A coordinator Cancel abandons a whole generation. Every job of that generation
still sitting on the QPU is access time being spent on a round the coordinator
has already thrown away, so the miner asks D-Wave to drop it.
"""

from __future__ import annotations

import threading

import numpy as np

import pytest

pytest.importorskip("numpy")  # OceanSampler imports numpy at module load

from dwave.cloud.exceptions import SolverOfflineError  # noqa: E402

from quip_solver_dwave.ocean import OceanSampler  # noqa: E402


class FakeCloudFuture:
    """Stand-in for dwave.cloud.computation.Future.

    Only the two calls the registry makes: ``done()`` to tell a live problem
    from a finished one, and ``cancel()`` to ask SAPI to drop it.
    """

    def __init__(self, done: bool = False):
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self.cancelled = True


def _sampler() -> OceanSampler:
    # mock=False keeps the real submit path (and its registry) without
    # contacting D-Wave; nothing here reaches ensure_connected.
    return OceanSampler(mock=False)


def test_cancel_drops_a_live_inflight_problem():
    s = _sampler()
    fut = FakeCloudFuture(done=False)
    s._register_inflight(b"\x01\x02", fut)

    live = s.cancel_inflight([b"\x01\x02"])

    assert fut.cancelled is True
    assert live == 1


def test_cancel_reports_a_problem_that_already_finished_as_not_live():
    # The anneal beat us. Asking SAPI to cancel is harmless but it did not
    # save any access time, and the counter must not claim it did.
    s = _sampler()
    fut = FakeCloudFuture(done=True)
    s._register_inflight(b"\x03", fut)

    live = s.cancel_inflight([b"\x03"])

    assert live == 0


def test_cancel_ignores_keys_that_were_never_registered():
    s = _sampler()
    assert s.cancel_inflight([b"\xff"]) == 0


def test_a_cancel_that_arrives_before_the_submit_lands_still_cancels():
    # The cloud Future only exists once _submit_sync has run on a pool thread
    # and SAPI has accepted the problem. A Cancel inside that window finds an
    # empty registry, and that window is exactly when cancelling pays best.
    s = _sampler()

    s.cancel_inflight([b"\x04"])  # nothing registered yet
    fut = FakeCloudFuture(done=False)
    s._register_inflight(b"\x04", fut)

    assert fut.cancelled is True


def test_a_later_job_reusing_a_cancelled_key_is_not_cancelled():
    # The pending-cancel note is consumed by the registration it applies to,
    # or a key would stay poisoned for the rest of the session.
    s = _sampler()
    s.cancel_inflight([b"\x05"])
    s._register_inflight(b"\x05", FakeCloudFuture())
    s._release_inflight(b"\x05")

    later = FakeCloudFuture(done=False)
    s._register_inflight(b"\x05", later)

    assert later.cancelled is False


def test_releasing_an_inflight_key_stops_a_later_cancel_touching_it():
    s = _sampler()
    fut = FakeCloudFuture(done=False)
    s._register_inflight(b"\x06", fut)
    s._release_inflight(b"\x06")

    assert s.cancel_inflight([b"\x06"]) == 0
    assert fut.cancelled is False


def test_registry_survives_concurrent_registration_and_cancellation():
    # Submits run on the sampler's own pool while the session thread handles
    # Cancel, so the registry is crossed by two threads by construction.
    s = _sampler()
    futures = [FakeCloudFuture(done=False) for _ in range(50)]
    keys = [bytes([i]) for i in range(50)]
    start = threading.Barrier(2)

    def register():
        start.wait()
        for k, f in zip(keys, futures):
            s._register_inflight(k, f)

    def cancel():
        start.wait()
        for _ in range(10):
            s.cancel_inflight(keys)

    threads = [threading.Thread(target=register), threading.Thread(target=cancel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every future is either cancelled outright or carries a pending note that
    # the final sweep honours; no key escapes both.
    s.cancel_inflight(keys)
    assert all(f.cancelled for f in futures)


class RecordingSampler:
    """Captures the kwargs handle_job passes down to the sampler."""

    def __init__(self):
        self.calls = []

    def sample(self, nodes, h, edges, j, **kwargs):
        from quip_solver_dwave.ocean import SampleResult

        self.calls.append(kwargs)
        return SampleResult(
            spins=np.array([[1, -1]], dtype=np.int8),
            variables=[0, 1],
            energies=[-1.0],
            device_access_time_us=46_000,
            num_reads=1,
        )


def _job(job_id: bytes, generation: int = 7):
    import time

    from quip_solver_core import miner_pb2, wire

    return miner_pb2.Job(
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


def test_handle_job_passes_the_job_id_as_the_cancel_key():
    # The session loop cancels by job id, so that is what has to reach the
    # sampler's registry; nonce_seed is the defect-clamp seed, not a handle.
    from quip_solver_dwave.job import handle_job

    rec = RecordingSampler()
    handle_job(
        _job(b"\xaa\xbb\xcc\xdd"),
        rec,
        session_nodes=[0, 1],
        session_edges=[(0, 1)],
    )

    assert rec.calls[0]["cancel_key"] == b"\xaa\xbb\xcc\xdd"


# --- Conservative accounting for submissions we cannot observe -------------
#
# A cancel that beats the anneal costs nothing; a cancel that loses is billed
# by D-Wave in full. The raised path carries no timing to tell them apart, so
# the ledger must assume the anneal ran rather than assume it did not.


class _Boom(Exception):
    pass


# The result-bearing attributes a real dwave.cloud.computation.Future exposes.
# All of them funnel through Future._load_result, which re-raises whatever
# exception SAPI reported for the problem, so a faithful double must raise on
# any of these, not just whichever one answer_view happens to read first.
_RESULT_ATTRS = {"samples", "energies", "num_occurrences", "variables", "timing"}


class CancelledFuture:
    """A cloud problem SAPI accepted and then cancelled: every result-bearing
    attribute raises."""

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        pass

    def __getattr__(self, name):
        # __getattr__ only runs for attributes not already found, so `record`
        # still raises AttributeError and answer_view's
        # getattr(raw, "record", None) probe still returns None cleanly.
        if name in _RESULT_ATTRS:
            raise _Boom("problem cancelled")
        raise AttributeError(name)


def _real_mode_sampler(submit_result):
    """A sampler on the real (non-mock) submit path, with the cloud stubbed.

    Everything up to _submit_encoded runs for real: the arrays are encoded
    against a solver ordering and wrapped in a submission body. Only the final
    hand-off to the cloud client is replaced, which is the one method that
    touches Ocean internals.
    """
    s = OceanSampler(mock=False)
    s._connected = True
    s._is_mock = False

    class _Identity:
        def dict(self):
            return {"name": "FakeSolver", "version": {"graph_id": "x"}}

    class _Solver:
        _encoding_qubits = [0, 1]
        _encoding_couplers = [(0, 1)]
        _params: dict = {}
        parameters = {"num_reads": None, "annealing_time": None, "label": None}
        identity = _Identity()

        def _format_params(self, type_, params):
            pass

    class _Sampler:
        solver = _Solver()

    s.sampler = _Sampler()
    submitted = []

    def _submit_encoded(solver, body, cancel_key):
        submitted.append(body)
        if isinstance(submit_result, Exception):
            raise submit_result
        if cancel_key is not None:
            s._register_inflight(cancel_key, submit_result)
        return submit_result

    s._submit_encoded = _submit_encoded  # type: ignore[method-assign]
    s.submitted = submitted  # type: ignore[attr-defined]
    return s


def _one_qubit_job(s, cancel_key):
    """The smallest problem the stub solver accepts."""
    return s.sample(
        np.array([0, 1]),
        np.array([1.0, -1.0]),
        np.array([(0, 1)]),
        np.array([0.5]),
        num_reads=1,
        cancel_key=cancel_key,
    )


def test_a_cancelled_problem_is_billed_the_conservative_estimate():
    # SAPI accepted it, so D-Wave may have annealed and charged for it. We
    # cannot see the timing, so the ledger assumes the worst.
    s = _real_mode_sampler(CancelledFuture())
    s._observe_access_us(46_000)  # a prior job establishes the going rate

    with pytest.raises(_Boom):
        _one_qubit_job(s, b"\x11")

    assert s.drain_unobserved_access_us() == 46_000


def test_draining_the_unobserved_charge_clears_it():
    # The session loop records it into the pacer exactly once.
    s = _real_mode_sampler(CancelledFuture())
    s._observe_access_us(46_000)
    with pytest.raises(_Boom):
        _one_qubit_job(s, b"\x12")

    s.drain_unobserved_access_us()

    assert s.drain_unobserved_access_us() == 0


class OfflineFuture(CancelledFuture):
    """A cloud problem SAPI accepted while the solver was down: the status
    comes back as failed with "Solver is offline" and nothing annealed."""

    def __getattr__(self, name):
        if name in _RESULT_ATTRS:
            raise SolverOfflineError("Solver is offline.")
        raise AttributeError(name)


def test_a_problem_refused_by_an_offline_solver_is_billed_nothing():
    # SAPI accepted the problem but the solver never ran it, so D-Wave charged
    # nothing. Booking the estimate here spent 342 s of budget on 7,623
    # rejects in one qblock during the Advantage2_system1 outage.
    s = _real_mode_sampler(OfflineFuture())
    s._observe_access_us(46_000)

    with pytest.raises(SolverOfflineError):
        _one_qubit_job(s, b"\x15")

    assert s.drain_unobserved_access_us() == 0


def test_a_submit_that_never_reached_sapi_is_billed_nothing():
    # The problem died in this process. D-Wave never saw it, so charging for
    # it would burn quota the QPU never spent.
    s = _real_mode_sampler(_Boom("network down"))
    s._observe_access_us(46_000)

    with pytest.raises(_Boom):
        _one_qubit_job(s, b"\x13")

    assert s.drain_unobserved_access_us() == 0


def test_the_estimate_uses_the_largest_access_time_seen_this_session():
    # Conservative among observations: a cancelled job is assumed to be at
    # least as expensive as the priciest one we have actually measured.
    s = _real_mode_sampler(CancelledFuture())
    s._observe_access_us(46_000)
    s._observe_access_us(120_000)
    s._observe_access_us(46_000)

    with pytest.raises(_Boom):
        _one_qubit_job(s, b"\x14")

    assert s.drain_unobserved_access_us() == 120_000


def test_a_cancelled_problem_is_registered_while_live_and_released_after():
    # Registered, or a Cancel has nothing to call. Released, or the next sweep
    # reports a finished problem as live and inflates the hit counter.
    seen = {}

    class WatchingFuture(CancelledFuture):
        def __getattr__(self, name):
            if name in _RESULT_ATTRS:
                seen["registered"] = b"\x15" in s._inflight
                raise _Boom("problem cancelled")
            raise AttributeError(name)

    s = _real_mode_sampler(WatchingFuture())
    with pytest.raises(_Boom):
        _one_qubit_job(s, b"\x15")

    assert seen["registered"] is True
    assert s._inflight == {}
