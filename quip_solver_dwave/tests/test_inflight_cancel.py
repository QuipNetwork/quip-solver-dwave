"""In-flight job tracking: the session-loop side of cancellation.

The sampler cancels by job id. Only the session loop knows which generation
each in-flight job belongs to, so it is what decides which ids a Cancel
reaches.
"""

from __future__ import annotations

import threading

from quip_solver_dwave.session_loop import CancelTally, InflightJobs


def test_a_cancel_reaches_only_jobs_at_or_below_the_watermark():
    inflight = InflightJobs()
    inflight.add(b"old", 5)
    inflight.add(b"at", 7)
    inflight.add(b"new", 9)

    assert sorted(inflight.abandoned(7)) == [b"at", b"old"]


def test_a_mempool_job_is_never_cancelled():
    # Generation 0 has no PoW cancellation scope, matching _is_abandoned.
    inflight = InflightJobs()
    inflight.add(b"mempool", 0)

    assert inflight.abandoned(10_000) == []


def test_a_released_job_is_no_longer_cancellable():
    # It already finished; asking SAPI to drop it would count a phantom.
    inflight = InflightJobs()
    inflight.add(b"done", 5)
    inflight.release(b"done")

    assert inflight.abandoned(7) == []


def test_releasing_a_job_that_was_never_added_is_harmless():
    # A job rejected before it ever reached the sampler still runs the same
    # release path on its way out.
    inflight = InflightJobs()
    inflight.release(b"ghost")

    assert inflight.abandoned(7) == []


def test_abandoned_does_not_consume_the_entries():
    # The worker threads own removal; a Cancel only reads. Dropping entries
    # here would strand their release and leak the map.
    inflight = InflightJobs()
    inflight.add(b"a", 3)

    assert inflight.abandoned(5) == [b"a"]
    assert inflight.abandoned(5) == [b"a"]


def test_tracking_survives_workers_finishing_during_a_cancel_sweep():
    # Jobs are added and released on pool threads while the session thread
    # sweeps for a Cancel, so the map is crossed by both by construction.
    inflight = InflightJobs()
    keys = [bytes([i]) for i in range(100)]
    for k in keys:
        inflight.add(k, 5)
    start = threading.Barrier(2)

    def release():
        start.wait()
        for k in keys:
            inflight.release(k)

    def sweep():
        start.wait()
        for _ in range(50):
            inflight.abandoned(7)

    threads = [threading.Thread(target=release), threading.Thread(target=sweep)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert inflight.abandoned(7) == []


def test_a_fresh_tally_reports_nothing_to_measure():
    assert CancelTally().summary() == "no cancels yet"


def test_the_tally_reports_how_often_the_anneal_beat_the_cancel():
    # The point of the counter: SAPI only refunds a problem it has not started
    # annealing, so this is the number that says whether cancelling pays.
    tally = CancelTally()
    tally.requested(4)
    tally.missed()

    assert "4 cancelled" in tally.summary()
    assert "1 annealed anyway" in tally.summary()
    assert "25%" in tally.summary()


def test_the_tally_accumulates_across_rounds():
    tally = CancelTally()
    tally.requested(2)
    tally.requested(2)
    tally.missed()
    tally.missed()

    assert "4 cancelled" in tally.summary()
    assert "2 annealed anyway" in tally.summary()
    assert "50%" in tally.summary()
