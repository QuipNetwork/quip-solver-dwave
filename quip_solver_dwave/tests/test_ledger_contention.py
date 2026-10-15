"""Ledger IO must not stall the thread that reads the coordinator stream.

The session loop handles every Cancel, Job, and Ping on one thread. Anything
it blocks on stalls all of them, which is how a sibling miner ended up idle
for up to 87 seconds after a round change (quip-solver#33). That miner's cause
was a session relaunch; this one's exposure is the usage ledger.

Billing a result writes SQLite and commits, which fsyncs. On the mounted
volume the deployment uses that is milliseconds, not microseconds, and at a
pipeline depth of 96 there are that many workers doing it. If billing holds
the lock the session loop needs to dispatch a job, the loop waits behind every
one of them.

Measured, the exposure is small: at the rate the chip can actually return
results (~23 jobs/s), a dispatch decision stays under 0.14 ms at p99 even with
a 5 ms fsync at depth 96. These tests exist because the shape is wrong rather
than because the current numbers hurt — the cost scales with depth and with
how slow the volume is, and both of those are deployment choices nobody should
have to know about to keep the loop responsive.
"""

from __future__ import annotations

import threading
import time

from quip_solver_dwave.budget import BudgetConfig, BudgetPacer
from quip_solver_dwave.usage import UsageLedger


class SlowLedger(UsageLedger):
    """A ledger whose commit is as slow as a network volume's fsync."""

    def __init__(self, delay_s: float):
        super().__init__(":memory:")
        self._delay = delay_s

    def record(self, access_time_us, *, now=None):
        # Inside the ledger's own lock, where a real fsync stalls.
        with self._lock:
            time.sleep(self._delay)
        super().record(access_time_us, now=now)


def _dispatch_latency_under_billing(bill_under_state_lock: bool) -> float:
    """How long a dispatch waits while exactly one result is billed.

    Models the two lock users in session_loop: a worker thread billing a
    result and the session-loop thread deciding whether a job may be sampled.
    Both take ``state_lock``; only one of them needs to. One billing, one
    measurement, no spinning, so the number means the same thing every run.
    """
    ledger = SlowLedger(delay_s=0.05)
    pacer = BudgetPacer(BudgetConfig(3000.0, 9), ledger)
    now = time.time()
    state_lock = threading.Lock()
    billing = threading.Event()

    def worker():
        if bill_under_state_lock:
            with state_lock:  # what the code did before the fix
                billing.set()
                pacer.record_access_time(43_200, now)
        else:
            billing.set()
            pacer.record_access_time(43_200, now)
            with state_lock:  # bookkeeping only, no IO
                pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    assert billing.wait(timeout=2), "worker never started billing"
    time.sleep(0.005)  # let the 50 ms write get under way

    t0 = time.perf_counter()
    with state_lock:  # what the session loop does per dispatched job
        pass
    waited = time.perf_counter() - t0
    t.join(timeout=2)
    return waited


def test_billing_under_the_dispatch_lock_stalls_the_session_loop():
    # Characterises the bug so the fix below has something to be better than.
    waited = _dispatch_latency_under_billing(bill_under_state_lock=True)
    assert waited > 0.02, (
        "expected dispatch to wait out the 50 ms write when billing holds the "
        f"same lock; it waited {waited * 1000:.1f} ms"
    )


def test_billing_outside_the_dispatch_lock_leaves_it_free():
    # The fix: the ledger has its own lock, so state_lock never needs to be
    # held across its IO.
    waited = _dispatch_latency_under_billing(bill_under_state_lock=False)
    assert waited < 0.01, (
        f"dispatch still blocked {waited * 1000:.1f} ms behind billing"
    )
