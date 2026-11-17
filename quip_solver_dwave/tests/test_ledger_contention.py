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

from quip_solver_core import miner_pb2

from quip_solver_dwave import budget
from quip_solver_dwave.tests.test_cancel_accounting import (
    JOB_GENERATION,
    BlockingSampler,
    _job,
)
from quip_solver_dwave.tests.test_family_leases import Coordinator, target
from quip_solver_dwave.usage import UsageLedger


def test_a_cancel_is_answered_while_a_worker_is_billing(monkeypatch, tmp_path):
    # The worker's billing write is held open. A session that billed under
    # state_lock would answer the Cancel only after the write, because the
    # Cancel handler takes state_lock to raise the watermark.
    billing = threading.Event()
    written = threading.Event()

    class HeldLedger(UsageLedger):
        def record(self, access_time_us, *, now=None):
            billing.set()
            assert written.wait(timeout=30)
            super().record(access_time_us, now=now)

    monkeypatch.setattr(budget, "UsageLedger", HeldLedger)
    sampler = BlockingSampler()
    sampler.release.set()
    coord = Coordinator(monkeypatch, sampler)
    topology = miner_pb2.Topology(nodes=[0, 1], edges=miner_pb2.EdgeList(u=[0], v=[1]), hash=b"")
    toml = f'budget = "3000s"\nusage_db = "{tmp_path / "usage.db"}"\n'
    coord.start(topology, target(), depth=3, toml=toml)
    # A boundary the gate can join on, so the job is not refused on budget.
    coord.send(cancel=miner_pb2.Cancel(max_generation=1))
    coord.inbox.put(_job())
    assert billing.wait(timeout=10), "the job was never billed"

    watermark = JOB_GENERATION + 1
    coord.send(cancel=miner_pb2.Cancel(max_generation=watermark))
    try:
        coord.wait_for(
            lambda sent: any(
                m.WhichOneof("msg") == "status" and m.status.abandoned_generation == watermark
                for m in sent
            ),
            timeout=5,
        )
    finally:
        written.set()
    coord.finish()
