"""Salt leases against a fake device: admission, accounting, and the stops.

Every lease takes one credit and gives one back with exactly one
``LeaseDone``. These tests pin what ``salts_done`` and ``best_energy_milli``
count, that no salt is drawn after a stop, that a stopped lease drops late
results while their access time is still billed, and that the problems are
the ones the canonical wheel draws.
"""

from __future__ import annotations

import queue
import re
import threading
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, cast

from dwave.graphs import zephyr_graph
import numpy as np
import pytest

from quip_solver_core import families, lease as lease_sdk, miner_pb2, scoring, wire

from quip_solver_dwave import leases, session_loop
from quip_solver_dwave.leases import LeaseRejected, LeaseTopology, SaltProblem, admit_lease
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.usage import UsageLedger

ACCESS_US = 40_000
GENERATION = 7
TWISTED = "quip/twisted-ice-dimer"
# Adjacency mode, weak_milli 45, pairs 6: admits on any Zephyr topology.
P_TWISTED_ADJ = bytes(
    [1, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0x2D, 0, 4, 0, 4, 0, 0, 0, 6, 0, 0, 0]
)
# The same layout with weak_milli 1000, outside the schema bounds (code 9).
P_TWISTED_W1000 = P_TWISTED_ADJ[:22] + bytes([0xE8, 0x03]) + P_TWISTED_ADJ[24:]


# -- inputs -------------------------------------------------------------------


def ring_topology() -> miner_pb2.Topology:
    """The conformance driver's lease topology: an eight-node ring with chords."""
    return miner_pb2.Topology(
        hash=b"\x33" * 32,
        nodes=list(range(8)),
        edges=miner_pb2.EdgeList(
            u=[0, 1, 2, 3, 4, 5, 6, 0, 0, 1, 2, 3], v=[1, 2, 3, 4, 5, 6, 7, 7, 4, 5, 6, 7]
        ),
        allowed_h_milli=[-1000, 0, 1000],
        allowed_j_milli=[-1000, 1000],
    )


def zephyr_topology(m: int = 2, t: int = 2) -> miner_pb2.Topology:
    g = zephyr_graph(m, t)
    edges = sorted((min(u, v), max(u, v)) for u, v in g.edges)
    return miner_pb2.Topology(
        hash=b"\x5a" * 32,
        nodes=sorted(g.nodes),
        edges=miner_pb2.EdgeList(u=[u for u, _ in edges], v=[v for _, v in edges]),
    )


def target(**kw) -> miner_pb2.SetTarget:
    fields = dict(max_energy_milli=0, min_solutions=1, max_proof_solutions=32, num_reads=2)
    fields.update(kw)
    return miner_pb2.SetTarget(**fields)


def generator(topo: miner_pb2.Topology, count: int = 4, **kw) -> miner_pb2.IsingProblemGenerator:
    gen = miner_pb2.IsingProblemGenerator(
        algorithm=leases.LEGACY,
        topology_hash=topo.hash,
        last_proof_block_hash=b"\x11" * 32,
        miner_account=b"\x22" * 32,
        base_salt=b"\x44" * 32,
        salt_start=10,
        salt_count=count,
    )
    for key, value in kw.items():
        if key == "family":
            gen.family.CopyFrom(value)
        else:
            setattr(gen, key, value)
    return gen


def family_generator(topo: miner_pb2.Topology, count: int = 3, params: bytes = P_TWISTED_ADJ, **kw):
    descriptor = kw.pop(
        "descriptor",
        miner_pb2.FamilyDescriptor(
            family_id=TWISTED, revision=1, params=params, params_digest=families.params_digest(params)
        ),
    )
    snapshot = LeaseTopology.from_proto(topo)
    kw.setdefault("topology_order_digest", snapshot.order_digest)
    return generator(topo, count, algorithm=leases.FAMILY, family=descriptor, **kw)


def lease_job(gen: miner_pb2.IsingProblemGenerator, job_id: bytes = b"lease-1", **kw) -> miner_pb2.Job:
    kw.setdefault("generation", GENERATION)
    return miner_pb2.Job(job_id=job_id, kind=miner_pb2.ISING_GENERATE, generator=gen, **kw)


def admit(job, topo=None, tgt=None, **kw):
    snapshot = LeaseTopology.from_proto(topo if topo is not None else ring_topology())
    return admit_lease(job, snapshot, tgt if tgt is not None else target(), max_nodes=10_000, max_edges=100_000, **kw)


def reason_of(job, topo=None, tgt=None, **kw) -> int:
    with pytest.raises(LeaseRejected) as info:
        admit(job, topo, tgt, **kw)
    return info.value.reason


# -- admission -----------------------------------------------------------------


def test_legacy_admission_runs_in_the_spec_order():
    ring = ring_topology()
    past = int(time.time() * 1000) - 1
    # Each failure shadows every later one, so each case also breaks the next check.
    assert reason_of(lease_job(generator(ring, base_salt=b"x"), deadline_ms=past), tgt=target()) == miner_pb2.EXPIRED
    with pytest.raises(LeaseRejected) as info:
        admit_lease(lease_job(generator(ring)), None, None, max_nodes=0, max_edges=0)
    assert info.value.reason == miner_pb2.TOPOLOGY_MISSING
    assert reason_of(lease_job(generator(ring)), tgt=target(max_proof_solutions=0)) == miner_pb2.TARGET_MISSING
    no_generator = miner_pb2.Job(job_id=b"x", kind=miner_pb2.ISING_GENERATE)
    assert reason_of(no_generator) == miner_pb2.MALFORMED
    for bad in (
        generator(ring, base_salt=b"short"),
        generator(ring, salt_count=0),
        generator(ring, salt_start=2**64 - 2, salt_count=4),
        generator(ring, algorithm=0),
        # FAM-48: a legacy lease carries no family data (code 2).
        generator(ring, topology_order_digest=b"\0" * 32),
        generator(ring, family=miner_pb2.FamilyDescriptor(family_id=TWISTED, revision=1)),
    ):
        assert reason_of(lease_job(bad, deadline_ms=0)) == miner_pb2.MALFORMED
    assert reason_of(lease_job(generator(ring, topology_hash=b"\x34" * 32))) == miner_pb2.TOPOLOGY_MISMATCH
    no_values = miner_pb2.Topology()
    no_values.CopyFrom(ring)
    del no_values.allowed_j_milli[:]
    assert reason_of(lease_job(generator(ring)), topo=no_values) == miner_pb2.MALFORMED
    repeated = miner_pb2.Topology()
    repeated.CopyFrom(ring)
    repeated.nodes[1] = 0
    assert reason_of(lease_job(generator(ring)), topo=repeated) == miner_pb2.MALFORMED
    snapshot = LeaseTopology.from_proto(ring)
    with pytest.raises(LeaseRejected) as info:
        admit_lease(lease_job(generator(ring)), snapshot, target(), max_nodes=4, max_edges=0)
    assert info.value.reason == miner_pb2.TOO_LARGE


def _descriptor(**kw) -> miner_pb2.FamilyDescriptor:
    descriptor = miner_pb2.FamilyDescriptor(
        family_id=TWISTED, revision=1, params=P_TWISTED_ADJ, params_digest=families.params_digest(P_TWISTED_ADJ)
    )
    for key, value in kw.items():
        setattr(descriptor, key, value)
    return descriptor


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        # Code 1: a family lease without a 32-byte order digest.
        ({"topology_order_digest": b"\0" * 31}, miner_pb2.MALFORMED),
        # Code 3 beats code 5: the id breaks the grammar before it is looked up.
        ({"descriptor": _descriptor(family_id="X")}, miner_pb2.MALFORMED),
        ({"descriptor": _descriptor(revision=0)}, miner_pb2.MALFORMED),
        # Codes 5 and 6 come from this miner's list, not the wheel's registry.
        ({"descriptor": _descriptor(family_id="quip/unregistered")}, miner_pb2.FAMILY_UNSUPPORTED),
        ({"descriptor": _descriptor(revision=2)}, miner_pb2.FAMILY_UNSUPPORTED),
        ({"descriptor": _descriptor(params=b"\x02", params_digest=families.params_digest(b"\x02"))}, miner_pb2.MALFORMED),
        ({"descriptor": _descriptor(params_digest=b"\0" * 32)}, miner_pb2.MALFORMED),
        (
            {"descriptor": _descriptor(params=P_TWISTED_W1000, params_digest=families.params_digest(P_TWISTED_W1000))},
            miner_pb2.MALFORMED,
        ),
        # Code 5 beats the topology hash, which beats code 11.
        (
            {"descriptor": _descriptor(family_id="quip/unregistered"), "topology_hash": b"\0" * 32},
            miner_pb2.FAMILY_UNSUPPORTED,
        ),
        ({"topology_hash": b"\0" * 32, "topology_order_digest": b"\1" * 32}, miner_pb2.TOPOLOGY_MISMATCH),
        ({"topology_order_digest": b"\1" * 32}, miner_pb2.TOPOLOGY_MISMATCH),
    ],
)
def test_family_admission_codes_map_to_their_wire_reasons(change, reason):
    topo = zephyr_topology()
    assert reason_of(lease_job(family_generator(topo, **change)), topo=topo) == reason


def test_a_family_that_cannot_build_on_the_topology_or_fit_the_limits_is_refused():
    topo = zephyr_topology()
    # Four nodes cannot hold a twisted-ice tile (code 12).
    tiny = miner_pb2.Topology(
        hash=topo.hash, nodes=[0, 1, 2, 3], edges=miner_pb2.EdgeList(u=[0, 1, 2], v=[1, 2, 3])
    )
    assert reason_of(lease_job(family_generator(tiny)), topo=tiny) == miner_pb2.FAMILY_UNSUPPORTED
    # A size bound above the advertised max_nodes (code 13 at admission).
    snapshot = LeaseTopology.from_proto(topo)
    with pytest.raises(LeaseRejected) as info:
        admit_lease(lease_job(family_generator(topo)), snapshot, target(), max_nodes=10, max_edges=0)
    assert info.value.reason == miner_pb2.TOO_LARGE


def test_an_unadvertised_family_is_refused_even_when_the_wheel_has_it():
    topo = zephyr_topology()
    job = lease_job(family_generator(topo))
    assert reason_of(job, topo=topo, generated=()) == miner_pb2.FAMILY_UNSUPPORTED


# -- the salt problems are the wheel's ------------------------------------------


def test_legacy_salts_are_the_canonical_draws():
    ring = ring_topology()
    gen = generator(ring)
    admitted = admit(lease_job(gen))
    reference = lease_sdk.Lease(gen.SerializeToString())
    edges = list(zip(ring.edges.u, ring.edges.v))
    for index in range(4):
        problem = admitted.problem(index)
        assert problem is not None
        assert problem.salt == bytes(reference.salt(index))
        assert problem.salt[:8] == (10 + index).to_bytes(8, "little")
        assert problem.nonce == bytes(lease_sdk.derive_nonce(b"\x11" * 32, b"\x22" * 32, problem.salt))
        h, j = lease_sdk.draw(problem.nonce, list(ring.nodes), edges, list(ring.allowed_h_milli), list(ring.allowed_j_milli))
        assert problem.h_milli.tolist() == h and problem.j_milli.tolist() == j
        assert problem.edges.tolist() == [list(e) for e in edges]
        assert problem.offset_milli == 0


def test_family_salts_are_the_canonical_instances_and_score_with_their_offset():
    topo = zephyr_topology()
    gen = family_generator(topo)
    admitted = admit(lease_job(gen), topo=topo)
    snapshot = LeaseTopology.from_proto(topo)
    rng = np.random.default_rng(5)
    for index in range(3):
        problem = admitted.problem(index)
        assert problem is not None
        instance = families.generate(
            TWISTED, 1, P_TWISTED_ADJ, families.params_digest(P_TWISTED_ADJ), snapshot.nodes, snapshot.edges, problem.nonce
        )
        assert problem.nodes.tolist() == list(instance.node_ids)
        assert problem.h_milli.tolist() == list(instance.h)
        assert problem.j_milli.tolist() == list(instance.j)
        assert problem.offset_milli == instance.offset_milli
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), (3, len(instance.node_ids)))
        edges = [tuple(e) for e in instance.edges]
        # scoring.energy_milli takes coefficients in units and returns milli.
        h_units = [v / 1000 for v in instance.h]
        j_units = [v / 1000 for v in instance.j]
        want = [
            scoring.energy_milli(row.tolist(), h_units, j_units, edges) + instance.offset_milli
            for row in spins
        ]
        assert problem.energies_milli(spins).tolist() == want


def test_a_later_topology_does_not_change_an_admitted_lease():
    ring = ring_topology()
    admitted = admit(lease_job(generator(ring)))
    before = admitted.problem(2)
    ring.allowed_h_milli[:] = [5]
    after = admitted.problem(2)
    assert before is not None and after is not None
    assert before.h_milli.tolist() == after.h_milli.tolist()


def test_scoring_a_batch_keeps_spin_workspaces_compact():
    # At 48 reads and full chip size, widening both gathered endpoints and
    # their product to int64 costs about 48 MB per call, on every job worker.
    n = 256
    positions = np.column_stack(np.triu_indices(n, k=1))
    problem = SaltProblem(
        index=0, salt=b"", nonce=b"", nodes=np.arange(n),
        h_milli=np.zeros(n, dtype=np.int64), positions=positions,
        j_milli=np.full(len(positions), 1000, dtype=np.int64),
    )
    spins = np.ones((48, n), dtype=np.int8)
    tracemalloc.start()
    try:
        energies = problem.energies_milli(spins)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert energies.tolist() == [len(positions) * 1000] * 48
    # Allow four bytes per read/coupler, plus fixed allocation overhead.
    # This bounds memory independently of machine speed or scheduler load.
    assert peak < 4 * 48 * len(positions) + 1_000_000


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_batch_scoring_accumulates_large_coefficients_exactly(layout):
    problem = SaltProblem(
        index=0, salt=b"", nonce=b"", nodes=np.array([8, 2, 7]),
        h_milli=np.array([2**31 - 1, -(2**31), 17], dtype=np.int64),
        positions=np.array([[0, 1], [1, 2], [0, 2]]),
        j_milli=np.array([2**31 - 1, -(2**31), 23], dtype=np.int64),
        offset_milli=31,
    )
    spins = np.array([[1, 1, 1], [-1, 1, -1]], dtype=np.int8, order="F" if layout == "F" else "C")
    if layout == "strided":
        spins = np.repeat(spins, 2, axis=1)[:, ::2]
    assert problem.energies_milli(spins).tolist() == [69, -4294967257]
    assert problem.energies_milli(spins[:0]).shape == (0,)


# -- the session, against a fake device ----------------------------------------


class FakeDevice:
    """The sampler surface the session uses, with per-salt control.

    ``hold`` names salt indices that wait on the device until released;
    ``reads`` maps a salt index to its read count (default 2); ``fail`` names
    salts whose sampling raises.
    """

    def __init__(self, *, hold=(), reads: Optional[Dict[int, int]] = None, fail=()):
        self.native_topology_hash = None
        self.lock = threading.Lock()
        self.calls: List[int] = []
        self.keys: List[bytes] = []
        self.cancelled: List[bytes] = []
        self.active = 0
        self.max_active = 0
        self.entered: Dict[int, threading.Event] = {i: threading.Event() for i in range(64)}
        self.release: Dict[int, threading.Event] = {i: threading.Event() for i in range(64)}
        for i in range(64):
            if i not in hold:
                self.release[i].set()
        self.reads = reads or {}
        self.fail = set(fail)

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
        return 0

    def sample(self, nodes, h, edges, j, **kwargs) -> SampleResult:
        match = re.search(r"-(\d+)$", kwargs["label"])
        assert match is not None
        index = int(match.group(1))
        with self.lock:
            self.calls.append(index)
            self.keys.append(kwargs["cancel_key"])
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.entered[index].set()
            assert self.release[index].wait(timeout=10)
            if index in self.fail:
                raise RuntimeError("device fault")
            n = self.reads.get(index, int(kwargs["num_reads"]))
            rng = np.random.default_rng(index)
            spins = rng.choice(np.array([-1, 1], dtype=np.int8), (n, len(nodes)))
            # A deliberately wrong energy: the lease path rescores exactly.
            return SampleResult(
                spins=spins,
                variables=[int(x) for x in nodes],
                energies=[0.0] * n,
                device_access_time_us=ACCESS_US,
                num_reads=n,
            )
        finally:
            with self.lock:
                self.active -= 1


class Coordinator:
    """A scripted coordinator: the test sends messages and waits on replies."""

    def __init__(self, monkeypatch, device: FakeDevice):
        self.inbox: "queue.Queue[Optional[miner_pb2.CoordMsg]]" = queue.Queue()
        self.sent: List[miner_pb2.MinerMsg] = []
        self.cond = threading.Condition()
        self.exit_code: Optional[int] = None

        outer = self

        class _Stub:
            def __init__(self, channel):
                pass

            def Session(self, request_iter):
                def drain():
                    for msg in request_iter:
                        with outer.cond:
                            outer.sent.append(msg)
                            outer.cond.notify_all()

                threading.Thread(target=drain, daemon=True).start()

                def responses():
                    while True:
                        msg = outer.inbox.get()
                        if msg is None:
                            return
                        yield msg

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

        def run():
            self.exit_code = session_loop.run_session(
                "unix:///tmp/nope.sock", "lease-test", cast(OceanSampler, device)
            )

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def send(self, **msg) -> None:
        self.inbox.put(miner_pb2.CoordMsg(**msg))

    def start(self, topo: miner_pb2.Topology, tgt: miner_pb2.SetTarget, *, depth: int = 4, toml: str = "") -> None:
        self.send(welcome=miner_pb2.Welcome(protocol_version=2))
        self.send(configure=miner_pb2.Configure(queue_depth=depth, idle_timeout_s=300, backend_toml=toml))
        self.send(topology=topo)
        self.send(set_target=tgt)

    def wait_for(self, predicate: Callable[[List[miner_pb2.MinerMsg]], bool], timeout: float = 10) -> None:
        with self.cond:
            assert self.cond.wait_for(lambda: predicate(self.sent), timeout=timeout), self.kinds()

    def kinds(self) -> List[str]:
        with self.cond:
            return [m.WhichOneof("msg") for m in self.sent]

    def finish(self, grace_ms: int = 1000) -> List[miner_pb2.MinerMsg]:
        self.send(shutdown=miner_pb2.Shutdown(grace_ms=grace_ms))
        self.thread.join(timeout=15)
        assert not self.thread.is_alive()
        with self.cond:
            return list(self.sent)


def has_done(job_id: bytes = b"lease-1") -> Callable[[List[miner_pb2.MinerMsg]], bool]:
    return lambda sent: any(m.WhichOneof("msg") == "lease_done" and m.lease_done.job_id == job_id for m in sent)


def lease_messages(sent: List[miner_pb2.MinerMsg], job_id: bytes = b"lease-1"):
    """``(results, dones, refunds after the first LeaseDone)`` of one lease."""
    results = [m.result for m in sent if m.WhichOneof("msg") == "result" and m.result.job_id == job_id]
    dones = [i for i, m in enumerate(sent) if m.WhichOneof("msg") == "lease_done" and m.lease_done.job_id == job_id]
    return results, dones


def assert_closed_once(sent: List[miner_pb2.MinerMsg], job_id: bytes = b"lease-1") -> miner_pb2.LeaseDone:
    """One LeaseDone, its one-credit refund right behind it, and no Result after it."""
    _, dones = lease_messages(sent, job_id)
    assert len(dones) == 1, [m.WhichOneof("msg") for m in sent]
    at = dones[0]
    follow = sent[at + 1]
    assert follow.WhichOneof("msg") == "job_request" and follow.job_request.credits == 1
    late = [m for m in sent[at:] if m.WhichOneof("msg") == "result" and m.result.job_id == job_id]
    assert not late
    return sent[at].lease_done


def check_results(results, admitted) -> int:
    """Every Result carries its salt, nonce, and exact energies. Returns the lowest."""
    lowest = leases.NO_ENERGY
    for result in results:
        index = admitted.lease.index_of(result.salt)
        assert index is not None
        problem = admitted.problem(index)
        assert result.nonce == problem.nonce
        for solution in result.solutions:
            spins = wire.decode_spins_packed(solution.spins, len(problem.nodes))
            assert solution.energy_milli == int(problem.energies_milli(np.array([spins]))[0])
            lowest = min(lowest, solution.energy_milli)
    return lowest


def test_a_whole_lease_reports_every_salt_and_closes_once(monkeypatch):
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target())
    job = lease_job(generator(ring, count=5))
    coord.send(job=job)
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    results, _ = lease_messages(sent)
    assert sorted(device.calls) == [0, 1, 2, 3, 4]
    assert len(results) == 5 and all(len(r.solutions) == 2 for r in results)
    assert all(len(r.salt) == 32 and len(r.nonce) == 32 for r in results)
    assert done.salts_done == 5
    assert done.best_energy_milli == check_results(results, admit(job))


def test_a_family_lease_packs_spins_in_instance_order_with_the_offset(monkeypatch):
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    topo = zephyr_topology()
    coord.start(topo, target())
    job = lease_job(family_generator(topo, count=3))
    coord.send(job=job)
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    results, _ = lease_messages(sent)
    admitted = admit(job, topo=topo)
    first = admitted.problem(0)
    assert first is not None
    assert len(results) == done.salts_done == 3
    assert all(len(s.spins) == (len(first.nodes) + 7) // 8 for r in results for s in r.solutions)
    assert done.best_energy_milli == check_results(results, admitted)


def test_an_empty_read_set_counts_as_done_and_sends_no_result(monkeypatch):
    device = FakeDevice(reads={1: 0})
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target())
    job = lease_job(generator(ring, count=3))
    coord.send(job=job)
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    results, _ = lease_messages(sent)
    assert done.salts_done == 3
    assert len(results) == 2
    assert {admit(job).lease.index_of(r.salt) for r in results} == {0, 2}


def test_a_failed_salt_is_not_done_and_the_lease_still_closes(monkeypatch):
    device = FakeDevice(fail={2})
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target())
    coord.send(job=lease_job(generator(ring, count=4)))
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    results, _ = lease_messages(sent)
    assert done.salts_done == 3 and len(results) == 3


def test_a_malformed_descriptor_gets_one_reject_and_one_refund(monkeypatch):
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    topo = zephyr_topology()
    coord.start(topo, target())
    coord.send(job=lease_job(family_generator(topo, descriptor=_descriptor(params_digest=b"\0" * 32))))
    coord.wait_for(lambda sent: any(m.WhichOneof("msg") == "reject" for m in sent))
    sent = coord.finish()
    at = next(i for i, m in enumerate(sent) if m.WhichOneof("msg") == "reject")
    assert sent[at].reject.reason == miner_pb2.MALFORMED
    assert sent[at + 1].WhichOneof("msg") == "job_request" and sent[at + 1].job_request.credits == 1
    assert not device.calls and not lease_messages(sent)[1]


def test_a_stale_lease_draws_nothing_and_closes_with_zero_salts(monkeypatch):
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target())
    coord.send(cancel=miner_pb2.Cancel(max_generation=GENERATION))
    coord.send(job=lease_job(generator(ring)))
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    assert (done.salts_done, done.best_energy_milli) == (0, leases.NO_ENERGY)
    assert not device.calls


def _budget_toml(tmp_path) -> str:
    return f'budget = "3000s"\nusage_db = "{tmp_path / "usage.db"}"\n'


def test_a_cancel_closes_the_lease_at_once_and_bills_the_late_salt(monkeypatch, tmp_path):
    device = FakeDevice(hold={1})
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    # A budget, so the ledger exists; a boundary first, so the gate joins.
    coord.start(ring, target(), depth=1, toml=_budget_toml(tmp_path))
    coord.send(cancel=miner_pb2.Cancel(max_generation=1))
    coord.send(job=lease_job(generator(ring, count=6)))
    assert device.entered[1].wait(timeout=10), "salt 1 never reached the device"
    coord.send(cancel=miner_pb2.Cancel(max_generation=GENERATION))
    # The LeaseDone goes out while salt 1 is still on the device.
    coord.wait_for(has_done())
    device.release[1].set()
    sent = coord.finish()
    done = assert_closed_once(sent)
    assert done.salts_done == 1
    results, _ = lease_messages(sent)
    assert len(results) == 1
    # No salt drawn after the cancel, and the cancel reached salt 1 on SAPI.
    assert device.calls == [0, 1]
    assert device.keys[1] in device.cancelled
    # Both salts annealed and both are billed, the dropped one included.
    assert UsageLedger(str(tmp_path / "usage.db")).spent_us_since(0.0) == 2 * ACCESS_US


def test_a_cancel_during_generation_drops_the_salt_before_sampling(monkeypatch, tmp_path):
    device = FakeDevice()
    generating, release = threading.Event(), threading.Event()
    problem = leases.AdmittedLease.problem

    def held_problem(self, index):
        if index == 1:
            generating.set()
            assert release.wait(5)
        return problem(self, index)

    monkeypatch.setattr(leases.AdmittedLease, "problem", held_problem)
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target(), depth=1, toml=_budget_toml(tmp_path))
    coord.send(cancel=miner_pb2.Cancel(max_generation=1))
    coord.send(job=lease_job(generator(ring, count=3)))
    try:
        assert generating.wait(5)
        coord.send(cancel=miner_pb2.Cancel(max_generation=GENERATION))
        coord.wait_for(lambda sent: any(
            m.WhichOneof("msg") == "status" and m.status.abandoned_generation == GENERATION
            for m in sent
        ))
    finally:
        release.set()
        coord.finish()
    assert device.calls == [0]
    assert UsageLedger(str(tmp_path / "usage.db")).spent_us_since(0.0) == ACCESS_US


def test_cancel_publishes_its_watermark_while_a_worker_holds_the_budget_gate(monkeypatch, tmp_path):
    deciding, release = threading.Event(), threading.Event()
    on_job = session_loop.ParticipationGate.on_job

    def held_decision(self, now):
        if threading.current_thread().name.startswith("dwave-lease-"):
            deciding.set()
            assert release.wait(5)
        return on_job(self, now)

    monkeypatch.setattr(session_loop.ParticipationGate, "on_job", held_decision)
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target(), toml=_budget_toml(tmp_path))
    coord.send(cancel=miner_pb2.Cancel(max_generation=1))
    coord.send(job=lease_job(generator(ring)))
    try:
        assert deciding.wait(5)
        coord.send(cancel=miner_pb2.Cancel(max_generation=GENERATION))
        coord.wait_for(lambda sent: any(
            m.WhichOneof("msg") == "status" and m.status.abandoned_generation == GENERATION
            for m in sent
        ), timeout=0.5)
    finally:
        release.set()
        coord.finish()
    assert not device.calls


def test_a_passed_deadline_closes_the_lease_like_a_cancel(monkeypatch):
    device = FakeDevice(hold={1})
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target(), depth=1)
    deadline = int(time.time() * 1000) + 1500
    coord.send(job=lease_job(generator(ring, count=6), deadline_ms=deadline))
    assert device.entered[1].wait(timeout=10)
    coord.wait_for(has_done(), timeout=10)
    device.release[1].set()
    sent = coord.finish()
    done = assert_closed_once(sent)
    assert done.salts_done == 1
    assert device.calls == [0, 1]


def test_shutdown_stops_the_draw_and_keeps_a_salt_that_finishes_in_the_window(monkeypatch):
    device = FakeDevice(hold={1})
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target(), depth=1)
    coord.send(job=lease_job(generator(ring, count=6)))
    assert device.entered[1].wait(timeout=10)
    coord.send(shutdown=miner_pb2.Shutdown(grace_ms=2000))
    time.sleep(0.2)
    device.release[1].set()
    coord.thread.join(timeout=15)
    with coord.cond:
        sent = list(coord.sent)
    done = assert_closed_once(sent)
    assert done.salts_done == 2
    assert device.calls == [0, 1]


def test_a_stream_that_closes_abandons_the_lease_without_a_lease_done(monkeypatch):
    device = FakeDevice(hold={0})
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target(), depth=1)
    coord.send(job=lease_job(generator(ring, count=3)))
    assert device.entered[0].wait(timeout=10)
    coord.inbox.put(None)
    device.release[0].set()
    coord.thread.join(timeout=15)
    with coord.cond:
        sent = list(coord.sent)
    assert not lease_messages(sent)[1]


def test_salts_in_flight_never_exceed_the_pipeline_depth(monkeypatch):
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    ring = ring_topology()
    coord.start(ring, target(), depth=2)
    coord.send(job=lease_job(generator(ring, count=8), job_id=b"lease-1"))
    coord.send(job=lease_job(generator(ring, count=8), job_id=b"lease-2"))
    coord.wait_for(lambda sent: has_done(b"lease-1")(sent) and has_done(b"lease-2")(sent))
    sent = coord.finish()
    assert device.max_active <= 2
    for job_id in (b"lease-1", b"lease-2"):
        assert assert_closed_once(sent, job_id).salts_done == 8
