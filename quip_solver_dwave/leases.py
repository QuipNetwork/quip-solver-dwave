"""Salt leases: ``ISING_GENERATE`` jobs, sampled salt by salt on the QPU.

A lease names the cached ``Topology`` and a range of salts. The problem of each
salt comes from the canonical Rust code in the ``quip_solver_core`` wheel, the
code the coordinator verifies with: ``lease.draw`` for the legacy
``BLAKE3_CHACHA8_V1`` generator, ``families.generate`` for a family lease.
Nothing here reimplements a draw.

This module admits a lease in the order SPEC "Generated jobs" and FAM-47 give,
builds the problem of each salt, and runs a lease to its one ``LeaseDone``
(:func:`run_lease`). The session loop supplies the QPU, the billing, and the
outbound queue through :class:`LeaseHooks`.

Accounting, which the tests in ``test_family_leases`` pin:

* One lease takes one credit and returns one, with its ``LeaseDone``, in a
  single outbound item, so nothing interleaves between the two.
* ``salts_done`` counts every salt that finished sampling, an empty read set
  included, and every salt that generation skipped. A salt whose sampling
  failed is not done.
* A ``Cancel`` that covers the lease, or a passed ``deadline_ms``, stops the
  draw and sends the ``LeaseDone`` at once. Results that finish later are
  dropped, but the session has already billed their access time: QPU quota
  follows the device, not the protocol.
* ``Shutdown`` stops the draw and accepts results until the lease close
  deadline. A closed stream abandons the lease without a ``LeaseDone``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quip_solver_core import (
    AdmissionError,
    GeneratorDefect,
    LeaseError,
    SaltSkipped,
    families,
    lease as lease_sdk,
    miner_pb2,
)

from quip_solver_dwave.job import now_unix_ms, pack_spin_rows

logger = logging.getLogger(__name__)

LEGACY = miner_pb2.GENERATOR_ALGORITHM_BLAKE3_CHACHA8_V1
FAMILY = miner_pb2.GENERATOR_ALGORITHM_FAMILY_CHACHA8_V1

#: ``LeaseDone.best_energy_milli`` when no read finished: ``i64::MAX``.
NO_ENERGY = 2**63 - 1

#: The family revisions this miner generates, as ``(family_id, revision)``:
#: every revision the wheel's registry implements, in registry order. The wheel
#: draws each instance, so a lease of any registered family runs on the QPU,
#: and the conformance ``--family`` gate covers each one.
GENERATED_FAMILIES: Tuple[Tuple[str, int], ...] = tuple(families.supported())

#: How often a lease thread checks for a stop while it waits.
TICK_S = 0.05


def generators() -> List["miner_pb2.GeneratorAlgorithm"]:
    """``Capabilities.generators``: the family generator only with a family (FAM-58)."""
    return [LEGACY, FAMILY] if GENERATED_FAMILIES else [LEGACY]


def generated_family_support() -> List[miner_pb2.FamilySupport]:
    """``Capabilities.generated_families``, every valid parameter set of each revision."""
    revisions: Dict[str, List[int]] = {}
    for family_id, revision in GENERATED_FAMILIES:
        revisions.setdefault(family_id, []).append(revision)
    return [
        miner_pb2.FamilySupport(family_id=family_id, revisions=revs)
        for family_id, revs in revisions.items()
    ]


class LeaseRejected(Exception):
    """Admission failed: the lease gets one ``Reject`` with ``reason`` and one refund."""

    def __init__(self, reason: "miner_pb2.RejectReason", detail: str):
        super().__init__(detail)
        self.reason = reason


def _family_reason(code: int) -> "miner_pb2.RejectReason":
    """The wire reason of a family code at admission (FAM-42)."""
    if code in (5, 6, 10, 12):
        return miner_pb2.FAMILY_UNSUPPORTED
    if code == 11:
        return miner_pb2.TOPOLOGY_MISMATCH
    if code == 13:
        return miner_pb2.TOO_LARGE
    return miner_pb2.MALFORMED


@dataclass(frozen=True, eq=False)
class LeaseTopology:
    """The cached ``Topology`` as a lease reads it: native ids in registration order."""

    hash: bytes
    nodes: List[int]
    edges: List[Tuple[int, int]]
    allowed_h_milli: List[int]
    allowed_j_milli: List[int]

    @classmethod
    def from_proto(cls, topo: miner_pb2.Topology) -> "LeaseTopology":
        edges = list(zip(topo.edges.u, topo.edges.v)) if topo.HasField("edges") else []
        return cls(
            hash=bytes(topo.hash),
            nodes=list(topo.nodes),
            edges=edges,
            allowed_h_milli=list(topo.allowed_h_milli),
            allowed_j_milli=list(topo.allowed_j_milli),
        )

    @cached_property
    def order_digest(self) -> bytes:
        """The FAM-50 digest a family lease must carry."""
        return bytes(families.topology_order_digest(self.nodes, self.edges))

    @cached_property
    def node_array(self) -> np.ndarray:
        return np.asarray(self.nodes, dtype=np.int64)

    @cached_property
    def positions(self) -> np.ndarray:
        """``(E, 2)`` positions of each edge's endpoints in ``nodes``.

        Read only after a draw succeeded, which rules out an unknown node.
        """
        at = {node: i for i, node in enumerate(self.nodes)}
        return np.asarray(
            [(at[u], at[v]) for u, v in self.edges], dtype=np.intp
        ).reshape(-1, 2)


@dataclass(frozen=True, eq=False)
class SaltProblem:
    """The model of one salt, as the QPU anneals it and the result reports it."""

    index: int
    salt: bytes
    nonce: bytes
    #: Native qubit ids, in ``Solution.spins`` order.
    nodes: np.ndarray
    h_milli: np.ndarray
    #: ``(E, 2)`` positions into ``nodes``.
    positions: np.ndarray
    j_milli: np.ndarray
    #: The family instance offset; zero on a legacy lease.
    offset_milli: int = 0
    #: The family instance metadata region (FAM-34); empty on a legacy lease.
    metadata: bytes = b""

    @property
    def edges(self) -> np.ndarray:
        """``(E, 2)`` native qubit ids."""
        return self.nodes[self.positions]

    def energies_milli(self, spins: np.ndarray) -> np.ndarray:
        """The exact energy of each ±1 row of ``spins``, offset included (FAM-33)."""
        s = np.asarray(spins, dtype=np.int64)
        u, v = self.positions[:, 0], self.positions[:, 1]
        return s @ self.h_milli + (s[:, u] * s[:, v]) @ self.j_milli + self.offset_milli


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    revision: int
    params: bytes
    params_digest: bytes


@dataclass(eq=False)
class AdmittedLease:
    """A lease that passed admission, with its topology and descriptor fixed.

    A later ``Topology`` or ``Job`` changes nothing here: every salt is drawn
    from the snapshot taken at admission.
    """

    job_id: bytes
    generation: int
    deadline_ms: int
    lease: lease_sdk.Lease
    topology: LeaseTopology
    family: Optional[FamilySpec]
    max_nodes: int
    max_edges: int
    #: Salt 0, built during admission. ``None`` when generation skipped it.
    first: Optional[SaltProblem]

    @property
    def salt_count(self) -> int:
        return int(self.lease.salt_count)

    def expired(self) -> bool:
        return bool(self.deadline_ms) and now_unix_ms() > self.deadline_ms

    def problem(self, index: int) -> Optional[SaltProblem]:
        """The problem of salt ``index``, or ``None`` when generation skips it."""
        if index == 0:
            return self.first
        return self._build(index)

    def _build(self, index: int) -> Optional[SaltProblem]:
        salt = bytes(self.lease.salt(index))
        nonce = bytes(self.lease.nonce(index))
        topo = self.topology
        if self.family is None:
            h, j = lease_sdk.draw(
                nonce, topo.nodes, topo.edges, topo.allowed_h_milli, topo.allowed_j_milli
            )
            return SaltProblem(
                index=index,
                salt=salt,
                nonce=nonce,
                nodes=topo.node_array,
                h_milli=np.asarray(h, dtype=np.int64),
                positions=topo.positions,
                j_milli=np.asarray(j, dtype=np.int64),
            )
        try:
            instance = self._generate(nonce)
        except SaltSkipped as exc:
            logger.debug("lease %s salt %d skipped: %s", self.job_id.hex(), index, exc.name)
            return None
        except (GeneratorDefect, AdmissionError) as exc:
            # Admission already passed on these inputs, so this is the
            # construction breaking its own contract. The salt is skipped, as
            # the Rust session does.
            logger.error(
                "lease %s salt %d: family construction broke its contract (%s); skipping it",
                self.job_id.hex(),
                index,
                exc.name,
            )
            return None
        return self._from_instance(index, salt, nonce, instance)

    def _generate(self, nonce: bytes) -> "families.Instance":
        fam = self.family
        assert fam is not None
        return families.generate(
            fam.family_id,
            fam.revision,
            fam.params,
            fam.params_digest,
            self.topology.nodes,
            self.topology.edges,
            nonce,
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
        )

    @staticmethod
    def _from_instance(
        index: int, salt: bytes, nonce: bytes, instance: "families.Instance"
    ) -> SaltProblem:
        return SaltProblem(
            index=index,
            salt=salt,
            nonce=nonce,
            nodes=np.asarray(instance.node_ids, dtype=np.int64),
            h_milli=np.asarray(instance.h, dtype=np.int64),
            positions=np.asarray(instance.edges, dtype=np.intp).reshape(-1, 2),
            j_milli=np.asarray(instance.j, dtype=np.int64),
            offset_milli=int(instance.offset_milli),
            metadata=bytes(instance.metadata),
        )


def admit_lease(
    job: miner_pb2.Job,
    topology: Optional[LeaseTopology],
    target: Optional[miner_pb2.SetTarget],
    *,
    max_nodes: int,
    max_edges: int,
    generated: Sequence[Tuple[str, int]] = GENERATED_FAMILIES,
) -> AdmittedLease:
    """Admit one lease, first failure wins (SPEC "Generated jobs", FAM-47).

    The order: ``EXPIRED``, ``TOPOLOGY_MISSING``, ``TARGET_MISSING``; the lease
    fields (``MALFORMED``, which covers family codes 1 and 2); codes 3 to 10;
    the topology hash; then codes 11 to 13 for a family lease, or the
    allowed-values and size checks for a legacy one. Salt 0 is drawn last,
    as part of the checks that need a draw, and kept.

    Raises:
        LeaseRejected: With the wire reason of the first failed check.
    """
    if job.deadline_ms and now_unix_ms() > job.deadline_ms:
        raise LeaseRejected(miner_pb2.EXPIRED, "deadline passed")
    if topology is None:
        raise LeaseRejected(miner_pb2.TOPOLOGY_MISSING, "no cached Topology")
    if target is None or target.max_proof_solutions == 0:
        raise LeaseRejected(miner_pb2.TARGET_MISSING, "no SetTarget with max_proof_solutions")
    if not job.HasField("generator"):
        raise LeaseRejected(miner_pb2.MALFORMED, "no generator")
    gen = job.generator
    try:
        lease = lease_sdk.Lease(gen.SerializeToString())
    except LeaseError as exc:
        raise LeaseRejected(miner_pb2.MALFORMED, exc.name) from None

    family: Optional[FamilySpec] = None
    if gen.algorithm == FAMILY:
        d = gen.family
        family = FamilySpec(d.family_id, int(d.revision), bytes(d.params), bytes(d.params_digest))
        _admit_descriptor(family, generated)

    if bytes(gen.topology_hash) != topology.hash:
        raise LeaseRejected(miner_pb2.TOPOLOGY_MISMATCH, "topology_hash differs from the cached Topology")

    if family is None:
        if (topology.nodes and not topology.allowed_h_milli) or (
            topology.edges and not topology.allowed_j_milli
        ):
            raise LeaseRejected(miner_pb2.MALFORMED, "the Topology has no allowed values")
        if (max_nodes and len(topology.nodes) > max_nodes) or (
            max_edges and len(topology.edges) > max_edges
        ):
            raise LeaseRejected(miner_pb2.TOO_LARGE, "the Topology exceeds the advertised size")
    elif bytes(gen.topology_order_digest) != topology.order_digest:
        raise LeaseRejected(miner_pb2.TOPOLOGY_MISMATCH, "code 11 TopologyOrderMismatch")

    admitted = AdmittedLease(
        job_id=bytes(job.job_id),
        generation=int(job.generation),
        deadline_ms=int(job.deadline_ms),
        lease=lease,
        topology=topology,
        family=family,
        max_nodes=max_nodes,
        max_edges=max_edges,
        first=None,
    )
    admitted.first = _first_salt(admitted)
    return admitted


def _admit_descriptor(family: FamilySpec, generated: Sequence[Tuple[str, int]]) -> None:
    """Codes 3 to 10 in ascending order.

    The wheel's ``admit`` checks the descriptor against every family it
    implements, so codes 5 and 6 come from this miner's own list: they sit
    after codes 3 and 4 and before codes 7 to 9.
    """
    try:
        families.admit(family.family_id, family.revision, family.params, family.params_digest)
        held: Optional[AdmissionError] = None
    except AdmissionError as exc:
        held = exc
    if held is not None and held.code in (3, 4):
        raise LeaseRejected(_family_reason(held.code), f"code {held.code} {held.name}")
    if (family.family_id, family.revision) not in generated:
        known = any(fid == family.family_id for fid, _ in generated)
        code, name = (6, "RevisionUnsupported") if known else (5, "FamilyUnsupported")
        raise LeaseRejected(miner_pb2.FAMILY_UNSUPPORTED, f"code {code} {name}")
    if held is not None:
        raise LeaseRejected(_family_reason(held.code), f"code {held.code} {held.name}")


def _first_salt(admitted: AdmittedLease) -> Optional[SaltProblem]:
    """Salt 0, which settles the admission checks that need the topology built.

    For a legacy lease, a draw fault means the cached ``Topology`` is
    malformed (a repeated or unknown node). For a family lease, the wheel
    raises ``AdmissionError`` for codes 12 and 13 before it draws, and
    ``SaltSkipped`` when salt 0 itself has no instance.
    """
    if admitted.family is None:
        try:
            return admitted._build(0)
        except LeaseError as exc:
            raise LeaseRejected(miner_pb2.MALFORMED, exc.name) from None
    salt = bytes(admitted.lease.salt(0))
    nonce = bytes(admitted.lease.nonce(0))
    try:
        instance = admitted._generate(nonce)
    except AdmissionError as exc:
        raise LeaseRejected(_family_reason(exc.code), f"code {exc.code} {exc.name}") from None
    except SaltSkipped as exc:
        logger.debug("lease %s salt 0 skipped: %s", admitted.job_id.hex(), exc.name)
        return None
    except GeneratorDefect as exc:
        logger.error(
            "lease %s salt 0: family construction broke its contract (%s); skipping it",
            admitted.job_id.hex(),
            exc.name,
        )
        return None
    return admitted._from_instance(0, salt, nonce, instance)


def lease_done(job_id: bytes, salts_done: int, best_energy_milli: int) -> miner_pb2.MinerMsg:
    return miner_pb2.MinerMsg(
        lease_done=miner_pb2.LeaseDone(
            job_id=job_id, salts_done=salts_done, best_energy_milli=best_energy_milli
        )
    )


def salt_result(
    lease: AdmittedLease,
    problem: SaltProblem,
    spins: np.ndarray,
    energies: np.ndarray,
    meta: miner_pb2.SamplerMeta,
) -> miner_pb2.MinerMsg:
    """One salt's ``Result``: every read, unfiltered by target, with its salt and nonce."""
    solutions = [
        miner_pb2.Solution(spins=packed, energy_milli=int(e))
        for packed, e in zip(pack_spin_rows(spins), energies.tolist())
    ]
    return miner_pb2.MinerMsg(
        result=miner_pb2.Result(
            job_id=lease.job_id,
            solutions=solutions,
            meta=meta,
            salt=problem.salt,
            nonce=problem.nonce,
        )
    )


class StopDrawing(Exception):
    """Raised by :attr:`LeaseHooks.sample_salt`: this salt failed, and so would the next."""


@dataclass
class LeaseHooks:
    """What :func:`run_lease` needs from the session.

    ``sample_salt`` runs one salt on the QPU and bills it, and returns its
    reads in ``problem.nodes`` order with their ``SamplerMeta``, or ``None``
    when sampling failed. ``send`` puts a list of messages on the outbound
    queue as one item. ``stopped`` is true once a ``Cancel`` covers the lease
    generation. ``close_at`` is the monotonic lease close deadline once a
    ``Shutdown`` arrived. ``aborted`` is true once the stream is gone.
    ``may_draw`` asks the budget whether another salt may go to the QPU.
    ``refund`` returns the credit refund to send with the ``LeaseDone``, or
    nothing when the session holds the credit back. ``sampled`` counts a
    sampled salt in ``Status.jobs_done``. ``postprocess``, when set, takes a
    salt's raw reads and a stop check, and returns final states to report
    beside them, or ``None`` when its trial did not succeed. ``in_reach``,
    when set, says whether a salt's raw reads are worth the postprocess.
    ``record``, when set, keeps the reads of a salt that had no postprocess.
    """

    sample_salt: Callable[[SaltProblem], Optional[Tuple[np.ndarray, miner_pb2.SamplerMeta]]]
    send: Callable[[List[miner_pb2.MinerMsg]], None]
    stopped: Callable[[], bool]
    close_at: Callable[[], Optional[float]]
    aborted: Callable[[], bool]
    may_draw: Callable[[], bool]
    refund: Callable[[], List[miner_pb2.MinerMsg]]
    sampled: Callable[[], None]
    postprocess: Optional[
        Callable[[SaltProblem, np.ndarray, Callable[[], bool]], Optional[np.ndarray]]
    ] = None
    in_reach: Optional[Callable[[SaltProblem, np.ndarray], bool]] = None
    record: Optional[Callable[[SaltProblem, np.ndarray], None]] = None


class _Ledger:
    """One lease's counts, and the guard that sends its ``LeaseDone`` once."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.dispatched = 0
        self.finished = 0
        self.salts_done = 0
        self.best = NO_ENERGY
        self.closed = False
        self.done_sent = False
        self._seen: set[int] = set()

    def complete(
        self,
        index: int,
        energies: np.ndarray,
        accept: Callable[[], bool],
        send_result: Callable[[], None],
    ) -> bool:
        """Count a sampled salt and send its ``Result``, unless the lease closed.

        The ``Result`` goes out under the ledger lock, so it cannot follow the
        ``LeaseDone``.
        """
        with self._cond:
            if self.done_sent or index in self._seen or not accept():
                return False
            self._seen.add(index)
            self.salts_done += 1
            if energies.size:
                self.best = min(self.best, int(energies.min()))
                send_result()
            return True

    def skip(self, index: int) -> None:
        with self._cond:
            if not self.done_sent and index not in self._seen:
                self._seen.add(index)
                self.salts_done += 1

    def dispatch(self) -> None:
        with self._cond:
            self.dispatched += 1

    def finish_one(self) -> None:
        with self._cond:
            self.finished += 1
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self.closed = True

    def take_done(self, force: bool) -> Optional[Tuple[int, int]]:
        """``(salts_done, best)`` once, when every salt finished or ``force``."""
        with self._cond:
            if self.done_sent or not self.closed:
                return None
            if not force and self.finished != self.dispatched:
                return None
            self.done_sent = True
            return self.salts_done, self.best

    def wait(self, timeout: float) -> None:
        with self._cond:
            self._cond.wait(timeout)


def run_lease(
    lease: AdmittedLease,
    hooks: LeaseHooks,
    *,
    slots: threading.Semaphore,
    submit: Callable[..., object],
    host_submit: Optional[Callable[..., object]] = None,
) -> None:
    """Sample every salt of ``lease``, then send its ``LeaseDone`` and refund.

    Runs on the lease's own thread. Each salt takes a permit from ``slots``,
    the session-wide bound on salts in flight, before it is drawn, and gives
    it back when its sampling ends. ``submit(fn, problem)`` runs a salt on the
    session's job pool. A salt with a postprocess hands its host work to
    ``host_submit(fn, ...)``, so the job pool worker, which stands for one
    QPU submission, is free for the next salt. When ``host_submit`` returns
    ``False`` the host queue is full, and the salt reports its raw reads at
    once. No salt is drawn after a stop.
    """
    ledger = _Ledger()
    no_more = threading.Event()

    def halted() -> bool:
        """A stop that sends the ``LeaseDone`` at once: a ``Cancel`` or the deadline."""
        return hooks.stopped() or lease.expired()

    def drawing() -> bool:
        return not (
            halted() or no_more.is_set() or hooks.aborted() or hooks.close_at() is not None
        )

    def finishing() -> bool:
        """The processor of a salt stops on any stop, a Shutdown close included."""
        close_at = hooks.close_at()
        return (
            halted()
            or hooks.aborted()
            or (close_at is not None and time.monotonic() >= close_at)
        )

    def report(problem: SaltProblem, spins: np.ndarray, meta: miner_pb2.SamplerMeta) -> None:
        energies = problem.energies_milli(spins)

        def send_result() -> None:
            hooks.send([salt_result(lease, problem, spins, energies, meta)])

        if ledger.complete(
            problem.index,
            energies,
            accept=lambda: not (halted() or hooks.aborted()),
            send_result=send_result,
        ):
            hooks.sampled()

    def host_work(problem: SaltProblem, spins: np.ndarray, meta: miner_pb2.SamplerMeta) -> None:
        try:
            assert hooks.postprocess is not None
            final = hooks.postprocess(problem, spins, finishing)
            if final is not None and final.shape[0]:
                # Final states first, then every raw read, unfiltered.
                spins = np.vstack([final.astype(np.int8), spins])
            report(problem, spins, meta)
        except Exception:  # noqa: BLE001 - a salt must always be counted finished
            logger.exception("lease %s salt %d postprocess failed", lease.job_id.hex(), problem.index)
        finally:
            ledger.finish_one()

    def work(problem: SaltProblem) -> None:
        handed_off = False
        try:
            try:
                outcome = hooks.sample_salt(problem)
            except StopDrawing:
                no_more.set()
                outcome = None
            finally:
                # The QPU part of the salt is over, and billed: its slot goes
                # to the next salt before any host work starts.
                slots.release()
            if outcome is None:
                return
            spins, meta = outcome
            if (
                hooks.postprocess is not None
                and spins.shape[0]
                and (hooks.in_reach is None or hooks.in_reach(problem, spins))
            ):
                run_host = host_submit or (lambda fn, *args: fn(*args))
                handed_off = True
                try:
                    accepted = run_host(host_work, problem, spins, meta)
                except RuntimeError:
                    # The host pool shut down: the session is ending.
                    handed_off = False
                    return
                if accepted is not False:
                    return
                # The host queue is full: this salt reports its raw reads now.
                handed_off = False
            report(problem, spins, meta)
            if hooks.record is not None and spins.shape[0]:
                hooks.record(problem, spins)
        except Exception:  # noqa: BLE001 - a salt must always be counted finished
            logger.exception("lease %s salt %d failed", lease.job_id.hex(), problem.index)
        finally:
            if not handed_off:
                ledger.finish_one()

    for index in range(lease.salt_count):
        acquired = False
        while drawing():
            if slots.acquire(timeout=TICK_S):
                acquired = True
                break
        if not acquired:
            break
        if not drawing() or not hooks.may_draw():
            slots.release()
            break
        problem = lease.problem(index)
        if problem is None:
            slots.release()
            ledger.skip(index)
            continue
        ledger.dispatch()
        try:
            submit(work, problem)
        except RuntimeError:
            # The pool shut down under the lease: the session is ending.
            slots.release()
            ledger.finish_one()
            break
    ledger.close()

    while not hooks.aborted():
        close_at = hooks.close_at()
        force = halted() or (close_at is not None and time.monotonic() >= close_at)
        done = ledger.take_done(force)
        if done is not None:
            salts_done, best = done
            hooks.send([lease_done(lease.job_id, salts_done, best), *hooks.refund()])
            logger.info(
                "[quip-solver-dwave] lease %s: %d of %d salts done, best %s",
                lease.job_id[:8].hex(),
                salts_done,
                lease.salt_count,
                "n/a" if best == NO_ENERGY else best // 1000,
            )
            return
        ledger.wait(TICK_S)
