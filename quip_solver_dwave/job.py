"""Job validation, sampling, and Result/Reject construction."""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quip_solver_core import miner_pb2, wire
from quip_solver_core.session import DEFAULT_NUM_SWEEPS

from quip_solver_dwave import MAX_EDGES, MAX_NODES
from quip_solver_dwave.config import SamplingDefaults
from quip_solver_dwave.ocean import SampleResult, SupportsSample, is_solver_offline
from quip_solver_dwave.qp import OffChipError
from quip_solver_dwave.schedule import ScheduleError
from quip_solver_dwave.warm import MalformedWarmStart, WarmStart, warm_start_from_ising

logger = logging.getLogger(__name__)


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def decode_coefficients(raw: bytes, scale: int) -> List[float]:
    """Decode one I32 coefficient array at ``scale`` into float units.

    Each element is ``value / scale`` units, and it must land exactly on a
    milli: the coordinator scores in i32 milli, so a value off that grid
    would be sampled as one problem and scored as another.

    Raises:
        ValueError: A partial element, or a value that is not exact i32 milli.
    """
    out: List[float] = []
    for value in wire.decode_i32_le(raw):
        milli, remainder = divmod(value * 1000, scale)
        if remainder or not -(2**31) <= milli < 2**31:
            raise ValueError(f"coefficient {value}/{scale} is not exact i32 milli")
        out.append(milli / 1000.0)
    return out


def edges_of(ising: miner_pb2.IsingProblem) -> List[Tuple[int, int]]:
    which = ising.WhichOneof("graph")
    if which == "edges":
        e = ising.edges
        return list(zip(e.u, e.v))
    return []


def resolve_graph(
    ising: miner_pb2.IsingProblem,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    n_h: int,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (nodes, edges) for a job.

    Inline EdgeList uses dense 0..n-1 node ids from ``n_h``; topology-hash
    jobs use the session-cached Topology.
    """
    which = ising.WhichOneof("graph")
    if which == "edges":
        return list(range(n_h)), edges_of(ising)
    return list(session_nodes), list(session_edges)


def spins_to_bytes(spins: Sequence[int]) -> bytes:
    return bytes(wire.encode_spins_packed(list(spins)))


def sample_dict_to_vector(
    sample: Dict[int, int], nodes: Sequence[int]
) -> List[int]:
    """Map a label-keyed sample into a dense spin vector in ``nodes`` order."""
    out: List[int] = []
    for n in nodes:
        s = sample.get(int(n), 1)
        out.append(1 if s >= 0 else -1)
    return out


class SolverUnavailable(Exception):
    """The solver is offline: SAPI refused the job before any anneal ran.

    Not answered here like other sampler failures, because the answer is not
    per-job. A reject with a refund re-dispatches the next staged job at once,
    and against an offline solver that loop runs at SAPI round-trip speed for
    the whole qblock. The session loop parks the credit instead and probes
    again at the next qblock boundary.
    """

    def __init__(self, job_id: bytes):
        super().__init__(job_id.hex())
        self.job_id = job_id


class _Rejected(Exception):
    """Raised by a validation step to abort the job with a reject reason.

    Lets the guards read as straight-line code and keeps the conversion to a
    Reject message in one place, in :func:`handle_job`.
    """

    def __init__(self, reason: miner_pb2.RejectReason):
        super().__init__(reason)
        self.reason = reason


def _reject(
    job_id: bytes, reason: miner_pb2.RejectReason
) -> List[miner_pb2.MinerMsg]:
    """Build the two-message reply that rejects ``job_id`` for ``reason``.

    A reject is terminal for the job, so it carries the same credit refund a
    Result does: the coordinator consumes a credit on dispatch and reclaims
    nothing on a bare Reject, so a reject without the follow-up JobRequest
    leaks one pipeline slot forever — the conformance driver's credit-ledger
    axis fails exactly this. ``reason`` is a ``miner_pb2`` reject-reason enum.
    """
    return [
        miner_pb2.MinerMsg(reject=miner_pb2.Reject(job_id=job_id, reason=reason)),
        miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1)),
    ]


def _validate_job(
    job: miner_pb2.Job,
    session_hash: Optional[bytes],
) -> Tuple[miner_pb2.IsingProblem, List[float], List[float]]:
    """Check job-level invariants and decode the milli arrays.

    Args:
        job: The job to validate.
        session_hash: Hash of the cached session ``Topology``, or ``None`` if
            no ``Topology`` has arrived yet.

    Returns:
        The job's ``IsingProblem`` and its decoded ``h`` and ``j`` values.

    Raises:
        _Rejected: Unsupported kind, undecodable ``h``/``j``, a deadline
            already passed, or a ``topology_hash`` the session cannot satisfy.
    """
    if job.kind not in (
        miner_pb2.ISING_SAMPLE,
        miner_pb2.JOB_KIND_UNSPECIFIED,
    ):
        raise _Rejected(miner_pb2.UNSUPPORTED_KIND)

    ising = job.ising if job.HasField("ising") else miner_pb2.IsingProblem()

    # MALFORMED: an encoding this backend does not advertise, a zero scale, a
    # partial element, or a value off the milli grid. Only I32 is advertised.
    if ising.encoding != miner_pb2.COEFFICIENT_ENCODING_I32 or ising.scale == 0:
        raise _Rejected(miner_pb2.MALFORMED)
    try:
        h = decode_coefficients(ising.h, ising.scale)
        j_vals = decode_coefficients(ising.j, ising.scale)
    except ValueError:
        raise _Rejected(miner_pb2.MALFORMED) from None

    # TOO_LARGE: the job exceeds the capability envelope this backend
    # advertises in Hello, Capabilities, and --capabilities. Sampling past it
    # would hand Ocean (or the exact mock solver) a problem the miner never
    # claimed to serve.
    if len(h) > MAX_NODES or len(j_vals) > MAX_EDGES:
        raise _Rejected(miner_pb2.TOO_LARGE)

    if job.deadline_ms and job.deadline_ms < now_unix_ms():
        raise _Rejected(miner_pb2.EXPIRED)

    if ising.WhichOneof("graph") == "topology_hash":
        if session_hash is None:
            raise _Rejected(miner_pb2.TOPOLOGY_MISSING)
        if bytes(ising.topology_hash) != bytes(session_hash):
            raise _Rejected(miner_pb2.TOPOLOGY_MISMATCH)

    return ising, h, j_vals


def _resolve_problem(
    ising: miner_pb2.IsingProblem,
    h: Sequence[float],
    j_vals: Sequence[float],
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    *,
    job_id: bytes,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the job's graph and build the sampler's ``h``/``J`` mappings.

    ``h`` and ``j`` are dense positional arrays over the
    resolved graph, so a length mismatch is a malformed job rather than
    something to trim to fit. Truncating would drop values the coordinator
    specified (or leave trailing nodes and edges untouched) and then report a
    credible energy for a different problem. A short ``j`` is not "the rest are
    zero" either: the wire format cannot express that, so it is a desync.

    Inline-EdgeList jobs derive their nodes from ``len(h)`` and so never trip
    the node check; it catches a job desynced from the session ``Topology``.

    Returns:
        ``(nodes, h, edges, j)`` as numpy arrays: the resolved node ordering,
        its biases, the resolved edge list, and its couplings.

    Raises:
        _Rejected: ``MALFORMED`` when ``h`` or ``j`` disagrees with the graph.
    """
    nodes, edges = resolve_graph(ising, session_nodes, session_edges, len(h))
    if not nodes and h:
        nodes = list(range(len(h)))

    if len(h) != len(nodes):
        logger.warning(
            "job %s: %d biases for a %d-node graph; rejecting MALFORMED",
            job_id.hex(),
            len(h),
            len(nodes),
        )
        raise _Rejected(miner_pb2.MALFORMED)

    if len(j_vals) != len(edges):
        logger.warning(
            "job %s: %d couplings for a %d-edge graph; rejecting MALFORMED",
            job_id.hex(),
            len(j_vals),
            len(edges),
        )
        raise _Rejected(miner_pb2.MALFORMED)

    # Arrays, not dicts. The submission payload is positional in the solver's
    # own ordering, so a dict keyed by qubit label exists only to be walked
    # once and thrown away — ~13 ms of GIL-bound work per job at production
    # size. quip_solver_dwave.qp maps these straight into the payload.
    return (
        np.asarray(nodes, dtype=np.int64),
        np.asarray(h, dtype=np.float64),
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(j_vals, dtype=np.float64),
    )


def _resolve_warm_start(
    ising: miner_pb2.IsingProblem,
    num_nodes: int,
    session_defaults: SamplingDefaults,
    *,
    job_id: bytes,
) -> Optional[WarmStart]:
    """The job's warm start, or ``None`` for a cold job.

    Raises:
        _Rejected: ``MALFORMED`` for a state of the wrong length, a set padding
            bit, or a reversal point of 1000 milli or more (SPEC section 3).
    """
    try:
        return warm_start_from_ising(ising, num_nodes, session_defaults)
    except MalformedWarmStart as exc:
        logger.warning("job %s: %s; rejecting MALFORMED", job_id.hex(), exc)
        raise _Rejected(miner_pb2.MALFORMED) from None


def _sampling_params(
    ising: miner_pb2.IsingProblem,
    session_target: Optional["miner_pb2.SetTarget"],
    session_sweeps: int,
    session_defaults: SamplingDefaults = SamplingDefaults(),
) -> Tuple[int, int, int]:
    """Resolve ``(num_reads, anneal_time_us, num_sweeps)`` for one job.

    All follow the same precedence: per-job override, then the session's
    ``SetTarget``, then ``session_defaults`` from ``Configure.backend_toml``,
    then the hard-coded default. Zero means "unset" at every rung, so an
    operator's ``anneal_time_us`` applies to jobs the coordinator left blank
    without ever overriding one it filled in. ``anneal_time_us`` resolving to 0
    means the solver's published default anneal. ``num_sweeps`` does not steer
    the QPU (an annealer runs anneals, not sweeps); it is the resolved budget
    the coordinator pinned, echoed in ``SamplerMeta.sweeps`` because the
    contract grades that echo verbatim (``sweeps_honoured``).

    (Full energy-based adapt for the QPU path is a follow-up; see quip-asx.* —
    it needs the shared GSE model and QPU credits.)
    """
    num_reads = int(ising.num_reads)
    if num_reads == 0 and session_target is not None and session_target.num_reads:
        num_reads = int(session_target.num_reads)
    if num_reads == 0:
        num_reads = session_defaults.num_reads
    if num_reads == 0:
        num_reads = 1

    anneal_time_us = int(ising.anneal_time_us)
    if (
        anneal_time_us == 0
        and session_target is not None
        and session_target.anneal_time_us
    ):
        anneal_time_us = int(session_target.anneal_time_us)
    if anneal_time_us == 0:
        anneal_time_us = session_defaults.anneal_time_us

    num_sweeps = int(ising.num_sweeps)
    if num_sweeps == 0 and session_target is not None and session_target.num_sweeps:
        num_sweeps = int(session_target.num_sweeps)
    if num_sweeps == 0:
        num_sweeps = session_sweeps

    return num_reads, anneal_time_us, num_sweeps


def spins_in_node_order(
    nodes: "np.ndarray | Sequence[int]", result: SampleResult
) -> np.ndarray:
    """``result.spins`` reordered from the sampler's columns into ``nodes``.

    One vectorised fancy-index rather than a dict per read. With no ``nodes``
    the order is the sampler's labels, sorted.
    """
    order = nodes if len(nodes) else sorted(result.variables)
    col_of = {v: i for i, v in enumerate(result.variables)}
    spins = result.spins
    n_cols = spins.shape[1]
    # A node the sampler never reported reads as +1, matching the dict path's
    # `sample.get(n, 1)`. Point those at one appended constant column so the
    # reorder stays a single fancy-index.
    idx = np.fromiter(
        (col_of.get(int(n), n_cols) for n in order), dtype=np.intp, count=len(order)
    )
    if idx.size and idx.max() == n_cols:
        spins = np.hstack([spins, np.ones((spins.shape[0], 1), dtype=np.int8)])
    return spins[:, idx] if idx.size else spins[:, :0]


def _build_result(
    job_id: bytes,
    nodes: "np.ndarray | Sequence[int]",
    result: SampleResult,
    num_sweeps: int,
) -> List[miner_pb2.MinerMsg]:
    """Turn a completed sample into a Result plus a follow-up JobRequest.

    Reports the sampler's own energy for the problem it actually annealed.
    Re-scoring here would buy nothing: the coordinator cannot trust a miner's
    energy regardless, so it re-scores whatever it accepts. ``energy_milli`` is
    an integer field, so the only transform is quantizing to milli.
    """
    # One vectorised reorder into session node order, then one vectorised
    # bit-pack. The wire packs node i into bit i % 8 of byte i / 8, lowest bit
    # first, with bit 1 for +1 (test_spin_encoding pins that equivalence
    # against wire.encode_spins_packed).
    ordered = spins_in_node_order(nodes, result)
    packed = np.packbits(ordered > 0, axis=1, bitorder="little")
    solutions = [
        miner_pb2.Solution(
            spins=row.tobytes(),
            energy_milli=int(round(qpu_e * 1000)),
        )
        for row, qpu_e in zip(packed, result.energies)
    ]

    meta = miner_pb2.SamplerMeta(
        reads=result.num_reads,
        # The resolved budget echo, not work performed: see _sampling_params.
        sweeps=num_sweeps,
        device_access_time_us=result.device_access_time_us,
        qpu_access_us=result.device_access_time_us,
        extra=result.extra,
    )
    return [
        miner_pb2.MinerMsg(
            result=miner_pb2.Result(
                job_id=job_id,
                solutions=solutions,
                meta=meta,
            )
        ),
        miner_pb2.MinerMsg(job_request=miner_pb2.JobRequest(credits=1)),
    ]


def handle_job(
    job: miner_pb2.Job,
    sampler: SupportsSample,
    *,
    session_nodes: Sequence[int],
    session_edges: Sequence[Tuple[int, int]],
    session_hash: Optional[bytes] = None,
    session_target: Optional["miner_pb2.SetTarget"] = None,
    session_sweeps: int = DEFAULT_NUM_SWEEPS,
    session_defaults: SamplingDefaults = SamplingDefaults(),
) -> List[miner_pb2.MinerMsg]:
    """Validate and solve one job; return Result+JobRequest or Reject messages.

    Validation runs in two steps that raise :class:`_Rejected` rather than
    returning early, so every reject reason converges on one exit here.
    :func:`_validate_job` covers the job itself (kind, decodable h/j, deadline,
    topology hash) and :func:`_resolve_problem` covers agreement between the
    arrays and the resolved graph. ``session_hash`` is the hash of the cached
    ``Topology``, ``None`` until one arrives; a ``topology_hash`` job with no
    cache rejects ``TOPOLOGY_MISSING`` and a differing hash rejects
    ``TOPOLOGY_MISMATCH``, matching the Rust miners.
    """
    job_id = job.job_id
    try:
        ising, h, j_vals = _validate_job(job, session_hash)
        nodes, h_arr, edges, j_arr = _resolve_problem(
            ising,
            h,
            j_vals,
            session_nodes,
            session_edges,
            job_id=job_id,
        )
        warm_start = _resolve_warm_start(
            ising, len(nodes), session_defaults, job_id=job_id
        )
    except _Rejected as exc:
        return _reject(job_id, exc.reason)

    num_reads, anneal_time_us, num_sweeps = _sampling_params(
        ising, session_target, session_sweeps, session_defaults
    )
    # Passed only when the job carries one, so a sampler double written before
    # warm starts existed still serves every cold job.
    warm_kwargs = {} if warm_start is None else {"warm_start": warm_start}
    try:
        result: SampleResult = sampler.sample(
            nodes,
            h_arr,
            edges,
            j_arr,
            num_reads=num_reads,
            # 0 means the solver's own default anneal.
            anneal_time_us=anneal_time_us or None,
            # Use job_id bytes as the defect-clamp seed when present.
            nonce_seed=bytes(job_id) if job_id else None,
            label=f"quip-{job_id.hex()[:8] if job_id else 'job'}",
            # The handle a coordinator Cancel reaches this submission by. The
            # session loop cancels by job id, so the seed cannot double as it:
            # a job with no defects is given no seed at all.
            cancel_key=bytes(job_id) if job_id else None,
            **warm_kwargs,
        )
    except ScheduleError as exc:
        # The job asked for an anneal this chip cannot run. Resending it
        # changes nothing, so it is not the transient OVERLOADED below.
        logger.warning("job %s: %s; rejecting MALFORMED", job_id.hex(), exc)
        return _reject(job_id, miner_pb2.MALFORMED)
    except OffChipError as exc:
        # A graph that does not fit this chip never will, so the coordinator
        # must route it elsewhere rather than resend it here. Expected input,
        # so no traceback.
        logger.warning(
            "job %s: %s; rejecting TOPOLOGY_MISMATCH", job_id.hex(), exc
        )
        return _reject(job_id, miner_pb2.TOPOLOGY_MISMATCH)
    except Exception as exc:
        if is_solver_offline(exc):
            raise SolverUnavailable(job_id) from exc
        # Ocean raises many exception types (Leap auth, network), and a job
        # worker's exception would otherwise die inside a discarded pool
        # Future: no Result, no Reject, a coordinator credit consumed forever,
        # and nothing in the log. Answer the job instead: OVERLOADED marks the
        # failure transient — the coordinator may resend elsewhere or later —
        # and the traceback reaches the operator.
        logger.exception(
            "job %s: sampler raised; rejecting OVERLOADED", job_id.hex()
        )
        return _reject(job_id, miner_pb2.OVERLOADED)
    return _build_result(job_id, nodes, result, num_sweeps)
