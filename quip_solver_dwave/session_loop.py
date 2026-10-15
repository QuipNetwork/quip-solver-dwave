"""gRPC session loop: Hello → Welcome → Configure → credit/job cycle.

Mirrors the ``quip-mock-miner`` reference miner (quip-solver repo) using the
``quip_solver_core`` Python SDK. Uses the synchronous gRPC client with a request queue (reliable
over UDS).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional, Tuple

import grpc

from quip_solver_core import miner_pb2, miner_pb2_grpc, session as session_sdk

from quip_solver_dwave import (
    ALGORITHM,
    BACKEND,
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_INTERNAL_FATAL,
    EXIT_TOKEN_REJECTED,
    FEATURES,
    MAX_EDGES,
    MAX_NODES,
)
from quip_solver_dwave.attempts import (
    default_attempts_dir,
    pickup_outcomes,
    seed_from_attempts,
)
from quip_solver_dwave.budget import (
    BudgetPacer,
    BudgetUnavailable,
    ParticipationDecision,
    budget_from_backend_toml,
    usage_db_from_backend_toml,
    warn_unknown_backend_keys,
)
from quip_solver_dwave.config import (
    SamplingDefaults,
    queue_depth_from_toml,
    sampling_defaults_from_toml,
)
from quip_solver_dwave.history import HistoryRecorder, JobSample
from quip_solver_dwave.job import SolverUnavailable, handle_job
from quip_solver_core.session import DEFAULT_NUM_SWEEPS, num_sweeps_from_toml
from quip_solver_dwave.ocean import OceanSampler, is_solver_unavailable
from quip_solver_dwave.profile import SnapshotRefresher
from quip_solver_dwave.strategy import (
    RoundDecision,
    RoundStrategy,
    describe_round_decision,
    strategy_config_from_toml,
)

logger = logging.getLogger(__name__)


def _surface_pool_failure(future) -> None:
    """Log a job-worker exception instead of letting the Future swallow it.

    ``handle_job`` answers every failure it can see (a sampler exception
    rejects OVERLOADED with a credit refund); this backstop catches the ones
    it cannot — a crash in the reply bookkeeping itself. Replies may be
    partially sent at that point, so no blind refund: the log line is the
    difference between a diagnosable incident and silence.
    """
    exc = future.exception()
    if exc is not None:
        logger.error("job worker crashed: %r", exc, exc_info=exc)

_STOP = object()

# How many submissions this backend keeps on the QPU at once when nobody says
# otherwise. Sized so that a QPU which becomes free can be fed at its own
# ceiling, which is Little's Law and has nothing to do with how contended the
# device happens to be today: a queue sized for a busy chip starves a free one.
#
#     depth = chip_throughput * round_trip
#
# Measured against Advantage2_system1 on a production-sized problem (4577
# nodes, 41514 couplers, num_reads=48): 43.2 ms of access time per job, so the
# chip tops out at 23.2 jobs/s, and an uncontended round trip is 1.57 s. That
# needs 36 in flight to saturate. Session logs show the round trip reaching
# 3.05 s under load, which needs 71.
#
# 96 holds the chip saturated through a 4.14 s round trip — 2.6x the clean
# figure and well past the worst observed — so a connectivity blip degrades
# throughput rather than stalling the device. Probed to 128 with no error at
# any depth: the Leap project's concurrency cap is nowhere near this.
#
# The cost of depth is reseed exposure: a Cancel catching a full pipeline
# strands this many jobs, and the ones D-Wave has already begun annealing are
# billed. That is affordable only because the Cancel path now reaches SAPI.
DEFAULT_QUEUE_DEPTH = 96

# Operator log token. Distinct from BACKEND ("dwave-qpu"), which is the
# protocol advertisement. Mixed-fleet lines use [quip-solver-dwave].
_LOG_BACKEND = "dwave"


def _short_job_id(job_id: bytes) -> str:
    """Render the leading bytes of a job id, matching quip-solver-core."""
    head = bytes(job_id[:8]).hex()
    if len(job_id) > 8:
        return head + ".."
    return head


def _energy_units(milli: int) -> int:
    """Floor-divide a milli-energy to whole units for a log line."""
    return milli // 1000


def _format_duration_ms(ms: int) -> str:
    """Render a millisecond duration with the shared miner buckets."""
    if ms < 1_000:
        return f"{ms}ms"
    if ms < 60_000:
        tenths = (ms + 50) // 100
        if tenths >= 600:
            return "1m 0s"
        secs = tenths // 10
        frac = tenths % 10
        return f"{secs}.{frac}s"
    if ms < 3_600_000:
        mins = ms // 60_000
        rem = ms % 60_000
        secs = (rem + 500) // 1_000
        if secs == 60:
            mins += 1
            secs = 0
        if mins >= 60:
            return f"{mins // 60}h {mins % 60}m"
        return f"{mins}m {secs}s"
    hours = ms // 3_600_000
    rem = ms % 3_600_000
    mins = (rem + 30_000) // 60_000
    if mins == 60:
        hours += 1
        mins = 0
    return f"{hours}h {mins}m"


def _reject_reason_name(reason: int) -> str:
    try:
        return miner_pb2.RejectReason.Name(reason)
    except ValueError:
        return str(reason)


@dataclass
class GateResult:
    """Outcome of asking the participation gate whether the QPU may sample."""

    allowed: bool
    changed: bool
    decision: ParticipationDecision
    # The strategy's verdict, when the budget allowed one to be asked for.
    round: Optional[RoundDecision] = None


class ParticipationGate:
    """Funds whole qblocks, never half of one.

    The miner signals participation by granting credits: the coordinator
    dispatches against credits alone, so withholding them is how the QPU sits a
    round out. Two rules govern when they move, and both consult the same
    budget line:

    * A join only ever happens on a qblock boundary. Joining mid-round buys a
      share of a round the field started ahead of us, and the coordinator
      cancels whatever is still staged at the boundary anyway, so the access
      time it costs is spent for nothing.
    * A joined round runs to its end. The pacing line is a target for the
      month, not a limit, and stopping at it mid-round throws away the rest
      of a round the miner committed to. The one mid-round stop is the
      period's allotment itself being spent; the next join still waits for
      a boundary.
    """

    def __init__(self, pacer: BudgetPacer, strategy: Optional[RoundStrategy] = None):
        self._pacer = pacer
        self._strategy = strategy
        self._participating = False
        self._boundary_generation = 0

    @property
    def participating(self) -> bool:
        return self._participating

    def use_strategy(self, strategy: RoundStrategy) -> None:
        """Attach the round strategy once history exists. Budget-only before."""
        self._strategy = strategy

    def on_qblock_boundary(
        self, generation: int, now: float
    ) -> Optional[GateResult]:
        """Re-decide at a new qblock. None when this is not a fresh boundary.

        Budget first, strategy second: the strategy is only asked whether a
        round the miner can afford is worth it. A skip parks credits the same
        way a budget line crossing does, and the next join waits for a
        boundary as before.
        """
        if generation <= self._boundary_generation:
            return None
        self._boundary_generation = generation
        decision = self._pacer.decide(now)
        allowed = decision.participate
        verdict: Optional[RoundDecision] = None
        if allowed and self._strategy is not None:
            verdict = self._strategy.decide(now, decision)
            allowed = verdict.join
        changed = allowed != self._participating
        self._participating = allowed
        return GateResult(
            allowed=allowed, changed=changed, decision=decision, round=verdict
        )

    def on_job(self, now: float) -> GateResult:
        """May this job be sampled? Yes while the round is joined and the
        period's allotment is not spent."""
        if not self._participating:
            return GateResult(
                allowed=False, changed=False, decision=self._pacer.decide(now)
            )
        decision = self._pacer.decide(now)
        changed = decision.exhausted
        if changed:
            self._participating = False
        return GateResult(
            allowed=not decision.exhausted, changed=changed, decision=decision
        )


class InflightJobs:
    """Which job is on the QPU right now, and for which generation.

    The sampler cancels by job id and knows nothing about generations; the
    coordinator abandons generations and knows nothing about what this miner
    still has in flight. This map is the join between the two.

    Reads never consume: a worker thread owns the removal of its own job, so a
    Cancel that dropped entries here would strand their release and leak the
    map for the life of the session.
    """

    def __init__(self) -> None:
        self._generations: dict[bytes, int] = {}
        self._lock = threading.Lock()

    def add(self, job_id: bytes, generation: int) -> None:
        with self._lock:
            self._generations[bytes(job_id)] = generation

    def release(self, job_id: bytes) -> None:
        with self._lock:
            self._generations.pop(bytes(job_id), None)

    def abandoned(self, watermark: int) -> list[bytes]:
        """Job ids belonging to a generation the coordinator reseeded past."""
        with self._lock:
            return [
                job_id
                for job_id, generation in self._generations.items()
                if _is_abandoned(generation, watermark)
            ]


class CancelTally:
    """How often a SAPI cancel actually beat the anneal.

    D-Wave only refunds a problem it has not started annealing, so the value
    of cancelling is an empirical question this answers: ``requested`` counts
    the problems handed to SAPI, ``missed`` the ones that came back with a
    full result anyway, having been charged in full.
    """

    def __init__(self) -> None:
        self._requested = 0
        self._missed = 0

    def requested(self, count: int) -> None:
        self._requested += count

    def missed(self) -> None:
        self._missed += 1

    def summary(self) -> str:
        if self._requested == 0:
            return "no cancels yet"
        pct = 100.0 * self._missed / self._requested
        return (
            f"{self._requested} cancelled, "
            f"{self._missed} annealed anyway ({pct:.0f}% missed)"
        )


def resolve_queue_depth(*, coordinator: int, configured: int) -> int:
    """Pick the pipeline depth from the three rungs that can set it.

    Mirrors the per-job ladder in :func:`quip_solver_dwave.job._sampling_params`:
    the operator's ``backend_toml`` wins because QPU-specific knowledge lives
    there, then whatever the coordinator sent, then this backend's own measured
    default. Zero means "not set" on the wire and in the config, and the floor
    of 1 keeps a misconfiguration from granting no credits and mining nothing.
    """
    if configured > 0:
        return configured
    if coordinator > 0:
        return coordinator
    return max(1, DEFAULT_QUEUE_DEPTH)


def _bill_unobserved(sampler, pacer: Optional[BudgetPacer]) -> None:
    """Record access time D-Wave charged for but never reported back.

    A cancelled submission raises instead of returning samples, so it carries
    no ``device_access_time_us``. The sampler estimates the charge rather than
    assuming none (see ``OceanSampler._charge_unobserved``); this drains that
    estimate into the ledger. Draining on a job that ran unbudgeted would
    throw the estimate away, so the pacer's absence is checked first.
    """
    if pacer is None:
        return
    owed = sampler.drain_unobserved_access_us()
    if owed:
        pacer.record_access_time(owed, time.time())


def _seed_history(
    recorder: HistoryRecorder,
    attempts_path: str,
    stop: threading.Event,
    miner_id: str,
    on_done: Optional[Callable[[], None]] = None,
) -> None:
    """Seed past rounds off the session thread. Best effort by design.

    A node with a thousand-plus attempt directories can take tens of seconds
    to seed, so shutdown cannot just wait it out: ``stop`` lets shutdown ask
    the seed to give up between directories, checked once per directory
    before that directory's writes begin, so it never leaves margins written
    for a directory it does not also mark seeded. ``miner_id`` keeps a
    second QPU miner on the same node out of this one's history. ``on_done``
    rebuilds the strategy's snapshot as soon as the seed is in, rather than
    waiting for the next scheduled refresh.
    """
    try:
        seed_from_attempts(
            recorder.store, attempts_path, time.time(), stop=stop.is_set, miner_id=miner_id
        )
    except Exception:  # noqa: BLE001 - history is optional, mining is not
        logger.warning("history: seeding from %s failed", attempts_path, exc_info=True)
    if on_done is not None:
        on_done()


def _pickup_history_outcomes(
    recorder: HistoryRecorder,
    attempts_path: str,
    miner_id: str,
    on_done: Optional[Callable[[], None]] = None,
) -> None:
    """Apply the coordinator's verdicts on recent rounds. Best effort."""
    try:
        pickup_outcomes(recorder.store, attempts_path, miner_id=miner_id)
    except Exception:  # noqa: BLE001 - see _seed_history
        logger.warning(
            "history: outcome pickup from %s failed", attempts_path, exc_info=True
        )
    if on_done is not None:
        on_done()


def _extra_int(meta, key: str) -> Optional[int]:
    """An integer the sampler stashed in ``SamplerMeta.extra``, or None."""
    if meta is None:
        return None
    raw = meta.extra.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _connect_solver(sampler, generation: Optional[int]) -> bool:
    """Connect to the QPU, or say why not. False means "try again next qblock".

    An offline solver is not fatal. The supervisor restarts a miner that
    exits six times with a backoff of 0.5 s doubling to 8 s and then gives
    up, so exiting here left the miner dead until an operator restarted it,
    even after D-Wave brought the solver back. Sitting the round out instead
    keeps the session alive and costs one SAPI probe per qblock. Any other
    connect error (credentials, network) still propagates and is fatal.
    """
    try:
        sampler.ensure_connected()
        return True
    except Exception as exc:
        if not is_solver_unavailable(exc):
            raise
        where = "at Configure" if generation is None else f"sitting out qblock {generation}"
        logger.warning(
            "[QPU] %s: solver offline (%s); the connect is retried once per "
            "qblock boundary and no credits go out until it succeeds",
            where,
            exc,
        )
        return False


def _log_qblock_joined(generation: int, decision: ParticipationDecision) -> None:
    logger.info(
        "[QPU] joining qblock %d: %.0fs of headroom on the budget line "
        "(spent %.0fs of %.0fs earned so far)",
        generation,
        decision.headroom_us / 1_000_000,
        decision.spent_us / 1_000_000,
        decision.allowance_us / 1_000_000,
    )


def _log_qblock_sat_out(generation: int, decision: ParticipationDecision) -> None:
    """One line when the gate shuts, with the reopening estimate.

    The per-attempt rejects sit at debug on purpose: a shut gate rejects every
    job the coordinator already staged, and narrating each one at warning
    buries the session log without adding anything this line did not say.
    """
    logger.info(
        "[QPU] sitting out qblock %d: %.0fs past the budget line, next window "
        "in %s",
        generation,
        -decision.headroom_us / 1_000_000,
        _format_duration_ms(int(decision.seconds_until_headroom * 1000)),
    )


def _log_allotment_spent(decision: ParticipationDecision) -> None:
    logger.info(
        "[QPU] period allotment spent mid-qblock (%.0fs of %.0fs); parking "
        "credits, the QPU rejoins at the first qblock after the reset in %s",
        decision.spent_us / 1_000_000,
        decision.budget_us / 1_000_000,
        _format_duration_ms(int(decision.seconds_until_headroom * 1000)),
    )


def _utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _log_budget_configured(pacer: BudgetPacer, now: float) -> None:
    """State the quota, then say plainly why the QPU is not mining yet.

    A budgeted miner is idle between Configure and the next qblock boundary,
    and idle for whole qblocks whenever spend is ahead of the line. Both are
    correct and both look like a hang, so each one says why it is waiting and
    how long the wait is.
    """
    stats = pacer.stats(now)
    logger.info(
        "[QPU] budget %.0fs per period, resets day %d (period %s -> %s); "
        "spent %.0fs of %.0fs earned so far; jobs this period: %d",
        stats["budget_seconds"],
        stats["reset_day"],
        _utc_day(stats["period_start"]),
        _utc_day(stats["period_end"]),
        stats["spent_seconds"],
        stats["allowance_seconds"],
        stats["jobs_this_period"],
    )
    if stats["headroom_seconds"] > 0:
        logger.info(
            "[QPU] waiting: %.0fs of headroom is available, but credits are "
            "held until the next qblock boundary so the QPU joins a whole "
            "round rather than part of one",
            stats["headroom_seconds"],
        )
    else:
        logger.info(
            "[QPU] waiting: %.0fs past the budget line, so no credits are "
            "granted. Next window in %s, then the QPU joins at the following "
            "qblock boundary",
            -stats["headroom_seconds"],
            _format_duration_ms(int(stats["seconds_until_headroom"] * 1000)),
        )


def _attach_round_strategy(
    gate: ParticipationGate, refresher: SnapshotRefresher, backend_toml: str
) -> None:
    """Give the gate a strategy once both a budget and history exist.

    Called from whichever of the two Configure branches runs second: the
    budget can name a gate before the recorder exists, or the recorder can
    exist before a budget names a gate. Either order ends with a strategy
    attached.
    """
    config = strategy_config_from_toml(backend_toml)
    gate.use_strategy(RoundStrategy(config, refresher))
    logger.info(
        "[QPU] round strategy attached: min_throughput_advantage=%.2f "
        "participation_chance=%.2f",
        config.min_throughput_advantage,
        config.participation_chance,
    )


def log_attempt(
    job_id: bytes,
    *,
    energy_milli: Optional[int] = None,
    valid: int = 0,
    total: int = 0,
    wall_ms: int = 0,
    device_ms: int = 0,
    rejected: Optional[str] = None,
    cancelled: bool = False,
) -> None:
    """Emit one per-attempt line in the shared miner format."""
    job = _short_job_id(job_id)
    wall = _format_duration_ms(wall_ms)
    if cancelled:
        logger.debug(
            "[quip-solver-%s] attempt %s: cancelled after %s",
            _LOG_BACKEND,
            job,
            wall,
        )
        return
    if rejected is not None:
        logger.warning(
            "[quip-solver-%s] attempt %s: rejected %s | %s wall",
            _LOG_BACKEND,
            job,
            rejected,
            wall,
        )
        return
    energy = "n/a" if energy_milli is None else str(_energy_units(energy_milli))
    device = _format_duration_ms(device_ms)
    logger.info(
        "[quip-solver-%s] attempt %s: energy %s, valid %d/%d | %s wall, %s device",
        _LOG_BACKEND,
        job,
        energy,
        valid,
        total,
        wall,
        device,
    )


def log_progress(
    *,
    jobs_done: int,
    elapsed_s: float,
    reads: int,
    sweeps: int,
    best_energy_milli: Optional[int],
    max_energy_milli: Optional[int],
    min_solutions: int,
) -> None:
    """Emit the shared progress line every N completed jobs."""
    rate = jobs_done / elapsed_s if elapsed_s > 0 else 0.0
    best = (
        "n/a"
        if best_energy_milli is None
        else str(_energy_units(best_energy_milli))
    )
    if max_energy_milli is None:
        requirement = "no target set"
    else:
        requirement = "requires energy<={}, solutions>={}".format(
            _energy_units(max_energy_milli),
            min_solutions,
        )
    logger.info(
        "[quip-solver-%s] progress: %d jobs | %.1f jobs/s | reads=%d sweeps=%d | best=%s | %s",
        _LOG_BACKEND,
        jobs_done,
        rate,
        reads,
        sweeps,
        best,
        requirement,
    )


def _status(
    miner_id: str, jobs_done: int = 0, abandoned: int = 0
) -> miner_pb2.MinerMsg:
    return miner_pb2.MinerMsg(
        status=miner_pb2.Status(
            miner_id=miner_id,
            utilization=0.0,
            jobs_done=jobs_done,
            abandoned_generation=abandoned,
        )
    )


def _is_abandoned(generation: int, watermark: int) -> bool:
    """True if a job's ``generation`` was abandoned by a Cancel(``watermark``).

    Generation ``0`` is a mempool job with no PoW cancellation scope and is
    never abandoned; a PoW job is abandoned once its generation is at or below
    the coordinator's reseed watermark. Mirrors the Rust ``CancelToken``.
    """
    return generation != 0 and generation <= watermark


def capabilities_message() -> miner_pb2.Capabilities:
    """Build the ``Capabilities`` message: what this backend supports.

    Must answer without touching the device, so it is a pure function of the
    same static numbers ``Hello`` advertises. ``--capabilities`` prints the
    protobuf JSON of this exact message and the in-session ``GetCapabilities``
    reply wraps it, so the two answers cannot drift apart (SPEC section 8).
    """
    return miner_pb2.Capabilities(
        backend=BACKEND,
        algorithm=ALGORITHM,
        supported_kinds=[miner_pb2.ISING_SAMPLE],
        max_nodes=MAX_NODES,
        max_edges=MAX_EDGES,
        features=list(FEATURES),
        protocol_version=session_sdk.PROTOCOL_VERSION,
        stream_width=1,
    )


def _unix_target(uri: str) -> str:
    """Normalize ``unix:///path`` or bare path to a grpc UDS target."""
    if uri.startswith("unix://"):
        path = uri[len("unix://") :]
    elif uri.startswith("unix:"):
        path = uri[len("unix:") :]
    else:
        path = uri
    if not path.startswith("/"):
        path = "/" + path
    # grpc-python accepts both unix:/abs and unix:///abs; prefer the triple-slash
    # form which matches tonic's unix:// advertisement after strip.
    return f"unix://{path}"


def run_session(
    coordinator_uri: str,
    miner_id: str,
    sampler: OceanSampler,
    *,
    budget: Optional[BudgetPacer] = None,
    attempts_dir: Optional[str] = None,
) -> int:
    """Run one miner session; return a process exit code."""
    target = _unix_target(coordinator_uri)
    try:
        hello = session_sdk.build_hello(
            miner_id,
            BACKEND,
            ALGORITHM,
            [miner_pb2.ISING_SAMPLE],
            MAX_NODES,
            MAX_EDGES,
            features=list(FEATURES),
        )
    except session_sdk.MissingToken:
        logger.error("QUIP_SESSION_TOKEN unset")
        return EXIT_TOKEN_REJECTED

    if sampler.native_topology_hash:
        hello.native_topology_hash = sampler.native_topology_hash

    out_q: queue.Queue = queue.Queue()
    # Pre-buffer Hello so the coordinator handshake has something immediately.
    out_q.put(miner_pb2.MinerMsg(hello=hello))

    def request_iter() -> Iterator[miner_pb2.MinerMsg]:
        while True:
            item = out_q.get()
            if item is _STOP:
                return
            yield item

    jobs_done = 0
    # Watermark set by Cancel(max_generation): any job at or below it belongs to
    # a generation the coordinator abandoned on reseed and is skipped, not sampled.
    cancel_watermark = 0
    grace_ms = 5000
    config: Optional[session_sdk.SessionConfig] = None
    session_nodes: list[int] = []
    session_edges: list[Tuple[int, int]] = []
    session_hash: Optional[bytes] = None
    session_target: Optional[miner_pb2.SetTarget] = None
    session_sweeps: int = DEFAULT_NUM_SWEEPS
    session_defaults = SamplingDefaults()
    pending_budget = budget
    # None until a budget is configured. While it is None the miner is
    # unmetered and mines every round, as it did before budget pacing.
    gate: Optional[ParticipationGate] = (
        ParticipationGate(pending_budget) if pending_budget is not None else None
    )
    queue_depth = DEFAULT_QUEUE_DEPTH
    # Pipeline: up to queue_depth QPU submissions in flight (overlaps cloud RTT).
    # jobs_done + pending_budget are shared with worker threads -> guard them.
    state_lock = threading.Lock()
    # What the QPU is chewing on right now, so a Cancel can reach it.
    inflight = InflightJobs()
    tally = CancelTally()
    # Credits held back because the solver is offline. A reject normally
    # refunds its credit and the coordinator dispatches the next staged job
    # at once; against an offline solver that is a reject loop at SAPI
    # round-trip speed for the whole qblock. The held credits go back out at
    # the next qblock boundary, one probe per round.
    parked_credits = 0
    # True while the QPU connect fails with an offline solver. The connect is
    # retried at each qblock boundary; until it succeeds no credits go out.
    solver_down = False
    job_pool: Optional[ThreadPoolExecutor] = None
    # History of throughput, rounds and margins. None until Configure names
    # the usage database, or forever if it cannot be opened. Optional in the
    # strict sense: nothing about mining depends on it.
    recorder: Optional[HistoryRecorder] = None
    attempts_path: Optional[str] = None
    seed_thread: Optional[threading.Thread] = None
    seed_stop = threading.Event()
    outcome_thread: Optional[threading.Thread] = None
    refresher: Optional[SnapshotRefresher] = None
    session_start = time.monotonic()
    best_energy_milli: Optional[int] = None
    PROGRESS_LOG_INTERVAL = 10

    def process_job(job, s_nodes, s_edges, s_hash, s_target, s_sweeps, s_defaults):
        # Runs on a pool thread: sample (blocking on the QPU), then enqueue
        # replies. Shared-state mutations are guarded by state_lock.
        nonlocal jobs_done, best_energy_milli, parked_credits
        started = time.monotonic()
        try:
            replies = handle_job(
                job,
                sampler,
                session_nodes=s_nodes,
                session_edges=s_edges,
                session_hash=s_hash,
                session_target=s_target,
                session_sweeps=s_sweeps,
                session_defaults=s_defaults,
            )
        except SolverUnavailable as exc:
            # The reject goes out without its refund: the credit is parked
            # until the next qblock boundary. Parked before the reject is
            # queued, so a boundary that sees the reject also sees the credit.
            with state_lock:
                parked_credits += 1
                held = parked_credits
            log = logger.warning if held == 1 else logger.debug
            log(
                "[QPU] solver offline (%s): parking the credit, %d held until "
                "the next qblock boundary",
                exc.__cause__,
                held,
            )
            replies = [
                miner_pb2.MinerMsg(
                    reject=miner_pb2.Reject(
                        job_id=job.job_id, reason=miner_pb2.OVERLOADED
                    )
                )
            ]
        finally:
            # Off the QPU one way or another: a later Cancel must not try to
            # drop a problem that has already finished.
            inflight.release(job.job_id)
            if recorder is not None:
                recorder.job_finished(time.time())
        _bill_unobserved(sampler, pending_budget)
        wall_ms = int((time.monotonic() - started) * 1000)
        for reply in replies:
            kind = reply.WhichOneof("msg")
            meta = reply.result.meta if kind == "result" else None
            if kind == "result" and pending_budget is not None and meta is not None:
                # Billed before the abandoned check, not after. D-Wave
                # charged for this anneal whatever the coordinator decided
                # to do with the answer, and a ledger that under-counts
                # hands the pacer headroom the QPU has already spent.
                #
                # Billed outside state_lock, too. This commits to SQLite,
                # which fsyncs the deployment's mounted volume, and the
                # ledger already has its own lock. state_lock is what the
                # session-loop thread takes to dispatch the next job, so
                # holding it across this put every Cancel, Job and Ping
                # behind one worker's disk write — at a pipeline depth of
                # 96, behind all of them.
                pending_budget.record_access_time(
                    meta.device_access_time_us, time.time()
                )
            job_best: Optional[int] = None
            valid = 0
            if kind == "result":
                job_best = min(
                    (s.energy_milli for s in reply.result.solutions),
                    default=None,
                )
                if s_target is not None:
                    valid = sum(
                        1
                        for s in reply.result.solutions
                        if s.energy_milli <= s_target.max_energy_milli
                    )
                else:
                    valid = len(reply.result.solutions)
            skip = False
            wasted = False
            with state_lock:
                abandoned = _is_abandoned(job.generation, cancel_watermark)
                if kind in ("result", "reject") and abandoned:
                    # SPEC section 5: no Result for an abandoned generation.
                    # The Reject goes the same way — a cancelled submission
                    # surfaces as one, and answering a job the coordinator has
                    # already reseeded past is noise it cannot act on.
                    if kind == "result":
                        # It came back with samples, so SAPI ran the anneal
                        # despite the cancel. That is the miss rate.
                        tally.missed()
                        wasted = True
                    skip = True
                elif kind == "result":
                    jobs_done += 1
                    if job_best is not None and (
                        best_energy_milli is None or job_best < best_energy_milli
                    ):
                        best_energy_milli = job_best
                    device_ms = (
                        meta.device_access_time_us // 1000
                        if meta is not None
                        else 0
                    )
                    log_attempt(
                        job.job_id,
                        energy_milli=job_best,
                        valid=valid,
                        total=len(reply.result.solutions),
                        wall_ms=wall_ms,
                        device_ms=device_ms,
                    )
                    if jobs_done % PROGRESS_LOG_INTERVAL == 0:
                        log_progress(
                            jobs_done=jobs_done,
                            elapsed_s=time.monotonic() - session_start,
                            reads=meta.reads if meta is not None else 0,
                            sweeps=meta.sweeps if meta is not None else 0,
                            best_energy_milli=best_energy_milli,
                            max_energy_milli=(
                                s_target.max_energy_milli
                                if s_target is not None
                                else None
                            ),
                            min_solutions=(
                                int(s_target.min_solutions)
                                if s_target is not None
                                else 0
                            ),
                        )
                elif kind == "reject":
                    log_attempt(
                        job.job_id,
                        rejected=_reject_reason_name(reply.reject.reason),
                        wall_ms=wall_ms,
                    )
                if not skip and kind == "job_request" and gate is not None:
                    # A finished job refills its own credit to keep the pipeline
                    # full mid-qblock. Dropping the refill is what parks the
                    # credits once the allotment is spent; the next grant
                    # waits for a qblock boundary after the reset.
                    result = gate.on_job(time.time())
                    if not result.allowed:
                        if result.changed:
                            _log_allotment_spent(result.decision)
                        skip = True
            # History goes after the lock for the same reason billing does:
            # it ends in a SQLite commit, and state_lock is what dispatches
            # the next job. A wasted result is a completed anneal the
            # coordinator threw away; the round trip counts, the job does not.
            if recorder is not None and kind == "result" and meta is not None:
                if wasted:
                    recorder.record_wasted(time.time())
                else:
                    recorder.record_job(
                        JobSample(
                            completed_at=time.time(),
                            generation=job.generation,
                            rtt_ms=wall_ms,
                            access_us=int(meta.device_access_time_us),
                            inflight_at_submit=_extra_int(meta, "inflight") or 0,
                            reads=int(meta.reads),
                            best_energy_milli=job_best,
                            target_milli=(
                                s_target.max_energy_milli
                                if s_target is not None
                                else None
                            ),
                            hits=valid if s_target is not None else 0,
                            sapi_ms=_extra_int(meta, "sapi_ms"),
                        )
                    )
            if skip:
                continue
            out_q.put(reply)

    def _stop_history_threads() -> None:
        # The refresher stops first, and its join is bounded at two seconds:
        # what that order buys is the join finishing before the recorder
        # below closes the store. A refresh still in flight when the store
        # does close is not silently lost -- its next query raises, and
        # refresh_now logs one warning and returns the snapshot already in
        # hand, nothing else.
        if refresher is not None:
            refresher.stop()
        # The seed can take tens of seconds on a node with a thousand-plus
        # attempt directories, so shutdown asks it to give up between
        # directories rather than block on it; a bounded join is the
        # backstop if it is already past that check. Outcome pickup only
        # ever reads two short directories, so a short join is enough there.
        if seed_thread is not None:
            seed_stop.set()
            seed_thread.join(timeout=5.0)
        if outcome_thread is not None:
            outcome_thread.join(timeout=2.0)

    exit_code = EXIT_CLEAN

    # tonic (Rust) rejects UDS streams whose :authority is the socket path;
    # pin a conventional authority so grpc-python ↔ tonic interop works.
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.default_authority", "localhost"),
            ("grpc.enable_http_proxy", 0),
        ],
    )
    try:
        # Wait briefly for the coordinator's listener (race with process spawn).
        deadline = time.time() + 10.0
        while True:
            try:
                grpc.channel_ready_future(channel).result(
                    timeout=max(0.05, deadline - time.time())
                )
                break
            except grpc.FutureTimeoutError:
                if time.time() >= deadline:
                    logger.error("timed out waiting for coordinator at %s", target)
                    return EXIT_INTERNAL_FATAL

        stub = miner_pb2_grpc.MinerServiceStub(channel)
        responses = stub.Session(request_iter())

        last_activity = time.monotonic()
        for cm in responses:
            last_activity = time.monotonic()
            which = cm.WhichOneof("msg")
            if which == "welcome":
                ver = cm.welcome.protocol_version
                if ver not in (0, 1):
                    logger.error("bad Welcome protocol_version=%s", ver)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            fatal=miner_pb2.Fatal(
                                exit_code=EXIT_INTERNAL_FATAL,
                                reason="bad welcome protocol_version",
                                restart_required=False,
                            )
                        )
                    )
                    exit_code = EXIT_INTERNAL_FATAL
                    break
            elif which == "configure":
                config = session_sdk.session_config_from_configure(
                    miner_id, cm.configure
                )
                # The coordinator has engaged us: connect to the QPU now (a
                # no-op in mock mode / when already connected).
                solver_down = not _connect_solver(sampler, None)
                # Uniform config handling: warn on any key the dwave schema
                # doesn't recognize before consuming the ones it does.
                warn_unknown_backend_keys(cm.configure.backend_toml)
                # Session-wide sweep budget: a top-level num_sweeps key, or
                # the SDK default. Echoed per Result in SamplerMeta.sweeps.
                session_sweeps = num_sweeps_from_toml(cm.configure.backend_toml)
                # Operator-set sampling defaults for jobs the coordinator left
                # blank. Lowest rung of the precedence ladder in job.py.
                session_defaults = sampling_defaults_from_toml(
                    cm.configure.backend_toml
                )
                if session_defaults != SamplingDefaults():
                    logger.info(
                        "[budget] sampling defaults from config: num_reads=%s "
                        "anneal_time_us=%s (per-job and SetTarget values still win)",
                        session_defaults.num_reads or "unset",
                        session_defaults.anneal_time_us or "unset",
                    )
                if pending_budget is None and cm.configure.backend_toml:
                    try:
                        pending_budget = budget_from_backend_toml(
                            cm.configure.backend_toml
                        )
                    except BudgetUnavailable as exc:
                        # No durable meter, no mining: an unmetered miner burns
                        # the period's quota in a day. Name the fix and stop.
                        logger.error("%s", exc)
                        out_q.put(
                            miner_pb2.MinerMsg(
                                fatal=miner_pb2.Fatal(
                                    exit_code=EXIT_CONFIG_INVALID,
                                    reason=str(exc),
                                    restart_required=False,
                                )
                            )
                        )
                        exit_code = EXIT_CONFIG_INVALID
                        break
                    if pending_budget is not None:
                        gate = ParticipationGate(pending_budget)
                        # The recorder (and its refresher) may already exist
                        # from an earlier Configure that named no budget.
                        if refresher is not None:
                            _attach_round_strategy(
                                gate, refresher, cm.configure.backend_toml
                            )
                out_q.put(miner_pb2.MinerMsg(ready=miner_pb2.Ready()))
                # Read straight off the wire, not off SessionConfig: the SDK
                # substitutes its own default of 3 for an unset field, which
                # would hide the "coordinator said nothing" case this backend
                # wants to answer with its own measured depth.
                depth = resolve_queue_depth(
                    coordinator=cm.configure.queue_depth,
                    configured=queue_depth_from_toml(cm.configure.backend_toml),
                )
                queue_depth = depth
                logger.info(
                    "[QPU] pipeline depth %d (coordinator asked for %s)",
                    depth,
                    cm.configure.queue_depth or "nothing",
                )
                if job_pool is None:
                    job_pool = ThreadPoolExecutor(
                        max_workers=max(1, depth), thread_name_prefix="dwave-job"
                    )
                # Local binding so the None check narrows: pending_budget is
                # captured by process_job, which blocks narrowing on it.
                pacer = pending_budget
                if solver_down:
                    # Nothing to dispatch against until the connect succeeds;
                    # the boundary that reconnects grants the depth.
                    pass
                elif pacer is None:
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=depth)
                        )
                    )
                else:
                    # Ready says the session is established; credits say the QPU
                    # is participating. They are deliberately not the same
                    # message. A budgeted miner grants nothing until a qblock
                    # boundary gives it a whole round to decide about.
                    _log_budget_configured(pacer, time.time())
                if recorder is None:
                    history_path = usage_db_from_backend_toml(cm.configure.backend_toml)
                    recorder = HistoryRecorder.open(history_path)
                    if recorder is not None:
                        attempts_path = attempts_dir or default_attempts_dir(history_path)
                        refresher = SnapshotRefresher(recorder.store)
                        refresher.start()
                        if gate is not None:
                            _attach_round_strategy(
                                gate, refresher, cm.configure.backend_toml
                            )
                        # Past rounds come from the coordinator's attempts
                        # files, hundreds of megabytes on a long-lived node:
                        # not work for the session thread. seed_stop lets
                        # shutdown cut it short between directories rather
                        # than block on it. The snapshot is rebuilt as soon
                        # as they are in.
                        seed_thread = threading.Thread(
                            target=_seed_history,
                            args=(recorder, attempts_path, seed_stop, miner_id),
                            kwargs={"on_done": refresher.request_refresh},
                            name="dwave-history-seed",
                            daemon=True,
                        )
                        seed_thread.start()
            elif which == "topology":
                topo = cm.topology
                session_nodes = list(topo.nodes)
                session_hash = bytes(topo.hash)
                if topo.HasField("edges"):
                    session_edges = list(zip(topo.edges.u, topo.edges.v))
                else:
                    session_edges = []
                sampler.set_session_topology(session_nodes, session_edges)
            elif which == "set_target":
                session_target = cm.set_target
                if recorder is not None:
                    recorder.round_target(cm.set_target.max_energy_milli)
            elif which == "job":
                with state_lock:
                    cancelled = _is_abandoned(cm.job.generation, cancel_watermark)
                if cancelled:
                    # Abandoned generation (coordinator reseeded): don't spend
                    # QPU access on it. Refund the credit so the coordinator
                    # keeps the pipeline full — mirrors the Rust miners' skip at
                    # dequeue. In-flight submissions are left to finish; the
                    # coordinator discards their stale-generation Results.
                    #
                    # The gate is deliberately not consulted: a skipped job bills
                    # no access time, so it must not be the thing that decides
                    # participation, and parking here would strand the refund.
                    log_attempt(cm.job.job_id, cancelled=True, wall_ms=0)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            job_request=miner_pb2.JobRequest(credits=1)
                        )
                    )
                    continue
                with state_lock:
                    gate_result = None if gate is None else gate.on_job(time.time())
                    budget_ok = gate_result is None or gate_result.allowed
                if not budget_ok:
                    logger.debug(
                        "[quip-solver-%s] attempt %s: rejected OVERLOADED (budget)",
                        _LOG_BACKEND,
                        _short_job_id(cm.job.job_id),
                    )
                    if gate_result is not None and gate_result.changed:
                        _log_allotment_spent(gate_result.decision)
                    out_q.put(
                        miner_pb2.MinerMsg(
                            reject=miner_pb2.Reject(
                                job_id=cm.job.job_id,
                                reason=miner_pb2.OVERLOADED,
                            )
                        )
                    )
                    # No replacement credit. The coordinator dispatches against
                    # credits alone, so refunding one here turns a shut gate into
                    # a reject/dispatch spin at memory speed — the credits come
                    # back at a qblock boundary, once the line allows it.
                    continue
                # Submit for concurrent sampling; the pool bounds in-flight to
                # queue_depth (credits keep the coordinator dispatching that many).
                args = (
                    cm.job,
                    list(session_nodes),
                    list(session_edges),
                    session_hash,
                    session_target,
                    session_sweeps,
                    session_defaults,
                )
                # Tracked before the submit, not inside the worker: a Cancel
                # arriving while the job waits for a pool thread must still
                # reach it, and the sampler holds the note until it registers.
                inflight.add(cm.job.job_id, cm.job.generation)
                if recorder is not None:
                    recorder.job_started(time.time())
                if job_pool is not None:
                    job_pool.submit(process_job, *args).add_done_callback(
                        _surface_pool_failure
                    )
                else:
                    process_job(*args)
            elif which == "cancel":
                # Raise the reseed watermark so every job at/below max_generation
                # is skipped at dequeue instead of sampled on the QPU. The Status
                # reply reports the real watermark, not just an ack flag, so the
                # coordinator can see the current cancel point outside a Cancel
                # round-trip too (Ping reports it the same way, below).
                with state_lock:
                    cancel_watermark = max(cancel_watermark, cm.cancel.max_generation)
                    done = jobs_done
                    watermark = cancel_watermark
                out_q.put(_status(miner_id, done, abandoned=watermark))
                # Every job of an abandoned generation still on the QPU is
                # access time buying a round the coordinator has thrown away.
                # Ask D-Wave to drop them; it refunds only the ones it has not
                # started annealing, which is what the tally measures.
                doomed = inflight.abandoned(watermark)
                if doomed:
                    live = sampler.cancel_inflight(doomed)
                    tally.requested(len(doomed))
                    logger.info(
                        "[QPU] cancel gen<=%d: asked D-Wave to drop %d in-flight "
                        "job(s), %d still running | session: %s",
                        watermark,
                        len(doomed),
                        live,
                        tally.summary(),
                    )
                # A reseed is the one qblock boundary the miner can see: it is
                # the only monotone round counter the coordinator sends, and it
                # arrives every round even while the QPU holds no credits. That
                # makes it the point at which participation is decided.
                reconnected = False
                if solver_down:
                    solver_down = not _connect_solver(sampler, cm.cancel.max_generation)
                    if not solver_down:
                        reconnected = True
                        # The session Topology arrived while the live graph
                        # was unknown, so it was taken as native. Bind it
                        # again to compute defects against the real QPU.
                        if session_nodes:
                            sampler.set_session_topology(session_nodes, session_edges)
                        logger.info(
                            "[QPU] qblock %d: solver back, connected",
                            cm.cancel.max_generation,
                        )
                boundary = None
                if gate is not None and not solver_down:
                    with state_lock:
                        boundary = gate.on_qblock_boundary(
                            cm.cancel.max_generation, time.time()
                        )
                    if boundary is not None:
                        if boundary.round is not None:
                            # One line per boundary either way; a skipped
                            # round names the slot it waits for.
                            logger.info(
                                "%s",
                                describe_round_decision(
                                    cm.cancel.max_generation,
                                    boundary.round,
                                    boundary.decision.headroom_us,
                                ),
                            )
                        if not boundary.allowed:
                            if boundary.round is None:
                                # Every skipped round says so, not just the
                                # first: a run of silent boundaries is exactly
                                # what made the old blackout unreadable.
                                _log_qblock_sat_out(
                                    cm.cancel.max_generation, boundary.decision
                                )
                        elif boundary.changed:
                            # Credits survive a reseed, so only the round that
                            # resumes mining needs a grant.
                            if boundary.round is None:
                                _log_qblock_joined(
                                    cm.cancel.max_generation, boundary.decision
                                )
                            out_q.put(
                                miner_pb2.MinerMsg(
                                    job_request=miner_pb2.JobRequest(
                                        credits=queue_depth
                                    )
                                )
                            )
                with state_lock:
                    parked, parked_credits = parked_credits, 0
                # A connect that just succeeded has nothing out at all, so
                # the whole depth goes out, not just what a reject parked.
                owed = queue_depth if reconnected else parked
                if owed and not solver_down:
                    # A round that just rejoined was granted the full depth
                    # above, and a round sat out gets the full depth when it
                    # rejoins; only a round still mining needs its credits
                    # back. Those are the probe: if the solver is still
                    # offline they park again, one reject each, this round.
                    regranted = (
                        boundary is not None and boundary.allowed and boundary.changed
                    )
                    mining = gate is None or gate.participating
                    if mining and not regranted:
                        logger.info(
                            "[QPU] qblock %d: granting %d credit(s) to probe "
                            "the solver",
                            cm.cancel.max_generation,
                            owed,
                        )
                        out_q.put(
                            miner_pb2.MinerMsg(
                                job_request=miner_pb2.JobRequest(credits=owed)
                            )
                        )
                if recorder is not None:
                    verdict = boundary.round if boundary is not None else None
                    if gate is None:
                        joined, reason = True, "unbudgeted"
                    # `boundary is not None` is implied by `verdict is not
                    # None` (verdict comes from boundary.round above), but
                    # pyright cannot narrow through that assignment, so the
                    # second conjunct is here only to satisfy it.
                    elif verdict is not None and boundary is not None:
                        joined, reason = boundary.allowed, verdict.reason
                    else:
                        # `boundary` is only None on a repeated watermark,
                        # which is exactly the condition that makes
                        # `round_boundary` below a no-op: it discards
                        # `joined`/`reason` without writing them, so
                        # `gate.participating` here is a safe placeholder.
                        joined = boundary.allowed if boundary is not None else gate.participating
                        reason = "budget" if joined else "budget-sat-out"
                    if solver_down:
                        joined, reason = False, "solver-offline"
                    fresh = recorder.round_boundary(
                        cm.cancel.max_generation,
                        time.time(),
                        joined=joined,
                        reason=reason,
                        expected_jobs=verdict.expected_jobs if verdict is not None else None,
                    )
                    # The coordinator's verdict on the round that just ended
                    # lands in its attempts file around now. One reader at a
                    # time; a boundary that finds it still busy waits for the
                    # next.
                    if (
                        fresh
                        and attempts_path is not None
                        and (outcome_thread is None or not outcome_thread.is_alive())
                    ):
                        outcome_thread = threading.Thread(
                            target=_pickup_history_outcomes,
                            args=(recorder, attempts_path, miner_id),
                            kwargs={
                                "on_done": (
                                    refresher.request_refresh
                                    if refresher is not None
                                    else None
                                )
                            },
                            name="dwave-history-outcomes",
                            daemon=True,
                        )
                        outcome_thread.start()
            elif which == "ping":
                with state_lock:
                    done = jobs_done
                    watermark = cancel_watermark
                out_q.put(_status(miner_id, done, abandoned=watermark))
            elif which == "get_capabilities":
                out_q.put(miner_pb2.MinerMsg(capabilities=capabilities_message()))
            elif which == "shutdown":
                grace_ms = cm.shutdown.grace_ms or 5000
                break

            # Soft idle timeout: if the coordinator stalls mid-session.
            idle_s = config.idle_timeout_s if config else 300
            if time.monotonic() - last_activity > idle_s:
                logger.info("idle timeout (%ss) — clean exit", idle_s)
                break

        # Drain in-flight submissions (they enqueue their Results) before the
        # end-of-outbound marker, bounded by the grace window.
        if job_pool is not None:
            job_pool.shutdown(wait=True)
        # Last sweep: a charge estimated after the final job's own drain would
        # otherwise die with the process and under-count the period.
        _bill_unobserved(sampler, pending_budget)
        # Signal end-of-outbound before spending the coordinator's grace
        # window on the history joins and close: every Result still queued
        # is an anneal already billed, and it is lost if the coordinator
        # closes the stream first.
        out_q.put(_STOP)
        _stop_history_threads()
        if recorder is not None:
            recorder.close()
        # Give the feeder thread a moment to flush (grace_ms).
        time.sleep(min(0.05, grace_ms / 1000.0))
        return exit_code
    except grpc.RpcError as exc:
        code = exc.code() if hasattr(exc, "code") else None
        if code == grpc.StatusCode.UNAUTHENTICATED:
            return EXIT_TOKEN_REJECTED
        logger.exception("rpc error: %s", exc)
        return EXIT_INTERNAL_FATAL
    except Exception:
        logger.exception("session failed")
        return EXIT_INTERNAL_FATAL
    finally:
        if job_pool is not None:
            job_pool.shutdown(wait=False, cancel_futures=True)
        try:
            out_q.put(_STOP)
        except Exception:  # noqa: BLE001
            pass
        _stop_history_threads()
        if recorder is not None:
            recorder.close()
        channel.close()
        sampler.close()


def run_session_sync(
    coordinator_uri: str,
    miner_id: str,
    sampler: OceanSampler,
    *,
    budget: Optional[BudgetPacer] = None,
    attempts_dir: Optional[str] = None,
) -> int:
    """Sync entry (session is already synchronous)."""
    return run_session(
        coordinator_uri, miner_id, sampler, budget=budget, attempts_dir=attempts_dir
    )
