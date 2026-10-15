"""Ocean SDK wrapper with v0.2 lessons and offline mock support.

Design points carried from ``QPU/dwave_sampler.py``:
- Thread-pooled async submits (GIL-bound encode/submit off the main path)
- The answer is read off the cloud future once it completes, never decoded
  into a SampleSet on the submit path
- Defect-qubit clamping before submit; reconstruction after decode
- Real ``device_access_time_us`` comes from the timing info already carried
  on the future

Offline mode (``QUIP_DWAVE_MOCK=1`` or an injected sampler) uses a dimod
sampler so unit/conformance tests never hit a real QPU.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from quip_solver_dwave.answer import AnswerView, answer_view
from quip_solver_dwave.qp import QpEncoder, build_submission_body
from quip_solver_dwave.defects import (
    DefectInfo,
    prepare_problem,
    reconstruct_samples,
)
from quip_solver_dwave.schedule import (
    FALLBACK_ANNEAL_US,
    forward_schedule,
    reverse_schedule,
)
from quip_solver_dwave.topology import native_topology_hash
from quip_solver_dwave.warm import WarmStart

logger = logging.getLogger(__name__)


# Ocean's canonical token variable, and the name the v0.2 stack shipped. The
# quip-node-manager compose files still deliver the Leap token as
# DWAVE_API_KEY, which Ocean never reads: DWaveSampler then raises "API token
# not defined" while a perfectly good token sits in the environment. Promote
# the old name at startup so an upgraded node keeps mining.
OCEAN_TOKEN_ENV = "DWAVE_API_TOKEN"
LEGACY_TOKEN_ENV = "DWAVE_API_KEY"


def adopt_legacy_token_env() -> bool:
    """Copy ``DWAVE_API_KEY`` to ``DWAVE_API_TOKEN`` when only the old name is set.

    Returns True when the promotion happened. ``DWAVE_API_TOKEN`` always wins:
    a value under the canonical name is never overwritten.
    """
    legacy = os.environ.get(LEGACY_TOKEN_ENV, "").strip()
    if not legacy or os.environ.get(OCEAN_TOKEN_ENV, "").strip():
        return False
    os.environ[OCEAN_TOKEN_ENV] = legacy
    logger.warning(
        "%s is deprecated and will be dropped; using its value as %s. "
        "Set %s instead.",
        LEGACY_TOKEN_ENV,
        OCEAN_TOKEN_ENV,
        OCEAN_TOKEN_ENV,
    )
    return True


def credentials_present() -> bool:
    """True if a D-Wave API token is available via env or SDK config file."""
    if os.environ.get(OCEAN_TOKEN_ENV):
        return True
    conf = os.path.expanduser("~/.config/dwave/dwave.conf")
    if os.path.isfile(conf):
        try:
            with open(conf, encoding="utf-8") as f:
                text = f.read()
            if "token" in text.lower() and "=" in text:
                return True
        except OSError:
            pass
    return False


def mock_mode_enabled() -> bool:
    return os.environ.get("QUIP_DWAVE_MOCK", "").strip() in ("1", "true", "yes")


def mock_backend() -> str:
    """Offline mock sampler backend: ``exact`` (default, brute-force — only
    tiny problems) or ``sa`` (SimulatedAnnealingSampler — scales to realistic
    topologies). Selected via ``QUIP_DWAVE_MOCK_BACKEND``."""
    b = os.environ.get("QUIP_DWAVE_MOCK_BACKEND", "").strip().lower()
    return "sa" if b == "sa" else "exact"


def is_solver_offline(exc: BaseException) -> bool:
    """True for Ocean's ``SolverOfflineError``: SAPI refused to run the problem.

    Matched by class name so the mock path keeps working without Ocean
    installed; this module never imports the SDK at load time.
    """
    return any(cls.__name__ == "SolverOfflineError" for cls in type(exc).__mro__)


def is_solver_unavailable(exc: BaseException) -> bool:
    """True when the solver cannot be connected to but may come back.

    ``SolverOfflineError`` is D-Wave saying the solver is down. The SDK also
    raises ``SolverNotFoundError`` for a solver that is listed but offline,
    because its default filter drops offline solvers, so both mean "wait".
    Matched by class name for the same reason as :func:`is_solver_offline`.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"SolverOfflineError", "SolverNotFoundError"})


def ocean_importable() -> bool:
    try:
        import dimod  # noqa: F401
        import dwave  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class SampleResult:
    """One decoded sample batch from the QPU/mock.

    Spins stay in the array the sampler produced. Materialising a dict per
    read, keyed by qubit label, cost ~220k Python-level operations per job and
    bought nothing: the only consumer wants a dense vector in session node
    order, which is one vectorised reorder away from this form.

    ``spins`` is ``(reads, len(variables))`` of int8, normalised to +1/-1, and
    ``variables[i]`` is the qubit label of column ``i``.
    """

    spins: "np.ndarray"
    variables: List[int]
    energies: List[float]
    device_access_time_us: int
    num_reads: int
    defect_info: Optional[DefectInfo] = None
    extra: Dict[str, str] = field(default_factory=dict)
    # SAPI's own clock: when the problem was accepted and when it was solved.
    # Their difference is D-Wave's service time, queues included. None when the
    # answer did not come from the cloud (mock, injected sampler).
    submitted_on_s: Optional[float] = None
    solved_on_s: Optional[float] = None
    # Problems this sampler already had on the QPU when this one was handed
    # over. With the round trip, Little's law turns it into throughput.
    inflight_at_submit: int = 0


class SupportsSample(Protocol):
    """The one call :func:`quip_solver_dwave.job.handle_job` makes on a sampler.

    Naming the surface instead of the concrete class is what lets the tests
    drive job.py with a recording or exploding double, which is the only way to
    cover the failure paths without a live QPU.
    """

    def sample(
        self,
        nodes: "np.ndarray",
        h: "np.ndarray",
        edges: "np.ndarray",
        j: "np.ndarray",
        *,
        num_reads: int = 1,
        anneal_time_us: Optional[int] = None,
        nonce_seed: Optional[bytes] = None,
        label: str = "quip-dwave-qa",
        cancel_key: Optional[bytes] = None,
        warm_start: Optional[WarmStart] = None,
    ) -> SampleResult: ...


class SupportsClose(Protocol):
    """The surface the SIGTERM handler needs: release the cloud client."""

    def close(self) -> None: ...


def descend_from(
    h: Dict[int, float], j: Dict[Tuple[int, int], float], state: Dict[int, int]
) -> Tuple[Dict[int, int], float]:
    """Zero-temperature single-spin descent from ``state``; the mock's reverse anneal.

    A reverse anneal searches near the state it starts from and never returns
    to a random one. Descent is the deterministic version of that: it flips a
    spin only when the flip lowers the energy, so a ground state comes back
    unchanged. That is what the conformance driver's seeded 4096-spin ring
    grades, and no exact enumeration could answer a problem that size.

    Returns the final state and its energy ``sum(h s) + sum(J s s)``.
    """
    nbrs: Dict[int, List[Tuple[int, float]]] = {v: [] for v in h}
    for (u, v), coupling in j.items():
        nbrs.setdefault(u, []).append((v, coupling))
        nbrs.setdefault(v, []).append((u, coupling))
    spins = {v: (1 if state.get(v, 1) >= 0 else -1) for v in nbrs}
    moved = True
    while moved:
        moved = False
        for v, around in nbrs.items():
            field = h.get(v, 0.0) + sum(c * spins[u] for u, c in around)
            if spins[v] * field > 0:
                spins[v] = -spins[v]
                moved = True
    energy = sum(bias * spins[v] for v, bias in h.items()) + sum(
        c * spins[u] * spins[v] for (u, v), c in j.items()
    )
    return spins, float(energy)


class MockSampler:
    """Offline sampler backed by dimod ExactSolver / SimulatedAnnealingSampler.

    Returns a sampleset-like object with a synthetic timing dict so
    ``device_access_time_us`` is non-zero in tests. A job that carries
    ``initial_state`` is answered by :func:`descend_from` on either backend.
    """

    def __init__(self, backend: str = "exact"):
        import dimod

        self._backend = backend
        if backend == "sa":
            self._sampler = dimod.SimulatedAnnealingSampler()
        else:
            self._sampler = dimod.ExactSolver()
        self.nodelist: List[int] = []
        self.edgelist: List[Tuple[int, int]] = []
        self.properties: Dict[str, Any] = {
            "chip_id": "mock-qpu",
            "num_qubits": 0,
        }

    def sample_ising(self, h, j, **kwargs):
        import dimod

        num_reads = int(kwargs.get("num_reads") or 1)
        t0 = time.monotonic()
        initial = kwargs.get("initial_state")
        if initial:
            # Every read of a reverse anneal starts from the one state, and
            # descent is deterministic, so the reads are one aggregated row.
            spins, energy = descend_from(h, j, initial)
            ss = dimod.SampleSet.from_samples(
                [spins], dimod.SPIN, energy=[energy], num_occurrences=[num_reads]
            )
        elif self._backend == "exact":
            ss = self._sampler.sample_ising(h, j)
            # ExactSolver returns every state; keep the lowest-energy rows.
            if len(ss) > num_reads:
                ss = ss.truncate(num_reads)
        else:
            ss = self._sampler.sample_ising(h, j, num_reads=num_reads)
        elapsed_us = max(1, int((time.monotonic() - t0) * 1_000_000))
        # Attach synthetic QPU timing so SamplerMeta.device_access_time_us is real-ish.
        # SampleSet.info is read-only; rebuild with the same samples + new info.
        info = dict(getattr(ss, "info", None) or {})
        info["timing"] = {
            "qpu_programming_time": 100,
            "qpu_sampling_time": elapsed_us,
        }
        return dimod.SampleSet(ss.record, ss.variables, info, ss.vartype)


def _parse_sapi_time(raw: Any) -> Optional[float]:
    """SAPI's ISO-8601 with a trailing Z, as a unix timestamp."""
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def server_timestamps(raw: Any) -> tuple[Optional[float], Optional[float]]:
    """``(submitted_on, solved_on)`` off a cloud Future, or ``(None, None)``.

    The SDK parks the SAPI problem JSON on ``Future._message`` and never
    parses these two fields (``time_received`` and ``time_solved`` exist on
    the class but nothing assigns them). Reading the private attribute is
    the only way to get D-Wave's service time per job; it is an SDK internal
    in the same sense as ``_submit_encoded``'s.
    """
    message = getattr(raw, "_message", None)
    if not isinstance(message, dict):
        return None, None
    return _parse_sapi_time(message.get("submitted_on")), _parse_sapi_time(
        message.get("solved_on")
    )


class OceanSampler:
    """Thin wrapper around a D-Wave (or mock) sampler.

    Callers always pass logical Ising problems. Defect clamping and sample
    reconstruction are handled here; energy scoring for consensus stays in
    the session layer via ``quip_solver_core.scoring``.
    """

    def __init__(
        self,
        *,
        sampler: Any = None,
        solver_name: Optional[str] = None,
        region: Optional[str] = None,
        token: Optional[str] = None,
        mock: Optional[bool] = None,
        submit_workers: int = 4,
        defective_qubits: Optional[Sequence[int]] = None,
        defective_edges: Optional[set] = None,
    ):
        self._submit_pool = ThreadPoolExecutor(
            max_workers=max(1, submit_workers),
            thread_name_prefix="dwave-submit",
        )
        # Live cloud problems, keyed by the caller's cancel key, so a
        # coordinator Cancel can reach a problem still sitting on the QPU.
        # Crossed by the submit pool and the session thread -> guard it.
        self._inflight: Dict[bytes, Any] = {}
        # Keys cancelled before their submit landed. The cloud Future only
        # exists once _submit_sync has run and SAPI has accepted the problem;
        # a Cancel inside that window has nothing to call yet, and that window
        # is exactly when cancelling saves the most access time.
        self._cancel_pending: set = set()
        self._inflight_lock = threading.Lock()
        # Access time D-Wave charged for problems whose timing we never saw.
        # See _charge_unobserved for why this exists and why it over-counts.
        self._unobserved_access_us = 0
        self._max_access_us = 0
        # Built on first submit, when the solver's ordering is known.
        self._encoder: Optional[QpEncoder] = None
        self._plan_cache = None
        self._defective_qubits: List[int] = list(defective_qubits or [])
        self._defective_edges: set = set(defective_edges or set())
        self._live_nodes: List[int] = []
        self._live_edges: List[Tuple[int, int]] = []
        self._native_hash: Optional[bytes] = None
        self._is_mock = False
        self._qpu_solver = None
        self._connected = False
        # Connection overrides; unset values fall through to D-Wave's native
        # config resolution (dwave.conf + standard env) in `_connect_real`.
        self._solver_name = solver_name
        self._region = region
        self._token = token

        use_mock = mock if mock is not None else mock_mode_enabled()
        if sampler is not None:
            self.sampler = sampler
            self._is_mock = use_mock or isinstance(sampler, MockSampler)
            self._connected = True
            self._apply_native_hash()
        elif use_mock:
            self.sampler = MockSampler(backend=mock_backend())
            self._is_mock = True
            self._connected = True
            logger.info(
                "[QPU] mock sampler active (QUIP_DWAVE_MOCK, backend=%s)",
                mock_backend(),
            )
            self._apply_native_hash()
        else:
            # Real QPU: do NOT contact D-Wave here. The connection is deferred
            # until the coordinator engages us (Configure calls ensure_connected)
            # or another mode forces it (--check), so an idle or unconnected
            # miner never opens a QPU session.
            self.sampler = None
            logger.info("[QPU] real sampler deferred until Configure")

    def ensure_connected(self) -> None:
        """Connect to the real QPU if not already (idempotent).

        Mock and injected samplers are ready at construction; a real sampler
        connects here — invoked when the coordinator sends Configure, or eagerly
        by --check. Safe to call repeatedly.
        """
        if self._connected:
            return
        self.sampler = self._connect_real(self._solver_name, self._region, self._token)
        self._detect_defects()
        self._connected = True
        self._apply_native_hash()

    def _apply_native_hash(self) -> None:
        if self._live_nodes and self._live_edges is not None:
            self._native_hash = native_topology_hash(self._live_nodes, self._live_edges)

    def _connect_real(self, solver_name, region, token):
        from dwave.system import DWaveSampler

        # Comply with D-Wave's own config: DWaveSampler resolves credentials
        # from ~/.config/dwave/dwave.conf and the canonical DWAVE_API_TOKEN /
        # DWAVE_API_SOLVER / DWAVE_API_REGION / DWAVE_API_ENDPOINT env vars. Pass
        # only explicit overrides a caller supplied; everything else is the SDK's.
        kwargs: Dict[str, Any] = {"request_timeout": (60, 300)}
        if solver_name is not None:
            kwargs["solver"] = solver_name
        if region is not None:
            kwargs["region"] = region
        if token is not None:
            kwargs["token"] = token
        base = DWaveSampler(**kwargs)
        self._qpu_solver = base
        self._live_nodes = sorted(int(n) for n in base.nodelist)
        self._live_edges = [(int(u), int(v)) for u, v in base.edgelist]
        logger.info(
            "[QPU] connected solver=%s qubits=%d",
            base.properties.get("chip_id", "?"),
            len(self._live_nodes),
        )
        return base

    def _detect_defects(self) -> None:
        # Real QPU: live graph is authoritative; no stored-topology diff here.
        # Defects are only injected when the session Topology lists qubits that
        # the live solver does not expose (handled per-job against Topology).
        self._defective_qubits = []
        self._defective_edges = set()

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    @property
    def native_topology_hash(self) -> Optional[bytes]:
        return self._native_hash

    @property
    def live_nodes(self) -> List[int]:
        return list(self._live_nodes)

    @property
    def live_edges(self) -> List[Tuple[int, int]]:
        return list(self._live_edges)

    def set_session_topology(
        self,
        nodes: Sequence[int],
        edges: Sequence[Tuple[int, int]],
    ) -> None:
        """Bind the session Topology; compute defects vs live QPU if known."""
        nodes_l = [int(n) for n in nodes]
        edges_l = [(int(u), int(v)) for u, v in edges]
        if self._live_nodes:
            live_set = set(self._live_nodes)
            self._defective_qubits = sorted(set(nodes_l) - live_set)
            live_edge_set = {(min(u, v), max(u, v)) for u, v in self._live_edges}
            def_q = set(self._defective_qubits)
            self._defective_edges = set()
            for u, v in edges_l:
                if u in def_q or v in def_q:
                    continue
                key = (min(u, v), max(u, v))
                if key not in live_edge_set:
                    self._defective_edges.add((u, v))
            self._log_topology_fit(len(nodes_l), len(edges_l))
        else:
            # Mock / no live hardware: treat session topology as native.
            self._live_nodes = nodes_l
            self._live_edges = edges_l
            self._defective_qubits = []
            self._defective_edges = set()
            self._native_hash = native_topology_hash(nodes_l, edges_l)

    def _log_topology_fit(self, session_nodes: int, session_edges: int) -> None:
        """Report how much of the session topology this solver can anneal.

        Qubits the live chip lacks are clamped to pseudorandom spins and the
        couplers it lacks are scored after the fact (see ``defects``), so a
        solver whose working graph is not the network's returns energies that
        are valid, accepted, and a fraction of what the graph allows. Without
        this line that shows up only as results that never reach target.
        """
        missing_nodes = len(self._defective_qubits)
        missing_edges = len(self._defective_edges)
        if not missing_nodes and not missing_edges:
            logger.info(
                "[QPU] session topology matches the live graph "
                "(%d nodes, %d couplers)",
                session_nodes,
                session_edges,
            )
            return
        node_pct = 100.0 * missing_nodes / session_nodes if session_nodes else 0.0
        edge_pct = 100.0 * missing_edges / session_edges if session_edges else 0.0
        # A production chip is missing a handful of qubits. Missing a tenth of
        # the graph is a different chip, not hardware defects.
        log = logger.error if max(node_pct, edge_pct) > 10.0 else logger.warning
        log(
            "[QPU] live solver is missing %d/%d nodes (%.1f%%) and %d/%d "
            "couplers (%.1f%%) of the session topology; the missing part is "
            "clamped, not annealed, so energies land short of target. Check "
            "that DWAVE_API_SOLVER names the chip the network's topology "
            "came from.",
            missing_nodes,
            session_nodes,
            node_pct,
            missing_edges,
            session_edges,
            edge_pct,
        )

    def _graph_plan(self, solver, nodes, edges):
        """The job graph's mapping into the solver's payload ordering.

        Cached on the graph's identity because a session sends the same graph
        every job: planning it costs ~10 ms, encoding against it ~0.6 ms.
        """
        key = (int(nodes.shape[0]), int(edges.shape[0]))
        with self._inflight_lock:
            cached = self._plan_cache
        encoder = self._encoder
        if encoder is None:
            encoder = QpEncoder(
                solver._encoding_qubits, solver._encoding_couplers
            )
            self._encoder = encoder
        if cached is not None and cached[0] == key:
            return encoder, cached[1]
        plan = encoder.plan(nodes, edges)
        with self._inflight_lock:
            self._plan_cache = (key, plan)
        return encoder, plan

    def _clamp_defects(self, nodes, h, edges, j, nonce_seed, warm_start=None):
        """Apply defect clamping, which still speaks dicts.

        The live graph matches the chip in the normal case, so this is a
        no-op and the arrays pass straight through. When it is not a no-op the
        conversion cost is paid on a path that only runs for a miner whose
        chip has lost qubits or couplers. The clamped qubits of a warm start
        take their spins from its state.
        """
        if not (self._defective_qubits or self._defective_edges):
            return nodes, h, edges, j, None
        h_dict = {int(n): float(b) for n, b in zip(nodes, h)}
        j_dict = {
            (int(u), int(v)): float(b) for (u, v), b in zip(edges.tolist(), j)
        }
        start_state = (
            None
            if warm_start is None
            else {int(n): int(s) for n, s in zip(nodes, warm_start.state)}
        )
        h_eff, j_eff, defect_info = prepare_problem(
            h_dict,
            j_dict,
            defective_qubits=self._defective_qubits,
            defective_edges=self._defective_edges,
            # Either kind of defect needs the reduction. Withholding the seed
            # when only couplers are missing skipped it entirely and sent the
            # QPU a graph it does not have.
            nonce_seed=nonce_seed,
            start_state=start_state,
        )
        nodes_eff = np.fromiter(h_eff.keys(), dtype=np.int64, count=len(h_eff))
        h_arr = np.fromiter(h_eff.values(), dtype=np.float64, count=len(h_eff))
        edge_keys = list(j_eff.keys())
        edges_eff = np.asarray(edge_keys, dtype=np.int64).reshape(-1, 2)
        j_arr = np.fromiter(j_eff.values(), dtype=np.float64, count=len(j_eff))
        return nodes_eff, h_arr, edges_eff, j_arr, defect_info

    @staticmethod
    def _restrict_warm_start(
        warm_start: WarmStart, full_nodes, kept_nodes
    ) -> WarmStart:
        """Cut a start state down to the qubits that survived clamping.

        SAPI wants a spin for every qubit in the submitted problem and for no
        other, and clamping both drops qubits and reorders the rest.
        """
        spin_of = {int(n): int(s) for n, s in zip(full_nodes, warm_start.state)}
        return replace(
            warm_start,
            state=np.fromiter(
                (spin_of[int(n)] for n in kept_nodes),
                dtype=np.int8,
                count=len(kept_nodes),
            ),
        )

    @staticmethod
    def _initial_state(props: Dict[str, Any], nodes, warm_start: WarmStart) -> Any:
        """The start state in the form the sampler behind ``props`` takes."""
        num_qubits = props.get("num_qubits")
        if not num_qubits:
            return {int(n): int(s) for n, s in zip(nodes, warm_start.state)}
        state = np.full(int(num_qubits), 3, dtype=np.int8)
        state[np.asarray(nodes, dtype=np.int64)] = warm_start.state
        return state.tolist()

    @staticmethod
    def _anneal_params(
        solver: Any,
        nodes,
        anneal_time_us: Optional[int],
        warm_start: Optional[WarmStart],
    ) -> Dict[str, Any]:
        """The anneal half of a submission: always a schedule, never a time.

        SAPI refuses ``annealing_time`` beside ``anneal_schedule``, and a
        reverse anneal has no ``annealing_time`` form, so one spelling covers
        every job. With no override the solver's own published default is
        written out as a schedule. A sampler that publishes none (the mock)
        gets no schedule at all and keeps its own default.

        SAPI takes ``initial_state`` as one entry per physical qubit, indexed
        by label, with 3 for a qubit the problem does not use. That list is
        built here for the cloud path. Ocean would expand a label-to-spin
        mapping itself, but the body is serialised by orjson, which refuses
        integer keys, so the mapping must never reach it. A dimod sampler has
        no physical qubits and takes the mapping.
        """
        props = getattr(solver, "properties", None) or {}
        time_range = props.get("annealing_time_range")
        anneal_us = anneal_time_us or props.get("default_annealing_time")
        if warm_start is None:
            if not anneal_us:
                return {}
            return {
                "anneal_schedule": forward_schedule(anneal_us, time_range=time_range)
            }
        return {
            "anneal_schedule": reverse_schedule(
                anneal_us or FALLBACK_ANNEAL_US,
                warm_start.reversal_s,
                warm_start.reversal_pause_us,
                time_range=time_range,
            ),
            "initial_state": OceanSampler._initial_state(props, nodes, warm_start),
            # D-Wave's default, stated because the miner depends on it: every
            # read restarts from the seed instead of from the read before it.
            "reinitialize_state": True,
        }

    def _register_inflight(self, key: bytes, future: Any) -> None:
        """Record a live cloud problem, or drop it if a Cancel beat it here."""
        with self._inflight_lock:
            if key in self._cancel_pending:
                # The note is consumed by the registration it applies to; a
                # key left poisoned would kill the next job that reuses it.
                self._cancel_pending.discard(key)
                doomed = future
            else:
                self._inflight[key] = future
                doomed = None
        if doomed is not None:
            doomed.cancel()

    def _release_inflight(self, key: bytes) -> None:
        """Forget a problem that has finished, cancelled or not."""
        with self._inflight_lock:
            self._inflight.pop(key, None)
            self._cancel_pending.discard(key)

    def cancel_inflight(self, keys: Sequence[bytes]) -> int:
        """Ask SAPI to drop these problems; return how many were still live.

        Best-effort by construction: D-Wave only refunds a problem it has not
        started annealing, so the return value counts the ones that had not
        finished when we asked, which is the ceiling on what was saved — not a
        confirmed refund. Keys whose submit has not landed are noted so
        :meth:`_register_inflight` cancels them the moment it does.
        """
        with self._inflight_lock:
            found = []
            for key in keys:
                future = self._inflight.pop(key, None)
                if future is None:
                    self._cancel_pending.add(key)
                else:
                    found.append(future)
        live = 0
        for future in found:
            if future.done():
                continue
            future.cancel()
            live += 1
        return live

    def _observe_access_us(self, access_us: int) -> None:
        """Note a measured charge, so an unmeasurable one can be estimated."""
        with self._inflight_lock:
            self._max_access_us = max(self._max_access_us, int(access_us))

    def _charge_unobserved(self) -> None:
        """Bill a problem SAPI accepted but whose timing we never saw.

        A cancel that beat the anneal costs nothing; a cancel that lost is
        charged by D-Wave in full. The raised path carries no timing to tell
        those apart, so the ledger assumes the anneal ran. That over-counts
        every successful cancel, which is the safe direction to be wrong:
        under-counting lets the pacer hand out headroom D-Wave has already
        spent, and the drift compounds across the period.

        The estimate is the largest charge measured this session, which is
        exact on a fleet whose jobs all carry the same num_reads. It is 0
        until the first job completes, so a cancel in the opening seconds of a
        session is under-billed; that window is bounded and does not recur.
        """
        with self._inflight_lock:
            self._unobserved_access_us += self._max_access_us

    def drain_unobserved_access_us(self) -> int:
        """Take the estimated charges accrued since the last call."""
        with self._inflight_lock:
            owed, self._unobserved_access_us = self._unobserved_access_us, 0
        return owed

    def close(self) -> None:
        self._submit_pool.shutdown(wait=False)
        if self._qpu_solver is not None:
            try:
                self._qpu_solver.client.close(wait=False)
            except Exception:  # noqa: BLE001 — best-effort
                pass

    def _submit_sync(
        self,
        nodes,
        h,
        edges,
        j,
        num_reads: int,
        label: str,
        anneal_time_us: Optional[int] = None,
        cancel_key: Optional[bytes] = None,
        warm_start: Optional[WarmStart] = None,
    ):
        """Run on a pool thread: build/submit only; do NOT touch .sampleset."""
        solver = getattr(self.sampler, "solver", None)
        params: Dict[str, Any] = {"num_reads": num_reads}
        # The proto's `anneal_time_us` and SAPI's schedule times are both
        # microseconds, so no conversion is needed.
        params.update(self._anneal_params(solver, nodes, anneal_time_us, warm_start))

        if self._is_mock or solver is None:
            # dimod samplers and injected doubles take dicts, and the problems
            # they see are tiny. Only the cloud path is worth encoding by hand.
            sample_fn = getattr(self.sampler, "sample_ising", None)
            if not callable(sample_fn):
                raise RuntimeError("sampler has no sample_ising")
            return sample_fn(
                {int(n): float(b) for n, b in zip(nodes, h)},
                {
                    (int(u), int(v)): float(b)
                    for (u, v), b in zip(edges.tolist(), j)
                },
                label=label,
                **params,
            )

        # Encode straight from the arrays into the solver's own ordering. The
        # Ocean path would build an h/J dict here and hand it to
        # encode_problem_as_qp, which walks it back into these same two dense
        # arrays: ~27 ms of GIL-bound work per job at production size, against
        # ~0.6 ms here. quip_solver_dwave.qp pins byte equality with that
        # function, so what reaches SAPI is unchanged.
        encoder, plan = self._graph_plan(solver, nodes, edges)
        data = encoder.encode(plan, h, j)
        body = build_submission_body(solver, data, params, label=label)

        computation = self._submit_encoded(solver, body, cancel_key)
        return computation

    def _submit_encoded(self, solver, body: bytes, cancel_key: Optional[bytes]):
        """Hand an encoded problem to the cloud client.

        Every Ocean internal this backend depends on lives in this method, so
        the blast radius of an SDK change is one function: ``Future`` and
        ``Present`` to build the computation, and ``client._submit`` to queue
        it. Everything upstream is our own arrays and our own encoder.
        """
        # Imported here so an offline/mock run never needs the cloud client.
        from dwave.cloud.computation import Future
        from dwave.cloud.concurrency import Present

        computation = Future(
            solver=solver,
            id_=None,
            # numpy, not lists. With return_matrix=False the decoder calls
            # .tolist() on a (reads x qubits) array, which is 220k Python
            # objects a job at production size. Safe only because nothing on
            # this path builds a SampleSet: the same flag makes
            # wait_sampleset's comprehension 4.4x slower.
            return_matrix=True,
        )
        # XXX carried on the Future until SAPI implements it, as Ocean does.
        computation._offset = 0
        # Registered before the submit, not after: _submit hands the problem to
        # the client's own threads, so the id can come back before this line
        # would otherwise run.
        if cancel_key is not None:
            self._register_inflight(cancel_key, computation)
        solver.client._submit(Present(result=body), computation)
        return computation

    @staticmethod
    def _decode_and_view(future_or_ss: Any) -> AnswerView:
        """Read the answer OFF the submit path (v0.2 lesson: never on it).

        Deliberately not ``.sampleset``. That property turns the decoded numpy
        arrays into Python lists, walks them with a nested comprehension over
        reads times variables, and hands them to dimod to convert back into
        numpy: about 38.8 ms per job at production size, to arrive at the arrays
        the decoder already had.
        """
        return answer_view(future_or_ss)

    def sample(
        self,
        nodes: "np.ndarray",
        h: "np.ndarray",
        edges: "np.ndarray",
        j: "np.ndarray",
        *,
        num_reads: int = 1,
        anneal_time_us: Optional[int] = None,
        nonce_seed: Optional[bytes] = None,
        label: str = "quip-dwave-qa",
        cancel_key: Optional[bytes] = None,
        warm_start: Optional[WarmStart] = None,
    ) -> SampleResult:
        """Submit one Ising problem and return decoded, reconstructed samples.

        Submit work runs on the thread pool. The answer is read off the future
        here, after it completes (v0.2 lesson: never decode on the submit
        path).
        ``anneal_time_us`` (microseconds) is the time a full ramp of the
        anneal takes; ``None``/``0`` means the solver's published default. It
        reaches SAPI as an ``anneal_schedule`` (see ``_anneal_params``).
        ``warm_start`` makes the job a reverse anneal from its state.

        ``cancel_key`` makes the submission reachable by
        :meth:`cancel_inflight` until it finishes. A cancelled problem raises
        out of here rather than returning samples, which is what the caller
        wants: there is no Result to send for a generation the coordinator has
        already abandoned.
        """
        full_nodes = nodes
        nodes, h, edges, j, defect_info = self._clamp_defects(
            nodes, h, edges, j, nonce_seed, warm_start
        )
        if warm_start is not None and defect_info is not None:
            warm_start = self._restrict_warm_start(warm_start, full_nodes, nodes)
        # Counted before the hand-off rather than inside _submit_encoded, so
        # the number needs no SDK object to carry it back.
        with self._inflight_lock:
            inflight_before = len(self._inflight)
        # Thread-pooled submit
        fut = self._submit_pool.submit(
            self._submit_sync,
            nodes,
            h,
            edges,
            j,
            max(1, int(num_reads)),
            label,
            anneal_time_us,
            cancel_key,
            warm_start,
        )
        # Two failure boundaries, billed differently. A failure here is the
        # submit itself dying in this process: D-Wave never saw the problem,
        # so charging for it would burn quota the QPU never spent.
        try:
            raw = fut.result()
        except BaseException:
            if cancel_key is not None:
                self._release_inflight(cancel_key)
            raise
        # Decode off the submit path. A failure here is a problem SAPI already
        # accepted — a cancelled one, most often — and it may have annealed.
        # The one exception is a solver that is offline: SAPI takes the
        # problem and then fails it without running it, and D-Wave charges
        # nothing. Booking the estimate there spent 342 s of budget on 7,623
        # rejects in a single qblock during the Advantage2_system1 outage.
        try:
            view = self._decode_and_view(raw)
        except BaseException as exc:
            if not is_solver_offline(exc):
                self._charge_unobserved()
            raise
        finally:
            if cancel_key is not None:
                self._release_inflight(cancel_key)
        access_us = view.access_time_us
        self._observe_access_us(access_us)
        submitted_on_s, solved_on_s = server_timestamps(raw)

        spins, variables, energies = reconstruct_samples(
            view.spins, view.variables, view.energies, defect_info
        )
        reads_done = view.reads

        # SamplerMeta.extra is the one channel that crosses handle_job into
        # the session loop, which is where the history is written. The
        # coordinator ignores keys it does not know.
        extra = {"mock": "1" if self._is_mock else "0", "inflight": str(inflight_before)}
        if submitted_on_s is not None and solved_on_s is not None:
            extra["sapi_ms"] = str(int(round((solved_on_s - submitted_on_s) * 1000)))

        return SampleResult(
            spins=spins,
            variables=variables,
            energies=energies,
            device_access_time_us=access_us,
            num_reads=reads_done,
            defect_info=defect_info,
            extra=extra,
            submitted_on_s=submitted_on_s,
            solved_on_s=solved_on_s,
            inflight_at_submit=inflight_before,
        )
