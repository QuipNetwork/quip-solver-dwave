#!/usr/bin/env python3
"""The QPU sidecar of the cascade miner.

The miner writes a ``survivor`` line for each model that passes its screen, and waits for ``deep`` or ``drop`` on
standard output. The first byte of the nonce picks the arm. ``qpu-cold`` and ``qpu-seeded`` sample the model on the
QPU and apply the QPU gate. ``gpu-256`` runs the GPU-only program at once, and so does a survivor that the pace, the
cap, or the queue keeps off the QPU. Standard output holds only the protocol, and standard error holds the rest.
``<out-dir>/events.jsonl`` journals the run for ``cascade_wins.py``. Energies on the wire are whole engine units.
The QPU reads, the gate table, and ``best_milli`` are milli energy. The spend journal counts microseconds of access.

One sidecar runs on an out-dir at a time, and a lock on ``<out-dir>/sidecar.lock`` makes sure of it. The cap, the
pace, and the time ``t`` of the events count from the first launch on the out-dir, so a restart does not reset them.
Give each card its own out-dir and its own share of the budget.

A line that the sidecar cannot take ends the run, because the miner and the sidecar then disagree. A failure of the
QPU side does not. The survivor runs on ``gpu-256``, a ``qpu_error`` event says why, and the QPU stays off until a
retry of the connection works. A call whose connection broke under it first goes again once, with a ``qpu_retry``
event. The cap and a mismatch of the lane order switch the QPU off for good.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Set, TextIO, Tuple

import numpy as np

from quip_miner_dwave import cascade_codec, cascade_flow, regime_io, replay


def _load_chain_qpu_capture():
    """``chain_qpu_capture.py`` as a module: ``scripts/`` is not a package. Its ``capture`` loop runs a fixed list of
    jobs, and survivors arrive over time, so only its estimate, margin, and energy recomputation are reused."""
    path = Path(__file__).resolve().parent / "chain_qpu_capture.py"
    spec = importlib.util.spec_from_file_location("chain_qpu_capture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chain_capture = _load_chain_qpu_capture()

ARMS = ("gpu-256", "qpu-cold", "qpu-seeded")
GPU_SALT = 3  # the kernel seed salt of the GPU arm. The seeded arm uses 10, and the cold arm 1000
MAX_COPIES = 8  # the most copies of a deep answer that the engine runs
MOCK_ACCESS_US = 50_000
ANSWERED_LIMIT = 4096  # deep legs remembered for the miner's later ``deep`` line. The oldest go first
EVENTS_FILE = "events.jsonl"
LOCK_FILE = "sidecar.lock"
LOCK_TIMEOUT_S = 60.0  # how long a launch waits for a predecessor that still drains its calls in flight
LOCK_POLL_S = 0.05
DRAIN_TIMEOUT_S = 30.0  # how long the end of input waits for the calls in flight, so a hung call cannot hold the lock
RETRY_MIN_S = 30.0  # the wait before the first retry of the connection after an error. A call that works resets it
RETRY_MAX_S = 600.0  # the wait doubles at each failed retry and at each recovery, up to this


def parse_arms(text: str) -> List[Tuple[str, int]]:
    """``name:weight`` pairs in order. A survivor takes an arm in proportion to its weight."""
    arms: List[Tuple[str, int]] = []
    for part in text.split(","):
        name, _, weight = part.strip().partition(":")
        if name not in ARMS or name in dict(arms) or not weight.isdigit() or int(weight) < 1:
            raise argparse.ArgumentTypeError(f"{part!r} is not name:weight with a name of {ARMS}, a positive whole "
                                             "weight, and no repeat")
        arms.append((name, int(weight)))
    return arms


def iso_utc(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def parse_time(text: str) -> datetime:
    """An ISO 8601 time. A time with no UTC offset is UTC."""
    when = datetime.fromisoformat(text)
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def load_gate(path: str, miss: float) -> int:
    """The cut, in milli energy, of the row of the QPU gate table that allows the miss ``miss``."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if table.get("unit") != "milli":
        raise ValueError(f"{path} is in the unit {table.get('unit')!r}, and the sidecar compares milli energy")
    for row in table["table"]:
        if math.isclose(row["miss_allowed"], miss, abs_tol=1e-12):
            return int(row["cut"])
    raise ValueError(f"{path} has no row that allows a miss of {miss}; its rows allow "
                     f"{[row['miss_allowed'] for row in table['table']]}")


def read_first_start(path: Path) -> Optional[Dict[str, Any]]:
    """The first ``start`` event of an events file, or ``None`` before any launch.

    A line that does not parse is a write that a crash cut short. Skipping it can only move the origin later, which
    makes the pace stricter, so it never lets the run spend more.
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event["type"] == "start":
                return event
    return None


def check_programs(path: str, names: Sequence[str]) -> None:
    held = {program["name"] for program in json.loads(Path(path).read_text(encoding="utf-8"))["programs"]}
    if missing := [name for name in names if name not in held]:
        raise ValueError(f"{path} holds no program named {missing}")


def choose_arm(nonce: str, arms: Sequence[Tuple[str, int]]) -> str:
    """The first byte of the nonce, modulo the weight sum, over the arms in order. Each arm spans its weight."""
    slot = int(nonce[:2], 16) % sum(weight for _, weight in arms)
    for name, weight in arms:
        if slot < weight:
            return name
        slot -= weight
    raise AssertionError("the slot is below the weight sum")


def lane_bests(energies: Sequence[int], qpu_lanes: int) -> Dict[str, int]:
    """The best energy of the lanes that the QPU seeded, and of the lanes that continued from the screen."""
    lanes = {"qpu_lane_best": energies[:qpu_lanes], "fill_lane_best": energies[qpu_lanes:]}
    return {name: min(values) for name, values in lanes.items() if len(values)}


@dataclass(frozen=True)
class Config:
    """The resolved settings of a run. The ``start`` event records them."""

    out_dir: str
    arms: List[Tuple[str, int]]
    gpu_program: str
    cold_program: str
    seeded_program: str
    qpu_miss: float
    qpu_gate_milli: int
    qpu_seconds: float
    spend_until_s: float  # on the time of the clock, which is the epoch seconds of the wall clock
    in_flight: int
    queue: int
    reads: int
    anneal_us: int
    j_scale: float
    mock: bool
    copies: int = 1  # independent deep units of each deep answer, which share one push


class Journal:
    """``events.jsonl``: one JSON object per line, each with its ``type`` and ``t``, the seconds since the origin.

    The origin is the ``started_at`` of the first ``start`` event in the file, so ``t`` and the pace keep counting
    across launches. A file with no ``start`` event has its origin at this launch.
    """

    def __init__(self, path: Path, clock: Callable[[], float]) -> None:
        self.path, self.clock = path, clock
        self.first_start = read_first_start(path)
        self.origin = clock() if self.first_start is None else datetime.fromisoformat(self.first_start["started_at"]).timestamp()
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8")

    def __enter__(self) -> "Journal":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    def event(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"type": kind, "t": round(self.clock() - self.origin, 3), **fields})
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()


class OutDirBusy(RuntimeError):
    """Another sidecar holds the out-dir."""


@contextmanager
def out_dir_lock(path: Path, timeout_s: float) -> Iterator[None]:
    """An exclusive lock on ``path`` for as long as the block runs, or the process lives.

    The cap is for the whole out-dir, and each sidecar keeps its ledger in memory, so two sidecars on one out-dir
    would each spend a full cap. A launch waits for a predecessor that still drains its calls in flight. Then it gives up.
    """
    with open(path, "a", encoding="utf-8") as handle:  # closing the file gives the lock back
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise OutDirBusy(f"another sidecar holds {path} and did not finish within {timeout_s:g} s. Use "
                                     "one sidecar for each out-dir, and give each card its own out-dir and its own "
                                     "share of the budget") from None
                time.sleep(LOCK_POLL_S)
        yield


class Ready(NamedTuple):
    nodes: int
    edges: int
    words: int
    lanes: int


class Unit(NamedTuple):
    """A deep leg that the sidecar started. The miner's ``deep`` line is journaled against it."""

    nonce: str
    arm: str
    qpu_lanes: int


@dataclass
class Job:
    """A survivor of a QPU arm, from its arrival until the sidecar answers it."""

    id: str
    nonce: str
    arm: str
    survivor: cascade_flow.Survivor
    void: bool = False
    answer: Any = None  # what ``sample`` keeps for ``decode``
    stage: str = "sample"  # how far the call got, which the ``qpu_error`` event of a failure names


class Skip(NamedTuple):
    """A survivor that stays off the QPU, and why. ``error`` holds the stage and the exception of a failure that the
    sidecar still has to journal as a ``qpu_error``."""

    job: Job
    reason: str
    error: Optional[Tuple[str, Exception]] = None


def rescore_lanes(spec: replay.TopologySpec, job: Job) -> np.ndarray:
    """The lanes of the survivor of ``job``, scored on the model of its nonce in ``spec``, in milli energy."""
    h, j = replay.model_from_nonce(spec, job.nonce)
    return chain_capture.full_energies_milli(job.survivor.states, spec.dense_edges, h, j)


class MockBackend:
    """A QPU that needs no network. The survivor's own states come back as the reads, each second one with its first
    spin flipped. The first read is 2 whole units below the survivor, and each later read is 1 milli shallower.

    A backend has ``connect()``, which opens a session and raises when the solver is offline. ``sample(job)`` sends the
    model and returns the access time in microseconds (0 when the solver sent none). ``decode(job)`` returns
    ``(spins, energies_milli)``, and the sidecar charges the access time before it decodes. ``rescore(job)`` scores the
    lanes of the survivor on its model, in milli energy, or returns ``None`` when the backend has no model.
    ``shape`` is the ``(nodes, edges)`` of the model that the backend samples, or ``None`` when it takes the model of
    the miner as it is. ``dropped(exc)`` says whether a failed ``sample`` lost its connection under the call, which
    the sidecar sends again once.

    With a ``spec`` the mock makes the same two checks as a live run: it has the ``shape`` of the spec, and it rescores
    the lanes on the model of the nonce. A mock run against the real miner then proves the model and the lane order at
    no cost.
    """

    shape: Optional[Tuple[int, int]] = None

    def __init__(self, reads: int, spec: Optional[replay.TopologySpec] = None) -> None:
        self.reads, self.spec = reads, spec
        if spec is not None:
            self.shape = (len(spec.nodes), len(spec.edges))

    def connect(self) -> None:
        pass

    def sample(self, job: Job) -> int:
        states = job.survivor.states
        spins = np.array([states[read % len(states)] for read in range(self.reads)], dtype=np.int8)
        spins[1::2, 0] *= -1
        job.answer = (spins, (job.survivor.energy - 2) * cascade_flow.MILLI + np.arange(self.reads, dtype=np.int64))
        return MOCK_ACCESS_US

    def decode(self, job: Job) -> Tuple[np.ndarray, np.ndarray]:
        return job.answer

    @staticmethod
    def dropped(exc: BaseException) -> bool:
        return False

    def rescore(self, job: Job) -> Optional[np.ndarray]:
        return None if self.spec is None else rescore_lanes(self.spec, job)

    def close(self) -> None:
        pass


@dataclass(eq=False)
class _Session:
    """One ``OceanSampler``, and the number of calls that run on it."""

    sampler: Any
    calls: int = 0


class OceanBackend:
    """The QPU path of ``chain_qpu_capture.main``: an ``OceanSampler`` on the topology of ``--spec``.

    A retry opens a new session. The session that it replaces closes when its last call in flight returns, so a long
    run does not keep one session, with its threads and sockets, for each recovery.
    """

    def __init__(self, spec: replay.TopologySpec, config: Config) -> None:
        self.spec, self.config = spec, config
        self.shape = (len(spec.nodes), len(spec.edges))
        self._lock = threading.Lock()
        self._session: Optional[_Session] = None  # the session that new calls use
        self._open: List[_Session] = []  # the sessions that are not closed: this one, and replaced ones with calls

    def connect(self) -> None:
        """Open a fresh session. A session that failed still says that it is connected, so a retry cannot reuse it."""
        from quip_miner_dwave.ocean import OceanSampler

        sampler = OceanSampler(mock=False, submit_workers=self.config.in_flight)
        try:
            sampler.ensure_connected()
            sampler.set_session_topology([int(n) for n in self.spec.nodes],
                                         [(int(u), int(v)) for u, v in self.spec.edges])
        except BaseException:
            sampler.close()
            raise
        session = _Session(sampler)
        with self._lock:
            replaced, self._session = self._session, session
            self._open.append(session)
        if replaced is not None:
            self._close_if_idle(replaced)

    def _close_if_idle(self, session: _Session) -> None:
        """Close a replaced session that no call runs on. The close runs on a thread of its own that nobody joins,
        because ``close`` can wait a minute for a poll at SAPI, and a call or a retry must not wait for it."""
        with self._lock:
            if session is self._session or session.calls or session not in self._open:
                return
            self._open.remove(session)
        threading.Thread(target=session.sampler.close, name="cascade-close", daemon=True).start()

    def sample(self, job: Job) -> int:
        config = self.config
        h, j = replay.model_from_nonce(self.spec, job.nonce)
        with self._lock:
            session = self._session
            assert session is not None  # the sidecar makes no call while its connection is down
            session.calls += 1
        try:
            result = session.sampler.sample(
                self.spec.nodes, h, self.spec.edges, j * config.j_scale,
                num_reads=config.reads, anneal_time_us=config.anneal_us, nonce_seed=bytes.fromhex(job.nonce),
                label="quip-cascade-sidecar", auto_scale=None if config.j_scale == 1.0 else False,
            )
        finally:
            with self._lock:
                session.calls -= 1
            self._close_if_idle(session)
        job.answer = (h, j, result)
        return int(result.device_access_time_us)

    @staticmethod
    def dropped(exc: BaseException) -> bool:
        """The connection broke under the call: urllib3's ``ProtocolError`` ("Connection aborted."), as when the server
        closed a pooled connection that sat idle between calls. A connection that cannot open is an outage instead."""
        import requests
        import urllib3

        return isinstance(exc, requests.exceptions.ConnectionError) and any(
            isinstance(arg, urllib3.exceptions.ProtocolError) for arg in exc.args)

    def decode(self, job: Job) -> Tuple[np.ndarray, np.ndarray]:
        h, j, result = job.answer
        spins = replay.spec_order(result.spins, result.variables, self.spec)
        return spins, chain_capture.full_energies_milli(spins, self.spec.dense_edges, h, j)

    def rescore(self, job: Job) -> np.ndarray:
        return rescore_lanes(self.spec, job)

    def close(self) -> None:
        """Close every session that is still open. ``main`` calls this after it has freed the lock of the out-dir."""
        with self._lock:
            sessions, self._open = self._open, []
        for session in sessions:
            session.sampler.close()


class Sidecar:
    """One run: the survivors that wait for a decision, the QPU calls in flight, and the spend ledger.

    The reader loop in ``run`` and the workers share the state behind one lock. Events and answers are written outside
    it. Only the spend journal is written under it, so that its lines follow the ledger.

    A line that the sidecar cannot take ends the run, because the miner and the sidecar then disagree. A failure of the
    QPU side does not. The survivor runs on ``gpu-256``, and the QPU stays off until a retry of the connection works,
    as it stays off at the cap. An exit would be a device fault to the miner, which loses the deep legs of the card,
    and the coordinator stops restarting a card that exits often.
    """

    def __init__(self, config: Config, backend: MockBackend | OceanBackend, journal: Journal, stdout: TextIO) -> None:
        self.config, self.backend, self.journal, self.stdout = config, backend, journal, stdout
        self._lock = threading.Lock()
        self._out_lock = threading.Lock()  # one answer line at a time on standard output
        self._threads: Set[threading.Thread] = set()  # the calls in flight, and a retry of the connection
        self._ready: Optional[Ready] = None
        self._pending: Dict[str, Job] = {}  # survivors with no answer yet, queued or in flight
        self._queue: OrderedDict[str, Job] = OrderedDict()  # the pending survivors that wait for a free call
        self._answered: OrderedDict[str, Unit] = OrderedDict()  # deep legs that the miner has not reported
        self._running = 0
        self._closing = False
        self._abandoned = False  # the drain at the end of input gave up on the threads that are left
        self._unfinished: List[threading.Thread] = []
        self._stdout_closed = False  # the miner closed its end. A line that comes after that stops the run
        self._capped = False  # the cap, for good
        self._lane_checked = False
        self._lane_off = False  # the lanes of the miner do not fit its model: off for good
        self._error_off = False  # a failure of the QPU side: off until a retry of the connection works
        self._probing = False
        self._retry_s, self._retry_at = RETRY_MIN_S, 0.0
        self._fatal: Optional[Exception] = None
        self._cap_us = int(config.qpu_seconds * 1e6)
        self._charged_us, self._ledger_known, self._reserved_us = 0, False, 0
        estimate_us = chain_capture.estimate_us(config.reads, config.anneal_us)
        self._reservation_us = int(estimate_us * chain_capture.RESERVATION_MARGIN)
        self._cell = f"cascade-r{config.reads}-s{config.j_scale:g}"

    def run(self, lines: Iterable[str]) -> None:
        """Serve the miner until its input ends, then write ``end``.

        A line that the sidecar cannot take, or a failure that it cannot report, raises after the calls in flight have
        finished and have their charges in the journal. The drain at the end of the input waits for them for
        ``DRAIN_TIMEOUT_S`` at most.
        """
        self._startup()
        try:
            for line in lines:
                if self._fatal is not None:
                    break
                if self._stdout_closed:
                    raise BrokenPipeError("the miner closed its standard output before its input ended")
                if line.strip():
                    self._handle(json.loads(line))
        finally:
            self._drain()
        if self._fatal is not None:
            raise self._fatal
        self.journal.event("end", charged_us=self._charged_us, unfinished=len(self._unfinished))

    def _startup(self) -> None:
        """Read the ledger and connect. A failure of either keeps the QPU off, and the sidecar still answers."""
        stage = "ledger"
        try:
            self._load_ledger()
            stage = "connect"
            self.backend.connect()
        except Exception as exc:
            with self._lock:
                self._latch_error()
            self._report_qpu_error(None, stage, exc)

    def _load_ledger(self) -> None:
        """The ledger starts from the spend journal, so a restart keeps the cap, and a call that an earlier run left
        open counts at its estimate, as ``read_spend`` counts it. The lock on the out-dir makes this process the only
        writer of the journal, so the ledger stays equal to it."""
        if not self._ledger_known:
            charged, _ = regime_io.read_spend(self.config.out_dir)
            with self._lock:
                self._charged_us, self._ledger_known = charged, True

    def _drain(self) -> None:
        """Submit nothing more, and wait for the calls in flight."""
        with self._lock:
            self._closing = True
            self._queue.clear()
            threads = list(self._threads)
        self._wait_for(threads)

    def _wait_for(self, threads: Sequence[threading.Thread]) -> None:
        """Wait for ``threads`` until ``DRAIN_TIMEOUT_S`` has passed. A call that has not finished by then keeps its
        reservation in the journal, so the cap still counts it, and its thread is left behind with nothing to write."""
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        with self._lock:
            self._abandoned = True
            self._unfinished = [thread for thread in threads if thread.is_alive()]
        if self._unfinished:
            print(f"cascade_sidecar: {len(self._unfinished)} calls did not finish in {DRAIN_TIMEOUT_S:g} s, and their "
                  "reservations stay open", file=sys.stderr)

    def _handle(self, msg: Dict[str, Any]) -> None:
        kind = msg["ev"]
        if kind == "ready":
            self._on_ready(msg)
        elif kind == "survivor":
            self._on_survivor(msg)
        elif kind == "deep":
            self._on_deep(msg)
        elif kind == "void":
            self._on_void(msg)
        elif kind in ("target", "stats"):
            self.journal.event(kind, **{key: value for key, value in msg.items() if key != "ev"})
        else:
            print(f"cascade_sidecar: ignoring the unknown event {kind!r}", file=sys.stderr)

    def _on_ready(self, msg: Dict[str, Any]) -> None:
        config = self.config
        if self._ready is not None:
            raise ValueError("the miner sent ready twice")
        if missing := [name for name in (config.gpu_program, config.cold_program, config.seeded_program)
                       if name not in msg["programs"]]:
            raise ValueError(f"the miner does not list the programs {missing}")
        ready = Ready(int(msg["nodes"]), int(msg["edges"]), int(msg["words"]), int(msg["lanes"]))
        if ready.lanes > cascade_codec.LANES_PER_WORD * ready.words:
            raise ValueError(f"the miner reports {ready.lanes} lanes in {ready.words} words, and a word holds "
                             f"{cascade_codec.LANES_PER_WORD}")
        shape = self.backend.shape
        if shape is not None and shape != (ready.nodes, ready.edges):
            raise ValueError(f"the miner reports a model of {ready.nodes} nodes and {ready.edges} edges, and --spec "
                             f"has {shape[0]} nodes and {shape[1]} edges")
        self._ready = ready

    def _on_survivor(self, msg: Dict[str, Any]) -> None:
        if self._ready is None:
            raise ValueError("the miner sent a survivor before ready")
        job_id, nonce, best = msg["id"], msg["nonce"], int(msg["best"])
        arm = choose_arm(nonce, self.config.arms)
        self.journal.event("survivor", id=job_id, nonce=nonce, best=best, arm=arm)
        if arm == "gpu-256":
            self._run_on_gpu(job_id, nonce)
            return
        if len(msg["energies"]) != self._ready.lanes:
            raise ValueError(f"survivor {job_id} has {len(msg['energies'])} energies for {self._ready.lanes} lanes")
        states = cascade_codec.decode_reads(msg["samples"], reads=self._ready.lanes, nodes=self._ready.nodes)
        survivor = cascade_flow.Survivor(nonce, best, states, np.asarray(msg["energies"], dtype=np.int64))
        job = Job(job_id, nonce, arm, survivor)
        if not self._lane_checked and not self._error_off:
            self._check_lanes(job)
        self._admit(job)

    def _check_lanes(self, job: Job) -> None:
        """Rescore the lanes of the first QPU-arm survivor on the model of its nonce. The energies that the miner sent
        must be 1,000 times the rescored ones, lane for lane. One pass proves the node order, the edge order, the bit
        order, and the sign of the lanes. No QPU call can show a mismatch, and a seeded leg would start from scrambled
        states with no sign of it."""
        try:
            rescored = self.backend.rescore(job)
        except Exception as exc:
            with self._lock:
                self._latch_error()
            self._report_qpu_error(job, "lane_check", exc)
            return
        self._lane_checked = True
        if rescored is None:  # a backend with no model to score on, as the mock has with no --spec
            return
        expected = job.survivor.energies * cascade_flow.MILLI
        bad = np.arange(len(expected)) if rescored.shape != expected.shape else np.flatnonzero(rescored != expected)
        if not bad.size:
            self.journal.event("lane_check", id=job.id, nonce=job.nonce, lanes=len(expected))
            return
        with self._lock:
            self._lane_off = True
        first = int(bad[0])
        self.journal.event("lane_order_mismatch", id=job.id, nonce=job.nonce, lanes=len(expected),
                           mismatched=int(bad.size), first_lane=first, first_expected_milli=int(expected[first]),
                           first_rescored_milli=int(rescored[first]) if rescored.shape == expected.shape else None)
        print(f"cascade_sidecar: the lanes of {job.id} do not match their energies on the model of the nonce "
              f"({bad.size} of {len(expected)} lanes differ). The lane order of the miner is not the order of --spec, "
              "so the QPU stays off for this launch", file=sys.stderr)

    def _on_void(self, msg: Dict[str, Any]) -> None:
        """The lease stopped before the decision. A queued survivor never reaches the QPU. A call in flight does."""
        with self._lock:
            job = self._pending.get(msg["id"])
            if job is not None:
                job.void = True
                if self._queue.pop(job.id, None) is not None:
                    del self._pending[job.id]
        self.journal.event("void", id=msg["id"])

    def _on_deep(self, msg: Dict[str, Any]) -> None:
        with self._lock:
            unit = self._answered.pop(msg["id"], None)
        nonce, arm, qpu_lanes = unit or (None, None, 0)
        fields: Dict[str, Any] = {
            "unit": msg["id"], "arm": arm, "nonce": nonce, "program": msg["program"], "status": msg["status"],
            "checkpoints": msg["checkpoints"], "best": msg["best"], "leg": msg["leg"], "seconds": msg["seconds"],
            "pushed": msg["pushed"], "qpu_lanes": qpu_lanes,
        }
        # A best with no leg is the survivor's own reads, whose lanes the QPU did not seed.
        if qpu_lanes and msg["leg"] is not None:
            fields.update(lane_bests(msg["energies"], qpu_lanes))
        self.journal.event("deep", **fields)

    def _reply(self, message: Dict[str, Any], unit: Optional[Unit] = None) -> None:
        """Write one answer line. ``unit`` is what the miner's later ``deep`` line is journaled against."""
        if unit is not None:
            with self._lock:
                self._answered[message["id"]] = unit
                if len(self._answered) > ANSWERED_LIMIT:
                    self._answered.popitem(last=False)
        with self._out_lock:
            try:
                self.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
                self.stdout.flush()
            except BrokenPipeError:
                # The miner closed its end. At the end of its input that is a normal shutdown, and the answer has no
                # reader. A line that comes after it is not normal, and ``run`` stops there.
                self._stdout_closed = True

    def _run_on_gpu(self, job_id: str, nonce: str) -> None:
        seed = cascade_flow.unit_seed(nonce, GPU_SALT)
        message: Dict[str, Any] = {"op": "deep", "id": job_id, "program": self.config.gpu_program, "seed": seed}
        if self.config.copies > 1:
            message["copies"] = self.config.copies
        self._reply(message, Unit(nonce, "gpu-256", 0))

    def _settle(self, job: Job) -> bool:
        """Close the decision of ``job``. False when a ``void`` came first, and then nothing may be answered."""
        with self._lock:
            self._pending.pop(job.id, None)
            return not job.void

    def _skip(self, skip: Skip) -> None:
        """The survivor stays off the QPU and runs the GPU-only program."""
        job = skip.job
        if skip.error is not None:
            self._report_qpu_error(job, *skip.error)
        self.journal.event("qpu_skip", id=job.id, nonce=job.nonce, arm=job.arm, reason=skip.reason)
        if self._settle(job):
            self._run_on_gpu(job.id, job.nonce)

    def _admit(self, job: Job) -> None:
        with self._lock:
            if self._closing:
                return
            self._pending[job.id] = job
            if self._running >= self.config.in_flight and len(self._queue) >= self.config.queue:
                skipped = [Skip(job, "queue")]
            else:
                self._queue[job.id] = job
                skipped = self._pump()
        for skip in skipped:
            self._skip(skip)

    def _pump(self) -> List[Skip]:
        """Start the queued survivors that the free slots allow. Returns those that must stay off the QPU.

        The cap is hard: a call goes only when its reservation fits under it, and the first refusal is for good. A
        mismatch of the lane order is for good too. An error is not: the first survivor that comes after the retry
        time starts a retry of the connection. No call goes before the lanes of the miner are checked, and a retry can
        land between the check and this test, so the test is here, under the lock. The pace is a target, so it ignores
        the reservation of the call it judges. A reservation that the spend journal cannot take is an error, and no
        call goes without one. The caller holds the lock.
        """
        skipped: List[Skip] = []
        while self._queue and self._running < self.config.in_flight:
            _, job = self._queue.popitem(last=False)
            if self._charged_us + self._reserved_us + self._reservation_us > self._cap_us:
                self._capped = True
            if self._capped:
                skipped.append(Skip(job, "cap"))
            elif self._lane_off:
                skipped.append(Skip(job, "lane_order"))
            elif self._error_off:
                self._retry_connection()
                skipped.append(Skip(job, "error"))
            elif not self._lane_checked:
                skipped.append(Skip(job, "lane_unchecked"))
            elif not self._on_pace():
                skipped.append(Skip(job, "pace"))
            else:
                try:
                    regime_io.append_spend(self.config.out_dir, regime_io.SUBMIT, self._cell, job.id,
                                           self.config.anneal_us, self._reservation_us)
                except Exception as exc:
                    self._latch_error()
                    skipped.append(Skip(job, "error", ("submit", exc)))
                    continue
                self._reserved_us += self._reservation_us
                self._running += 1
                self._start(self._run, job)
        return skipped

    def _on_pace(self) -> bool:
        """The spend of the out-dir is within its share of the cap so far. The share grows evenly from 0 at the origin
        of the run to the whole cap at ``spend_until``, and nothing goes after that. A restart does not reset it."""
        elapsed = self.journal.clock() - self.journal.origin
        window = self.config.spend_until_s - self.journal.origin
        if elapsed >= window:
            return False
        return self._charged_us + self._reserved_us <= self._cap_us * elapsed / window

    def _start(self, target: Callable[..., None], *args: Any) -> None:
        """Run ``target`` on a thread of its own. The thread does not hold the process back when a call hangs. The
        caller holds the lock."""
        thread = threading.Thread(target=target, args=args, name="cascade-qpu", daemon=True)
        thread.start()
        self._threads.add(thread)  # the thread needs the lock to leave the set, so it cannot leave before it is in

    def _latch_error(self) -> None:
        """Keep the QPU off until a retry of the connection works. The caller holds the lock."""
        if not self._error_off:
            self._error_off = True
            self._retry_at = self.journal.clock() + self._retry_s

    def _retry_connection(self) -> None:
        """Start a retry of the connection when its time has come. The caller holds the lock."""
        if not self._probing and not self._closing and self.journal.clock() >= self._retry_at:
            self._probing = True
            self._start(self._probe)

    def _probe(self) -> None:
        """One retry of the connection, on its own thread because a connection can take seconds. When it fails, the wait
        before the next retry doubles, up to ``RETRY_MAX_S``. When it works, the QPU comes back, and the wait doubles
        for the next failure too. A solver that connects and then fails every call is then tried less and less often.
        A call that works sets the wait back to ``RETRY_MIN_S``."""
        stage = "ledger"
        try:
            self._load_ledger()
            stage = "connect"
            self.backend.connect()
        except Exception as exc:
            with self._lock:
                self._probing = False
                self._retry_s = min(2 * self._retry_s, RETRY_MAX_S)
                self._retry_at = self.journal.clock() + self._retry_s
                self._threads.discard(threading.current_thread())
            if not self._abandoned:
                self._report_qpu_error(None, stage, exc)
            return
        with self._lock:
            self._probing = False
            self._error_off = False
            retry_s = self._retry_s
            self._retry_s = min(2 * retry_s, RETRY_MAX_S)
            self._threads.discard(threading.current_thread())
        if not self._abandoned:
            self.journal.event("qpu_recovered", retry_s=retry_s)
            print("cascade_sidecar: the connection to the QPU is back, and the QPU is on again", file=sys.stderr)

    def _report_qpu_error(self, job: Optional[Job], stage: str, exc: Exception) -> None:
        """Journal and print a failure of the QPU side."""
        fields: Dict[str, Any] = {"stage": stage, "exception": type(exc).__name__, "message": str(exc)}
        if job is not None:
            fields.update(id=job.id, nonce=job.nonce, arm=job.arm)
        self.journal.event("qpu_error", **fields)
        print(f"cascade_sidecar: QPU error at {stage}: {type(exc).__name__}: {exc}. The QPU stays off until a retry of "
              "the connection works", file=sys.stderr)

    def _run(self, job: Job) -> None:
        try:
            self._serve(job)
        except Exception as exc:
            try:
                self._call_failed(job, exc)
            except Exception as inner:
                self._fail(inner)
        try:
            with self._lock:
                self._running -= 1
                self._threads.discard(threading.current_thread())
                skipped = self._pump()
            for skip in skipped:
                self._skip(skip)
        except Exception as exc:
            self._fail(exc)

    def _call_failed(self, job: Job, exc: Exception) -> None:
        """A call failed. Its reservation stays open, which the cap counts. The survivor runs on ``gpu-256``, and the
        QPU stays off until a retry of the connection works."""
        if self._abandoned:
            return
        with self._lock:
            self._latch_error()
        self._skip(Skip(job, "error", (job.stage, exc)))

    def _serve(self, job: Job) -> None:
        """One QPU call: the call, its charge, the QPU gate, and the answer. A call whose connection broke goes again
        once, with a ``qpu_retry`` event, before it counts as a failure."""
        config = self.config
        try:
            access_us = int(self.backend.sample(job))
        except Exception as exc:
            if self._abandoned or not self.backend.dropped(exc):
                raise
            self.journal.event("qpu_retry", id=job.id, nonce=job.nonce, arm=job.arm, exception=type(exc).__name__,
                               message=str(exc))
            access_us = int(self.backend.sample(job))
        # SAPI sends no timing record when something is wrong. The reservation stands in for the access time, so that
        # the cap has something to count, and the call counts as a failure.
        charge_us = access_us if access_us > 0 else self._reservation_us
        job.stage = "charge"
        with self._lock:
            regime_io.append_spend(config.out_dir, regime_io.CHARGE, self._cell, job.id, config.anneal_us, charge_us)
            self._reserved_us -= self._reservation_us
            self._charged_us += charge_us
        if self._abandoned:  # the run has ended, and only the charge was left to write
            return
        if access_us <= 0:
            job.stage = "access_time"
            raise RuntimeError(f"the solver reported no QPU access time, so the journal charges its reservation of "
                               f"{charge_us} us")
        job.stage = "decode"
        spins, energies_milli = self.backend.decode(job)
        plans = cascade_flow.qpu_plans(
            job.survivor, spins, energies_milli, qpu_gate_milli=config.qpu_gate_milli, arms=[job.arm],
            programs={"seeded": config.seeded_program, "cold": config.cold_program},
            lanes=len(job.survivor.states))
        message: Dict[str, Any] = {"op": "drop", "id": job.id}
        unit = None
        if plans:  # one arm asked for a plan, so the gate leaves one plan or none
            plan = plans[0]
            message = {"op": "deep", "id": job.id, "program": plan.program, "seed": plan.seed}
            if config.copies > 1:
                message["copies"] = config.copies
            if plan.states is not None:
                assert self._ready is not None
                message["lanes"] = cascade_codec.encode_lanes(plan.states, self._ready.words)
                message["seeded"] = len(plan.states)
            unit = Unit(job.nonce, plan.arm, plan.qpu_lanes)
        with self._lock:
            self._retry_s = RETRY_MIN_S  # a call went through, so the next error waits the shortest time again
        live = self._settle(job)
        self.journal.event("qpu", id=job.id, nonce=job.nonce, arm=job.arm, kept=bool(plans),
                           best_milli=int(np.min(energies_milli)), access_us=charge_us, reads=len(energies_milli),
                           **({} if live else {"void": True}))
        if live:
            self._reply(message, unit)

    def _fail(self, exc: Exception) -> None:
        """Submit nothing more. ``run`` raises the first failure after the calls in flight are charged. This is for a
        failure that the sidecar cannot report as a QPU error, such as one in its own journal."""
        with self._lock:
            if self._fatal is None:
                self._fatal = exc
            self._closing = True
            self._queue.clear()
        print(f"cascade_sidecar: {type(exc).__name__}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out-dir", required=True,
                        help="holds events.jsonl, the spend journal, and a lock. One sidecar for each out-dir: give "
                             "each card its own out-dir and its own share of the budget")
    parser.add_argument("--programs", required=True, help="the programs file of the engine")
    parser.add_argument("--qpu-gates", required=True, help="the table that `cascade_gates.py qpu` wrote")
    parser.add_argument("--arms", type=parse_arms, default="gpu-256:1,qpu-cold:1,qpu-seeded:1",
                        help="name:weight for each arm, in order; a survivor takes an arm by its first nonce byte")
    parser.add_argument("--qpu-seconds", type=float, required=True,
                        help="the cap on the QPU access time that this out-dir may spend in all, in seconds")
    parser.add_argument("--spend-until", type=parse_time, required=True,
                        help="spread the cap evenly from the first launch on this out-dir to this ISO 8601 time. A time with no "
                             "offset is UTC")
    parser.add_argument("--in-flight", type=int, default=16, help="QPU calls in flight at once")
    parser.add_argument("--queue", type=int, default=64,
                        help="survivors that wait for a free call; the rest run on the GPU")
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--anneal-us", type=int, default=80)
    parser.add_argument("--j-scale", type=float, default=0.5,
                        help="multiply the couplings and turn SAPI scaling off; 1.0 keeps SAPI scaling")
    parser.add_argument("--qpu-miss", type=float, default=0.01,
                        help="the row of the gate table that allows this share of winner runs to be lost")
    parser.add_argument("--cold-program", default="cold-1048576")
    parser.add_argument("--seeded-program", default="seeded-1048576-b0250")
    parser.add_argument("--gpu-program", default="gpu-256")
    parser.add_argument("--copies", type=int, default=1,
                        help=f"independent deep units for each deep answer, 1 to {MAX_COPIES}; the engine pushes the "
                             "first hit of any of them, or their best when the last one ends")
    parser.add_argument("--spec", help="the topology spec of the QPU runs; needed unless --mock. With --mock it makes "
                                       "the sidecar check the model and the lane order of the miner, as a live run does")
    parser.add_argument("--mock", action="store_true",
                        help="a fake QPU that needs no Ocean. Without --spec it skips the checks of the model and of "
                             "the lane order")
    parser.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    return parser


def resolve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Config:
    """The settings of the run, or a usage error."""
    if not args.mock and not args.spec:
        parser.error("--spec is required unless --mock")
    for name in ("qpu_seconds", "j_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be a finite positive number")
    for name in ("in_flight", "reads", "anneal_us"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.queue < 0:
        parser.error("--queue must not be negative")
    if not 1 <= args.copies <= MAX_COPIES:
        parser.error(f"--copies must be 1 to {MAX_COPIES}")
    try:
        cut = load_gate(args.qpu_gates, args.qpu_miss)
    except (OSError, ValueError) as exc:
        parser.error(f"--qpu-gates: {exc}")
    return Config(
        out_dir=args.out_dir, arms=args.arms, gpu_program=args.gpu_program, cold_program=args.cold_program,
        seeded_program=args.seeded_program, qpu_miss=args.qpu_miss, qpu_gate_milli=cut,
        qpu_seconds=args.qpu_seconds, spend_until_s=args.spend_until.timestamp(), in_flight=args.in_flight,
        queue=args.queue, reads=args.reads, anneal_us=args.anneal_us, j_scale=args.j_scale, mock=args.mock,
        copies=args.copies)


def serve(
    args: argparse.Namespace, config: Config, journal: Journal, stdin: Iterable[str], stdout: TextIO,
    after_unlock: ExitStack,
) -> int:
    """One launch, under the lock of the out-dir. Returns the exit code.

    The close of the backend goes to ``after_unlock``, which ``main`` runs once the lock is free. A real session can
    wait a minute for a poll at SAPI when it closes, and that wait must not keep the next launch out.
    """
    out_dir = Path(config.out_dir)
    try:
        first = journal.first_start
        if first is not None and first.get("mock") != config.mock:
            # A mock launch books fake access time against the cap, and it fixes the origin of the pace. A live launch
            # on the same out-dir would inherit both, and ``cascade_wins`` would count the fake events as real data.
            raise ValueError(f"{journal.path} was started with mock={first.get('mock')}, and this launch has "
                             f"mock={config.mock}. Give a mock launch an out-dir of its own")
        try:
            spent_us: Optional[int] = regime_io.read_spend(out_dir)[0]
        except Exception:  # the sidecar reports it as a ``qpu_error`` when it starts, and it keeps the QPU off
            spent_us = None
        journal.event("start", **asdict(config), started_at=iso_utc(journal.origin),
                      spend_until=iso_utc(config.spend_until_s), window_s=config.spend_until_s - journal.origin,
                      spent_us=spent_us)
        check_programs(args.programs, [config.gpu_program, config.cold_program, config.seeded_program])
        if config.mock:  # with a spec, the mock makes the checks of a live run. Without one, it skips them
            backend = MockBackend(config.reads, replay.load_spec(args.spec) if args.spec else None)
        else:
            backend = OceanBackend(replay.load_spec(args.spec), config)
        after_unlock.callback(backend.close)
        Sidecar(config, backend, journal, stdout).run(stdin)
    except Exception as exc:
        traceback.print_exc()
        journal.event("error", message=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    stdin: Optional[Iterable[str]] = None,
    stdout: Optional[TextIO] = None,
    *,
    clock: Callable[[], float] = time.time,
    lock_timeout_s: float = LOCK_TIMEOUT_S,
) -> int:
    """Run the sidecar. ``clock`` is the wall clock in seconds, which the pace and the time ``t`` of the events use."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = resolve(parser, args)
    arms = ",".join(f"{name}:{weight}" for name, weight in config.arms)
    print(f"QPU cap {config.qpu_seconds:g} s of access, spent evenly until {iso_utc(config.spend_until_s)}; "
          f"arms {arms}; {config.in_flight} calls in flight, {config.queue} queued", file=sys.stderr)
    if not args.yes:
        print("pass --yes to spend it", file=sys.stderr)
        return 2

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as after_unlock:  # it exits last, when the lock is free, and it closes the backend then
        try:
            with out_dir_lock(out_dir / LOCK_FILE, lock_timeout_s), Journal(out_dir / EVENTS_FILE, clock) as journal:
                return serve(args, config, journal, sys.stdin if stdin is None else stdin,
                             sys.stdout if stdout is None else stdout, after_unlock)
        except OutDirBusy as exc:
            print(f"cascade_sidecar: {exc}", file=sys.stderr)
            # The other sidecar owns the file, and one line is a single atomic write.
            with Journal(out_dir / EVENTS_FILE, clock) as journal:
                journal.event("error", message=f"{type(exc).__name__}: {exc}")
            return 1


if __name__ == "__main__":
    code = main()
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        # The miner is gone, and an answer that it never read is still in the buffer. Point standard output at nothing,
        # so that Python does not fail again at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(code)
