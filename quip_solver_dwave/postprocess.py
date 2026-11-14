"""Family postprocess of QPU reads, with a record that keeps every state.

Every salt's raw QPU reads are written to a record, with the nonce that
rebuilds its model, whether or not a processor runs on them. A salt without a
postprocess leaves a ``raw`` record.

A family salt's raw QPU reads go to the processor its family's policy names,
and the final states come back to the salt's ``Result`` beside the raw reads.
Only a salt whose raw reads leave the target in reach goes to the processor
(:class:`ReachGate`). When the processor ends, the record gains the final
states, the processor's identity, the policy seed, the measured host time, and
the outcome. A processor that fails or is stopped leaves a ``failed`` or
``incomplete`` record. It never leaves a record that looks like a successful
trial with raw reads alone.

Nothing here touches billing. The session bills a salt's QPU access when the
sampler returns, before the postprocess starts. Processor startup and wall
time go only to the trial record, never to the quota ledger.

The postprocess runs by default. Without a ``[postprocess]`` table in
``Configure.backend_toml``, each family in :data:`DEFAULT_POLICIES` gets its
policy, and a host without the processor binary reports raw reads alone.
``postprocess = false`` turns the postprocess off. A table replaces the
default::

    [postprocess]
    records = "/var/lib/quip/qpu-reads"     # default: qpu-reads beside usage_db
    threads = 1                             # processors at once, default 1
    "quip/twisted-ice-dimer" = "qpp-race768/1"

A family the table does not name is not postprocessed. A named family with
an unknown policy, or a processor binary that cannot be found, makes the
configuration invalid.

The records go to one directory for each UTC day. ``records_max_gb`` (top
level, default 20) caps their size: the oldest days are deleted first, and the
current day stops taking records once it alone passes the cap.

``qpp-race768/1`` is the read race of the 2026-11-11 study
(``~/quip-data/regimes/qpu-aware-pp-20261111``) at 768 sweeps, stages
``4:64,1:512`` with beta 6 to 20, every raw read in the population.
``qpp-race2048-repair32/1``, the default, is the race at 2,048 sweeps, stages
``8:64,2:256,1:1024``, then the defect repair on its best state with chains of
up to 32 flips. Both run in ``quip-dwave-qpp`` (``crates/qpp``) through
:mod:`quip_solver_dwave.qpp`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import struct
import threading
import time
import tomllib
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from quip_solver_dwave import qpp
from quip_solver_dwave.budget import usage_db_from_backend_toml

logger = logging.getLogger(__name__)

TWISTED_DIMER = "quip/twisted-ice-dimer"

OK = "ok"
FAILED = "failed"
INCOMPLETE = "incomplete"
#: The reads of a salt that no processor ran on.
RAW = "raw"

#: The records directory beside the usage database, when no table names one.
DEFAULT_RECORDS_DIR_NAME = "qpu-reads"
#: Most gigabytes of records kept.
DEFAULT_RECORDS_MAX_GB = 20.0


class PostprocessConfigInvalid(ValueError):
    """The ``[postprocess]`` table of ``backend_toml`` cannot be used."""


@dataclass(frozen=True)
class Policy:
    """One versioned postprocess policy: the processor and its fixed options."""

    name: str
    family_id: str
    mode: str
    params: Dict[str, Any]


#: Every policy this miner runs, by name. A new policy is a new name.
POLICIES: Dict[str, Policy] = {
    "qpp-race768/1": Policy(
        name="qpp-race768/1",
        family_id=TWISTED_DIMER,
        mode="race",
        params={"stages": "4:64,1:512", "b0": 6, "b1": 20, "single_every": 4},
    ),
    "qpp-race2048-repair32/1": Policy(
        name="qpp-race2048-repair32/1",
        family_id=TWISTED_DIMER,
        mode="race",
        params={
            "stages": "8:64,2:256,1:1024",
            "b0": 6,
            "b1": 20,
            "single_every": 4,
            "repair_depth": 32,
        },
    ),
}

#: The policy of each family when ``backend_toml`` has no ``[postprocess]``.
DEFAULT_POLICIES: Dict[str, Policy] = {
    TWISTED_DIMER: POLICIES["qpp-race2048-repair32/1"],
}

#: Most processors at once under the default configuration.
DEFAULT_MAX_THREADS = 4


@dataclass(frozen=True)
class RecordsConfig:
    #: Directory of the records.
    root: Path
    #: Most bytes of records kept.
    max_bytes: int


def records_config_from_toml(toml_text: str) -> RecordsConfig:
    """Where the records go and how much disk they may take.

    The directory is ``[postprocess] records`` when the table names one, else
    ``qpu-reads`` beside the usage database.

    Raises:
        PostprocessConfigInvalid: ``records_max_gb`` is not a positive number.
    """
    data: Dict[str, Any] = {}
    if toml_text and toml_text.strip():
        try:
            data = tomllib.loads(toml_text)
        except tomllib.TOMLDecodeError:
            data = {}
    table = data.get("postprocess")
    named = table.get("records") if isinstance(table, dict) else None
    if isinstance(named, str) and named:
        root = Path(named)
    else:
        root = (
            Path(usage_db_from_backend_toml(toml_text)).parent
            / DEFAULT_RECORDS_DIR_NAME
        )
    max_gb = data.get("records_max_gb", DEFAULT_RECORDS_MAX_GB)
    if isinstance(max_gb, bool) or not isinstance(max_gb, (int, float)) or max_gb <= 0:
        raise PostprocessConfigInvalid("records_max_gb must be a positive number")
    return RecordsConfig(root=root, max_bytes=int(max_gb * 1e9))


@dataclass(frozen=True)
class PostprocessConfig:
    threads: int
    #: Policy of each family that is postprocessed.
    policies: Dict[str, Policy]
    #: ``True`` when a ``[postprocess]`` table asked for this configuration.
    explicit: bool = True
    #: Most salts queued or running at once. A salt past the limit reports
    #: raw reads. ``None`` sets no limit.
    max_pending: Optional[int] = None


def usable_processors() -> int:
    """Processors this process may run on: its affinity, which a container cpuset limits."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def default_postprocess_config() -> PostprocessConfig:
    """The default policies, all usable processors but one, and a short queue.

    The queue limit keeps the postprocess from delaying results: when the
    host cannot keep up with the QPU, the salts past the limit report their
    raw reads at once.
    """
    threads = min(DEFAULT_MAX_THREADS, max(1, usable_processors() - 1))
    return PostprocessConfig(
        threads=threads,
        policies=dict(DEFAULT_POLICIES),
        explicit=False,
        max_pending=2 * threads,
    )


def postprocess_config_from_toml(toml_text: str) -> Optional[PostprocessConfig]:
    """The ``[postprocess]`` table, or the default when the document has none.

    ``postprocess = false`` gives ``None``: no postprocess.

    Raises:
        PostprocessConfigInvalid: The table is present but not usable.
    """
    if not toml_text or not toml_text.strip():
        return default_postprocess_config()
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError:
        return default_postprocess_config()
    table = data.get("postprocess")
    if table is None:
        return default_postprocess_config()
    if table is False:
        return None
    if not isinstance(table, dict):
        raise PostprocessConfigInvalid("postprocess must be a table or false")
    records = table.get("records")
    if records is not None and (not isinstance(records, str) or not records):
        raise PostprocessConfigInvalid("postprocess.records must name a directory")
    threads = table.get("threads", 1)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise PostprocessConfigInvalid("postprocess.threads must be a positive integer")
    policies: Dict[str, Policy] = {}
    for family_id, name in table.items():
        if family_id in ("records", "threads"):
            continue
        policy = POLICIES.get(name) if isinstance(name, str) else None
        if policy is None:
            raise PostprocessConfigInvalid(
                f"postprocess.{family_id!r} names {name!r}, not one of {sorted(POLICIES)}"
            )
        if policy.family_id != family_id:
            raise PostprocessConfigInvalid(
                f"policy {policy.name} is for {policy.family_id}, not {family_id}"
            )
        policies[family_id] = policy
    if not policies:
        raise PostprocessConfigInvalid("postprocess names no family")
    return PostprocessConfig(threads=threads, policies=policies)


# -- the dimer model from an instance -------------------------------------------


def _sections(metadata: bytes) -> Dict[int, bytes]:
    """The FAM-34 sections of a metadata region, by kind.

    The wheel built and validated the instance, so a layout fault here is a
    defect, not a coordinator mistake.
    """
    (count,) = struct.unpack_from("<H", metadata, 0)
    at = 2
    out: Dict[int, bytes] = {}
    for _ in range(count):
        kind, length = struct.unpack_from("<HI", metadata, at)
        at += 6
        out[kind] = metadata[at : at + length]
        at += length
    if at != len(metadata):
        raise ValueError("metadata has bytes after its last section")
    return out


def _counted(payload: bytes, width: int) -> np.ndarray:
    (count,) = struct.unpack_from("<I", payload, 0)
    rows = np.frombuffer(payload, dtype="<u4", offset=4)
    if rows.size != count * width:
        raise ValueError(f"a section holds {rows.size} values for {count} rows of {width}")
    return rows.reshape(count, width).astype(np.int64)


def dimer_model(
    nodes: np.ndarray, positions: np.ndarray, h_milli: np.ndarray, j_milli: np.ndarray, metadata: bytes
) -> qpp.DimerModel:
    """The qpp model of a twisted-ice dimer instance: kind 1 tiles, kind 3 pairs.

    Raises:
        ValueError: The instance has a field, which the qpp model cannot hold,
            or its metadata lacks the tiles or the pairs.
    """
    if np.any(h_milli):
        raise ValueError("a twisted-ice dimer instance with a field is outside the qpp model")
    sections = _sections(metadata)
    if 1 not in sections or 3 not in sections:
        raise ValueError("the metadata lacks the tiles or the pairs section")
    return qpp.DimerModel(
        n=int(len(nodes)),
        edges=np.asarray(positions, dtype=np.int64).reshape(-1, 2),
        j_milli=np.asarray(j_milli, dtype=np.int64),
        tiles=_counted(sections[1], 4),
        pairs=_counted(sections[3], 2),
    )


# -- the trial record -------------------------------------------------------------


@dataclass
class Trial:
    """The reads of one salt and what a processor made of them, as the record keeps it."""

    job_id: bytes
    #: The lease salt, or ``None`` for an explicit family job.
    salt_index: Optional[int]
    family_id: str
    #: The nonce that rebuilds the salt's model.
    nonce: bytes
    policy: str
    processor: str
    seed: int
    raw_spins: np.ndarray
    raw_energies: np.ndarray
    outcome: str = INCOMPLETE
    final_spins: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int8))
    final_energies: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: Wall time of the processor run, process startup included.
    host_s: float = 0.0
    #: Processor seconds the processor reported, when it finished.
    processor_cpu_s: Optional[float] = None
    error: Optional[str] = None
    #: The target the salt mined against, when the session knew one.
    target_milli: Optional[int] = None
    #: Unix time the reads arrived. It names the record's day directory.
    ts_s: float = field(default_factory=time.time)

    @property
    def day(self) -> str:
        return time.strftime("%Y%m%d", time.gmtime(self.ts_s))

    @property
    def stem(self) -> str:
        if self.salt_index is None:
            return self.job_id.hex()
        return f"{self.job_id.hex()}.{self.salt_index}"

    def summary(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id.hex(),
            "salt_index": self.salt_index,
            "family_id": self.family_id,
            "nonce": self.nonce.hex(),
            "ts_s": self.ts_s,
            "target_milli": self.target_milli,
            "policy": self.policy,
            "processor": self.processor,
            "seed": self.seed,
            "outcome": self.outcome,
            "raw_reads": int(self.raw_spins.shape[0]),
            "raw_best_milli": int(self.raw_energies.min()) if self.raw_energies.size else None,
            "final_best_milli": int(self.final_energies.min()) if self.final_energies.size else None,
            "host_s": self.host_s,
            "processor_cpu_s": self.processor_cpu_s,
            "error": self.error,
        }


def _pack_spins(spins: np.ndarray) -> np.ndarray:
    """±1 rows as bits, one row of bytes per read; a set bit is +1."""
    return np.packbits(np.atleast_2d(spins) > 0, axis=1)


def _save_record(path: Path, **arrays: Any) -> None:
    """Write an uncompressed npz through a temp file in the same directory, then replace.

    Packed spins are already at one bit each, and deflate on 48 chip reads
    costs about 25 ms of worker time for a larger file.
    """
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_record(path: Path) -> dict[str, Any]:
    """One record as ``raw_spins``, ``raw_energies``, ``final_spins``, ``final_energies`` and ``summary``."""
    with np.load(path) as z:
        n = int(z["n_spins"])

        def spins(bits: np.ndarray) -> np.ndarray:
            rows = np.unpackbits(bits, axis=1, count=n).astype(np.int8)
            return rows * 2 - 1

        return {
            "raw_spins": spins(z["raw_bits"]),
            "raw_energies": z["raw_energies"],
            "final_spins": spins(z["final_bits"]),
            "final_energies": z["final_energies"],
            "summary": json.loads(str(z["summary"])),
        }


class TrialStore:
    """``<records>/<day>/<job>.<salt>.npz`` with every state, and ``<day>/trials.jsonl``.

    The npz of a postprocessed salt is written twice: with the raw reads
    before the processor starts, then with the final states once it ends. Each
    write replaces the file whole, so a kill leaves the raw-only file with
    outcome ``incomplete``.

    The store never raises for a full or failing disk: a record that cannot
    be written is counted and logged, and the salt's result goes out as usual.
    """

    def __init__(self, root: Path, max_bytes: Optional[int] = None):
        self.root = root
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        #: Bytes held by each day directory.
        self._day_bytes: Dict[str, int] = {
            d.name: sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            for d in self.root.iterdir()
            if d.is_dir() and d.name.isdigit()
        }
        #: Records not written: the cap was reached or the write failed.
        self.dropped = 0

    def path(self, trial: Trial) -> Path:
        return self.root / trial.day / f"{trial.stem}.npz"

    def _drop(self, why: str) -> None:
        self.dropped += 1
        if self.dropped == 1 or self.dropped % 1000 == 0:
            logger.warning("[QPU] %d read records not written: %s", self.dropped, why)

    def _room(self, day: str) -> bool:
        """Delete the oldest days until the records fit. False when ``day`` alone is too large."""
        if self.max_bytes is None:
            return True
        while sum(self._day_bytes.values()) > self.max_bytes:
            oldest = min(self._day_bytes)
            if oldest >= day:
                return False
            shutil.rmtree(self.root / oldest, ignore_errors=True)
            del self._day_bytes[oldest]
            logger.info(
                "[QPU] read records of %s deleted to stay under the cap", oldest
            )
        return True

    def write(self, trial: Trial) -> bool:
        """Write or replace the record of ``trial``. False when it was not written."""
        path = self.path(trial)
        with self._lock:
            if not self._room(trial.day):
                self._drop(
                    f"day {trial.day} alone passes the cap of {self.max_bytes} bytes"
                )
                return False
        try:
            path.parent.mkdir(exist_ok=True)
            before = path.stat().st_size if path.exists() else 0
            _save_record(
                path,
                n_spins=np.int64(trial.raw_spins.shape[1]),
                raw_bits=_pack_spins(trial.raw_spins),
                raw_energies=trial.raw_energies,
                final_bits=_pack_spins(trial.final_spins),
                final_energies=trial.final_energies,
                summary=np.str_(json.dumps(trial.summary())),
            )
            grown = path.stat().st_size - before
        except OSError as exc:
            with self._lock:
                self._drop(f"{type(exc).__name__}: {exc}")
            return False
        with self._lock:
            self._day_bytes[trial.day] = self._day_bytes.get(trial.day, 0) + grown
        return True

    def finish(self, trial: Trial) -> None:
        """Write the final record of ``trial`` and its line in the day's ``trials.jsonl``."""
        if not self.write(trial):
            return
        line = json.dumps(trial.summary()) + "\n"
        try:
            with self._lock, (self.root / trial.day / "trials.jsonl").open("a") as f:
                f.write(line)
                self._day_bytes[trial.day] = self._day_bytes.get(trial.day, 0) + len(
                    line
                )
        except OSError as exc:
            with self._lock:
                self._drop(f"{type(exc).__name__}: {exc}")

    def record_raw(
        self,
        *,
        job_id: bytes,
        salt_index: Optional[int],
        family_id: str,
        nonce: bytes,
        raw_spins: np.ndarray,
        raw_energies: np.ndarray,
        target_milli: Optional[int] = None,
    ) -> None:
        """Record the reads of a salt that no processor runs on."""
        self.finish(
            Trial(
                job_id=job_id,
                salt_index=salt_index,
                family_id=family_id,
                nonce=nonce,
                policy="",
                processor="",
                seed=0,
                raw_spins=np.asarray(raw_spins, dtype=np.int8),
                raw_energies=np.asarray(raw_energies, dtype=np.int64),
                outcome=RAW,
                target_milli=target_milli,
            )
        )


#: Trials a gate needs before it rules a salt out.
REACH_MIN_TRIALS = 16
#: Gains a gate remembers.
REACH_WINDOW = 256
#: Milli-units added to the largest remembered gain.
REACH_MARGIN_MILLI = 3000
#: One salt in this many goes to the processor although the gate ruled it out.
REACH_PROBE_EVERY = 64


class ReachGate:
    """Sends a salt to the processor only when its raw reads leave the target in reach.

    The gain of a trial is its best raw energy less its best final energy. The
    reach is the largest of the last :data:`REACH_WINDOW` gains plus
    :data:`REACH_MARGIN_MILLI`. A salt is in reach when its best raw energy
    less the reach is at or below the target. Until :data:`REACH_MIN_TRIALS`
    gains are known, and for a salt with no target, every salt is in reach.
    One ruled-out salt in :data:`REACH_PROBE_EVERY` still runs, so the reach
    follows the models the chain sets.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gains: "deque[int]" = deque(maxlen=REACH_WINDOW)
        self._ruled_out = 0

    def reach_milli(self) -> Optional[int]:
        with self._lock:
            if len(self._gains) < REACH_MIN_TRIALS:
                return None
            return max(self._gains) + REACH_MARGIN_MILLI

    def learn(self, raw_best_milli: int, final_best_milli: int) -> None:
        with self._lock:
            self._gains.append(raw_best_milli - final_best_milli)

    def admits(self, raw_best_milli: int, target_milli: Optional[int]) -> bool:
        reach = self.reach_milli()
        if (
            reach is None
            or target_milli is None
            or raw_best_milli - reach <= target_milli
        ):
            return True
        with self._lock:
            self._ruled_out += 1
            if self._ruled_out == 1 or self._ruled_out % 1000 == 0:
                logger.info(
                    "[QPU] postprocess gate: %d salts ruled out so far (reach %d, raw best %d, "
                    "target %d, milli-units)",
                    self._ruled_out,
                    reach,
                    raw_best_milli,
                    target_milli,
                )
            return self._ruled_out % REACH_PROBE_EVERY == 0


def processor_identity(binary: Path) -> str:
    """``quip-dwave-qpp sha256:<digest>`` of the binary that runs the policy."""
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    return f"{qpp.BINARY_NAME} sha256:{digest}"


def policy_seed(nonce: bytes) -> int:
    """The processor seed of a salt: the first 8 nonce bytes, little-endian."""
    return int.from_bytes(nonce[:8], "little")


class Postprocessor:
    """Runs the configured policy on one salt's reads and records the trial."""

    def __init__(
        self,
        config: PostprocessConfig,
        *,
        store: Optional[TrialStore] = None,
        binary: Optional[Path] = None,
    ):
        exe = binary if binary is not None else qpp.find_binary()
        if exe is None or not exe.is_file():
            raise PostprocessConfigInvalid(
                f"{qpp.BINARY_NAME} not found: build crates/qpp or set {qpp.BINARY_ENV}"
            )
        self.config = config
        self.binary = exe
        self.identity = processor_identity(exe)
        self.store = store
        #: One gate for each family: the gain depends on the family's models.
        self.gates: Dict[str, ReachGate] = {fid: ReachGate() for fid in config.policies}
        # The host work of every salt runs here, ``threads`` at a time, off
        # the job pool whose workers stand for QPU submissions.
        self.executor = ThreadPoolExecutor(
            max_workers=config.threads, thread_name_prefix="dwave-postprocess"
        )
        self._lock = threading.Lock()
        self._pending = 0
        #: Salts that reported raw reads because the queue was full.
        self.skipped = 0

    def submit(self, fn: Callable[..., None], *args: Any) -> "Optional[Future[None]]":
        """Queue the host work of one salt, or return ``None`` when the queue is full."""
        limit = self.config.max_pending
        with self._lock:
            if limit is not None and self._pending >= limit:
                self.skipped += 1
                if self.skipped == 1 or self.skipped % 1000 == 0:
                    logger.info(
                        "[QPU] postprocess queue full (%d pending): %d salts so far "
                        "reported raw reads alone",
                        self._pending,
                        self.skipped,
                    )
                return None
            self._pending += 1
        try:
            future = self.executor.submit(fn, *args)
        except BaseException:
            self._release()
            raise
        future.add_done_callback(lambda _f: self._release())
        return future

    def _release(self) -> None:
        with self._lock:
            self._pending -= 1

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def policy_for(self, family_id: Optional[str]) -> Optional[Policy]:
        return self.config.policies.get(family_id) if family_id else None

    def run(
        self,
        *,
        job_id: bytes,
        salt_index: Optional[int],
        family_id: str,
        nonce: bytes,
        nodes: np.ndarray,
        positions: np.ndarray,
        h_milli: np.ndarray,
        j_milli: np.ndarray,
        metadata: bytes,
        raw_spins: np.ndarray,
        raw_energies: np.ndarray,
        stop: Callable[[], bool],
        target_milli: Optional[int] = None,
    ) -> Trial:
        """Postprocess one salt. The returned trial says how it ended.

        ``raw_energies`` are the exact energies of ``raw_spins``. Final
        energies are rescored on the host and leave out the instance offset,
        as qpp's own model does; the caller adds it.
        """
        policy = self.config.policies[family_id]
        trial = Trial(
            job_id=job_id,
            salt_index=salt_index,
            family_id=family_id,
            nonce=nonce,
            policy=policy.name,
            processor=self.identity,
            seed=policy_seed(nonce),
            raw_spins=np.asarray(raw_spins, dtype=np.int8),
            raw_energies=np.asarray(raw_energies, dtype=np.int64),
            target_milli=target_milli,
        )
        # Raw evidence first, so a crash or a kill below still leaves it.
        if self.store is not None:
            self.store.write(trial)
        try:
            if trial.raw_spins.shape[0] == 0:
                raise ValueError("no raw reads to postprocess")
            model = dimer_model(nodes, positions, h_milli, j_milli, metadata)
            if stop():
                raise qpp.QppStopped("stopped before the processor started")
            started = time.monotonic()
            try:
                result = qpp.run(
                    model,
                    trial.raw_spins,
                    policy.mode,
                    binary=self.binary,
                    stop=stop,
                    seed=trial.seed,
                    pop_first=0,
                    pop_count=trial.raw_spins.shape[0],
                    **policy.params,
                )
            finally:
                trial.host_s = time.monotonic() - started
        except qpp.QppStopped as exc:
            trial.outcome, trial.error = INCOMPLETE, str(exc)
        except (qpp.QppError, ValueError, OSError) as exc:
            trial.outcome, trial.error = FAILED, f"{type(exc).__name__}: {exc}"
            logger.warning(
                "postprocess %s (%s): %s", trial.stem[:24], policy.name, trial.error
            )
        else:
            trial.outcome = OK
            trial.final_spins = result.state[None].astype(np.int8)
            trial.final_energies = np.array([result.best_milli], dtype=np.int64)
            trial.processor_cpu_s = result.cpu_s
            self.gates[family_id].learn(
                int(trial.raw_energies.min()), int(result.best_milli)
            )
        if self.store is not None:
            self.store.finish(trial)
        return trial

