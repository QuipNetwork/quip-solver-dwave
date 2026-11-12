"""Family postprocess of QPU reads, with a trial record that keeps every state.

A family salt's raw QPU reads go to the processor its family's policy names,
and the final states come back to the salt's ``Result`` beside the raw reads.
Before the processor starts, the raw reads are written to a trial record.
When it ends, the record gains the final states, the processor's identity,
the policy seed, the measured host time, and the outcome. A processor that
fails or is stopped leaves a ``failed`` or ``incomplete`` record. It never
leaves a record that looks like a successful trial with raw reads alone.

Nothing here touches billing. The session bills a salt's QPU access when the
sampler returns, before the postprocess starts. Processor startup and wall
time go only to the trial record, never to the quota ledger.

The postprocess runs only when ``Configure.backend_toml`` asks for it::

    [postprocess]
    records = "/var/lib/quip/postprocess"   # required
    threads = 1                             # processors at once, default 1
    "quip/twisted-ice-dimer" = "qpp-race768/1"

A family the table does not name is not postprocessed. A named family with
an unknown policy, a missing ``records``, or a processor binary that cannot
be found makes the configuration invalid.

The one policy is ``qpp-race768/1``: the read race of the 2026-11-11 study
(``~/quip-data/regimes/qpu-aware-pp-20261111``) at 768 sweeps, stages
``4:64,1:512`` with beta 6 to 20, every raw read in the population. It runs in
``quip-dwave-qpp`` (``crates/qpp``) through :mod:`quip_solver_dwave.qpp`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
import time
import tomllib
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from quip_solver_dwave import qpp
from quip_solver_dwave.regime_io import atomic_savez

logger = logging.getLogger(__name__)

TWISTED_DIMER = "quip/twisted-ice-dimer"

OK = "ok"
FAILED = "failed"
INCOMPLETE = "incomplete"


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
}


@dataclass(frozen=True)
class PostprocessConfig:
    records: Path
    threads: int
    #: Policy of each family that is postprocessed.
    policies: Dict[str, Policy]


def postprocess_config_from_toml(toml_text: str) -> Optional[PostprocessConfig]:
    """The ``[postprocess]`` table, or ``None`` when the document has none.

    Raises:
        PostprocessConfigInvalid: The table is present but not usable.
    """
    if not toml_text or not toml_text.strip():
        return None
    try:
        data = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError:
        return None
    table = data.get("postprocess")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise PostprocessConfigInvalid("postprocess must be a table")
    records = table.get("records")
    if not isinstance(records, str) or not records:
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
    return PostprocessConfig(records=Path(records), threads=threads, policies=policies)


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
    """One postprocess of one salt's reads, as the record keeps it."""

    job_id: bytes
    salt_index: int
    family_id: str
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

    @property
    def stem(self) -> str:
        return f"{self.job_id.hex()}.{self.salt_index}"

    def summary(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id.hex(),
            "salt_index": self.salt_index,
            "family_id": self.family_id,
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


class TrialStore:
    """``<records>/<job>.<salt>.npz`` with every state, and ``trials.jsonl``.

    The npz is written twice: with the raw reads before the processor starts,
    then with the final states once it ends. Each write replaces the file
    whole, so a kill leaves the raw-only file with outcome ``incomplete``.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def path(self, trial: Trial) -> Path:
        return self.root / f"{trial.stem}.npz"

    def write(self, trial: Trial) -> None:
        summary = trial.summary()
        atomic_savez(
            self.path(trial),
            raw_spins=trial.raw_spins,
            raw_energies=trial.raw_energies,
            final_spins=trial.final_spins,
            final_energies=trial.final_energies,
            summary=np.str_(json.dumps(summary)),
        )

    def finish(self, trial: Trial) -> None:
        self.write(trial)
        with self._lock, (self.root / "trials.jsonl").open("a") as f:
            f.write(json.dumps(trial.summary()) + "\n")


def processor_identity(binary: Path) -> str:
    """``quip-dwave-qpp sha256:<digest>`` of the binary that runs the policy."""
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    return f"{qpp.BINARY_NAME} sha256:{digest}"


def policy_seed(nonce: bytes) -> int:
    """The processor seed of a salt: the first 8 nonce bytes, little-endian."""
    return int.from_bytes(nonce[:8], "little")


class Postprocessor:
    """Runs the configured policy on one salt's reads and records the trial."""

    def __init__(self, config: PostprocessConfig, *, binary: Optional[Path] = None):
        exe = binary if binary is not None else qpp.find_binary()
        if exe is None or not exe.is_file():
            raise PostprocessConfigInvalid(
                f"{qpp.BINARY_NAME} not found: build crates/qpp or set {qpp.BINARY_ENV}"
            )
        self.config = config
        self.binary = exe
        self.identity = processor_identity(exe)
        self.store = TrialStore(config.records)
        # The host work of every salt runs here, ``threads`` at a time, off
        # the job pool whose workers stand for QPU submissions.
        self.executor = ThreadPoolExecutor(
            max_workers=config.threads, thread_name_prefix="dwave-postprocess"
        )

    def submit(self, fn: Callable[..., None], *args: Any) -> "Future[None]":
        return self.executor.submit(fn, *args)

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def policy_for(self, family_id: Optional[str]) -> Optional[Policy]:
        return self.config.policies.get(family_id) if family_id else None

    def run(
        self,
        *,
        job_id: bytes,
        salt_index: int,
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
            policy=policy.name,
            processor=self.identity,
            seed=policy_seed(nonce),
            raw_spins=np.asarray(raw_spins, dtype=np.int8),
            raw_energies=np.asarray(raw_energies, dtype=np.int64),
        )
        # Raw evidence first, so a crash or a kill below still leaves it.
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
                "postprocess %s salt %d (%s): %s", job_id.hex()[:16], salt_index, policy.name, trial.error
            )
        else:
            trial.outcome = OK
            trial.final_spins = result.state[None].astype(np.int8)
            trial.final_energies = np.array([result.best_milli], dtype=np.int64)
            trial.processor_cpu_s = result.cpu_s
        self.store.finish(trial)
        return trial

