"""QPU-aware postprocess for the twisted-ice dimer cells.

The numeric work runs in ``quip-dwave-qpp``, the Rust binary in ``crates/qpp``
(a bit-for-bit port of the 2026-11-11 research prototype, with
its cluster root lookup fixed). This module writes
its input files, runs it, and rescores the returned state on the host, so a
result never reaches a caller with an energy the host did not confirm.

The modes are the prototype's. ``race`` is the selected design: polish a
population of reads, then race them through ``keep:sweeps[:b0]`` stages,
each lineage restarting from its best state. ``chain`` continues one state,
``ratchet`` merges short excursions cluster by cluster, and ``pa`` is
population annealing. A ``mask`` frees part of the spins and pins the rest,
which the subset anneals use.

Energies are integer milli, as the coordinator scores them. The tile and
dimer weights (``tw``, ``dw``) are milli as well.

The processor is specific to the QPU path: it starts from QPU reads and the
tile and pair structure of the cell, so it is not a generic CPU solver.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

BINARY_ENV = "QUIP_DWAVE_QPP"
BINARY_NAME = "quip-dwave-qpp"


class QppError(RuntimeError):
    """The binary refused the input (exit 1) or found a bookkeeping defect (exit 2)."""


class QppStopped(QppError):
    """``stop`` became true while the binary ran, and the run was killed."""


#: How often :func:`run` checks ``stop`` while the binary runs.
POLL_S = 0.02


@dataclass(frozen=True)
class DimerModel:
    """A twisted-ice dimer model: couplings plus its tile and pair structure.

    ``edges`` is ``(E, 2)`` spin indices, ``j_milli`` the ``E`` couplings,
    ``tiles`` is ``(T, 4)`` and ``pairs`` is ``(P, 2)``, all in ``0..n``.
    """

    n: int
    edges: np.ndarray
    j_milli: np.ndarray
    tiles: np.ndarray
    pairs: np.ndarray

    def _check(self) -> None:
        for name in ("edges", "tiles", "pairs"):
            arr = np.asarray(getattr(self, name))
            if arr.size and (arr.min() < 0 or arr.max() >= self.n):
                raise ValueError(f"{name} has a spin outside 0..{self.n}")

    def gauge(self) -> np.ndarray:
        """``(T, 4)`` signs: tile spin ``k`` takes the sign of its coupling to spin 0.

        Spin 0 of each tile has sign +1. Every tile spin must couple to spin 0.
        """
        self._check()
        weight = {
            (min(u, v), max(u, v)): j
            for (u, v), j in zip(np.asarray(self.edges).tolist(), np.asarray(self.j_milli).tolist())
        }
        tiles = np.asarray(self.tiles, dtype=np.int64)
        gauge = np.ones(tiles.shape, dtype=np.int8)
        for t, nodes in enumerate(tiles.tolist()):
            for k in range(1, 4):
                a, b = sorted((nodes[0], nodes[k]))
                if (a, b) not in weight:
                    raise ValueError(f"tile {t} has no coupling between spins {a} and {b}")
                gauge[t, k] = 1 if weight[(a, b)] > 0 else -1
        return gauge


@dataclass(frozen=True)
class QppResult:
    mode: str
    start_milli: int
    best_milli: int
    #: ``[work in sweeps, processor s, best milli, current milli]`` per checkpoint.
    trace: List[Tuple[int, float, int, int]]
    cpu_s: float
    #: The best state, ``n`` spins of +1/-1.
    state: np.ndarray
    #: Mode-specific fields of the binary's report (``mean_clusters`` and so on).
    extra: dict


def write_model(path: "str | Path", model: DimerModel) -> None:
    """Write the binary's model file: ``i32 n, T, P, E``, edge rows, tiles, gauge, pairs."""
    gauge = model.gauge()
    edges = np.asarray(model.edges, dtype=np.int64).reshape(-1, 2)
    rows = np.column_stack([edges, np.asarray(model.j_milli, dtype=np.int64)])
    tiles = np.asarray(model.tiles, dtype=np.int64).reshape(-1, 4)
    pairs = np.asarray(model.pairs, dtype=np.int64).reshape(-1, 2)
    with open(path, "wb") as f:
        f.write(np.array([model.n, len(tiles), len(pairs), len(edges)], dtype="<i4").tobytes())
        f.write(rows.astype("<i4").tobytes())
        f.write(tiles.astype("<i4").tobytes())
        f.write(gauge.astype(np.int8).tobytes())
        f.write(pairs.astype("<i4").tobytes())


def energies_milli(model: DimerModel, states: np.ndarray) -> np.ndarray:
    """Energy in milli of each ±1 row of ``states``."""
    s = np.asarray(states, dtype=np.int64)
    edges = np.asarray(model.edges, dtype=np.int64).reshape(-1, 2)
    j = np.asarray(model.j_milli, dtype=np.int64)
    return (s[:, edges[:, 0]] * s[:, edges[:, 1]]) @ j


def find_binary() -> Optional[Path]:
    """``$QUIP_DWAVE_QPP``, else ``quip-dwave-qpp`` on ``PATH``, else the crate's release build."""
    env = os.environ.get(BINARY_ENV)
    if env:
        return Path(env)
    on_path = shutil.which(BINARY_NAME)
    if on_path:
        return Path(on_path)
    built = Path(__file__).resolve().parents[1] / "crates" / "qpp" / "target" / "release" / BINARY_NAME
    return built if built.is_file() else None


def run(
    model: DimerModel,
    seeds: np.ndarray,
    mode: str,
    *,
    mask: Optional[np.ndarray] = None,
    binary: "str | Path | None" = None,
    workdir: "str | Path | None" = None,
    stop: Optional[Callable[[], bool]] = None,
    **params: Any,
) -> QppResult:
    """Run one postprocess on ``seeds`` (``K`` rows of ``n`` ±1 spins).

    ``params`` are the binary's ``key=value`` options (``sweeps``, ``b0``,
    ``b1``, ``seed``, ``seed_index``, ``stages``, ``pop_first``,
    ``pop_count`` and the rest). ``mask`` marks the free spins. ``stop`` is
    polled while the binary runs; when it turns true the binary is killed.

    Raises:
        QppStopped: ``stop`` turned true.
        QppError: The binary failed, or its best state does not rescore on
            the host to the energy it reported.
    """
    exe = Path(binary) if binary is not None else find_binary()
    if exe is None:
        raise QppError(f"{BINARY_NAME} not found: build crates/qpp or set {BINARY_ENV}")
    states = np.asarray(seeds)
    if states.ndim != 2 or states.shape[1] != model.n:
        raise ValueError(f"seeds must be (K, {model.n}), got {states.shape}")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        d = Path(tmp)
        write_model(d / "model.bin", model)
        np.where(states > 0, 1, -1).astype(np.int8).tofile(d / "seeds.bin")
        args = [f"{k}={v}" for k, v in params.items()]
        if mask is not None:
            free = np.asarray(mask, dtype=bool)
            if free.shape != (model.n,):
                raise ValueError(f"mask must have {model.n} entries, got {free.shape}")
            free.astype(np.uint8).tofile(d / "mask.bin")
            args.append(f"mask={d / 'mask.bin'}")
        cmd = [str(exe), str(d / "model.bin"), str(d / "seeds.bin"), str(d / "out.bin"), mode, *args]
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as proc:
            while True:
                try:
                    stdout, stderr = proc.communicate(timeout=POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    if stop is not None and stop():
                        proc.kill()
                        proc.communicate()
                        raise QppStopped(f"{BINARY_NAME} stopped before it finished") from None
        if proc.returncode != 0:
            raise QppError(f"{BINARY_NAME} exited {proc.returncode}: {stderr.strip()}")
        try:
            report = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise QppError(f"{BINARY_NAME} wrote no JSON report: {exc}") from None
        state = np.fromfile(d / "out.bin", dtype=np.int8)
    if state.shape != (model.n,):
        raise QppError(f"{BINARY_NAME} returned {state.size} spins for a {model.n}-spin model")
    rescored = int(energies_milli(model, state[None])[0])
    if rescored != report["best"]:
        raise QppError(f"host rescore {rescored} differs from the reported best {report['best']}")
    known = {"mode", "start", "trace", "best", "cpu"}
    return QppResult(
        mode=report["mode"],
        start_milli=int(report["start"]),
        best_milli=int(report["best"]),
        trace=[(int(w), float(c), int(b), int(x)) for w, c, b, x in report["trace"]],
        cpu_s=float(report["cpu"]),
        state=state,
        extra={k: v for k, v in report.items() if k not in known},
    )
