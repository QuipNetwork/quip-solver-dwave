"""Decisions of the hybrid pipeline, with no device in them.

The screen hands over survivors. This module decides which deep legs a model
gets after the QPU has sampled it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from quip_miner_dwave import msa

MILLI = 1000


@dataclass
class Survivor:
    """A model that passed the screen. Energies are whole energy units."""

    nonce: str
    energy: int
    states: np.ndarray  # int8 [lanes, nodes], the screen's final states
    energies: np.ndarray  # int [lanes]
    found_s: float = 0.0


@dataclass(frozen=True)
class DeepPlan:
    """One deep leg to run. ``states`` seeds lanes ``0..len(states)``; ``None`` is a cold leg."""

    unit_id: str
    arm: str
    nonce: str
    program: str
    seed: int
    priority: int
    states: Optional[np.ndarray] = field(default=None, compare=False)
    qpu_lanes: int = 0


def unit_seed(nonce: str, salt: int) -> int:
    """A kernel seed that is stable for a nonce and differs by ``salt``."""
    return (int(nonce[:15], 16) * 1_000_003 + salt) & 0x7FFF_FFFF_FFFF_FFFF


def qpu_plans(
    survivor: Survivor,
    qpu_spins: np.ndarray,
    qpu_energies_milli: np.ndarray,
    *,
    qpu_gate_milli: int,
    arms: Sequence[str],
    programs: Dict[str, str],
    lanes: int,
    rep: int = 0,
) -> List[DeepPlan]:
    """Deep legs for a survivor the QPU sampled. Empty when the QPU gate drops it.

    ``programs`` names the engine program of each role: ``seeded`` and ``cold``.
    A unit id is the nonce, the program, and ``rep``, so one model can run
    several programs and several independent repeats of one program.

    The seeded leg takes the best ``lanes // 2`` distinct QPU reads first and
    fills the other lanes with the screen's own states. The fill lanes are the
    control inside every job.
    """
    best_milli = int(np.min(qpu_energies_milli))
    if best_milli > qpu_gate_milli:
        return []
    plans: List[DeepPlan] = []
    if "qpu-seeded" in arms:
        order = np.argsort(qpu_energies_milli, kind="stable")[: lanes // 2]
        states, qpu_count = msa.pack_lanes(
            qpu_spins[order], np.asarray(qpu_energies_milli)[order], survivor.states, survivor.energies, lanes=lanes
        )
        plans.append(
            DeepPlan(f"{survivor.nonce}:{programs['seeded']}:{rep}", "qpu-seeded", survivor.nonce,
                     programs["seeded"], unit_seed(survivor.nonce, 10 + rep), best_milli // MILLI, states, qpu_count)
        )
    if "qpu-cold" in arms:
        plans.append(cold_plan(survivor.nonce, programs, best_milli // MILLI, rep))
    return plans


def cold_plan(nonce: str, programs: Dict[str, str], priority: int, rep: int = 0) -> DeepPlan:
    """The cold deep leg of a model. Every arm that wants it shares one unit id, so it runs once."""
    return DeepPlan(f"{nonce}:{programs['cold']}:{rep}", "cold", nonce, programs["cold"],
                    unit_seed(nonce, 1000 + rep), priority)
