"""The step 5 arithmetic of QUI-1387: what an MSA-lite ceiling lets through.

An attempt passes a ceiling when its MSA-lite best energy is at or below it.
The low-energy set is the lowest 1% of the QPU's own recorded energies. A
ceiling is judged by four numbers: the share of all attempts that pass, the
low-energy attempts that pass, the low-energy attempts that do not, and the
false positives, which pass but are outside the low-energy set. Without the
miss count a lower ceiling always looks better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


def low_energy_set(attempts: Dict[str, int], share: float = 0.01) -> Set[str]:
    """Nonces in the lowest ``share`` of QPU energies. Ties at the cut stay in."""
    ordered = sorted(attempts.values())
    count = max(1, int(len(ordered) * share))
    cut = ordered[count - 1]
    return {nonce for nonce, energy in attempts.items() if energy <= cut}


def ceiling_for_share(lite: Dict[str, int], weights: Dict[str, float], share: float) -> int:
    """The highest ceiling whose weighted pass share does not exceed ``share``."""
    total = sum(weights[n] for n in lite)
    running = 0.0
    ceiling = min(lite.values())
    for energy in sorted(set(lite.values())):
        running += sum(weights[n] for n, e in lite.items() if e == energy)
        if running / total > share:
            break
        ceiling = energy
    return ceiling


@dataclass(frozen=True)
class FilterRow:
    sweeps: int
    ceiling_milli: int
    pass_share: float
    low_pass: int
    low_miss: int
    false_positive_estimate: float
    recall: float


def filter_row(
    sweeps: int,
    ceiling_milli: int,
    lite: Dict[str, int],
    weights: Dict[str, float],
    low: Set[str],
) -> FilterRow:
    passed = {n for n, e in lite.items() if e <= ceiling_milli}
    low_here = low & set(lite)
    low_pass = len(passed & low_here)
    total = sum(weights[n] for n in lite)
    return FilterRow(
        sweeps=sweeps,
        ceiling_milli=ceiling_milli,
        pass_share=sum(weights[n] for n in passed) / total,
        low_pass=low_pass,
        low_miss=len(low_here) - low_pass,
        false_positive_estimate=sum(weights[n] for n in passed - low_here),
        recall=low_pass / len(low_here) if low_here else 0.0,
    )


def choose_sweeps(rows: List[FilterRow], tolerance: float = 0.02) -> FilterRow:
    """The lowest sweep count whose recall is within ``tolerance`` of the longest run's."""
    ordered = sorted(rows, key=lambda r: r.sweeps)
    target = ordered[-1].recall - tolerance
    return next(r for r in ordered if r.recall >= target)
