"""The CPU multi-spin annealing kernel, as the miner calls it.

``quip_msa`` is the PyO3 binding built from quip-miner-cpu. It is optional:
the QPU path never needs it, and a miner without it still mines. Only the
MSA-lite and MSA-heavy stages of QUI-1387 call into this module.

The kernel packs 64 reads into one machine word, and a word costs the same
with one lane in use or with all 64. :func:`pack_lanes` fills the word: QPU
reads first, then the best MSA-lite reads of the same model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    # An optional binding: the stub in typings/ types it, and an install
    # without the compiled module is a supported state, not a defect.
    import quip_msa  # pyright: ignore[reportMissingModuleSource]

# One replica word of the kernel. More reads than this start a second word,
# which doubles the cost of the job.
LANES = 64


class MsaUnavailable(RuntimeError):
    """The ``quip_msa`` binding is not installed."""


def load_kernel() -> "quip_msa.Msa":
    """A new ``quip_msa.Msa``. Keep one per topology: it caches the colouring.

    Raises:
        MsaUnavailable: the binding is not installed.
    """
    try:
        import quip_msa  # pyright: ignore[reportMissingModuleSource]
    except ImportError as exc:
        raise MsaUnavailable(
            "quip_msa is not installed; build it from quip-miner-cpu/py "
            "with `maturin develop --release`"
        ) from exc
    return quip_msa.Msa()


def _best_first(spins: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Rows of ``spins`` by ascending energy. Stable, so ties keep their order."""
    states = np.asarray(spins, dtype=np.int8)
    scores = np.asarray(energies)
    if states.ndim != 2 or scores.shape != (states.shape[0],):
        raise ValueError(
            f"spins must be 2-D with one energy each; got spins.shape="
            f"{states.shape}, energies.shape={scores.shape}"
        )
    return states[np.argsort(scores, kind="stable")]


def pack_lanes(
    qpu_spins: np.ndarray,
    qpu_energies: np.ndarray,
    lite_spins: Optional[np.ndarray] = None,
    lite_energies: Optional[np.ndarray] = None,
    *,
    lanes: int = LANES,
) -> Tuple[np.ndarray, int]:
    """Start states for one seeded MSA-heavy job, and how many came from the QPU.

    The QPU reads go first, best first. The best MSA-lite reads of the same
    model fill the lanes that remain. A state that is already packed is
    dropped: every lane shares one acceptance threshold, so two lanes that
    start identical stay identical and the second one is wasted work.

    The fill lanes are the control inside every job. They show whether lanes
    seeded from the QPU finish lower than lanes that continued from MSA-lite.

    Returns:
        ``(states, qpu_count)``. ``states`` is int8 of shape ``(k, nodes)``
        with ``k <= lanes``, and rows ``0..qpu_count`` came from the QPU.
    """
    if (lite_spins is None) != (lite_energies is None):
        raise ValueError("lite_spins and lite_energies are passed together or not at all")

    rows = [_best_first(qpu_spins, qpu_energies)]
    if lite_spins is not None and lite_energies is not None:
        rows.append(_best_first(lite_spins, lite_energies))

    packed: list[np.ndarray] = []
    seen: set[bytes] = set()
    qpu_count = 0
    for source, block in enumerate(rows):
        for state in block:
            if len(packed) == lanes:
                break
            key = state.tobytes()
            if key in seen:
                continue
            seen.add(key)
            packed.append(state)
            if source == 0:
                qpu_count += 1
    width = np.asarray(qpu_spins).shape[1]
    states = np.vstack(packed) if packed else np.empty((0, width), dtype=np.int8)
    return states, qpu_count
