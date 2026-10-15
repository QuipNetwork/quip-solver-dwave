"""Read a finished job's answer without building a dimod.SampleSet.

``OceanSampler`` needs five things off an answer: spins, variable labels,
energies, occurrence counts and timing. A ``dwave.cloud.computation.Future``
carries all five already decoded.

Asking for ``.sampleset`` instead costs about 38.8 ms per job at production
size, and every millisecond of it is spent undoing work. The decoder turns its
numpy arrays into Python lists, ``wait_sampleset`` walks those with a nested
comprehension over reads times variables, and dimod converts the result back
into numpy. Nothing downstream wants a SampleSet; this backend reads the
arrays and writes them to the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class AnswerView:
    """One finished job's answer, in the form the miner actually uses.

    ``spins`` is ``(reads, len(variables))`` of int8 normalised to +1/-1, and
    ``variables[i]`` is the qubit label of column ``i``.
    """

    spins: np.ndarray
    variables: List[int]
    energies: List[float]
    reads: int
    access_time_us: int


def _access_time_us(timing: Any) -> int:
    """Programming plus sampling, which is what D-Wave bills."""
    timing = timing or {}
    prog = timing.get("qpu_programming_time") or 0
    sample = timing.get("qpu_sampling_time") or 0
    return int(prog) + int(sample)


def _normalise(samples: Any) -> np.ndarray:
    """Coerce to int8 +1/-1. Offline samplers can hand back 0/1."""
    arr = np.asarray(samples)
    return np.where(arr >= 1, 1, -1).astype(np.int8)


def answer_view(raw: Any) -> AnswerView:
    """Read a cloud ``Future`` or a ``dimod.SampleSet`` into an AnswerView.

    Never touches ``Future.sampleset``. ``Future.variables`` returns the
    sampleset's variables when one already exists, so building one anywhere
    on this path puts back the cost this function exists to avoid.
    """
    record = getattr(raw, "record", None)
    if record is not None:
        # A dimod.SampleSet: the mock and injected-sampler paths.
        samples = record.sample
        energies = record.energy
        occurrences = getattr(record, "num_occurrences", None)
        variables = list(raw.variables)
        timing = (getattr(raw, "info", None) or {}).get("timing")
    else:
        # A cloud Future, read before anything builds a SampleSet.
        samples = np.asarray(raw.samples)
        energies = raw.energies
        occurrences = raw.num_occurrences
        variables = list(raw.variables)
        timing = raw.timing
        # The decoder pads the solution matrix out to the solver's full
        # physical qubit count and writes the decoded bits into only the
        # active columns (``solutions[:, active_variables] = bits``), while
        # ``variables`` carries the active labels alone. Those labels are the
        # column indices into the padded matrix, so selecting by them both
        # drops the padding and restores the 1:1 pairing this view promises.
        if samples.ndim == 2 and samples.shape[1] != len(variables):
            samples = samples[:, variables]
        elif samples.ndim == 1 and samples.size == 0:
            # Zero solutions: np.asarray([]) is 1-D and skips the guard above,
            # but every downstream consumer expects (reads, len(variables)).
            samples = samples.reshape(0, len(variables))

    spins = _normalise(samples)
    energy_list = [float(e) for e in energies]
    # The cloud client folds identical reads into one row carrying
    # num_occurrences, so the row count is distinct solutions rather than
    # anneals performed. Offline samplers do not aggregate, where the sum
    # degrades to the row count anyway.
    reads = (
        int(sum(int(o) for o in occurrences))
        if occurrences is not None
        else len(energy_list)
    )
    return AnswerView(
        spins=spins,
        variables=[int(v) for v in variables],
        energies=energy_list,
        reads=reads,
        access_time_us=_access_time_us(timing),
    )
