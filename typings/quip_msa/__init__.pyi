"""Stub for the optional ``quip_msa`` binding (quip-miner-cpu/py).

Hand-written to match ``py/src/lib.rs``. The binding is not a dependency of
this package, so without the stub Pyright cannot resolve the import at all.
"""

from typing import Optional, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

class ResearchMetadata(TypedDict):
    requested_kernel: str
    observed_kernel: str
    representation: str
    rng_scheme: str
    seeded_reads: int
    workspace_bytes: Optional[int]

class Msa:
    def __init__(self) -> None: ...
    def sample(
        self,
        h: npt.NDArray[np.float64],
        edges: npt.NDArray[np.int64],
        j: npt.NDArray[np.float64],
        *,
        num_sweeps: int,
        num_reads: int = 64,
        seed: int = 0,
        beta_range: Optional[Tuple[float, float]] = None,
        initial_spins: Optional[npt.NDArray[np.int8]] = None,
        start_beta: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.int8], npt.NDArray[np.int64]]: ...
    def sample_research(
        self,
        h: npt.NDArray[np.float64],
        edges: npt.NDArray[np.int64],
        j: npt.NDArray[np.float64],
        *,
        kernel: str,
        num_sweeps: int,
        num_reads: int = 64,
        seed: int = 0,
        beta_range: Optional[Tuple[float, float]] = None,
        initial_spins: Optional[npt.NDArray[np.int8]] = None,
        start_beta: Optional[float] = None,
    ) -> Tuple[npt.NDArray[np.int8], npt.NDArray[np.float64], ResearchMetadata]:
        """Anneal on exactly `kernel` and report what ran.

        `kernel` is one of `cpu-sa`, `cpu-msa-unit`, `cpu-msa-f64` or
        `cpu-msa`. A model the named kernel cannot run raises `ValueError`
        instead of falling back, unlike `sample`. `cpu-msa` runs the unit
        arm when the model fits it and the float arm otherwise;
        `observed_kernel` in the returned metadata names the arm that ran.
        Returns `(spins, energies, metadata)`: int8 spins of shape
        `(num_reads, len(h))`, float64 energies of the original model of
        shape `(num_reads,)`, and a metadata dict.
        """
        ...

def draw_ising(
    nonce: bytes,
    n_nodes: int,
    n_edges: int,
    allowed_h_milli: list[int],
    allowed_j_milli: list[int],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
def default_beta_range(
    h: npt.NDArray[np.float64],
    edges: npt.NDArray[np.int64],
    j: npt.NDArray[np.float64],
) -> Tuple[float, float]: ...
