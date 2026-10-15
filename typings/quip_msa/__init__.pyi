"""Stub for the optional ``quip_msa`` binding (quip-solver-cpu/py).

Hand-written to match ``py/src/lib.rs``. The binding is not a dependency of
this package, so without the stub Pyright cannot resolve the import at all.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

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
