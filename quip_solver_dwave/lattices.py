"""Lattice graphs of King et al., "Beyond-classical computation in quantum simulation".

The site numbering and the bond order follow the authors' ``quenchlib``
(doi:10.5281/zenodo.14063693, CC-BY-4.0), so an edge list built here compares
equal to theirs. Only the graphs come from that code. The couplings come from
the protocol draw (see :mod:`quip_solver_dwave.regimes`).

Two lattices:

``diamond``      one spin for each site, degree 4 or less, open in x and y,
                 periodic in z.
``cubic dimer``  two spins for each site of a simple cubic lattice, tied by a
                 strong ferromagnetic dimer. Open in x and y, periodic in z
                 when ``Lz > 2``.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

import numpy as np

Dims = Tuple[int, int, int]

# Column (x, y) of the diamond lattice holds one site in each block of four
# levels, at level 4 zc + _COLUMN_LEVEL[x mod 2][y mod 2]. Read the other way,
# level z holds the columns with (x mod 2, y mod 2) equal to entry z mod 4 of
# [(0, 1), (0, 0), (1, 0), (1, 1)].
_COLUMN_LEVEL = [[1, 0], [2, 3]]

DIMER_MILLI = -2000


def diamond_site_count(dims: Dims) -> int:
    lx, ly, lz = dims
    return lx * ly * lz // 4


def diamond_sites(dims: Dims) -> np.ndarray:
    """``(x, y, level)`` of each site, shape ``(n, 3)``, in site-number order."""
    lx, ly, lz = dims
    sites = [
        (x, y, 4 * zc + _COLUMN_LEVEL[x % 2][y % 2])
        for y, zc, x in itertools.product(range(ly), range(lz // 4), range(lx))
    ]
    return np.asarray(sites, dtype=np.int64).reshape(-1, 3)


def _diamond_site_numbers(dims: Dims) -> Dict[Tuple[int, int, int], int]:
    return {(x, y, level): i for i, (x, y, level) in enumerate(diamond_sites(dims).tolist())}


def diamond_edges(dims: Dims) -> np.ndarray:
    """Bonds of the diamond lattice, shape ``(m, 2)``, sorted by site pair."""
    lz = dims[2]
    if lz % 4 != 0 or lz < 4:
        raise ValueError(f"Lz must be a multiple of 4; got {lz}")
    numbers = _diamond_site_numbers(dims)
    edges: List[Tuple[int, int]] = []
    for (x, y, z), u in numbers.items():
        d = z & 1
        for dx, dy in ((d, 1 - d), (-d, d - 1)):
            v = numbers.get((x + dx, y + dy, (z + 1) % lz))
            if v is not None:
                edges.append((min(u, v), max(u, v)))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)


def cubic_logical_bonds(dims: Dims) -> Tuple[np.ndarray, np.ndarray]:
    """Bonds of the simple cubic lattice and the axis of each one.

    Returns ``(bonds, axis)``: ``bonds`` is ``(m, 2)`` site pairs in draw
    order (all x bonds, then y, then z, each group looping z, y, x), and
    ``axis`` holds 0, 1, or 2.
    """
    lx, ly, lz = dims
    if min(lx, ly) < 2:
        raise ValueError(f"Lx and Ly must be at least 2; got {dims}")

    def site(x: int, y: int, z: int) -> int:
        return x + y * lx + z * lx * ly

    bonds: List[Tuple[int, int]] = []
    axis: List[int] = []
    cells = list(itertools.product(range(lz), range(ly), range(lx)))
    for z, y, x in cells:
        if x < lx - 1:
            bonds.append((site(x, y, z), site(x + 1, y, z)))
            axis.append(0)
    for z, y, x in cells:
        if y < ly - 1:
            bonds.append((site(x, y, z), site(x, y + 1, z)))
            axis.append(1)
    for z, y, x in cells:
        if z < lz - 1:
            bonds.append((site(x, y, z), site(x, y, z + 1)))
            axis.append(2)
        elif lz > 2:
            bonds.append((site(x, y, z), site(x, y, 0)))
            axis.append(2)
    return np.asarray(bonds, dtype=np.int64).reshape(-1, 2), np.asarray(axis, dtype=np.int64)


def cubic_dimer_edges(dims: Dims, bond_milli: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True edges and couplings of the cubic dimer lattice, in milli units.

    Site ``u`` becomes spin ``u`` and spin ``u + n``. The x bonds join the
    first spins, the y bonds join the second spins, and a z bond splits into
    two half-strength bonds that cross between them. ``bond_milli`` holds one
    drawn coupling for each bond of :func:`cubic_logical_bonds` and must be
    even, so the halves stay whole numbers.

    Returns ``(edges, j_milli, bond)`` with the dimers first. ``bond[e]`` is
    the logical bond that made edge ``e``, or -1 for a dimer.
    """
    lx, ly, lz = dims
    n = lx * ly * lz
    bonds, axis = cubic_logical_bonds(dims)
    drawn = np.asarray(bond_milli, dtype=np.int64)
    if drawn.shape != (len(bonds),):
        raise ValueError(f"need one coupling for each of {len(bonds)} bonds; got {drawn.shape}")
    if np.any(drawn % 2):
        raise ValueError("drawn couplings must be even so that a z half is a whole number")
    edges: List[Tuple[int, int]] = [(u, u + n) for u in range(n)]
    couplings: List[int] = [DIMER_MILLI] * n
    made_by: List[int] = [-1] * n
    for index, ((u, v), ax, j) in enumerate(zip(bonds.tolist(), axis.tolist(), drawn.tolist())):
        if ax == 0:
            edges.append((u, v))
            couplings.append(j)
            made_by.append(index)
        elif ax == 1:
            edges.append((u + n, v + n))
            couplings.append(j)
            made_by.append(index)
        else:
            edges += [(u, v + n), (u + n, v)]
            couplings += [j // 2, j // 2]
            made_by += [index, index]
    return (
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        np.asarray(couplings, dtype=np.int64),
        np.asarray(made_by, dtype=np.int64),
    )
