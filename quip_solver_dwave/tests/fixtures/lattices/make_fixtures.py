#!/usr/bin/env python3
"""Write the lattice fixtures from the authors' own generator.

The fixtures pin :mod:`quip_solver_dwave.lattices` to ``quenchlib`` of King et
al. (doi:10.5281/zenodo.14063693, CC-BY-4.0). Run this only to regenerate
them:

    curl -L -o /tmp/src.zip "https://zenodo.org/records/14063693/files/src.zip?download=1"
    echo "6ac13954d69559edd297abf75ab60167  /tmp/src.zip" | md5sum -c
    unzip -q -o /tmp/src.zip -d /tmp/quench-src
    PYTHONPATH=/tmp/quench-src/src python make_fixtures.py

The cubic dimer fixture gives bond ``k`` of the authors' cubic lattice the
value ``2 (k + 1)``, so every physical edge shows which bond made it and
whether it is a half.
"""

from __future__ import annotations

import json
from pathlib import Path

import dimod
# The authors' code is not a dependency. It is on the path only when this script runs.
from quenchlib.instances import _3ddimer, _diamond  # pyright: ignore[reportMissingImports]

HERE = Path(__file__).resolve().parent
PERIODIC = (False, False, True)

for dims in ((3, 3, 8), (4, 4, 8), (6, 6, 16)):
    edges = _diamond.generate_edges(dims, PERIODIC)
    name = "diamond_{}_{}_{}.json".format(*dims)
    (HERE / name).write_text(json.dumps({"dims": dims, "edges": [list(e) for e in edges]}) + "\n")

for dims in ((3, 2, 2), (3, 3, 3), (4, 3, 5)):
    bonds = list(_3ddimer.generate_edges(dims, PERIODIC))
    cubic = dimod.BQM(vartype="SPIN")
    cubic.add_quadratic_from({(u, v): 2 * (k + 1) for k, (u, v) in enumerate(bonds)})
    dimer = _3ddimer.make_bicubic_bqm_from_cubic_bqm(dims, cubic)
    physical = sorted([min(u, v), max(u, v), value] for (u, v), value in dimer.quadratic.items())
    name = "cubic_dimer_{}_{}_{}.json".format(*dims)
    (HERE / name).write_text(
        json.dumps({"dims": dims, "logical_bonds": [list(b) for b in bonds], "physical": physical}) + "\n"
    )
