"""Regenerate golden.tsv from the C++ prototype the Rust binary ports.

    python3 crates/qpp/tests/fixtures/make_golden.py PATH/TO/qpp

The reference is the prototype ``quip-data/regimes/qpu-aware-pp-20261111/src/qpp.cpp``
with its one root-lookup fix (line 145, ``int r = parent[i];`` becomes
``int r = find(i);``), built with ``g++ -O3 -march=native`` (GCC 12). At 192
tiles the fix changes the ``chain`` and ``race_sibling`` cases, so the
vectors fail a port that keeps the old lookup. The model is synthetic: corner-sharing four-spin tiles
with ice-like couplings, dimer pairs, and weak random edges, so the fixture
stays small and needs no research data.

Each line of golden.tsv is ``name, args, stdout, best state hex`` separated by
tabs. The processor-time fields of stdout read ``C`` (see ``mask_clock``), the
same masking ``tests/golden.rs`` applies to the Rust output.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = {
    "chain": ["chain", "sweeps=96", "b0=6", "b1=20", "seed=3"],
    "chain_mask_k2": ["chain", "sweeps=64", "ksweeps=2", "single_every=3", "seed_index=1", "mask=MASK"],
    "race": ["race", "stages=8:16,3:32:5,1:64", "pop_first=0", "pop_count=8", "b0=6"],
    "race_sibling": ["race", "stages=8:8,1:16", "pop_first=0", "pop_count=8", "rounds=2", "children=2", "rsweeps=16"],
    "ratchet": ["ratchet", "rounds=3", "children=3", "sweeps=12", "seed=9"],
    "ratchet_nosplit": ["ratchet", "rounds=2", "children=2", "sweeps=12", "split=0"],
    "pa": ["pa", "steps=4", "sweeps=4", "pop_first=0", "pop_count=8", "replicas=12", "seed=13"],
}


def model_bytes(tiles: int = 192) -> bytes:
    """Corner-sharing tiles: every spin sits in two tiles, as in the ice cells.

    Shared spins let the tile links grow clusters across many tiles, which is
    what builds the deep union-find trees the root lookup must handle.
    """
    rng = np.random.default_rng(20261112)
    n = 2 * tiles
    while True:
        slots = rng.permutation(np.repeat(np.arange(n), 2)).reshape(tiles, 4)
        if all(len(set(row)) == 4 for row in slots.tolist()):
            break
    gauge = rng.choice(np.array([-1, 1], dtype=np.int8), (tiles, 4))
    gauge[:, 0] = 1
    j = {}
    for t, nodes in enumerate(slots.tolist()):
        for a in range(4):
            for b in range(a + 1, 4):
                j.setdefault((min(nodes[a], nodes[b]), max(nodes[a], nodes[b])), 1000 * int(gauge[t, a] * gauge[t, b]))
    spins = rng.permutation(n)
    pairs = [(int(spins[2 * k]), int(spins[2 * k + 1])) for k in range(n // 4)]
    for u, v in pairs:
        j[(min(u, v), max(u, v))] = -1000
    for _ in range(n):
        u, v = sorted(int(x) for x in rng.choice(n, 2, replace=False))
        j.setdefault((u, v), int(rng.choice([-250, 250])))
    rows = [(u, v, w) for (u, v), w in j.items()]
    out = np.array([n, tiles, len(pairs), len(rows)], dtype="<i4").tobytes()
    out += np.array(rows, dtype="<i4").tobytes() + slots.astype("<i4").tobytes()
    return out + gauge.tobytes() + np.array(pairs, dtype="<i4").tobytes()


def mask_clock(stdout: str) -> str:
    """Replace each trace entry's processor seconds, and the final ``cpu``, by ``C``."""
    stdout = re.sub(r"\[(-?\d+),[0-9.]+,(-?\d+),(-?\d+)\]", r"[\1,C,\2,\3]", stdout)
    return re.sub(r'"cpu":[0-9.]+', '"cpu":C', stdout).rstrip("\n")


def main(cpp: str) -> None:
    rng = np.random.default_rng(7)
    model = model_bytes()
    n = int(np.frombuffer(model[:4], dtype="<i4")[0])
    seeds = rng.choice(np.array([-1, 1], dtype=np.int8), (8, n))
    mask = (rng.random(n) < 0.5).astype(np.uint8)
    (HERE / "model.bin").write_bytes(model)
    (HERE / "seeds.bin").write_bytes(seeds.tobytes())
    (HERE / "mask.bin").write_bytes(mask.tobytes())
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, args in CASES.items():
            run_args = [a.replace("MASK", str(HERE / "mask.bin")) for a in args]
            out = Path(tmp) / "out.bin"
            proc = subprocess.run(
                [cpp, str(HERE / "model.bin"), str(HERE / "seeds.bin"), str(out), *run_args],
                capture_output=True, text=True, check=True,
            )
            lines.append("\t".join([name, " ".join(args), mask_clock(proc.stdout), out.read_bytes().hex()]))
    (HERE / "golden.tsv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
