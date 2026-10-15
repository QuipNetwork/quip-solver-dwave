#!/usr/bin/env python3
"""Submit recorded models to the QPU again and keep every read: step 7 input.

The attempts log holds no spins, so the seed test needs fresh reads. Each
model goes through the miner's own ``OceanSampler``, so the defect reduction,
the anneal schedule, and the reconstruction of clamped qubits are the
production code. One ``.npz`` file per model holds the reads in spec node
order, with energies recomputed on the full model in milli units.

``--dry-run`` prints the model count and the estimated spend and submits
nothing. A real run needs ``--yes``. The spend does not pass through the
miner's usage ledger, so it is printed at the end for the operator to record.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import Any, List

import numpy as np

from quip_solver_dwave import replay
from quip_solver_dwave.ocean import OceanSampler

EST_ACCESS_US = 46_100


def atomic_savez_compressed(path: str, **arrays: Any) -> None:
    """Write an npz atomically: temp file in the same directory, then replace.

    ``np.savez_compressed`` appends ``.npz`` to a path that lacks it, so the
    temp name must already end in ``.npz`` to land at the right final name
    after ``os.replace``. A kill mid-write leaves only the temp file behind,
    which the resume check (looking for ``<nonce>.npz``) does not see.
    """
    tmp_path = f"{path}.{os.getpid()}.tmp.npz"
    try:
        with open(tmp_path, "wb") as fh:
            np.savez_compressed(fh, **arrays)  # pyright: ignore[reportArgumentType]
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def check_budget(model_count: int, max_qpu_seconds: float) -> None:
    """Refuse to run if the estimated spend for the remaining models is too high."""
    estimated_seconds = model_count * EST_ACCESS_US / 1e6
    if estimated_seconds > max_qpu_seconds:
        raise SystemExit(
            f"refusing to run: {model_count} models would cost about "
            f"{estimated_seconds:.1f} s of QPU access, over the "
            f"--max-qpu-seconds limit of {max_qpu_seconds:.1f} s"
        )


def pick(heavy_csv: str, per_set: int, seed: int) -> List[str]:
    by_set: dict = {}
    with open(heavy_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_set.setdefault(row["set"], []).append(row["nonce"])
    rng = random.Random(seed)
    chosen: List[str] = []
    for name in sorted(by_set):
        nonces = sorted(by_set[name])
        chosen += rng.sample(nonces, min(per_set, len(nonces)))
    return chosen


def energies_milli(spins: np.ndarray, dense_edges: np.ndarray, h: np.ndarray, j: np.ndarray) -> np.ndarray:
    s = spins.astype(np.int64)
    field = s @ np.rint(h * 1000).astype(np.int64)
    coupling = (s[:, dense_edges[:, 0]] * s[:, dense_edges[:, 1]]) @ np.rint(j * 1000).astype(np.int64)
    return field + coupling


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--heavy", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--per-set", type=int, default=250)
    parser.add_argument("--reads", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1387)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="confirm the QPU spend")
    parser.add_argument(
        "--max-qpu-seconds", type=float, default=30.0,
        help="refuse to run if the estimated spend for the remaining models exceeds this",
    )
    args = parser.parse_args()

    spec = replay.load_spec(args.spec)
    os.makedirs(args.out_dir, exist_ok=True)
    nonces = [n for n in pick(args.heavy, args.per_set, args.seed)
              if not os.path.exists(f"{args.out_dir}/{n}.npz")]
    print(f"{len(nonces)} models to capture, about {len(nonces) * EST_ACCESS_US / 1e6:.1f} s of QPU access")
    if args.dry_run:
        return 0
    check_budget(len(nonces), args.max_qpu_seconds)
    if not args.yes:
        print("pass --yes to spend it")
        return 2

    sampler = OceanSampler(mock=False)
    sampler.ensure_connected()
    sampler.set_session_topology(
        [int(n) for n in spec.nodes],
        [(int(u), int(v)) for u, v in spec.edges],
    )
    spent_us = 0
    skipped = 0
    try:
        for count, nonce in enumerate(nonces, 1):
            h, j = replay.model_from_nonce(spec, nonce)
            result = sampler.sample(
                spec.nodes, h, spec.edges, j,
                num_reads=args.reads, nonce_seed=bytes.fromhex(nonce), label="quip-seed-capture",
            )
            try:
                spins = replay.spec_order(result.spins, result.variables, spec)
            except ValueError as exc:
                print(f"skipping nonce {nonce}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            # The sampler's own energies are scored against the (possibly
            # defect-reduced) problem it actually ran; recompute here from the
            # spins on the full spec model so the values are comparable across
            # runs regardless of which qubits were clamped.
            atomic_savez_compressed(
                f"{args.out_dir}/{nonce}.npz",
                spins=spins,
                energies_milli=energies_milli(spins, spec.dense_edges, h, j),
                access_us=np.int64(result.device_access_time_us),
            )
            spent_us += int(result.device_access_time_us)
            if count % 50 == 0:
                print(f"{count} of {len(nonces)}  spent {spent_us / 1e6:.1f} s")
            if spent_us / 1e6 >= args.max_qpu_seconds:
                print(
                    f"stopping: {count} of {len(nonces)} models done, spent "
                    f"{spent_us / 1e6:.1f} s, at the --max-qpu-seconds cap of "
                    f"{args.max_qpu_seconds:.1f} s",
                    file=sys.stderr,
                )
                return 3
    finally:
        close = getattr(sampler, "close", None)
        if close:
            close()
    captured = len(nonces) - skipped
    print(f"captured {captured} models, skipped {skipped}, QPU access spent {spent_us / 1e6:.2f} s")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
