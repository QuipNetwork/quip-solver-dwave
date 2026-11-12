#!/usr/bin/env python3
"""The step 5 tables of QUI-1387, and the two nonce lists for step 6.

Table mode (the default) reads the sampled MSA-lite data set and prints, for
each sweep count and each pass share, the ceiling, the low-energy attempts
that pass and miss, the estimated false positives, and the recall. It then
names the sweep count the choice rule picks.

List mode (``--full``) reads a data set that covers every nonce at one sweep
count, applies one ceiling, and writes the low-energy set and the
false-positive set as nonce lists.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict

from quip_solver_dwave import replay
from quip_solver_dwave.lite_filter import (
    ceiling_for_share,
    choose_sweeps,
    filter_row,
    low_energy_set,
)

SHARES = (0.22, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002)


def read_lite(path: str) -> Dict[int, Dict[str, int]]:
    by_sweeps: Dict[int, Dict[str, int]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_sweeps[int(row["sweeps"])][row["nonce"]] = int(row["lite_best_milli"])
    return by_sweeps


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--lite", required=True)
    parser.add_argument("--choice-share", type=float, default=0.05)
    parser.add_argument("--full", action="store_true", help="the data set covers every nonce")
    parser.add_argument("--sweeps", type=int, help="list mode: the sweep count to cut")
    parser.add_argument("--ceiling-milli", type=int, help="list mode: the ceiling")
    parser.add_argument("--out-dir", help="list mode: where the nonce lists go")
    parser.add_argument("--table-out", help="table mode: also write the table as CSV")
    args = parser.parse_args()

    attempts = replay.load_attempts(args.attempts)
    low = low_energy_set(attempts)
    by_sweeps = read_lite(args.lite)
    print(f"attempts {len(attempts)}  low-energy set {len(low)}  QPU cut {max(attempts[n] for n in low)}")

    if args.full:
        lite = by_sweeps[args.sweeps]
        missing = set(attempts) - set(lite)
        if missing:
            example = sorted(missing)[0]
            print(
                f"the data set lacks {len(missing)} nonces at {args.sweeps} sweeps, "
                f"e.g. {example}", file=sys.stderr,
            )
            return 1
        passed = {n for n, e in lite.items() if e <= args.ceiling_milli}
        false_positive = sorted(passed - low)
        with open(f"{args.out_dir}/low-energy.nonces", "w", encoding="utf-8") as fh:
            fh.write("\n".join(sorted(low)) + "\n")
        with open(f"{args.out_dir}/false-positive.nonces", "w", encoding="utf-8") as fh:
            fh.write("\n".join(false_positive) + "\n")
        print(
            f"ceiling {args.ceiling_milli}: pass {len(passed)} ({len(passed) / len(lite):.3%})  "
            f"low pass {len(passed & low)}  low miss {len(low - passed)}  false positive {len(false_positive)}"
        )
        return 0

    rows = []
    chosen_rows = []
    print(f"{'sweeps':>6} {'share':>6} {'ceiling':>10} {'low pass':>9} {'low miss':>9} {'false pos':>10} {'recall':>7}")
    for sweeps in sorted(by_sweeps):
        lite = by_sweeps[sweeps]
        others = [n for n in lite if n not in low]
        weight = (len(attempts) - len(low)) / len(others)
        weights = {n: (1.0 if n in low else weight) for n in lite}
        for share in SHARES:
            row = filter_row(sweeps, ceiling_for_share(lite, weights, share), lite, weights, low)
            rows.append((share, row))
            if share == args.choice_share:
                chosen_rows.append(row)
            print(
                f"{sweeps:6d} {share:6.1%} {row.ceiling_milli:10d} {row.low_pass:9d} "
                f"{row.low_miss:9d} {row.false_positive_estimate:10.0f} {row.recall:7.1%}"
            )
    chosen = choose_sweeps(chosen_rows)
    print(
        f"choice at a {args.choice_share:.0%} pass share: {chosen.sweeps} sweeps, "
        f"ceiling {chosen.ceiling_milli}, recall {chosen.recall:.1%}"
    )
    if args.table_out:
        with open(args.table_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sweeps", "target_share", "ceiling_milli", "pass_share", "low_pass", "low_miss", "false_positive_estimate", "recall"])
            for share, row in rows:
                writer.writerow([row.sweeps, share, row.ceiling_milli, f"{row.pass_share:.5f}", row.low_pass, row.low_miss, f"{row.false_positive_estimate:.0f}", f"{row.recall:.4f}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
