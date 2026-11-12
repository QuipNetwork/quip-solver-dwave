#!/usr/bin/env python3
"""Pull the QPU attempts from a node into one CSV.

The attempts log is one JSON object per line, in one file per qblock under
``~/nodes.quip.network/data/attempts/<qblock_id>/attempts.jsonl``. The filter
runs on the node, so only three short columns cross the network and the 20 GB
data directory stays where it is.

Kept: lines whose ``miner_type`` starts with ``QPU``. Written: ``nonce`` (the
``job_id``), ``raw_best_energy_milli``, ``qblock_id``. ``best_energy_milli`` is
a sentinel on almost every line and is never read. Repeated nonces stay in the
CSV; ``replay.load_attempts`` keeps the lowest energy of each.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

REMOTE_FILTER = r"""
import glob, json, sys
for path in sorted(glob.glob(sys.argv[1] + "/*/attempts.jsonl")):
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not str(row.get("miner_type", "")).startswith("QPU"):
                continue
            energy = row.get("raw_best_energy_milli")
            if energy is None:
                continue
            print(f'{row["job_id"]},{energy},{row.get("qblock_id", "")}')
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--host", default="qpu-1.nodes.quip.network")
    parser.add_argument("--remote-dir", default="nodes.quip.network/data/attempts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("nonce,raw_best_energy_milli,qblock_id\n")
        fh.flush()
        done = subprocess.run(
            ["ssh", args.host, "python3", "-", args.remote_dir],
            input=REMOTE_FILTER.encode(),
            stdout=fh,
            check=False,
        )
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
