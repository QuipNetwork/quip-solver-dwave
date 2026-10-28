#!/usr/bin/env python3
"""Fetch the latest qblocks of the chain and write them with their rounds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

from quip_miner_dwave import chain


def one(post: chain.Post, url: str, method: str) -> str:
    return post(url, [chain.state_call(method, "0x", 1)])[0]["result"]


def main(argv: Optional[Sequence[str]] = None, post: chain.Post = chain.http_post) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--rpc", default=chain.DEFAULT_RPC)
    parser.add_argument("--count", type=int, default=600, help="qblocks to fetch, ending at the latest")
    parser.add_argument("--topology-hash", required=True, help="64 hex characters that every qblock must carry")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    latest = chain.decode_latest_id(one(post, args.rpc, "QuantumPowApi_latest_qblock_id"))
    difficulty = chain.decode_difficulty(one(post, args.rpc, "QuantumPowApi_current_difficulty"))
    qblocks: List[chain.QBlock] = chain.fetch_qblocks(latest - args.count + 1, latest, args.rpc, post)
    foreign = [q.qblock_id for q in qblocks if q.topology_hash != args.topology_hash]
    if foreign:
        raise SystemExit(f"refusing: qblocks {foreign[:5]} use another topology than {args.topology_hash[:8]}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "qblocks.jsonl", "w", encoding="utf-8") as handle:
        for qblock in qblocks:
            handle.write(json.dumps(asdict(qblock)) + "\n")
    rows = chain.rounds(qblocks)
    summary = {"latest_qblock_id": latest, "current_difficulty": difficulty,
               "block_seconds": chain.BLOCK_SECONDS, "rounds": rows}
    (out / "rounds.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    seconds = sorted(row["blocks"] * chain.BLOCK_SECONDS for row in rows)
    median = f"{seconds[len(seconds) // 2]:.0f} s" if seconds else "none"
    print(f"{len(qblocks)} qblocks {qblocks[0].qblock_id}..{latest}; "
          f"current gate {difficulty['gate_milli']}; median round {median}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
