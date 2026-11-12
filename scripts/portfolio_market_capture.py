"""Capture hourly market bars for the production portfolio universe from assets-api.

The production trading game built every race's covariance and expected return
from assets-api hourly bars. The service returns history only back from now,
capped at 2,160 hours, so a window cannot be fetched twice. This script saves
the raw response once, with its hash, request, and fetch time, and every
portfolio model built later reads that file instead of the network.

The universe is every asset that appears in the read-only production race
export. The script refuses to write a capture that is missing any ticker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quip_solver_dwave import regime_io

BASE_URL = "https://asset-tracker.quip.network"
MAX_HOURS = 2160


def production_universe(races_path: Path) -> list[str]:
    races = json.loads(races_path.read_text(encoding="utf-8"))
    return sorted({asset for race in races for asset in race["assets"]})


def bar_counts(body: dict[str, Any], tickers: Sequence[str]) -> dict[str, int]:
    """Bars per ticker. Raises when a ticker is missing or has no bars."""
    bars = body.get("bars") or {}
    missing = [ticker for ticker in tickers if not bars.get(ticker)]
    if missing:
        raise ValueError(f"assets-api returned no bars for {', '.join(missing)}")
    return {ticker: len(bars[ticker]) for ticker in tickers}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--races", type=Path, required=True, help="production race export JSON")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--hours", type=int, default=MAX_HOURS)
    args = parser.parse_args(argv)

    tickers = production_universe(args.races)
    query = urllib.parse.urlencode({"tickers": ",".join(tickers), "hours": args.hours})
    url = f"{BASE_URL}/v1/history?{query}"
    fetched_at = datetime.now(UTC).isoformat()
    with urllib.request.urlopen(url, timeout=300) as response:
        raw = response.read()
    body = json.loads(raw)
    counts = bar_counts(body, tickers)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = fetched_at[:13].replace("-", "")
    bars_path = args.out / f"bars-{stamp}.json"
    bars_path.write_bytes(raw)
    provenance = {
        "url": url,
        "fetched_at_utc": fetched_at,
        "hours": args.hours,
        "interval": body.get("interval"),
        "tickers": tickers,
        "bar_counts": counts,
        "first_bar_utc": min(bars[0]["t"] for bars in body["bars"].values()),
        "last_bar_utc": max(bars[-1]["t"] for bars in body["bars"].values()),
        "bars_file": bars_path.name,
        "bars_sha256": hashlib.sha256(raw).hexdigest(),
        "races_file": str(args.races),
        "races_sha256": hashlib.sha256(args.races.read_bytes()).hexdigest(),
    }
    regime_io.atomic_write_json(args.out / f"bars-{stamp}.provenance.json", provenance)
    print(json.dumps({k: provenance[k] for k in ("bars_file", "bars_sha256", "first_bar_utc", "last_bar_utc")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
