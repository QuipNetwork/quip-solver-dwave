"""Seed and refresh round history from the coordinator's attempts files.

The coordinator appends one JSON line per validated result under
``<data_dir>/<qblock_id>/attempts.jsonl`` (quip-coordinator ``attempt.rs``).
Those lines carry what the miner cannot see on the wire: whether the
coordinator accepted the attempt against the decayed threshold, and whether
it won the qblock. They also reach back before this history existed, so a
first start inherits every past round.

A directory is complete once a higher-numbered one exists. Complete
directories seed once, tracked in ``seeded_dirs``. The two newest are
re-read at every boundary for outcomes only, because a win confirms
shortly after the Cancel it causes.

A directory the miner was live for — a generation the live recorder
opened, or an hour within the directory's own span that a live hourly row
covers — only contributes outcomes and any round row a live Cancel never
opened; its margins and hourly sums are already in the live tables and
would double count if seeded again.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from quip_solver_dwave.history import (
    AttemptRoundSummary,
    HistoryStore,
    day_floor,
    margin_unit,
    split_by_hour,
)
from quip_solver_dwave.usage import hour_floor

logger = logging.getLogger(__name__)

ATTEMPTS_FILE = "attempts.jsonl"
DEFAULT_DIR_NAME = "attempts"
# A gap longer than this between two completions means the miner was parked,
# not waiting on the QPU; busy time inferred from attempts stops there.
BUSY_GAP_S = 30.0
# The live round plus the one before it: a win lands in the previous round's
# file after the Cancel that it caused.
NEWEST_DIRS_FOR_OUTCOMES = 2


def default_attempts_dir(usage_db_path: str) -> str:
    """``attempts`` beside the usage database.

    The deployment renders both under one data directory (``/data`` in
    Docker, the node manager's data directory natively), so the ledger's
    directory is the one place the coordinator's files are known to be.
    """
    return str(Path(usage_db_path).parent / DEFAULT_DIR_NAME)


@dataclass(frozen=True)
class Attempt:
    """The fields of one attempts.jsonl line this module uses."""

    ts_ms: int
    generation: int
    miner_type: str
    miner_id: str
    raw_best_energy_milli: int
    threshold_milli: int
    accepted: bool
    won: bool
    device_access_time_us: int

    @property
    def is_qpu(self) -> bool:
        return self.miner_type.upper().startswith("QPU")


def parse_attempt(line: str) -> Optional[Attempt]:
    """One JSON line to an Attempt, or None for anything unreadable."""
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    try:
        return Attempt(
            ts_ms=int(obj["ts_ms"]),
            generation=int(obj["generation"]),
            miner_type=str(obj.get("miner_type") or ""),
            miner_id=str(obj.get("miner_id") or ""),
            # best_energy_milli is i64::MAX when no row cleared the gate; the
            # raw field is the best the miner found either way.
            raw_best_energy_milli=int(obj["raw_best_energy_milli"]),
            threshold_milli=int(obj["threshold_milli"]),
            accepted=bool(obj.get("accepted", False)),
            won=obj.get("chain_block_number") is not None,
            device_access_time_us=int(obj.get("device_access_time_us") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return None


def read_attempts(path: Path, *, miner_id: Optional[str] = None) -> Tuple[List[Attempt], int]:
    """QPU attempts in one file, and how many lines the file had.

    ``miner_id``, given, drops lines from any other miner the same way a
    non-QPU line is dropped: the coordinator writes one file per qblock for
    every miner it sees, and a node running more than one QPU miner would
    otherwise absorb a sibling miner's jobs into this one's history.
    """
    attempts: List[Attempt] = []
    lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            lines += 1
            att = parse_attempt(line)
            if att is None or not att.is_qpu:
                continue
            if miner_id is not None and att.miner_id != miner_id:
                continue
            attempts.append(att)
    return attempts, lines


def summarise_rounds(attempts: Iterable[Attempt]) -> List[AttemptRoundSummary]:
    """Group attempts by generation, oldest generation first."""
    groups: Dict[int, List[Attempt]] = {}
    for att in attempts:
        groups.setdefault(att.generation, []).append(att)
    out: List[AttemptRoundSummary] = []
    for generation, atts in sorted(groups.items()):
        atts.sort(key=lambda a: a.ts_ms)
        margins: Dict[Tuple[int, int], int] = {}
        for a in atts:
            # Each attempt against the threshold in force when it ran: the
            # threshold steps within a round as difficulty decays.
            key = (day_floor(a.ts_ms / 1000.0), margin_unit(a.raw_best_energy_milli, a.threshold_milli))
            margins[key] = margins.get(key, 0) + 1
        out.append(
            AttemptRoundSummary(
                generation=generation,
                first_ts_s=atts[0].ts_ms // 1000,
                last_ts_s=atts[-1].ts_ms // 1000,
                jobs=len(atts),
                hits_coord=sum(1 for a in atts if a.accepted),
                won=any(a.won for a in atts),
                best_energy_milli=min(a.raw_best_energy_milli for a in atts),
                threshold_milli=atts[0].threshold_milli,
                access_us=sum(a.device_access_time_us for a in atts),
                margins=margins,
            )
        )
    return out


def hourly_from_attempts(
    attempts: Iterable[Attempt], *, before_hour: int
) -> Dict[int, Tuple[int, int, int]]:
    """Approximate hour rows: ``hour -> (jobs, busy_ms, access_us)``.

    Busy time is the sum of gaps between consecutive completions no longer
    than BUSY_GAP_S, split at hour boundaries. Hours at or after
    ``before_hour`` belong to the live recorder and are left out.
    """
    rows: Dict[int, List[int]] = {}
    prev: Optional[float] = None
    for a in sorted(attempts, key=lambda a: a.ts_ms):
        t = a.ts_ms / 1000.0
        hour = hour_floor(t)
        if hour < before_hour:
            row = rows.setdefault(hour, [0, 0, 0])
            row[0] += 1
            row[2] += a.device_access_time_us
        if prev is not None and 0.0 < t - prev <= BUSY_GAP_S:
            for piece_hour, ms in split_by_hour(prev, t):
                if piece_hour < before_hour:
                    rows.setdefault(piece_hour, [0, 0, 0])[1] += ms
        prev = t
    return {hour: (r[0], r[1], r[2]) for hour, r in rows.items()}


def _numbered_dirs(attempts_dir: Path) -> List[Path]:
    """Round directories in ascending qblock order. ``pending`` is skipped."""
    if not attempts_dir.is_dir():
        return []
    dirs = [p for p in attempts_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


@dataclass
class SeedReport:
    dirs_seeded: int = 0
    rounds_inserted: int = 0
    rounds_updated: int = 0
    dirs_skipped: int = 0


def seed_from_attempts(
    store: HistoryStore,
    attempts_dir: str,
    now: float,
    *,
    stop: Optional[Callable[[], bool]] = None,
    miner_id: Optional[str] = None,
) -> SeedReport:
    """Seed every complete, not-yet-seeded directory. Safe on every start.

    A node with a thousand-plus attempt directories can take tens of seconds
    to seed; ``stop`` is checked once per directory, before that directory's
    writes begin. One directory's round applications, margin writes, hourly
    writes and ``mark_seeded`` share one transaction (``store.batch()``), so
    a shutdown mid-seed, a hard kill, or any other failure partway through a
    directory never leaves it with margins written but not marked seeded —
    that would double-count them on the next start, since ``seed_margin`` is
    additive.

    ``miner_id``, given, restricts every directory to this miner's own
    lines: the coordinator writes one attempts file per qblock for every
    miner it sees.
    """
    report = SeedReport()
    dirs = _numbered_dirs(Path(attempts_dir))
    # The newest directory is the round in progress; it is picked up later.
    for d in dirs[:-1]:
        if stop is not None and stop():
            break
        if store.is_seeded(d.name):
            continue
        path = d / ATTEMPTS_FILE
        if not path.is_file():
            store.mark_seeded(d.name)
            continue
        try:
            attempts, _ = read_attempts(path, miner_id=miner_id)
        except OSError as exc:
            logger.warning("attempts: cannot read %s: %s", path, exc)
            report.dirs_skipped += 1
            continue
        rounds = summarise_rounds(attempts)
        # A generation the live recorder already opened means this process
        # (or a previous run of it) recorded the round as it happened. A
        # live hourly row inside the directory's own span covers the case
        # where the miner was running but never opened a round for one of
        # its generations (a restart mid-qblock, before the recorder saw
        # the Cancel that would have opened it). Either way, only the
        # outcomes and any missing round row are new: margins and hourly
        # sums are already in the live tables.
        covered = any(
            store.find_live_round(r.generation, r.first_ts_s) is not None for r in rounds
        )
        if not covered and attempts:
            first_hour = hour_floor(min(a.ts_ms for a in attempts) / 1000.0)
            last_hour = hour_floor(max(a.ts_ms for a in attempts) / 1000.0)
            covered = store.has_live_hours(first_hour, last_hour)
        with store.batch():
            for r in rounds:
                # A round row for a generation the live recorder never
                # opened is not double counting even when the directory is
                # otherwise covered; only margins and hourly rows are.
                outcome = store.apply_attempt_round(r, insert_missing=True)
                if outcome == "inserted":
                    report.rounds_inserted += 1
                elif outcome == "updated":
                    report.rounds_updated += 1
            if not covered:
                for r in rounds:
                    for (day, margin), jobs in r.margins.items():
                        store.seed_margin(day, margin, jobs)
                hourly = hourly_from_attempts(attempts, before_hour=hour_floor(now))
                for hour, (jobs, busy_ms, access_us) in hourly.items():
                    store.seed_hourly(hour, jobs=jobs, busy_ms=busy_ms, access_us=access_us)
            store.mark_seeded(d.name)
        report.dirs_seeded += 1
    if report.dirs_seeded:
        logger.info(
            "[QPU] history seeded from %d attempts directories: %d rounds added, "
            "%d live rounds given their outcome",
            report.dirs_seeded,
            report.rounds_inserted,
            report.rounds_updated,
        )
    return report


def pickup_outcomes(
    store: HistoryStore, attempts_dir: str, *, miner_id: Optional[str] = None
) -> int:
    """Apply accepted and won outcomes from the newest directories to live rounds."""
    updated = 0
    for d in _numbered_dirs(Path(attempts_dir))[-NEWEST_DIRS_FOR_OUTCOMES:]:
        path = d / ATTEMPTS_FILE
        if not path.is_file():
            continue
        try:
            attempts, _ = read_attempts(path, miner_id=miner_id)
        except OSError as exc:
            logger.warning("attempts: cannot read %s: %s", path, exc)
            continue
        for r in summarise_rounds(attempts):
            if store.apply_attempt_round(r, insert_missing=False) == "updated":
                updated += 1
    return updated
