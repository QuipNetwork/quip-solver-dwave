"""QPU budget with even distribution across the quota period.

Replaces the v0.2/v0.3 daily reservoir. That model accumulated credit and then
burnt it in one continuous run, which left the miner dark for a contiguous
multi-hour block once the pool drained — long enough for the rest of the field
to close the gap while the QPU sat idle.

This model paces instead of bursting. The allowance at any instant is the flat
share of the budget that the elapsed part of the period has earned::

    allowance = budget * (now - period_start) / (period_end - period_start)

Mining is allowed while cumulative spend sits under that line. Spend comes from
:mod:`quip_solver_dwave.usage`, so it survives a restart, and the period follows
D-Wave's own quota semantics: a fixed day of the month, UTC, clamped to the
month's length.

The line is consulted at qblock boundaries only: a joined round runs to its
end, and the miner then sits out boundaries until the rising line has covered
what that round cost. So the miner idles for a handful of qblocks at a time
rather than one night at a time. Mid-round, the one thing that stops it is the
period's whole allotment being spent.
"""

from __future__ import annotations

import calendar
import logging
import threading
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from quip_solver_dwave.config import warn_unknown_fields
from quip_solver_dwave.usage import UsageLedger

logger = logging.getLogger(__name__)

# Where the usage ledger lives when the coordinator does not name a path. The
# deployment already mounts /data for config.toml and the attempts dashboard,
# so it is the one directory known to outlive the container.
DEFAULT_USAGE_DB = "/data/qpu-usage.db"

# How long a pacer may serve this period's spend from memory before consulting
# the durable record again. The per-job participation check runs on the
# session-loop thread under the dispatch lock, so a SQLite read there stalls
# Cancel, Job and Ping handling; at the chip's ceiling of ~23 jobs/s this turns
# 23 reads a second into one every few seconds. The window is what another
# writer sharing usage_db can add without this pacer seeing it, which at that
# same ceiling is about a second of QPU time against a monthly quota.
SPEND_REFRESH_S = 5.0

# Config keys the dwave backend recognizes in Configure.backend_toml. Anything
# else (outside SESSION_KEYS) is a typo and gets warned about, uniform with the
# Rust backends' unknown-field handling. Connection credentials (token/solver/
# region) are NOT here: they come from D-Wave's own config (dwave.conf + env),
# not the coordinator.
DWAVE_CONFIG_KEYS = frozenset(
    {
        "budget",
        "budget_seconds",
        "budget_reset_day",
        "usage_db",
        "anneal_time_us",
        "num_reads",
        "reversal_s_milli",
        "reversal_pause_us",
        "queue_depth",
        "min_throughput_advantage",
        "participation_chance",
    }
)


class BudgetUnavailable(Exception):
    """The budget cannot be enforced, so the miner must not mine.

    Raised for an invalid quota shape or a ledger that will not open. Both mean
    the same thing operationally: spend would go unmetered.
    """


def warn_unknown_backend_keys(toml_text: str) -> None:
    """Parse ``Configure.backend_toml`` and warn on keys outside the dwave
    schema, matching the Rust backends' unknown-field warnings."""
    if not toml_text or not toml_text.strip():
        return
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return
    warn_unknown_fields("dwave", data.keys(), DWAVE_CONFIG_KEYS)


def parse_duration(duration_str: str) -> float:
    """Parse a duration string (``30s``, ``5m``, ``2h``, ``1d``, ``1w``) to seconds."""
    s = duration_str.strip().lower()
    if s.endswith("s") and not s.endswith("ms"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    if s.endswith("h"):
        return float(s[:-1]) * 3600.0
    if s.endswith("d"):
        return float(s[:-1]) * 86400.0
    if s.endswith("w"):
        return float(s[:-1]) * 604800.0
    return float(s)


def _reset_at(year: int, month: int, day: int) -> datetime:
    """The reset instant for one month, clamped to that month's length.

    A reset day of 31 has to mean "the 28th" in February; clamping keeps every
    month in the year exactly one period long with no gaps or overlaps.
    """
    last = calendar.monthrange(year, month)[1]
    return datetime(year, month, min(day, last), tzinfo=timezone.utc)


def period_bounds(now: float, reset_day: int) -> tuple[float, float]:
    """UTC ``[start, end)`` of the quota month containing ``now``."""
    dt = datetime.fromtimestamp(now, timezone.utc)
    start = _reset_at(dt.year, dt.month, reset_day)
    if dt < start:
        # This month's reset has not happened yet, so the live period opened
        # with last month's.
        year, month = (dt.year - 1, 12) if dt.month == 1 else (dt.year, dt.month - 1)
        start = _reset_at(year, month, reset_day)
    year, month = (
        (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    )
    return start.timestamp(), _reset_at(year, month, reset_day).timestamp()


@dataclass
class BudgetConfig:
    """Quota shape: how much QPU time per reset period."""

    budget_seconds: float
    reset_day: int = 1
    usage_db: str = DEFAULT_USAGE_DB


@dataclass
class ParticipationDecision:
    """Whether the pacer will fund the next qblock, and why."""

    participate: bool
    headroom_us: float
    allowance_us: float
    spent_us: float
    period_start: float
    period_end: float
    seconds_until_headroom: float
    # Allowance earned per wall second: the budget spread flat over the period.
    accrual_us_per_s: float = 0.0
    # The whole period's allotment, and whether spend has reached it. Past
    # that point every submission would come back rejected, so it is the
    # one thing that stops a round the miner already joined.
    budget_us: float = 0.0
    exhausted: bool = False


class BudgetPacer:
    """Even-distribution gate over one period's QPU allotment.

    Spend for the live period is held in memory. :meth:`decide` runs on the
    session-loop thread, under the lock that dispatches jobs, once per job, so
    a SQLite query there sits on the path that also handles Cancel, Job and
    Ping — and the ledger is on a mounted volume in the deployment. Nothing
    else adds to this period's spend, so the durable record only has to be
    consulted when the period rolls over, or when this process starts and
    inherits whatever a previous run already spent.
    """

    def __init__(self, config: BudgetConfig, ledger: UsageLedger):
        self.config = config
        self.ledger = ledger
        # Guards the cached total only. Never held across ledger IO.
        self._spend_lock = threading.Lock()
        self._cached_period_start: Optional[float] = None
        self._cached_spent_us = 0.0
        self._cached_at = 0.0

    def _spent_us(self, period_start: float, now: float) -> float:
        """This period's billed access time, from memory where possible.

        Spend this process bills is added to the cache as it happens, so the
        cache is exact for its own work. It is not the only writer, though:
        ``usage_db`` defaults to a shared path, and two miners drawing on one
        D-Wave account must share a ledger or they will collectively overrun
        the quota. So the durable record is re-read on a short interval, which
        bounds how much of another writer's spend this pacer can be blind to
        while still keeping SQLite off the per-job path.

        Staleness is measured on the caller's own clock, the one the period
        maths already uses, so a simulated month ages the cache exactly as a
        real one does. A clock that steps backwards only holds the cache a
        little longer, which is the safe direction.
        """
        with self._spend_lock:
            fresh = (
                self._cached_period_start == period_start
                and 0.0 <= now - self._cached_at < SPEND_REFRESH_S
            )
            if fresh:
                return self._cached_spent_us
        # Read outside the lock so a slow volume cannot block a billing.
        spent = self.ledger.spent_us_since(period_start)
        with self._spend_lock:
            if self._cached_period_start != period_start:
                # A new period starts from whatever the record holds, which is
                # also how a restart mid-period inherits a previous run.
                self._cached_period_start = period_start
                self._cached_spent_us = spent
            else:
                # Same period: another writer may have added to the record,
                # and this pacer may have billed since the read began. Take
                # the larger, because under-counting hands out headroom the
                # QPU has already spent.
                self._cached_spent_us = max(self._cached_spent_us, spent)
            self._cached_at = now
            return self._cached_spent_us

    def decide(self, now: float) -> ParticipationDecision:
        """Evaluate spend against the flat allowance line at ``now``."""
        start, end = period_bounds(now, self.config.reset_day)
        span = end - start
        elapsed = min(max(now - start, 0.0), span)
        budget_us = self.config.budget_seconds * 1_000_000
        allowance_us = budget_us * elapsed / span
        spent_us = self._spent_us(start, now)
        headroom_us = allowance_us - spent_us

        rate_us_per_s = budget_us / span
        if headroom_us > 0:
            until = 0.0
        else:
            # Time for the rising line to reach current spend; never longer
            # than the wait for the reset, which zeroes spend outright.
            catch_up = (
                -headroom_us / rate_us_per_s if rate_us_per_s > 0 else float("inf")
            )
            until = min(catch_up, max(0.0, end - now))

        return ParticipationDecision(
            participate=headroom_us > 0,
            headroom_us=headroom_us,
            allowance_us=allowance_us,
            spent_us=spent_us,
            period_start=start,
            period_end=end,
            seconds_until_headroom=until,
            accrual_us_per_s=rate_us_per_s,
            budget_us=budget_us,
            exhausted=spent_us >= budget_us,
        )

    def record_access_time(self, qpu_access_time_us: float, now: float) -> None:
        """Bill a completed job against the period.

        The cached total is raised before the durable write, not after, so the
        two can only ever disagree in the safe direction: a decision taken
        while the write is in flight sees the charge rather than missing it.
        Under-counting is what hands the pacer headroom the QPU has already
        spent.
        """
        start, _ = period_bounds(now, self.config.reset_day)
        with self._spend_lock:
            if self._cached_period_start == start:
                self._cached_spent_us += float(qpu_access_time_us)
            # Otherwise the period has rolled over or nothing has been read
            # yet, and the next _spent_us reloads from the record below.
        self.ledger.record(qpu_access_time_us, now=now)

    def stats(self, now: float) -> Dict[str, Any]:
        decision = self.decide(now)
        return {
            "budget_seconds": self.config.budget_seconds,
            "reset_day": self.config.reset_day,
            "period_start": decision.period_start,
            "period_end": decision.period_end,
            "allowance_seconds": decision.allowance_us / 1_000_000,
            "spent_seconds": decision.spent_us / 1_000_000,
            "headroom_seconds": decision.headroom_us / 1_000_000,
            "jobs_this_period": self.ledger.jobs_since(decision.period_start),
            "seconds_until_headroom": decision.seconds_until_headroom,
        }


def usage_db_from_backend_toml(toml_text: str) -> str:
    """The ledger path ``Configure.backend_toml`` names, or the default.

    History shares the ledger's file, and an unbudgeted miner still keeps
    history, so the path resolves on its own rather than through the pacer.
    """
    if not toml_text or not toml_text.strip():
        return DEFAULT_USAGE_DB
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return DEFAULT_USAGE_DB
    return str(data.get("usage_db") or DEFAULT_USAGE_DB)


def budget_from_backend_toml(toml_text: str) -> Optional[BudgetPacer]:
    """Build a pacer from ``Configure.backend_toml``, or None if unbudgeted.

    Raises :class:`BudgetUnavailable` when a budget is configured but cannot
    be enforced: an unmetered miner would spend the period's quota in a day, so
    refusing to start is the safe failure.
    """
    if not toml_text or not toml_text.strip():
        return None
    try:
        data = tomllib.loads(toml_text)
    except Exception as exc:
        # A document that will not parse is not "no budget configured" — it is
        # a budget nobody can read. Returning None here would mine unmetered on
        # a typo, which is the exact failure this module exists to prevent.
        # `budget = 250m` (a bare duration) is the common one: TOML needs
        # `budget = "250m"`.
        raise BudgetUnavailable(
            f"cannot parse the coordinator's backend config: {exc}. "
            "A duration must be quoted, as in budget = \"250m\"; the miner "
            "will not mine while its budget is unreadable."
        ) from exc

    raw = data.get("budget") or data.get("budget_seconds")
    if raw is None:
        return None
    amount = float(raw) if isinstance(raw, (int, float)) else parse_duration(str(raw))

    reset_day = int(data.get("budget_reset_day", 1))
    if not 1 <= reset_day <= 31:
        raise BudgetUnavailable(
            f"budget_reset_day must be 1-31, got {reset_day}"
        )

    db_path = usage_db_from_backend_toml(toml_text)
    try:
        ledger = UsageLedger(db_path)
    except Exception as exc:
        raise BudgetUnavailable(
            f"cannot open the QPU usage ledger at {db_path}: {exc}. "
            "Set usage_db to a writable path on a persistent volume; the miner "
            "will not mine without a durable record of spend."
        ) from exc

    return BudgetPacer(
        BudgetConfig(
            budget_seconds=amount,
            reset_day=reset_day,
            usage_db=db_path,
        ),
        ledger,
    )
