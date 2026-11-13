"""The round strategy: is this qblock worth the headroom, or is a faster hour coming?

Runs after the budget said yes, once per qblock boundary, and answers from a
memory snapshot only. The design and its reasoning are in
docs/superpowers/specs/2026-09-11-qpu-time-of-week-strategy-design.md.

Winning a round depends on the round's difficulty, which the protocol sets
and the miner cannot see ahead, and on how many models the QPU evaluates
while the round is open. Only the second varies with the hour, so the
decision compares slots by deliverable jobs and nothing else. The one
exception is a target the QPU already reaches: recent jobs that beat it say
a single job wins the round, so the hour no longer matters and the round is
joined.
"""

from __future__ import annotations

import logging
import random
import threading
import tomllib
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

from quip_solver_dwave.budget import ParticipationDecision
from quip_solver_dwave.profile import Snapshot, SnapshotRefresher, slot_label, slot_of

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyConfig:
    """Operator knobs from ``Configure.backend_toml``.

    With the defaults and no history the miner behaves as it does today:
    every round the budget allows is joined. ``min_throughput_advantage``
    is how far the bar must sit above this round's jobs before the round
    is skipped for it, a tolerance against chasing noise in the profile.
    ``participation_chance`` is the share of rounds joined regardless of
    the verdict, so every slot keeps getting measured.
    ``easy_target_share`` is the share of recent QPU jobs that must reach
    the current target for the round to count as easy and be joined in any
    slot. Zero turns the check off.
    """

    min_throughput_advantage: float = 0.25
    participation_chance: float = 0.10
    easy_target_share: float = 0.5


# key -> (lower bound, upper bound or None)
STRATEGY_KEYS = {
    "min_throughput_advantage": (0.0, None),
    "participation_chance": (0.0, 1.0),
    "easy_target_share": (0.0, 1.0),
}


def strategy_config_from_toml(toml_text: str) -> StrategyConfig:
    """Lenient like the sampling defaults: the budget parser reports a bad document."""
    defaults = StrategyConfig()
    if not toml_text or not toml_text.strip():
        return defaults
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return defaults
    values = {}
    for key, (low, high) in STRATEGY_KEYS.items():
        default = getattr(defaults, key)
        raw = data.get(key)
        if raw is None:
            values[key] = default
            continue
        ok = (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and low <= float(raw)
            and (high is None or float(raw) <= high)
        )
        if not ok:
            bound = f"between {low:g} and {high:g}" if high is not None else f"at least {low:g}"
            logger.warning("ignoring %s=%r from config: expected a number %s", key, raw, bound)
            values[key] = default
            continue
        values[key] = float(raw)
    return StrategyConfig(**values)


REASON_NO_DATA = "no-data"
REASON_EASY_TARGET = "easy-target"
REASON_EXPLORE = "explore"
REASON_SATURATED = "saturated"
REASON_FAST_SLOT = "fast-slot"
REASON_SLOW_SLOT = "slow-slot"


# The reach check reads at most this many recent jobs, none older than the
# age limit, and needs at least the minimum before it calls a round easy.
REACH_WINDOW_JOBS = 16
REACH_MAX_AGE_S = 3600.0
REACH_MIN_JOBS = 4


@dataclass(frozen=True)
class TargetReach:
    """How many recent QPU jobs reached the latest round target."""

    jobs: int
    reached: int
    target_milli: int

    @property
    def share(self) -> float:
        return self.reached / self.jobs if self.jobs else 0.0


class TargetReachTracker:
    """Recent job best energies, judged against the latest target.

    Each job is compared with the newest target, not the one it ran under,
    so a target that tightened since the job ran is accounted for. Job
    threads write and the session thread reads, hence the lock.
    """

    def __init__(
        self, window_jobs: int = REACH_WINDOW_JOBS, max_age_s: float = REACH_MAX_AGE_S
    ):
        self._bests: Deque[Tuple[float, int]] = deque(maxlen=window_jobs)
        self._max_age_s = max_age_s
        self._target_milli: Optional[int] = None
        self._lock = threading.Lock()

    def set_target(self, target_milli: int) -> None:
        with self._lock:
            self._target_milli = int(target_milli)

    def observe(self, completed_at: float, best_energy_milli: Optional[int]) -> None:
        if best_energy_milli is None:
            return
        with self._lock:
            self._bests.append((completed_at, int(best_energy_milli)))

    def seed(self, rounds: Iterable[Tuple[float, int, Optional[int]]]) -> None:
        """Recent rounds from history, oldest first: (end time, best, target).

        A restart would otherwise start with no evidence and sit out the
        first round even when every recent job won. Each round adds one
        entry, its best job, and the newest round's target stands in until
        the coordinator sends the live one.
        """
        newest_target: Optional[int] = None
        with self._lock:
            for completed_at, best, target in rounds:
                self._bests.append((completed_at, int(best)))
                if target is not None:
                    newest_target = int(target)
            if self._target_milli is None:
                self._target_milli = newest_target

    def summary(self, now: float) -> Optional[TargetReach]:
        """None until a target and a recent job both exist."""
        with self._lock:
            target = self._target_milli
            recent = [b for ts, b in self._bests if now - ts <= self._max_age_s]
        if target is None or not recent:
            return None
        return TargetReach(
            jobs=len(recent),
            reached=sum(1 for b in recent if b <= target),
            target_milli=target,
        )


def is_easy(reach: Optional[TargetReach], config: StrategyConfig) -> bool:
    """Recent jobs reach the target often enough that one job wins the round."""
    return (
        reach is not None
        and config.easy_target_share > 0.0
        and reach.jobs >= REACH_MIN_JOBS
        and reach.share >= config.easy_target_share
    )


@dataclass(frozen=True)
class RoundDecision:
    """The verdict on one qblock boundary and the numbers behind it."""

    join: bool
    reason: str
    slot: int
    # None on a no-data verdict: no history means no prediction was made,
    # not a prediction of zero.
    expected_jobs: Optional[float]
    jobs_per_s: Optional[float]
    # The throughput bar: jobs per round of the slowest round the rest of
    # the period's funds still cover. Zero when they cover every round.
    bar_jobs: float
    rounds_funded: float
    rounds_remaining: float
    # The recent jobs against the target, when any exist.
    reach: Optional[TargetReach] = None


def _easy(slot: int, reach: TargetReach) -> RoundDecision:
    return RoundDecision(
        join=True,
        reason=REASON_EASY_TARGET,
        slot=slot,
        expected_jobs=None,
        jobs_per_s=None,
        bar_jobs=0.0,
        rounds_funded=0.0,
        rounds_remaining=0.0,
        reach=reach,
    )


def _no_data(slot: int) -> RoundDecision:
    return RoundDecision(
        join=True,
        reason=REASON_NO_DATA,
        slot=slot,
        expected_jobs=None,
        jobs_per_s=None,
        bar_jobs=0.0,
        rounds_funded=0.0,
        rounds_remaining=0.0,
    )


def rounds_by_slot(now: float, period_end: float, round_length_s: float) -> Dict[int, float]:
    """How many rounds of the rest of the period fall in each slot.

    Walks hour by hour rather than round by round: a month of one-minute
    rounds is forty thousand rounds but only seven hundred hours, and this
    runs on the session thread under the dispatch lock.
    """
    counts: Dict[int, float] = {}
    t = now
    while t < period_end:
        hour_end = min((int(t) // 3600 + 1) * 3600.0, period_end)
        slot = slot_of(t)
        counts[slot] = counts.get(slot, 0.0) + (hour_end - t) / round_length_s
        t = hour_end
    return counts


def decide_round(
    *,
    now: float,
    headroom_us: float,
    accrual_us_per_s: float,
    period_end: float,
    snapshot: Optional[Snapshot],
    config: StrategyConfig,
    explore_draw: float,
    reach: Optional[TargetReach] = None,
) -> RoundDecision:
    """Join this round, or hold the funds for faster hours.

    Runs after the budget said yes. A joined round runs to its end, so a
    join costs a whole round at this slot's rate, and the funds the rest of
    the period will have (headroom now plus accrual to the reset) cover only
    some of the rounds left in it. The best use of a fixed allotment across
    hours of varying rate is to spend it in the fastest ones: rank the
    remaining rounds by the jobs they would deliver, walk down until their
    cost exhausts the funds, and join now when this round clears that bar.

    An easy round skips the ranking: when recent jobs reach the target, the
    round's first job is likely to win it, and the round ends with that win
    after a few jobs, not a whole round of them.
    """
    slot = slot_of(now)
    if reach is not None and is_easy(reach, config):
        return _easy(slot, reach)
    if snapshot is None or all(st.jobs_per_s is None for st in snapshot.slots):
        return _no_data(slot)

    length = snapshot.round_length_s
    access_us = snapshot.access_s_per_job * 1_000_000.0

    def jobs_in(s: int) -> float:
        st = snapshot.slots[s]
        if st.jobs_per_s is None:
            return 0.0
        return st.jobs_per_s * max(0.0, length - (st.rtt_s or 0.0))

    jobs_now = jobs_in(slot)
    funds_us = headroom_us + accrual_us_per_s * max(0.0, period_end - now)
    counts = rounds_by_slot(now, period_end, length)
    remaining = sum(counts.values())

    bar, funded, spent_us = 0.0, 0.0, 0.0
    for s in sorted(counts, key=jobs_in, reverse=True):
        jobs = jobs_in(s)
        cost_us = jobs * access_us
        if cost_us <= 0.0:
            # Free rounds cannot exhaust anything: everything from here on
            # is covered.
            funded = remaining
            break
        affordable = (funds_us - spent_us) / cost_us
        if affordable < counts[s]:
            bar = jobs
            funded += max(0.0, affordable)
            break
        spent_us += counts[s] * cost_us
        funded += counts[s]

    def verdict(join: bool, reason: str) -> RoundDecision:
        return RoundDecision(
            join=join,
            reason=reason,
            slot=slot,
            expected_jobs=jobs_now,
            jobs_per_s=snapshot.slots[slot].jobs_per_s,
            bar_jobs=bar,
            rounds_funded=funded,
            rounds_remaining=remaining,
            reach=reach,
        )

    if explore_draw < config.participation_chance:
        return verdict(True, REASON_EXPLORE)
    if bar <= 0.0:
        return verdict(True, REASON_SATURATED)
    if jobs_now * (1.0 + config.min_throughput_advantage) >= bar:
        return verdict(True, REASON_FAST_SLOT)
    return verdict(False, REASON_SLOW_SLOT)


def describe_round_decision(generation: int, d: RoundDecision, headroom_us: float) -> str:
    """One log line per boundary, with the numbers the verdict rests on."""
    head = f"[QPU] gen {generation}: {'join' if d.join else 'skip'} ({d.reason})"
    if d.reason == REASON_NO_DATA:
        return f"{head} | no history yet, mining every round the budget allows"
    headroom = f"headroom {headroom_us / 1_000_000:.0f}s"
    if d.reason == REASON_EASY_TARGET:
        assert d.reach is not None
        return (
            f"{head} | {d.reach.reached} of {d.reach.jobs} recent jobs reached the target "
            f"{d.reach.target_milli / 1000:.0f} | {headroom}"
        )
    # Every other reason comes from a snapshot with evidence, so decide_round
    # always sets this to a real float here; narrow for the format calls below.
    assert d.expected_jobs is not None
    rate = f"{d.jobs_per_s:.2f}" if d.jobs_per_s is not None else "?"
    now = f"{d.expected_jobs:.0f} jobs at {rate} jobs/s in {slot_label(d.slot)}"
    funds = f"funds cover {d.rounds_funded:.0f} of {d.rounds_remaining:.0f} rounds left"
    if d.reason == REASON_SATURATED:
        return f"{head} | {now}; every round is covered | {headroom}"
    if d.join:
        return f"{head} | {now}, bar {d.bar_jobs:.0f} | {funds} | {headroom}"
    return f"{head} | {now} is under the bar of {d.bar_jobs:.0f} | {funds} | {headroom}"


class RoundStrategy:
    """The gate's adapter: config, the latest snapshot, and the explore draw."""

    def __init__(
        self,
        config: StrategyConfig,
        snapshots: SnapshotRefresher,
        rng: Optional[random.Random] = None,
        reach: Optional[TargetReachTracker] = None,
    ):
        self._config = config
        self._snapshots = snapshots
        self._rng = rng or random.Random()
        self._reach = reach

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def easy_reach(self, now: float) -> Optional[TargetReach]:
        """The recent-job evidence, when it makes the current round easy."""
        reach = self._reach.summary(now) if self._reach is not None else None
        return reach if is_easy(reach, self._config) else None

    def easy_join(self, now: float) -> Optional[RoundDecision]:
        """An easy-target verdict to join mid-round, or None."""
        reach = self.easy_reach(now)
        return _easy(slot_of(now), reach) if reach is not None else None

    def decide(self, now: float, budget: ParticipationDecision) -> RoundDecision:
        return decide_round(
            now=now,
            headroom_us=budget.headroom_us,
            accrual_us_per_s=budget.accrual_us_per_s,
            period_end=budget.period_end,
            snapshot=self._snapshots.latest(),
            config=self._config,
            explore_draw=self._rng.random(),
            reach=self._reach.summary(now) if self._reach is not None else None,
        )
