"""The participation decision must not touch the disk.

``decide`` runs on the session-loop thread, under the lock that dispatches
jobs, once per job. Reading the ledger there put a SQLite query on the path
that also handles Cancel, Job and Ping, and the ledger lives on a mounted
volume in the deployment.

Spend this process bills is tracked in memory exactly, so the durable record
only has to be consulted when the period rolls over, when the process starts
and inherits a previous run, or on a short interval to pick up a sibling miner
sharing the same usage_db. The ledger stays the source of truth.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from quip_solver_dwave.budget import BudgetConfig, BudgetPacer, period_bounds
from quip_solver_dwave.usage import UsageLedger


class CountingLedger(UsageLedger):
    """Counts how often the durable record is actually consulted."""

    def __init__(self):
        super().__init__(":memory:")
        self.reads = 0
        self.writes = 0

    def spent_us_since(self, start_ts):
        self.reads += 1
        return super().spent_us_since(start_ts)

    def record(self, access_time_us, *, now=None):
        self.writes += 1
        super().record(access_time_us, now=now)


def ts(y, mo, d, h=0):
    return datetime(y, mo, d, h, tzinfo=timezone.utc).timestamp()


MID_PERIOD = ts(2026, 9, 19)
RESET_DAY = 9


def _pacer():
    led = CountingLedger()
    return BudgetPacer(BudgetConfig(3000.0, RESET_DAY), led), led


def test_the_ledger_is_read_once_for_a_period_not_once_per_decision():
    # One decision per dispatched job at depth 96 is a lot of SELECTs.
    pacer, led = _pacer()

    for _ in range(50):
        pacer.decide(MID_PERIOD)

    assert led.reads == 1


def test_billing_is_visible_to_the_next_decision_without_a_reload():
    pacer, led = _pacer()
    before = pacer.decide(MID_PERIOD).spent_us

    pacer.record_access_time(43_200, MID_PERIOD)
    after = pacer.decide(MID_PERIOD)

    assert after.spent_us == before + 43_200
    assert led.reads == 1  # still no second query
    assert led.writes == 1  # but it did reach the durable record


def test_spend_still_reaches_the_durable_record():
    # The cache is an optimisation, not a replacement: a restart has to see it.
    pacer, led = _pacer()
    pacer.record_access_time(43_200, MID_PERIOD)

    start, _ = period_bounds(MID_PERIOD, RESET_DAY)
    assert UsageLedger.spent_us_since(led, start) == 43_200


def test_a_fresh_pacer_picks_up_spend_the_ledger_already_holds():
    # What a restart mid-period does. Starting from zero would hand out a
    # month's headroom that was already spent.
    led = CountingLedger()
    led.record(500_000, now=MID_PERIOD)
    pacer = BudgetPacer(BudgetConfig(3000.0, RESET_DAY), led)

    assert pacer.decide(MID_PERIOD).spent_us == 500_000


def test_crossing_into_a_new_period_reloads_from_the_ledger():
    # Spend resets on the quota boundary, so the cached total for the old
    # period must not carry over.
    pacer, led = _pacer()
    pacer.record_access_time(900_000, MID_PERIOD)
    assert pacer.decide(MID_PERIOD).spent_us == 900_000

    next_period = ts(2026, 10, 19)
    decision = pacer.decide(next_period)

    assert decision.spent_us == 0.0
    assert led.reads == 2  # one per period, not per call


def test_the_cache_leads_the_ledger_so_it_can_only_over_count():
    # Under-counting hands the pacer headroom the QPU already spent. If the
    # two ever disagree, the in-memory total must be the higher one.
    pacer, led = _pacer()
    pacer.decide(MID_PERIOD)
    seen = []

    real = led.record

    def slow(access_time_us, *, now=None):
        # Whatever decide() sees while the write is in flight.
        seen.append(pacer.decide(MID_PERIOD).spent_us)
        real(access_time_us, now=now)

    led.record = slow  # type: ignore[method-assign]
    pacer.record_access_time(43_200, MID_PERIOD)

    assert seen == [43_200]


def test_decisions_stay_consistent_while_workers_bill():
    # Depth 96 means the session loop decides while many workers bill.
    pacer, _ = _pacer()
    pacer.decide(MID_PERIOD)
    stop = threading.Event()
    billed = [0]

    def worker():
        while not stop.is_set():
            pacer.record_access_time(1_000, MID_PERIOD)
            billed[0] += 1

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    readings = [pacer.decide(MID_PERIOD).spent_us for _ in range(200)]
    stop.set()
    for t in threads:
        t.join(timeout=2)

    # Monotone: spend never goes backwards, which a torn read would show.
    assert readings == sorted(readings)
    assert pacer.decide(MID_PERIOD).spent_us > 0


# --- another writer on the same ledger -------------------------------------
#
# usage_db defaults to a shared path. Two miners drawing on one D-Wave account
# have to share a ledger or they will collectively overrun the quota, so a
# cache that only ever trusts its own billing is wrong.


def test_another_writers_spend_is_picked_up_within_the_refresh_window():
    from quip_solver_dwave.budget import SPEND_REFRESH_S

    pacer, led = _pacer()
    assert pacer.decide(MID_PERIOD).spent_us == 0.0

    # A sibling miner bills against the same file.
    UsageLedger.record(led, 700_000, now=MID_PERIOD)

    # Inside the window this pacer may still be serving from memory.
    assert pacer.decide(MID_PERIOD).spent_us == 0.0
    # Past it, the durable record wins.
    later = MID_PERIOD + SPEND_REFRESH_S + 0.1
    assert pacer.decide(later).spent_us == 700_000


def test_a_refresh_during_an_in_flight_write_does_not_lose_the_charge():
    # The window the max() guards: this pacer has raised its own total, but
    # the durable write has not landed, so a refresh reads a lower number.
    # Taking it would hand back headroom the QPU has already spent.
    from quip_solver_dwave.budget import SPEND_REFRESH_S

    pacer, led = _pacer()
    pacer.decide(MID_PERIOD)

    writing = threading.Event()
    release = threading.Event()
    real = led.record

    def slow(access_time_us, *, now=None):
        writing.set()
        release.wait(timeout=2)
        real(access_time_us, now=now)

    led.record = slow  # type: ignore[method-assign]
    t = threading.Thread(
        target=pacer.record_access_time, args=(43_200, MID_PERIOD), daemon=True
    )
    t.start()
    assert writing.wait(timeout=2)

    # Refresh while the ledger still reads 0.
    later = MID_PERIOD + SPEND_REFRESH_S + 0.1
    during = pacer.decide(later).spent_us
    release.set()
    t.join(timeout=2)

    assert during == 43_200, (
        f"a refresh mid-write lost the charge: saw {during}, expected 43200"
    )


def test_a_clock_that_steps_backwards_does_not_wedge_the_cache():
    # NTP can move the clock. A negative age must not read as "fresh forever".
    from quip_solver_dwave.budget import SPEND_REFRESH_S

    pacer, led = _pacer()
    pacer.decide(MID_PERIOD)
    UsageLedger.record(led, 700_000, now=MID_PERIOD)

    pacer.decide(MID_PERIOD - 60)  # clock jumps back
    forward = MID_PERIOD + SPEND_REFRESH_S + 0.1
    assert pacer.decide(forward).spent_us == 700_000
