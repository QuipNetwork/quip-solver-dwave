"""Tests of the QPU sidecar: its protocol, its pace and cap rules, and its journals. None of them reaches the QPU."""

import errno
import fcntl
import http.client
import importlib.util
import io
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import requests
import urllib3
import urllib3.connection

from quip_solver_dwave import cascade_codec, cascade_flow, regime_io, replay
from quip_solver_dwave.ocean import SampleResult

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cascade_sidecar.py"

NODES, LANES, WORDS = 40, 128, 2
PROGRAMS = ["screen", "leg-256", "gpu-256", "cold-1048576", "seeded-1048576-b0250"]
GATE_MILLI = -101_000  # keeps a survivor with best -100: its mock reads reach -102_000
ACCESS_US = 50_000  # what one mock call reports
RESERVATION_US = 59_861  # int(1.2 * (34_640 + 64 * (80 + 158.2))): the reservation of 64 reads at 80 us
STATES = np.random.default_rng(7).choice(np.array([-1, 1], dtype=np.int8), size=(LANES, NODES))
STATES3 = np.random.default_rng(3).choice(np.array([-1, 1], dtype=np.int8), size=(LANES, 3))  # a 3-node spec


@pytest.fixture
def sidecar(monkeypatch):
    spec = importlib.util.spec_from_file_location("cascade_sidecar", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cascade_sidecar", module)  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Output:
    """A standard output that a test can wait on. It keeps one entry for each line the sidecar writes."""

    def __init__(self):
        self.lines: list[str] = []
        self.closed = False  # the reader has gone, and a write raises as it does on a pipe
        self._changed = threading.Condition()

    def write(self, text: str) -> int:
        if self.closed:
            raise BrokenPipeError(errno.EPIPE, "Broken pipe")
        with self._changed:
            self.lines.extend(text.splitlines())
            self._changed.notify_all()
        return len(text)

    def flush(self) -> None:
        pass

    def wait_for(self, count: int) -> None:
        with self._changed:
            assert self._changed.wait_for(lambda: len(self.lines) >= count, timeout=5), f"{len(self.lines)} of {count}"

    def messages(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]


class EventLog:
    """The kinds of the events that a journal has written, which a test can wait for."""

    def __init__(self):
        self.kinds: list[str] = []
        self._changed = threading.Condition()

    def add(self, kind: str) -> None:
        with self._changed:
            self.kinds.append(kind)
            self._changed.notify_all()

    def wait_for(self, kind: str, count: int = 1) -> None:
        with self._changed:
            assert self._changed.wait_for(lambda: self.kinds.count(kind) >= count, timeout=5), (kind, self.kinds)


class Dropped(Exception):
    """What ``Gate.sample`` raises for a call whose connection broke under it."""


class Gate:
    """Wraps a backend. ``sample`` waits until the test opens the gate, and the wrapper records what it saw. Each
    entry of ``drop_sample`` breaks the connection under one call of its job."""

    def __init__(self, inner, *, hold=True, fail_sample=(), drop_sample=(), zero_access=(), fail_decode=False,
                 hang=(), connect_failures=0, rescore=None):
        self.inner = inner
        self.shape = inner.shape
        self.opened = threading.Event()
        if not hold:
            self.opened.set()
        self.entered = threading.Semaphore(0)
        self.calls: list[str] = []
        self.live = self.peak = 0
        self.fail_sample, self.zero_access, self.fail_decode = set(fail_sample), set(zero_access), fail_decode
        self.drops = Counter(drop_sample)
        self.hang, self.hung = set(hang), threading.Event()  # these calls wait for ``hung``, which the test sets last
        self.connects, self.connect_failures = 0, connect_failures  # the first connections raise, as an offline solver
        self.rescore_with = rescore
        self._lock = threading.Lock()

    def connect(self):
        with self._lock:
            self.connects += 1
            offline = self.connects <= self.connect_failures
        if offline:
            raise RuntimeError("the solver is offline")
        self.inner.connect()

    def rescore(self, job):
        return self.inner.rescore(job) if self.rescore_with is None else self.rescore_with(job)

    def sample(self, job):
        with self._lock:
            self.calls.append(job.id)
            self.live += 1
            self.peak = max(self.peak, self.live)
        self.entered.release()
        try:
            assert self.opened.wait(5), "the test never opened the gate"
            if job.id in self.hang:
                assert self.hung.wait(10), "the test never released the hung call"
            if job.id in self.fail_sample:
                raise RuntimeError(f"the solver is offline for {job.id}")
            with self._lock:
                drop = self.drops[job.id] > 0
                if drop:
                    self.drops[job.id] -= 1
            if drop:
                raise Dropped(f"the server closed the connection under {job.id}")
            return 0 if job.id in self.zero_access else self.inner.sample(job)
        finally:
            with self._lock:
                self.live -= 1

    @staticmethod
    def dropped(exc):
        return isinstance(exc, Dropped)

    def decode(self, job):
        if self.fail_decode:
            raise ValueError("the reads lack spec nodes")
        return self.inner.decode(job)

    def close(self) -> None:
        pass


def nonce_of(first_byte: int, index: int) -> str:
    return f"{first_byte:02x}{index:02x}" + "cd" * 30


def ready_line(programs=PROGRAMS, nodes=NODES, edges=90, lanes=LANES) -> str:
    return json.dumps({"ev": "ready", "nodes": nodes, "edges": edges, "words": WORDS, "lanes": lanes, "programs": programs})


def survivor_line(index: int, first_byte: int, best: int = -100, states=None, energies=None) -> str:
    states = STATES if states is None else states
    energies = [best + lane for lane in range(LANES)] if energies is None else energies
    return json.dumps({
        "ev": "survivor", "id": f"7:{index}", "nonce": nonce_of(first_byte, index),
        "checkpoints": [best + 60, best + 20, best], "best": best, "lanes": LANES,
        "energies": energies, "samples": cascade_codec.encode_reads(states),
    })


def survivor_of(index: int, first_byte: int, best: int = -100) -> cascade_flow.Survivor:
    energies = np.array([best + lane for lane in range(LANES)])
    return cascade_flow.Survivor(nonce_of(first_byte, index), best, STATES, energies)


def mock_reads(reads: int = 64) -> np.ndarray:
    """The reads of the mock QPU for ``STATES``: the states, with the first spin of each second read flipped."""
    spins = np.array([STATES[read % LANES] for read in range(reads)], dtype=np.int8)
    spins[1::2, 0] *= -1
    return spins


def unpack_lanes(lanes_hex: str, nodes: int, count: int) -> np.ndarray:
    """``int8[count, nodes]`` from seed lanes: lane ``r`` is bit ``r % 64`` of word ``r // 64``, and a set bit is -1."""
    words = np.frombuffer(bytes.fromhex(lanes_hex), dtype="<u8").reshape(nodes, WORDS)
    bits = [(words[:, lane // 64] >> np.uint64(lane % 64)) & np.uint64(1) for lane in range(count)]
    return np.array([1 - 2 * bit.astype(np.int8) for bit in bits])


def config(module, tmp_path, **override):
    values = {
        "out_dir": str(tmp_path), "arms": [("gpu-256", 1), ("qpu-cold", 1), ("qpu-seeded", 1)],
        "gpu_program": "gpu-256", "cold_program": "cold-1048576", "seeded_program": "seeded-1048576-b0250",
        "qpu_miss": 0.01, "qpu_gate_milli": GATE_MILLI, "qpu_seconds": 100.0,
        "spend_until_s": 1000.0, "in_flight": 4, "queue": 8, "reads": 64, "anneal_us": 80, "j_scale": 0.5,
        "mock": True,
    }
    values.update(override)
    return module.Config(**values)


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]


def spend_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / regime_io.SPEND_JOURNAL).read_text().splitlines()]


def run_sidecar(module, tmp_path, lines, *, backend=None, clock=None, out=None, log=None, **override):
    """Run a sidecar over ``lines``. Returns the answers it wrote and the events it journaled.

    Without a ``clock`` the run sits halfway through its window, where the pace allows 50 s of access. ``log``, an
    ``EventLog``, learns of each event once the journal has written it.
    """
    out = Output() if out is None else out
    cfg = config(module, tmp_path, **override)
    fake = FakeClock() if clock is None else clock
    with module.Journal(tmp_path / "events.jsonl", fake) as journal:
        if log is not None:
            write = journal.event

            def tap(kind, **fields):
                write(kind, **fields)
                log.add(kind)

            journal.event = tap
        if clock is None:
            fake.now = cfg.spend_until_s / 2
        module.Sidecar(cfg, backend or module.MockBackend(cfg.reads), journal, out).run(lines)
    return out.messages(), read_events(tmp_path)


def test_each_arm_answers_a_survivor_with_one_deep_or_drop(sidecar, tmp_path):
    # The first nonce byte 0, 1, 2 lands on gpu-256, qpu-cold, qpu-seeded. Survivors 3 and 4 have a best of -90.
    lines = [ready_line(), survivor_line(0, 0), survivor_line(1, 1), survivor_line(2, 2),
             survivor_line(3, 1, best=-90), survivor_line(4, 2, best=-90)]
    answers, _ = run_sidecar(sidecar, tmp_path, lines)
    by_id = {answer["id"]: answer for answer in answers}
    assert len(answers) == 5 and sorted(by_id) == [f"7:{i}" for i in range(5)]
    assert all("nonce" not in answer for answer in answers)

    gpu_seed = cascade_flow.unit_seed(nonce_of(0, 0), 3)  # the GPU arm has the salt 3, the seeded arm 10, the cold arm 1000
    assert by_id["7:0"] == {"op": "deep", "id": "7:0", "program": "gpu-256", "seed": gpu_seed}
    cold_seed = cascade_flow.cold_plan(nonce_of(1, 1), {"cold": "cold-1048576"}, 0).seed
    assert by_id["7:1"] == {"op": "deep", "id": "7:1", "program": "cold-1048576", "seed": cold_seed}

    # The QPU gate keeps a best of -102_000 milli at the cut -101_000, and drops -92_000.
    assert by_id["7:3"] == {"op": "drop", "id": "7:3"} and by_id["7:4"] == {"op": "drop", "id": "7:4"}

    reads = mock_reads()
    plan = cascade_flow.qpu_plans(
        survivor_of(2, 2), reads, -102_000 + np.arange(64), qpu_gate_milli=GATE_MILLI, arms=["qpu-seeded"],
        programs={"seeded": "seeded-1048576-b0250", "cold": "cold-1048576"}, lanes=LANES)[0]
    assert plan.states is not None
    seeded = by_id["7:2"]
    assert seeded == {"op": "deep", "id": "7:2", "program": "seeded-1048576-b0250", "seed": plan.seed,
                      "lanes": cascade_codec.encode_lanes(plan.states, WORDS), "seeded": len(plan.states)}
    assert seeded["seeded"] == LANES
    rows = unpack_lanes(seeded["lanes"], NODES, seeded["seeded"])
    assert np.array_equal(rows[:64], reads)  # the 64 QPU reads, best first
    own = {state.tobytes() for state in STATES}
    assert all(row.tobytes() in own for row in rows[64:])  # the fill is the survivor's own states
    assert len({row.tobytes() for row in rows}) == LANES


def test_copies_ride_on_every_deep_answer_and_never_on_a_drop(sidecar, tmp_path):
    # The first nonce byte 0, 1, 2 lands on gpu-256, qpu-cold, qpu-seeded; survivor 3 has a best the QPU gate drops.
    lines = [ready_line(), survivor_line(0, 0), survivor_line(1, 1), survivor_line(2, 2),
             survivor_line(3, 2, best=-90)]
    answers, _ = run_sidecar(sidecar, tmp_path, lines, copies=4)
    by_id = {answer["id"]: answer for answer in answers}
    assert [by_id[f"7:{i}"]["copies"] for i in range(3)] == [4, 4, 4]
    assert by_id["7:3"] == {"op": "drop", "id": "7:3"}
    (tmp_path / "single").mkdir()
    single, _ = run_sidecar(sidecar, tmp_path / "single", lines)
    assert all("copies" not in answer for answer in single)


def test_copies_outside_one_to_the_engine_limit_are_a_usage_error(sidecar, tmp_path):
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"unit": "milli", "table": [{"miss_allowed": 0.01, "cut": GATE_MILLI}]}))
    parser = sidecar.build_parser()
    base = ["--out-dir", str(tmp_path), "--programs", "p.json", "--qpu-gates", str(gates), "--qpu-seconds", "10",
            "--spend-until", "2026-10-09T00:00:00Z", "--mock"]
    assert sidecar.resolve(parser, parser.parse_args(base + ["--copies", "4"])).copies == 4
    assert sidecar.resolve(parser, parser.parse_args(base)).copies == 1
    for bad in ("0", str(sidecar.MAX_COPIES + 1)):
        with pytest.raises(SystemExit):
            sidecar.resolve(parser, parser.parse_args(base + ["--copies", bad]))


def test_the_journal_records_each_survivor_and_the_qpu_verdict(sidecar, tmp_path):
    lines = [ready_line(), survivor_line(0, 0), survivor_line(1, 1), survivor_line(2, 2),
             survivor_line(3, 2, best=-90)]
    _, events = run_sidecar(sidecar, tmp_path, lines)
    assert all({"type", "t"} <= set(event) for event in events)
    survivors = [(e["id"], e["nonce"], e["best"], e["arm"]) for e in events if e["type"] == "survivor"]
    assert survivors == [("7:0", nonce_of(0, 0), -100, "gpu-256"), ("7:1", nonce_of(1, 1), -100, "qpu-cold"),
                         ("7:2", nonce_of(2, 2), -100, "qpu-seeded"), ("7:3", nonce_of(2, 3), -90, "qpu-seeded")]
    qpu = sorted((e for e in events if e["type"] == "qpu"), key=lambda e: e["id"])
    assert [(e["id"], e["nonce"], e["arm"], e["kept"], e["best_milli"], e["access_us"], e["reads"]) for e in qpu] == [
        ("7:1", nonce_of(1, 1), "qpu-cold", True, -102_000, ACCESS_US, 64),
        ("7:2", nonce_of(2, 2), "qpu-seeded", True, -102_000, ACCESS_US, 64),
        ("7:3", nonce_of(2, 3), "qpu-seeded", False, -92_000, ACCESS_US, 64),
    ]
    assert all("void" not in e for e in qpu)
    assert not any(e["type"] in ("lane_check", "lane_order_mismatch", "qpu_error") for e in events)  # no spec to score on
    # Each QPU call journals a reservation before the call and its charge after it.
    spend = spend_lines(tmp_path)
    assert [line["event"] for line in spend].count(regime_io.SUBMIT) == 3
    assert sorted(line["nonce"] for line in spend if line["event"] == regime_io.CHARGE) == ["7:1", "7:2", "7:3"]
    assert {line["us"] for line in spend if line["event"] == regime_io.SUBMIT} == {RESERVATION_US}
    assert regime_io.read_spend(tmp_path) == (3 * ACCESS_US, 0)


def test_a_void_line_cancels_a_pending_decision(sidecar, tmp_path):
    gate = Gate(sidecar.MockBackend(64))

    def feed():
        yield ready_line()
        yield survivor_line(0, 2)
        yield survivor_line(1, 2)
        for _ in range(2):
            assert gate.entered.acquire(timeout=5)  # both calls are on the QPU
        yield json.dumps({"ev": "void", "id": "7:0"})
        gate.opened.set()

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate)
    assert [answer["id"] for answer in answers] == ["7:1"]
    qpu = {e["id"]: e for e in events if e["type"] == "qpu"}
    assert qpu["7:0"]["void"] is True and "void" not in qpu["7:1"]
    assert [e["id"] for e in events if e["type"] == "void"] == ["7:0"]
    # The call of the voided survivor still reached the QPU, so its charge is journaled.
    assert regime_io.read_spend(tmp_path) == (2 * ACCESS_US, 0)


def test_a_void_for_a_queued_survivor_never_reaches_the_qpu(sidecar, tmp_path):
    out = Output()
    gate = Gate(sidecar.MockBackend(64))

    def feed():
        yield ready_line()
        for index in range(3):
            yield survivor_line(index, 1)
        assert gate.entered.acquire(timeout=5)  # 7:0 is on the QPU, and 7:1 and 7:2 wait for its slot
        yield json.dumps({"ev": "void", "id": "7:1"})
        gate.opened.set()
        out.wait_for(2)  # the input stays open until 7:2 has had its turn, since the end of the input drops the queue

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=1, queue=4,
                                  arms=[("qpu-cold", 1)])
    assert gate.calls == ["7:0", "7:2"] and [answer["id"] for answer in answers] == ["7:0", "7:2"]
    assert [e["id"] for e in events if e["type"] == "qpu"] == ["7:0", "7:2"]
    assert [e["id"] for e in events if e["type"] == "void"] == ["7:1"]
    assert regime_io.read_spend(tmp_path) == (2 * ACCESS_US, 0)


def test_a_miner_deep_line_becomes_a_deep_event_with_the_arm_and_the_lane_bests(sidecar, tmp_path):
    out = Output()
    # The 64 QPU lanes reach -120 and the 64 fill lanes reach -110.
    energies = [-120 + lane for lane in range(64)] + [-110 + lane for lane in range(64)]
    programs = ["gpu-256", "cold-1048576", "seeded-1048576-b0250"]

    def feed():
        yield ready_line()
        for index in range(3):
            yield survivor_line(index, index)
        out.wait_for(3)
        for index, program in enumerate(programs):
            yield json.dumps({"ev": "deep", "id": f"7:{index}", "program": program, "status": "passed",
                              "checkpoints": [-100, -120], "best": -120, "leg": 1, "energies": energies,
                              "seconds": 12.5, "pushed": 1})

    _, events = run_sidecar(sidecar, tmp_path, feed(), out=out)
    deep = {e["unit"]: e for e in events if e["type"] == "deep"}
    assert [deep[f"7:{i}"]["arm"] for i in range(3)] == ["gpu-256", "cold", "qpu-seeded"]
    seeded = deep["7:2"]
    assert (seeded["nonce"], seeded["program"], seeded["status"], seeded["best"], seeded["leg"], seeded["seconds"],
            seeded["pushed"], seeded["checkpoints"]) == (nonce_of(2, 2), programs[2], "passed", -120, 1, 12.5, 1,
                                                         [-100, -120])
    assert seeded["qpu_lanes"] == 64 and seeded["qpu_lane_best"] == -120 and seeded["fill_lane_best"] == -110
    for unit in ("7:0", "7:1"):
        assert deep[unit]["nonce"] == nonce_of(int(unit[-1]), int(unit[-1])) and deep[unit]["qpu_lanes"] == 0
        assert "qpu_lane_best" not in deep[unit] and "fill_lane_best" not in deep[unit]


def test_a_best_that_no_leg_reached_has_no_lane_bests(sidecar, tmp_path):
    out = Output()

    def feed():
        yield ready_line()
        yield survivor_line(2, 2)
        out.wait_for(1)
        # The target admitted the survivor's own best before a leg beat it: the energies are the screen's lanes.
        yield json.dumps({"ev": "deep", "id": "7:2", "program": "seeded-1048576-b0250", "status": "hit",
                          "checkpoints": [], "best": -130, "leg": None, "energies": [-130] * 128, "seconds": 0.5,
                          "pushed": 1})

    _, events = run_sidecar(sidecar, tmp_path, feed(), out=out)
    (deep,) = [e for e in events if e["type"] == "deep"]
    assert (deep["arm"], deep["qpu_lanes"], deep["leg"]) == ("qpu-seeded", 64, None)
    assert "qpu_lane_best" not in deep and "fill_lane_best" not in deep


def test_the_unit_records_are_bounded_and_a_deep_line_with_no_record_has_no_arm(sidecar, tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "ANSWERED_LIMIT", 2)
    out = Output()

    def deep_line(index):
        return json.dumps({"ev": "deep", "id": f"7:{index}", "program": "gpu-256", "status": "dropped",
                           "checkpoints": [], "best": -50, "leg": None, "energies": [], "seconds": 1.0, "pushed": 0})

    def feed():
        yield ready_line()
        for index in range(3):
            yield survivor_line(index, 0)
        out.wait_for(3)
        yield deep_line(0)  # its record was the oldest of three, so it went first
        yield deep_line(2)

    _, events = run_sidecar(sidecar, tmp_path, feed(), out=out, arms=[("gpu-256", 1)])
    deep = {e["unit"]: e for e in events if e["type"] == "deep"}
    assert (deep["7:0"]["arm"], deep["7:0"]["nonce"], deep["7:0"]["qpu_lanes"]) == (None, None, 0)
    assert (deep["7:2"]["arm"], deep["7:2"]["nonce"]) == ("gpu-256", nonce_of(0, 2))


def test_the_pace_rule_skips_when_ahead_and_resumes_when_behind(sidecar, tmp_path):
    clock, out = FakeClock(), Output()
    # A cap of 100 s over a window of 1,000 s allows 100_000 us of access for each second that has passed.
    # A call charges 50_000 us, and a call in flight holds its reservation of 59_861 us.
    steps = [0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1000.0]

    def feed():
        yield ready_line()
        for index, now in enumerate(steps):
            clock.now = now
            yield survivor_line(index, 1)
            out.wait_for(index + 1)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), clock=clock, out=out, arms=[("qpu-cold", 1)])
    # 7:0 runs (nothing spent). 7:1 and 7:2 are ahead (50_000 against 0 and 40_000). 7:3 and 7:4 run, because
    # the spend (50_000 and 100_000) is at or below the allowance of 100_000. 7:5 is ahead (150_000).
    # 7:6 comes when spend-until has passed.
    assert sorted(e["id"] for e in events if e["type"] == "qpu") == ["7:0", "7:3", "7:4"]
    skips = [(e["id"], e["nonce"], e["arm"], e["reason"]) for e in events if e["type"] == "qpu_skip"]
    assert skips == [(f"7:{i}", nonce_of(1, i), "qpu-cold", "pace") for i in (1, 2, 5, 6)]
    programs = {answer["id"]: answer["program"] for answer in answers}
    assert [programs[f"7:{i}"] for i in range(7)] == (
        ["cold-1048576", "gpu-256", "gpu-256", "cold-1048576", "cold-1048576", "gpu-256", "gpu-256"])
    assert regime_io.read_spend(tmp_path) == (3 * ACCESS_US, 0)


def test_the_cap_stops_submission_and_journals_the_jobs_in_flight(sidecar, tmp_path):
    clock, out = FakeClock(), Output()
    gate = Gate(sidecar.MockBackend(64))

    def feed():
        yield ready_line()
        clock.now = 900.0  # the pace allows 144_000 us by now, so only the cap can refuse a call
        for index in range(4):
            yield survivor_line(index, 1)
        for _ in range(2):
            assert gate.entered.acquire(timeout=5)
        out.wait_for(2)  # 7:2 and 7:3 meet the cap of 160_000 us: two reservations fit, and a third would pass it
        gate.opened.set()
        out.wait_for(4)
        # The two calls charged 100_000 us, so 7:4 would fit (159_861 us with its reservation). The cap is for good.
        yield survivor_line(4, 1)
        out.wait_for(5)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), clock=clock, out=out, backend=gate, qpu_seconds=0.16,
                                  arms=[("qpu-cold", 1)])
    assert sorted(gate.calls) == ["7:0", "7:1"]
    skips = [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"]
    assert skips == [("7:2", "cap"), ("7:3", "cap"), ("7:4", "cap")]
    programs = {answer["id"]: answer["program"] for answer in answers}
    assert [programs[f"7:{i}"] for i in (2, 3, 4)] == ["gpu-256"] * 3
    # The two calls that were in flight finished after the cap, and the journal holds their charges.
    spend = spend_lines(tmp_path)
    assert [line["us"] for line in spend if line["event"] == regime_io.CHARGE] == [ACCESS_US, ACCESS_US]
    assert regime_io.read_spend(tmp_path) == (2 * ACCESS_US, 0)


@pytest.mark.parametrize("carried_us, qpu_seconds, now, reason", [
    (140_000, 0.15, 0.0, "cap"),  # the journal already holds the cap, with a call left open
    (50_000, 100.0, 0.0, "pace"),  # at the origin the pace allows nothing, so the carried spend is ahead of it
    (50_000, 100.0, 0.5, None),  # half a second on, a cap of 100 s over 1,000 s allows 50_000 us, and the spend is not ahead
])
def test_the_ledger_starts_from_the_spend_journal(sidecar, tmp_path, carried_us, qpu_seconds, now, reason):
    # An earlier run charged ``carried_us``, and it died with one call open, which counts at its estimate.
    regime_io.append_spend(tmp_path, regime_io.SUBMIT, "cell", "6:0", 80, RESERVATION_US)
    regime_io.append_spend(tmp_path, regime_io.CHARGE, "cell", "6:0", 80, carried_us)
    if reason == "cap":
        regime_io.append_spend(tmp_path, regime_io.SUBMIT, "cell", "6:1", 80, RESERVATION_US)
    clock = FakeClock()

    def feed():
        yield ready_line()
        clock.now = now  # seconds after the origin of the window
        yield survivor_line(0, 1)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), qpu_seconds=qpu_seconds, clock=clock,
                                  arms=[("qpu-cold", 1)])
    skips = [e["reason"] for e in events if e["type"] == "qpu_skip"]
    assert skips == ([reason] if reason else [])
    assert answers[0]["program"] == ("gpu-256" if reason else "cold-1048576")


def test_survivors_past_the_queue_run_as_gpu_256_and_the_window_holds(sidecar, tmp_path):
    out = Output()
    gate = Gate(sidecar.MockBackend(64))

    def feed():
        yield ready_line()
        for index in range(4):  # 7:0 and 7:1 take the two slots, 7:2 waits, and 7:3 finds the queue full
            yield survivor_line(index, 1)
        for _ in range(2):
            assert gate.entered.acquire(timeout=5)
        out.wait_for(1)  # 7:3 is answered while both calls are still on the QPU
        gate.opened.set()
        out.wait_for(4)  # the input stays open until 7:2 has had its turn, since the end of the input drops the queue

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=2, queue=1,
                                  arms=[("qpu-cold", 1)])
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [("7:3", "queue")]
    assert out.messages()[0]["id"] == "7:3" and out.messages()[0]["program"] == "gpu-256"
    assert sorted(answer["id"] for answer in answers) == ["7:0", "7:1", "7:2", "7:3"]
    assert gate.peak == 2 and gate.calls[2] == "7:2"


def test_target_and_stats_lines_are_journaled_with_their_time(sidecar, tmp_path):
    clock = FakeClock()
    stats = {"salts": 123456, "passed": [900, 40, 7], "survivors": 7, "deep_done": 2, "deep_active": 1,
             "deep_queued": 0, "drops": 3}

    def feed():
        yield ready_line()
        clock.now = 5.0
        yield json.dumps({"ev": "target", "energy": -14300})
        clock.now = 60.0
        yield json.dumps({"ev": "stats", **stats})
        clock.now = 61.5
        yield json.dumps({"ev": "target", "energy": None})

    _, events = run_sidecar(sidecar, tmp_path, feed(), clock=clock)
    assert [e for e in events if e["type"] == "target"] == [
        {"type": "target", "t": 5.0, "energy": -14300}, {"type": "target", "t": 61.5, "energy": None}]
    assert [e for e in events if e["type"] == "stats"] == [{"type": "stats", "t": 60.0, **stats}]


def test_a_failed_call_runs_its_survivor_on_gpu_256_and_keeps_the_qpu_off(sidecar, tmp_path, capsys):
    out = Output()
    gate = Gate(sidecar.MockBackend(64), fail_sample={"7:0"})

    def feed():
        yield ready_line()
        yield survivor_line(0, 1)
        yield survivor_line(1, 1)  # on the QPU beside 7:0
        for _ in range(2):
            assert gate.entered.acquire(timeout=5)
        gate.opened.set()
        out.wait_for(2)  # 7:0 fails and runs on gpu-256, and 7:1 comes back from the QPU
        yield survivor_line(2, 1)  # the QPU is off, so 7:2 runs on gpu-256 at once
        out.wait_for(3)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=2, queue=0,
                                  arms=[("qpu-cold", 1)])
    assert {answer["id"]: answer["program"] for answer in answers} == {
        "7:0": "gpu-256", "7:1": "cold-1048576", "7:2": "gpu-256"}
    (error,) = [e for e in events if e["type"] == "qpu_error"]
    assert (error["id"], error["nonce"], error["arm"], error["stage"]) == ("7:0", nonce_of(1, 0), "qpu-cold", "sample")
    assert error["exception"] == "RuntimeError" and "offline for 7:0" in error["message"]
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [("7:0", "error"), ("7:2", "error")]
    assert gate.calls == ["7:0", "7:1"] and events[-1]["type"] == "end"
    # The failed call keeps its reservation open, and the cap counts it. The call beside it was charged.
    assert regime_io.read_spend(tmp_path) == (ACCESS_US + RESERVATION_US, 1)
    assert "QPU error at sample: RuntimeError" in capsys.readouterr().err


def test_a_call_whose_connection_broke_goes_again_once_and_the_qpu_stays_on(sidecar, tmp_path, capsys):
    out = Output()
    gate = Gate(sidecar.MockBackend(64), hold=False, drop_sample=["7:0"])

    def feed():
        yield ready_line()
        yield survivor_line(0, 1)
        out.wait_for(1)  # 7:0 comes back from its second call
        yield survivor_line(1, 1)  # the QPU is still on, so 7:1 goes to the QPU as well
        out.wait_for(2)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=1, queue=0,
                                  arms=[("qpu-cold", 1)])
    assert {answer["id"]: answer["program"] for answer in answers} == {"7:0": "cold-1048576", "7:1": "cold-1048576"}
    assert gate.calls == ["7:0", "7:0", "7:1"]
    (retry,) = [e for e in events if e["type"] == "qpu_retry"]
    assert (retry["id"], retry["nonce"], retry["arm"], retry["exception"]) == (
        "7:0", nonce_of(1, 0), "qpu-cold", "Dropped")
    assert "closed the connection under 7:0" in retry["message"]
    assert not [e for e in events if e["type"] in ("qpu_error", "qpu_skip")]
    assert regime_io.read_spend(tmp_path) == (2 * ACCESS_US, 0)  # one charge for each survivor
    assert "QPU error" not in capsys.readouterr().err


def test_a_call_whose_connection_breaks_twice_fails_and_turns_the_qpu_off(sidecar, tmp_path):
    out = Output()
    gate = Gate(sidecar.MockBackend(64), hold=False, drop_sample=["7:0", "7:0"])

    def feed():
        yield ready_line()
        yield survivor_line(0, 1)
        out.wait_for(1)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=1, queue=0,
                                  arms=[("qpu-cold", 1)])
    assert [(answer["id"], answer["program"]) for answer in answers] == [("7:0", "gpu-256")]
    assert gate.calls == ["7:0", "7:0"]
    assert [e["type"] for e in events if e["type"].startswith("qpu_")] == ["qpu_retry", "qpu_error", "qpu_skip"]
    assert regime_io.read_spend(tmp_path) == (RESERVATION_US, 1)


def test_the_ocean_backend_counts_only_a_connection_that_broke_under_a_call_as_dropped(sidecar):
    broke = requests.exceptions.ConnectionError(urllib3.exceptions.ProtocolError(
        "Connection aborted.", http.client.RemoteDisconnected("Remote end closed connection without response")))
    # The message of each of the 5 connection errors that the live run journaled on 2026-10-07.
    assert str(broke) == "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))"
    reset = requests.exceptions.ConnectionError(urllib3.exceptions.ProtocolError(
        "Connection aborted.", ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")))
    host = "na-west-1.cloud.dwavesys.com"  # nothing here opens a connection
    refused = requests.exceptions.ConnectionError(urllib3.exceptions.MaxRetryError(
        urllib3.HTTPSConnectionPool(host), "/sapi/v2/problems/",
        urllib3.exceptions.NewConnectionError(urllib3.connection.HTTPSConnection(host), "Connection refused")))
    assert sidecar.OceanBackend.dropped(broke) and sidecar.OceanBackend.dropped(reset)
    assert not sidecar.OceanBackend.dropped(refused)  # no connection at all: an outage, which turns the QPU off
    assert not sidecar.OceanBackend.dropped(RuntimeError("the solver went offline"))
    assert not sidecar.MockBackend.dropped(broke)


def test_a_failed_call_sends_the_queued_survivors_to_gpu_256(sidecar, tmp_path):
    out = Output()
    gate = Gate(sidecar.MockBackend(64), fail_sample={"7:0"})

    def feed():
        yield ready_line()
        for index in range(3):  # 7:0 takes the one slot, and 7:1 and 7:2 wait for it
            yield survivor_line(index, 1)
        assert gate.entered.acquire(timeout=5)
        gate.opened.set()
        out.wait_for(3)  # the QPU is off, so the two that waited run on gpu-256 as well

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=1, queue=4,
                                  arms=[("qpu-cold", 1)])
    assert gate.calls == ["7:0"]
    assert sorted((answer["id"], answer["program"]) for answer in answers) == [
        ("7:0", "gpu-256"), ("7:1", "gpu-256"), ("7:2", "gpu-256")]
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [
        ("7:0", "error"), ("7:1", "error"), ("7:2", "error")]
    assert regime_io.read_spend(tmp_path) == (RESERVATION_US, 1)


@pytest.mark.parametrize("kind, victim", [
    pytest.param("qpu_error", "7:0", id="the journal cannot take the qpu_error of the call that failed"),
    pytest.param("qpu_skip", "7:1", id="the journal cannot take the qpu_skip of a survivor that waited"),
])
def test_a_failure_that_the_sidecar_cannot_report_as_a_qpu_error_still_ends_the_run(
    sidecar, tmp_path, monkeypatch, kind, victim
):
    out = Output()
    gate = Gate(sidecar.MockBackend(64), fail_sample={"7:0"})
    raised = threading.Event()
    write = sidecar.Journal.event

    def full_disk(self, name, **fields):
        if name == kind and fields.get("id") == victim:
            raised.set()
            raise OSError(errno.ENOSPC, "No space left on device")
        return write(self, name, **fields)

    monkeypatch.setattr(sidecar.Journal, "event", full_disk)

    def feed():
        yield ready_line()
        yield survivor_line(0, 1)
        yield survivor_line(1, 1)  # waits for the one slot
        assert gate.entered.acquire(timeout=5)
        gate.opened.set()
        assert raised.wait(5)  # the input stays open until the journal has failed

    with pytest.raises(OSError, match="No space left on device"):
        run_sidecar(sidecar, tmp_path, feed(), backend=gate, out=out, in_flight=1, queue=2, arms=[("qpu-cold", 1)])
    assert gate.calls == ["7:0"]


def test_end_of_input_submits_nothing_more_and_waits_for_the_calls_in_flight(sidecar, tmp_path, monkeypatch):
    gate = Gate(sidecar.MockBackend(64))
    wait_for = sidecar.Sidecar._wait_for

    def open_the_gate_then_wait(self, threads):
        gate.opened.set()  # the input has ended and the sidecar is closing: let the call on the QPU finish
        return wait_for(self, threads)

    monkeypatch.setattr(sidecar.Sidecar, "_wait_for", open_the_gate_then_wait)

    def feed():
        yield ready_line()
        yield survivor_line(0, 1)
        yield survivor_line(1, 1)
        assert gate.entered.acquire(timeout=5)  # 7:0 is on the QPU, and 7:1 waits for the slot

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, in_flight=1, queue=4,
                                  arms=[("qpu-cold", 1)])
    assert gate.calls == ["7:0"] and [answer["id"] for answer in answers] == ["7:0"]
    assert events[-1]["type"] == "end" and [e["id"] for e in events if e["type"] == "qpu"] == ["7:0"]
    assert regime_io.read_spend(tmp_path) == (ACCESS_US, 0)


def test_a_hung_call_does_not_hold_the_end_of_input_for_longer_than_the_drain_bound(
    sidecar, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sidecar, "DRAIN_TIMEOUT_S", 0.3)
    out = Output()
    gate = Gate(sidecar.MockBackend(64), hold=False, hang={"7:1"})
    cfg = config(sidecar, tmp_path, in_flight=2, queue=0, arms=[("qpu-cold", 1)])
    clock = FakeClock()

    def feed():
        yield ready_line()
        yield survivor_line(0, 1)
        yield survivor_line(1, 1)
        for _ in range(2):
            assert gate.entered.acquire(timeout=5)
        out.wait_for(1)  # 7:0 is done, and 7:1 is still on the QPU when the input ends

    started = time.monotonic()
    with sidecar.Journal(tmp_path / "events.jsonl", clock) as journal:  # it stays open while the late call finishes
        clock.now = cfg.spend_until_s / 2
        try:
            sidecar.Sidecar(cfg, gate, journal, out).run(feed())
            assert time.monotonic() - started < 4  # the hung call would hold the run for 10 s
            events = read_events(tmp_path)
            assert [answer["id"] for answer in out.messages()] == ["7:0"]
            assert events[-1]["type"] == "end" and events[-1]["unfinished"] == 1
            # 7:0 has its charge, and the call that did not finish keeps its reservation, so the cap counts it.
            assert regime_io.read_spend(tmp_path) == (ACCESS_US + RESERVATION_US, 1)
            assert "1 calls did not finish in 0.3 s" in capsys.readouterr().err
        finally:
            gate.hung.set()
            for thread in threading.enumerate():
                if thread.name == "cascade-qpu":
                    thread.join(5)
        # The late call books its charge. The run has ended, so it writes no event and sends no answer.
        assert regime_io.read_spend(tmp_path) == (2 * ACCESS_US, 0)
        assert [e["type"] for e in read_events(tmp_path)] == ["survivor", "survivor", "qpu", "end"]
        assert [answer["id"] for answer in out.messages()] == ["7:0"]


def test_a_hung_call_does_not_hold_the_lock_of_the_out_dir_against_the_next_launch(sidecar, tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "DRAIN_TIMEOUT_S", 0.3)
    write_inputs(tmp_path)
    entered, hung = threading.Event(), threading.Event()
    sample = sidecar.MockBackend.sample

    def hang(self, job):
        entered.set()
        assert hung.wait(10)
        return sample(self, job)

    monkeypatch.setattr(sidecar.MockBackend, "sample", hang)
    clock = FakeClock()

    def feed():
        yield ready_line()
        clock.now = 500.0
        yield survivor_line(0, 1)
        assert entered.wait(5)

    flags = ["--mock", "--yes", "--arms", "qpu-cold:1"]
    try:
        assert sidecar.main(argv(tmp_path, *flags, cap="100", until=UNTIL), stdin=feed(), stdout=Output(),
                            clock=clock) == 0
        # The call of the first launch is still on the QPU, and the second launch takes the lock all the same.
        assert sidecar.main(argv(tmp_path, *flags, cap="100", until=UNTIL), stdin=io.StringIO(""), stdout=Output(),
                            clock=clock, lock_timeout_s=0.2) == 0
    finally:
        hung.set()
        for thread in threading.enumerate():
            if thread.name == "cascade-qpu":
                thread.join(5)
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "survivor", "end", "start", "end"]
    assert events[2]["unfinished"] == 1 and events[3]["spent_us"] == RESERVATION_US  # the open reservation counts


def test_the_lock_of_the_out_dir_is_free_while_the_backend_closes(sidecar, tmp_path, monkeypatch):
    write_inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def blocked(self):  # the close of a real session can wait a minute for a poll at SAPI
        if threading.current_thread().name == "first-launch":
            entered.set()
            assert release.wait(10), "the test never released the close"

    monkeypatch.setattr(sidecar.MockBackend, "close", blocked)
    flags = ["--mock", "--yes"]
    first = []

    def launch():
        first.append(sidecar.main(argv(tmp_path, *flags), stdin=io.StringIO(""), stdout=Output()))

    thread = threading.Thread(target=launch, name="first-launch")
    thread.start()
    try:
        assert entered.wait(10)  # the first launch has ended its run, and its close waits
        assert sidecar.main(argv(tmp_path, *flags), stdin=io.StringIO(""), stdout=Output(), lock_timeout_s=0.2) == 0
    finally:
        release.set()
        thread.join(10)
    assert first == [0]
    assert [e["type"] for e in read_events(tmp_path / "out")] == ["start", "end", "start", "end"]


def test_a_call_with_no_access_time_is_charged_its_reservation_and_runs_on_gpu_256(sidecar, tmp_path):
    gate = Gate(sidecar.MockBackend(64), hold=False, zero_access={"7:0"})
    answers, events = run_sidecar(sidecar, tmp_path, [ready_line(), survivor_line(0, 1)], backend=gate,
                                  arms=[("qpu-cold", 1)])
    assert regime_io.read_spend(tmp_path) == (RESERVATION_US, 0)
    assert [line["us"] for line in spend_lines(tmp_path) if line["event"] == regime_io.CHARGE] == [RESERVATION_US]
    (error,) = [e for e in events if e["type"] == "qpu_error"]
    assert error["stage"] == "access_time" and "no QPU access time" in error["message"]
    assert [answer["program"] for answer in answers] == ["gpu-256"]
    assert not any(e["type"] == "qpu" for e in events) and events[-1]["type"] == "end"


def test_the_charge_is_journaled_before_the_decode_and_survives_its_failure(sidecar, tmp_path):
    gate = Gate(sidecar.MockBackend(64), hold=False, fail_decode=True)
    answers, events = run_sidecar(sidecar, tmp_path, [ready_line(), survivor_line(0, 1)], backend=gate,
                                  arms=[("qpu-cold", 1)])
    assert regime_io.read_spend(tmp_path) == (ACCESS_US, 0)
    (error,) = [e for e in events if e["type"] == "qpu_error"]
    assert (error["stage"], error["exception"]) == ("decode", "ValueError") and "lack spec nodes" in error["message"]
    assert [answer["program"] for answer in answers] == ["gpu-256"]


def test_the_qpu_comes_back_when_a_retry_of_the_connection_works(sidecar, tmp_path):
    clock, out, log = FakeClock(), Output(), EventLog()
    gate = Gate(sidecar.MockBackend(64), hold=False, fail_sample={"7:0"})

    def feed():
        yield ready_line()
        clock.now = 500.0  # halfway through the window, so only the error can keep the QPU off
        yield survivor_line(0, 1)  # the call fails, and the retry is due 30 s from now
        out.wait_for(1)
        clock.now = 520.0
        yield survivor_line(1, 1)  # too early for a retry
        out.wait_for(2)
        assert gate.connects == 1
        clock.now = 531.0
        yield survivor_line(2, 1)  # the retry starts at this survivor, which still runs on gpu-256
        log.wait_for("qpu_recovered")
        out.wait_for(3)
        yield survivor_line(3, 1)  # the QPU is on again
        out.wait_for(4)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, clock=clock, out=out, log=log,
                                  arms=[("qpu-cold", 1)])
    programs = {answer["id"]: answer["program"] for answer in answers}
    assert [programs[f"7:{i}"] for i in range(4)] == ["gpu-256", "gpu-256", "gpu-256", "cold-1048576"]
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [
        ("7:0", "error"), ("7:1", "error"), ("7:2", "error")]
    assert gate.connects == 2  # the start, and one retry
    assert [e["type"] for e in events if e["type"] in ("qpu_error", "qpu_recovered")] == ["qpu_error", "qpu_recovered"]
    assert gate.calls == ["7:0", "7:3"]


def test_a_failed_retry_doubles_the_wait_before_the_next_one(sidecar, tmp_path):
    clock, out, log = FakeClock(), Output(), EventLog()
    gate = Gate(sidecar.MockBackend(64), hold=False, connect_failures=2)  # the start fails, and so does the first retry

    def feed():
        yield ready_line()
        clock.now = 10.0  # the start failed at 0 s, so the first retry is due at 30 s
        yield survivor_line(0, 1)
        out.wait_for(1)
        clock.now = 31.0
        yield survivor_line(1, 1)  # the first retry starts, and it fails: the next one is due 60 s later, at 91 s
        log.wait_for("qpu_error", 2)
        out.wait_for(2)
        clock.now = 90.0
        yield survivor_line(2, 1)  # not due yet
        out.wait_for(3)
        assert gate.connects == 2
        clock.now = 92.0
        yield survivor_line(3, 1)  # the second retry works
        log.wait_for("qpu_recovered")
        out.wait_for(4)
        yield survivor_line(4, 1)
        out.wait_for(5)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, clock=clock, out=out, log=log,
                                  arms=[("qpu-cold", 1)])
    assert gate.connects == 3
    errors = [(e["stage"], e["message"]) for e in events if e["type"] == "qpu_error"]
    assert errors == [("connect", "the solver is offline")] * 2
    programs = {answer["id"]: answer["program"] for answer in answers}
    assert [programs[f"7:{i}"] for i in range(5)] == ["gpu-256"] * 4 + ["cold-1048576"]
    assert [e["retry_s"] for e in events if e["type"] == "qpu_recovered"] == [60.0]


def test_a_call_that_works_resets_the_wait_before_the_next_retry(sidecar, tmp_path):
    clock, out, log = FakeClock(), Output(), EventLog()
    gate = Gate(sidecar.MockBackend(64), hold=False, connect_failures=2, fail_sample={"7:4"})

    def feed():
        yield ready_line()
        clock.now = 31.0
        yield survivor_line(0, 1)  # the first retry fails, so the wait doubles to 60 s, and the next one is due at 91 s
        log.wait_for("qpu_error", 2)
        out.wait_for(1)
        clock.now = 92.0
        yield survivor_line(1, 1)  # the second retry works
        log.wait_for("qpu_recovered")
        out.wait_for(2)
        clock.now = 93.0
        yield survivor_line(2, 1)  # a call that works: the wait is 30 s again
        out.wait_for(3)
        clock.now = 94.0
        yield survivor_line(3, 1)  # still a call that works
        out.wait_for(4)
        yield survivor_line(4, 1)  # this call fails at 94 s, so the next retry is due at 124 s, not at 154 s
        log.wait_for("qpu_error", 3)
        out.wait_for(5)
        clock.now = 125.0
        yield survivor_line(5, 1)  # the retry starts here: it is the fourth connection
        log.wait_for("qpu_recovered", 2)
        out.wait_for(6)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, clock=clock, out=out, log=log,
                                  arms=[("qpu-cold", 1)])
    assert gate.connects == 4
    assert [e["retry_s"] for e in events if e["type"] == "qpu_recovered"] == [60.0, 30.0]


def test_a_failed_call_after_a_recovery_doubles_the_wait_before_the_next_retry(sidecar, tmp_path):
    clock, out, log = FakeClock(), Output(), EventLog()
    # The connection works each time, and the calls of 7:0, 7:2, and 7:5 fail: a solver that connects and then refuses.
    gate = Gate(sidecar.MockBackend(64), hold=False, fail_sample={"7:0", "7:2", "7:5"})

    def feed():
        yield ready_line()
        clock.now = 500.0  # halfway through the window, so only the error can keep the QPU off
        yield survivor_line(0, 1)  # the first call fails at 500 s, so the first retry is due 30 s later, at 530 s
        out.wait_for(1)
        clock.now = 531.0
        yield survivor_line(1, 1)  # the retry starts at this survivor, and it works
        log.wait_for("qpu_recovered", 1)
        out.wait_for(2)
        clock.now = 532.0
        yield survivor_line(2, 1)  # the second call fails at 532 s, so the wait doubles: the retry is due at 592 s
        out.wait_for(3)
        clock.now = 591.0
        yield survivor_line(3, 1)  # not due yet. With a wait of 30 s, the retry would have started at 562 s
        out.wait_for(4)
        assert gate.connects == 2
        clock.now = 593.0
        yield survivor_line(4, 1)  # the second retry starts here, and it works
        log.wait_for("qpu_recovered", 2)
        out.wait_for(5)
        clock.now = 594.0
        yield survivor_line(5, 1)  # the third call fails at 594 s, so the wait doubles again: the retry is due at 714 s
        out.wait_for(6)
        clock.now = 713.0
        yield survivor_line(6, 1)  # not due yet
        out.wait_for(7)
        assert gate.connects == 3
        clock.now = 715.0
        yield survivor_line(7, 1)  # the third retry starts here, and it works
        log.wait_for("qpu_recovered", 3)
        out.wait_for(8)
        yield survivor_line(8, 1)  # the QPU is on, and this call works
        out.wait_for(9)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, clock=clock, out=out, log=log,
                                  arms=[("qpu-cold", 1)])
    assert [e["retry_s"] for e in events if e["type"] == "qpu_recovered"] == [30.0, 60.0, 120.0]
    assert gate.connects == 4  # the start, and one retry after each failed call
    assert gate.calls == ["7:0", "7:2", "7:5", "7:8"]
    programs = {answer["id"]: answer["program"] for answer in answers}
    assert [programs[f"7:{i}"] for i in range(9)] == ["gpu-256"] * 8 + ["cold-1048576"]
    # Three failed calls leave three open reservations. A wait of 30 s each time would have left more of them.
    assert regime_io.read_spend(tmp_path) == (ACCESS_US + 3 * RESERVATION_US, 3)


def test_the_wait_stops_doubling_at_the_cap_after_a_failed_retry_and_after_a_recovery(sidecar, tmp_path, monkeypatch):
    monkeypatch.setattr(sidecar, "RETRY_MAX_S", 90.0)
    clock, out, log = FakeClock(), Output(), EventLog()
    gate = Gate(sidecar.MockBackend(64), hold=False, connect_failures=3, fail_sample={"7:3"})

    def feed():
        yield ready_line()
        clock.now = 31.0
        yield survivor_line(0, 1)  # the start failed at 0 s. The retry at 30 s fails, so the wait doubles to 60 s: 91 s
        log.wait_for("qpu_error", 2)
        out.wait_for(1)
        clock.now = 92.0
        yield survivor_line(1, 1)  # that retry fails too. A wait of 120 s is over the cap, so it is 90 s: due at 182 s
        log.wait_for("qpu_error", 3)
        out.wait_for(2)
        clock.now = 183.0
        yield survivor_line(2, 1)  # the third retry works
        log.wait_for("qpu_recovered", 1)
        out.wait_for(3)
        clock.now = 184.0
        yield survivor_line(3, 1)  # the call fails at 184 s. A wait of 180 s is over the cap, so it is 90 s: due at 274 s
        out.wait_for(4)
        clock.now = 275.0
        yield survivor_line(4, 1)  # the retry starts here
        log.wait_for("qpu_recovered", 2)
        out.wait_for(5)

    run_sidecar(sidecar, tmp_path, feed(), backend=gate, clock=clock, out=out, log=log, arms=[("qpu-cold", 1)])
    assert gate.connects == 5
    assert [e["retry_s"] for e in read_events(tmp_path) if e["type"] == "qpu_recovered"] == [90.0, 90.0]


def test_the_real_backend_opens_a_new_session_after_a_failed_call(sidecar, tmp_path, fake_sampler):
    pytest.importorskip("quip_msa")
    spec = replay.load_spec(write_spec(tmp_path))

    def survivor(index):
        energies = lane_energies(spec, nonce_of(2, index), STATES3)
        return survivor_line(index, 2, best=min(energies), states=STATES3, energies=energies)

    fake_sampler.sample_failures = 1  # the first call raises
    clock, out, log = FakeClock(), Output(), EventLog()
    backend = sidecar.OceanBackend(spec, config(sidecar, tmp_path, arms=[("qpu-seeded", 1)]))

    def feed():
        yield ready_line(nodes=3, edges=2)
        clock.now = 500.0
        yield survivor(0)  # the call fails, and the retry is due at 530 s
        out.wait_for(1)
        clock.now = 531.0
        yield survivor(1)  # the retry opens a new session
        log.wait_for("qpu_recovered")
        out.wait_for(2)
        assert fake_sampler.sessions[0].closed.wait(5)  # no call runs on the old session, so it closes at once
        yield survivor(2)  # and the new session takes the call
        out.wait_for(3)

    try:
        answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=backend, clock=clock, out=out, log=log,
                                      arms=[("qpu-seeded", 1)], qpu_gate_milli=10_000_000)
    finally:
        backend.close()
    assert [answer["program"] for answer in answers] == ["gpu-256", "gpu-256", "seeded-1048576-b0250"]
    # Two sessions, each with its own topology. The old one closes at the recovery, when no call runs on it. The new
    # one closes when the sidecar ends.
    assert [call[0] for call in fake_sampler] == [
        "init", "connect", "topology", "sample", "init", "connect", "topology", "close", "sample", "close"]


def test_each_recovery_closes_the_session_that_it_replaces(sidecar, tmp_path, fake_sampler):
    pytest.importorskip("quip_msa")
    spec = replay.load_spec(write_spec(tmp_path))
    fake_sampler.sample_failures = 1  # the first call raises
    clock, out, log = FakeClock(), Output(), EventLog()
    backend = sidecar.OceanBackend(spec, config(sidecar, tmp_path, arms=[("qpu-seeded", 1)]))

    def feed():
        yield ready_line(nodes=3, edges=2)
        clock.now = 500.0
        yield spec_survivor(spec, 0)  # the call fails on the first session, and the retry is due at 530 s
        out.wait_for(1)
        clock.now = 531.0
        yield spec_survivor(spec, 1)  # the first retry opens the second session
        log.wait_for("qpu_recovered", 1)
        out.wait_for(2)
        assert fake_sampler.sessions[0].closed.wait(5)
        fake_sampler.sample_failures = 1  # the next call raises too
        clock.now = 532.0
        yield spec_survivor(spec, 2)  # the call fails on the second session, and the retry is due at 592 s
        out.wait_for(3)
        clock.now = 593.0
        yield spec_survivor(spec, 3)  # the second retry opens the third session
        log.wait_for("qpu_recovered", 2)
        out.wait_for(4)
        assert fake_sampler.sessions[1].closed.wait(5)
        yield spec_survivor(spec, 4)  # the third session takes the call
        out.wait_for(5)

    try:
        answers, _ = run_sidecar(sidecar, tmp_path, feed(), backend=backend, clock=clock, out=out, log=log,
                                 arms=[("qpu-seeded", 1)], qpu_gate_milli=10_000_000)
        # After two recoveries only the live session is open. A call that ends on it does not close it.
        assert [session.closed.is_set() for session in fake_sampler.sessions] == [True, True, False]
    finally:
        backend.close()
    assert [answer["program"] for answer in answers] == ["gpu-256"] * 4 + ["seeded-1048576-b0250"]
    assert fake_sampler.sessions[2].closed.is_set() and [call[0] for call in fake_sampler].count("close") == 3


def test_a_replaced_session_with_a_call_in_flight_closes_when_that_call_returns(sidecar, tmp_path, fake_sampler):
    pytest.importorskip("quip_msa")
    spec = replay.load_spec(write_spec(tmp_path))
    fake_sampler.hold_nonces = {bytes.fromhex(nonce_of(2, 0))}  # the first call waits at the gate
    fake_sampler.fail_nonces = {bytes.fromhex(nonce_of(2, 1))}  # the second call raises
    clock, out, log = FakeClock(), Output(), EventLog()
    cfg = config(sidecar, tmp_path, arms=[("qpu-seeded", 1)], in_flight=2)
    backend = sidecar.OceanBackend(spec, cfg)

    def feed():
        yield ready_line(nodes=3, edges=2)
        clock.now = 500.0
        yield spec_survivor(spec, 0)
        assert fake_sampler.entered.acquire(timeout=5)  # the first call runs on the first session
        yield spec_survivor(spec, 1)  # the second call fails on the first session, and the retry is due at 530 s
        out.wait_for(1)
        clock.now = 531.0
        yield spec_survivor(spec, 2)  # the retry opens the second session while the first call still runs
        log.wait_for("qpu_recovered")
        out.wait_for(2)
        fake_sampler.gate.set()  # the first call returns
        out.wait_for(3)
        assert fake_sampler.sessions[0].closed.wait(5)

    try:
        answers, _ = run_sidecar(sidecar, tmp_path, feed(), backend=backend, clock=clock, out=out, log=log, in_flight=2,
                                 arms=[("qpu-seeded", 1)], qpu_gate_milli=10_000_000)
    finally:
        backend.close()
    assert sorted(answer["id"] for answer in answers) == ["7:0", "7:1", "7:2"]
    # The first session closes after its call has returned, and not before. The second closes when the sidecar ends.
    assert [call[0] for call in fake_sampler] == [
        "init", "connect", "topology", "sample", "sample", "init", "connect", "topology", "released", "close", "close"]


def test_a_call_that_returns_after_the_launch_has_closed_the_sessions_closes_nothing_again(
    sidecar, tmp_path, fake_sampler
):
    pytest.importorskip("quip_msa")
    spec = replay.load_spec(write_spec(tmp_path))
    backend = sidecar.OceanBackend(spec, config(sidecar, tmp_path))
    nonce = nonce_of(2, 0)
    job = sidecar.Job("7:0", nonce, "qpu-cold", cascade_flow.Survivor(nonce, -5, STATES3, np.zeros(LANES, dtype=np.int64)))
    fake_sampler.hold_nonces = {bytes.fromhex(nonce)}
    outcome = []

    def run_call():
        try:
            outcome.append(backend.sample(job))
        except Exception as exc:  # the test reports it below
            outcome.append(exc)

    backend.connect()
    thread = threading.Thread(target=run_call)
    thread.start()
    assert fake_sampler.entered.acquire(timeout=5)  # the call runs on the first session
    backend.connect()  # a retry replaces the first session, and the call still runs on it
    backend.close()  # the launch ends: both sessions close
    assert [session.closed.is_set() for session in fake_sampler.sessions] == [True, True]
    fake_sampler.gate.set()
    thread.join(5)
    assert outcome == [ACCESS_US]  # the late call returns its access time, and it closes nothing
    assert [call[0] for call in fake_sampler].count("close") == 2


def test_a_closed_standard_output_at_the_end_of_the_input_is_a_normal_end(sidecar, tmp_path, monkeypatch):
    write_inputs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    out = Output()
    sample = sidecar.MockBackend.sample

    def held(self, job):
        entered.set()
        assert release.wait(5)
        return sample(self, job)

    monkeypatch.setattr(sidecar.MockBackend, "sample", held)
    clock = FakeClock()

    def feed():
        yield ready_line()
        clock.now = 500.0
        yield survivor_line(0, 1)
        assert entered.wait(5)
        out.closed = True  # the miner stops, and both of its pipes close
        release.set()

    flags = ["--mock", "--yes", "--arms", "qpu-cold:1"]
    assert sidecar.main(argv(tmp_path, *flags, cap="100", until=UNTIL), stdin=feed(), stdout=out, clock=clock) == 0
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "survivor", "qpu", "end"]  # an end, and no error
    assert out.lines == [] and regime_io.read_spend(tmp_path / "out") == (ACCESS_US, 0)


def test_a_closed_standard_output_before_the_input_ends_stops_the_run_at_the_next_line(sidecar, tmp_path):
    out = Output()
    out.closed = True
    lines = [ready_line(), survivor_line(0, 0), json.dumps({"ev": "stats", "salts": 1})]
    with pytest.raises(BrokenPipeError, match="closed its standard output"):
        run_sidecar(sidecar, tmp_path, lines, out=out)


def test_a_survivor_before_ready_is_a_protocol_error(sidecar, tmp_path):
    with pytest.raises(ValueError, match="before ready"):
        run_sidecar(sidecar, tmp_path, [survivor_line(0, 0)])


# The tests below run ``main``: the command line, the files it reads, the exit codes, and the journal it opens.


def write_inputs(tmp_path, programs=PROGRAMS, cut=GATE_MILLI, unit="milli"):
    (tmp_path / "programs.json").write_text(json.dumps({"programs": [{"name": name} for name in programs]}))
    rows = [{"miss_allowed": miss, "cut": value, "winners_lost": 0, "winner_miss": 0.0, "control_pass": 0.5}
            for miss, value in ((0.0, cut + 2000), (0.01, cut), (0.05, cut - 2000))]
    (tmp_path / "gates.json").write_text(json.dumps({"unit": unit, "winners": 100, "controls": 100, "table": rows}))


def argv(tmp_path, *extra, cap="10", until="2099-01-01T00:00:00Z"):
    flags = ["--out-dir", str(tmp_path / "out"), "--programs", str(tmp_path / "programs.json"),
             "--qpu-gates", str(tmp_path / "gates.json")]
    if cap is not None:
        flags += ["--qpu-seconds", cap]
    if until is not None:
        flags += ["--spend-until", until]
    return [*flags, *extra]


class SolverNotFoundError(Exception):
    """The class that Ocean raises for an offline solver."""


class Calls(list):
    connect_failures = 0  # how many of the first connections raise, as an offline solver makes them
    sample_failures = 0  # how many of the first calls raise

    def __init__(self):
        super().__init__()
        self.sessions = []  # each fake sampler, in the order that the script opened them
        self.fail_nonces = set()  # a call with one of these nonce seeds raises at once
        self.hold_nonces = set()  # a call with one of these nonce seeds waits at the gate
        self.gate = threading.Event()
        self.entered = threading.Semaphore(0)  # one release for each call that waits at the gate


@pytest.fixture
def fake_sampler(monkeypatch):
    """Put a fake in place of ``OceanSampler``. The list it returns holds every call that the script makes."""
    calls = Calls()

    class Fake:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            calls.sessions.append(self)
            self.closed = threading.Event()

        def ensure_connected(self):
            calls.append(("connect",))
            if calls.connect_failures:
                calls.connect_failures -= 1
                raise SolverNotFoundError("the solver is offline")

        def set_session_topology(self, nodes, edges):
            calls.append(("topology", list(nodes), list(edges)))

        def sample(self, nodes, h, edges, j, **kwargs):
            calls.append(("sample", h, j, kwargs))
            if calls.sample_failures:
                calls.sample_failures -= 1
                raise RuntimeError("the solver went offline")
            if kwargs["nonce_seed"] in calls.fail_nonces:
                raise RuntimeError("the solver went offline")
            if kwargs["nonce_seed"] in calls.hold_nonces:
                calls.entered.release()
                assert calls.gate.wait(5), "the test never opened the gate"
                calls.append(("released",))
            # The nodes come back in reverse order, so the script has to put them back in the order of the spec.
            reads = np.tile(np.array([1, 1, -1], dtype=np.int8), (kwargs["num_reads"], 1))
            return SampleResult(spins=reads, variables=[2, 1, 0], energies=[0.0] * kwargs["num_reads"],
                                device_access_time_us=ACCESS_US, num_reads=kwargs["num_reads"])

        def close(self):
            calls.append(("close",))
            self.closed.set()

    monkeypatch.setattr("quip_solver_dwave.ocean.OceanSampler", Fake)
    return calls


def test_without_yes_a_run_prints_its_cap_and_never_connects(sidecar, tmp_path, capsys, fake_sampler):
    write_inputs(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    assert sidecar.main(argv(tmp_path, "--spec", str(spec))) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # standard output is only the protocol
    for text in ("cap 10 s", "2099-01-01T00:00:00+00:00", "gpu-256:1,qpu-cold:1,qpu-seeded:1", "--yes"):
        assert text in captured.err
    assert fake_sampler == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("flags, named", [
    ({"cap": None}, "--qpu-seconds"), ({"cap": "nan"}, "--qpu-seconds"), ({"cap": "inf"}, "--qpu-seconds"),
    ({"cap": "0"}, "--qpu-seconds"), ({"cap": "-5"}, "--qpu-seconds"),
    ({"until": None}, "--spend-until"), ({"until": "tomorrow"}, "--spend-until"),
])
def test_a_run_needs_a_finite_positive_cap_and_a_spend_until_time(sidecar, tmp_path, capsys, flags, named):
    write_inputs(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        sidecar.main(argv(tmp_path, "--mock", "--yes", **flags))
    assert stopped.value.code == 2 and named in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_real_run_needs_a_spec(sidecar, tmp_path, capsys, fake_sampler):
    write_inputs(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        sidecar.main(argv(tmp_path, "--yes"))
    assert stopped.value.code == 2 and "--spec" in capsys.readouterr().err
    assert fake_sampler == []


@pytest.mark.parametrize("arms", ["gpu-256:1,gpu-256:2", "qpu-warm:1", "qpu-cold:0", "qpu-cold", "qpu-cold:x", ""])
def test_arms_name_known_arms_once_with_positive_weights(sidecar, tmp_path, capsys, arms):
    write_inputs(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        sidecar.main(argv(tmp_path, "--mock", "--yes", "--arms", arms))
    assert stopped.value.code == 2 and "--arms" in capsys.readouterr().err


def test_the_gate_is_the_cut_of_the_row_that_qpu_miss_names(sidecar, tmp_path, capsys):
    write_inputs(tmp_path)
    gates = str(tmp_path / "gates.json")
    assert [sidecar.load_gate(gates, miss) for miss in (0.0, 0.01, 0.05)] == [GATE_MILLI + 2000, GATE_MILLI,
                                                                              GATE_MILLI - 2000]
    with pytest.raises(SystemExit):
        sidecar.main(argv(tmp_path, "--mock", "--yes", "--qpu-miss", "0.02"))
    assert "0.02" in capsys.readouterr().err
    write_inputs(tmp_path, unit="micro")
    with pytest.raises(SystemExit):
        sidecar.main(argv(tmp_path, "--mock", "--yes"))
    assert "milli" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_clean_run_journals_its_configuration_and_ends(sidecar, tmp_path, capsys):
    write_inputs(tmp_path)
    out = Output()
    assert sidecar.main(argv(tmp_path, "--mock", "--yes", "--qpu-miss", "0.05"), stdin=io.StringIO(""), stdout=out) == 0
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "end"]
    start = events[0]
    assert (start["qpu_gate_milli"], start["qpu_miss"], start["qpu_seconds"], start["mock"]) == (
        GATE_MILLI - 2000, 0.05, 10.0, True)
    assert start["arms"] == [["gpu-256", 1], ["qpu-cold", 1], ["qpu-seeded", 1]]
    assert (start["in_flight"], start["queue"], start["reads"], start["anneal_us"], start["j_scale"]) == (
        16, 64, 64, 80, 0.5)
    assert (start["gpu_program"], start["cold_program"], start["seeded_program"]) == (
        "gpu-256", "cold-1048576", "seeded-1048576-b0250")
    assert start["spend_until"] == "2099-01-01T00:00:00+00:00" and start["window_s"] > 0 and start["spent_us"] == 0
    assert start["started_at"].endswith("+00:00") and start["spend_until_s"] == 4070908800.0
    assert out.lines == [] and capsys.readouterr().out == ""


UNTIL = "1970-01-01T00:16:40Z"  # 1,000 s after the start of the time of ``FakeClock``


def launch(module, tmp_path, clock, now, lines, *flags, cap="10"):
    """One launch of ``main`` at the fake time ``now``, with the cap ``cap`` over the window that ends at ``UNTIL``."""
    clock.now = now
    out = Output()
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    code = module.main(argv(tmp_path, "--mock", "--yes", *flags, cap=cap, until=UNTIL), stdin=stdin, stdout=out,
                       clock=clock)
    return code, out.messages()


def test_a_restart_keeps_the_origin_of_the_pace_and_of_the_time_axis(sidecar, tmp_path):
    write_inputs(tmp_path)
    clock = FakeClock()
    # A cap of 10 s over 1,000 s allows 10_000 us of access for each second since the origin. A call charges 50_000 us.
    # Launch 0 is at the origin, so its call is within the pace. Launch 1 restarts at the same instant. The origin is
    # stored, so the elapsed time is 0, the allowance is 0, and the spend of 50_000 us is ahead of it: the call skips.
    # A pace that restarted at each launch would let it run. Launch 2 comes 1 s after the origin, with an allowance of
    # 10_000 us, so it skips. Launch 3 comes 6 s after the origin, with an allowance of 60_000 us, so it runs.
    results = [launch(sidecar, tmp_path, clock, now, [ready_line(), survivor_line(index, 1)], "--arms", "qpu-cold:1")
               for index, now in enumerate((0.0, 0.0, 1.0, 6.0))]
    assert [code for code, _ in results] == [0, 0, 0, 0]
    assert [answers[0]["program"] for _, answers in results] == ["cold-1048576", "gpu-256", "gpu-256", "cold-1048576"]
    out = tmp_path / "out"
    events = read_events(out)
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [("7:1", "pace"), ("7:2", "pace")]
    assert regime_io.read_spend(out) == (2 * ACCESS_US, 0)
    # Each launch reads the origin from the first ``start`` event, so ``t`` keeps counting from it.
    starts = [e for e in events if e["type"] == "start"]
    assert [e["t"] for e in starts] == [0.0, 0.0, 1.0, 6.0]
    assert {e["started_at"] for e in starts} == {"1970-01-01T00:00:00+00:00"}
    times = [e["t"] for e in events]
    assert times == sorted(times) and times[-1] == 6.0


def test_a_torn_line_in_the_events_file_does_not_hide_the_origin(sidecar, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # A write cut short leaves half a line. The origin is the first ``start`` event that can be read.
    (out_dir / "events.jsonl").write_text('{"type": "error", "t": 0.0, "mess\n'
                                          '{"type": "start", "t": 0.0, "started_at": "1970-01-01T00:00:10+00:00"}\n')
    assert sidecar.read_first_start(out_dir / "events.jsonl")["started_at"] == "1970-01-01T00:00:10+00:00"
    assert sidecar.read_first_start(out_dir / "missing.jsonl") is None
    (out_dir / "events.jsonl").write_text('{"type": "survivor", "t": 1.0}\n')
    assert sidecar.read_first_start(out_dir / "events.jsonl") is None  # no launch has written its ``start`` yet


def test_a_second_sidecar_on_a_locked_out_dir_ends_with_an_error_and_spends_nothing(sidecar, tmp_path, capsys):
    write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = Output()
    stdin = io.StringIO(ready_line() + "\n" + survivor_line(0, 1) + "\n")
    with open(out_dir / "sidecar.lock", "a") as holder:  # another sidecar holds the out-dir
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        flags = ["--mock", "--yes", "--arms", "qpu-cold:1"]
        assert sidecar.main(argv(tmp_path, *flags), stdin=stdin, stdout=out, lock_timeout_s=0.2) == 1
    events = read_events(out_dir)
    assert [e["type"] for e in events] == ["error"] and "sidecar.lock" in events[0]["message"]
    assert "one sidecar for each out-dir" in events[0]["message"] and "sidecar.lock" in capsys.readouterr().err
    assert not (out_dir / regime_io.SPEND_JOURNAL).exists() and out.lines == [] and stdin.tell() == 0
    # The lock is gone with its holder, and a run takes it, finishes, and gives it back.
    assert sidecar.main(argv(tmp_path, "--mock", "--yes"), stdin=io.StringIO(""), stdout=out, lock_timeout_s=0.2) == 0
    assert sidecar.main(argv(tmp_path, "--mock", "--yes"), stdin=io.StringIO(""), stdout=out, lock_timeout_s=0.2) == 0
    assert [e["type"] for e in read_events(out_dir)] == ["error", "start", "end", "start", "end"]


def test_a_sidecar_waits_for_a_predecessor_that_is_still_draining(sidecar, tmp_path):
    write_inputs(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    holder = open(out_dir / "sidecar.lock", "a")  # the timer closes it, and the finally clause closes it again
    fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    release = threading.Timer(0.2, holder.close)  # closing the file gives the lock back
    release.start()
    try:
        assert sidecar.main(argv(tmp_path, "--mock", "--yes"), stdin=io.StringIO(""), stdout=Output(),
                            lock_timeout_s=10.0) == 0
    finally:
        release.join()
        holder.close()
    assert [e["type"] for e in read_events(out_dir)] == ["start", "end"]


@pytest.mark.parametrize("nodes, edges, named", [(4, 2, "4 nodes and 2 edges"), (3, 5, "3 nodes and 5 edges")])
def test_a_model_that_differs_from_the_spec_ends_the_run_before_any_call(
    sidecar, tmp_path, fake_sampler, nodes, edges, named
):
    write_inputs(tmp_path)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]],
                                     "allowed_h_milli": [-1000, 0, 1000], "allowed_j_milli": [-1000, 1000]}))
    out = Output()
    stdin = io.StringIO(ready_line(nodes=nodes, edges=edges) + "\n" + survivor_line(0, 2) + "\n")
    flags = ["--spec", str(spec_path), "--yes", "--arms", "qpu-seeded:1"]
    assert sidecar.main(argv(tmp_path, *flags), stdin=stdin, stdout=out) == 1
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "error"]
    assert named in events[1]["message"] and "3 nodes and 2 edges" in events[1]["message"]
    assert not [call for call in fake_sampler if call[0] == "sample"] and out.lines == []
    assert not (tmp_path / "out" / regime_io.SPEND_JOURNAL).exists()


@pytest.mark.parametrize("ready, survivor, named", [
    pytest.param(ready_line(lanes=256), survivor_line(0, 1), "256 lanes in 2 words", id="lanes beyond the words"),
    pytest.param(ready_line(), survivor_line(0, 1, energies=[-100] * 127), "127 energies", id="short energies"),
    pytest.param(ready_line(), survivor_line(0, 1, states=STATES[:127]), "do not hold 128 reads", id="short samples"),
])
def test_a_line_that_cannot_run_fails_before_the_call_and_the_charge(sidecar, tmp_path, ready, survivor, named):
    gate = Gate(sidecar.MockBackend(64), hold=False)
    with pytest.raises(ValueError, match=named):  # the seeded arm is the one that reads the energies and the states
        run_sidecar(sidecar, tmp_path, [ready, survivor], backend=gate, arms=[("qpu-seeded", 1)])
    assert gate.calls == [] and not (tmp_path / regime_io.SPEND_JOURNAL).exists()


def test_an_unknown_program_in_ready_ends_the_run_with_an_error(sidecar, tmp_path, capsys):
    write_inputs(tmp_path)
    out = Output()
    stdin = io.StringIO(ready_line(programs=["screen", "gpu-256", "seeded-1048576-b0250"]) + "\n")
    assert sidecar.main(argv(tmp_path, "--mock", "--yes"), stdin=stdin, stdout=out) == 1
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "error"] and "cold-1048576" in events[1]["message"]
    assert out.lines == [] and "cold-1048576" in capsys.readouterr().err


def test_a_program_the_file_lacks_ends_the_run_with_an_error(sidecar, tmp_path):
    write_inputs(tmp_path)
    stdin = io.StringIO(ready_line() + "\n")
    assert sidecar.main(argv(tmp_path, "--mock", "--yes", "--cold-program", "cold-999"), stdin=stdin,
                        stdout=Output()) == 1
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "error"] and "cold-999" in events[1]["message"]
    assert stdin.tell() == 0  # it stopped before it read the miner


def test_a_spend_journal_write_that_fails_keeps_the_qpu_off_and_the_survivor_runs_on_gpu_256(
    sidecar, tmp_path, monkeypatch, capsys
):
    write_inputs(tmp_path)
    clock, out = FakeClock(), Output()
    entered, release, failed = threading.Event(), threading.Event(), threading.Event()
    reader = threading.current_thread()
    sample, append_spend = sidecar.MockBackend.sample, regime_io.append_spend

    def held(self, job):  # 7:0 stays on the QPU until the test lets it go
        entered.set()
        assert release.wait(5)
        return sample(self, job)

    def full_disk(round_dir, event, cell, nonce, anneal_us, micros):
        if event == regime_io.SUBMIT and threading.current_thread() is not reader:
            failed.set()  # the line of 7:1, which the worker of 7:0 writes when it starts the next survivor
            raise OSError(errno.ENOSPC, "No space left on device")
        return append_spend(round_dir, event, cell, nonce, anneal_us, micros)

    monkeypatch.setattr(sidecar.MockBackend, "sample", held)
    monkeypatch.setattr(regime_io, "append_spend", full_disk)

    def feed():
        yield ready_line()
        clock.now = 500.0  # halfway through the window, so the pace allows both calls
        yield survivor_line(0, 1)
        yield survivor_line(1, 1)  # waits for the one slot
        assert entered.wait(5)
        release.set()
        assert failed.wait(5)  # the input stays open until the worker has met the failure

    flags = ["--mock", "--yes", "--arms", "qpu-cold:1", "--in-flight", "1"]
    assert sidecar.main(argv(tmp_path, *flags, cap="100", until=UNTIL), stdin=feed(), stdout=out, clock=clock) == 0
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "survivor", "survivor", "qpu", "qpu_error", "qpu_skip", "end"]
    error = events[4]
    assert (error["id"], error["stage"], error["exception"]) == ("7:1", "submit", "OSError")
    assert "No space left on device" in error["message"] and "No space left on device" in capsys.readouterr().err
    assert (events[5]["id"], events[5]["reason"]) == ("7:1", "error")
    # No call went without a reservation in the journal, so 7:1 runs on gpu-256.
    assert {message["id"]: message["program"] for message in out.messages()} == {
        "7:0": "cold-1048576", "7:1": "gpu-256"}
    assert regime_io.read_spend(tmp_path / "out") == (ACCESS_US, 0)


def test_a_spend_journal_that_cannot_be_read_keeps_the_qpu_off_and_the_sidecar_answers_every_survivor(
    sidecar, tmp_path, capsys
):
    write_inputs(tmp_path)
    (tmp_path / "out").mkdir()
    torn = '{"event": "submit", "ce'  # a write cut short
    (tmp_path / "out" / regime_io.SPEND_JOURNAL).write_text(torn)
    out = Output()
    stdin = io.StringIO(ready_line() + "\n" + survivor_line(0, 1) + "\n" + survivor_line(1, 2) + "\n")
    assert sidecar.main(argv(tmp_path, "--mock", "--yes"), stdin=stdin, stdout=out) == 0
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "qpu_error", "survivor", "qpu_skip", "survivor", "qpu_skip", "end"]
    assert events[0]["spent_us"] is None and events[1]["stage"] == "ledger"
    assert [e["reason"] for e in events if e["type"] == "qpu_skip"] == ["error", "error"]
    assert [message["program"] for message in out.messages()] == ["gpu-256", "gpu-256"]
    assert (tmp_path / "out" / regime_io.SPEND_JOURNAL).read_text() == torn  # the sidecar wrote nothing to it
    assert "QPU error at ledger" in capsys.readouterr().err


def test_a_failed_call_does_not_end_the_run(sidecar, tmp_path, monkeypatch, capsys):
    write_inputs(tmp_path)

    def offline(self, job):
        raise RuntimeError("the solver is offline")

    monkeypatch.setattr(sidecar.MockBackend, "sample", offline)
    out = Output()
    stdin = io.StringIO(ready_line() + "\n" + survivor_line(0, 1) + "\n")
    assert sidecar.main(argv(tmp_path, "--mock", "--yes", "--arms", "qpu-cold:1"), stdin=stdin, stdout=out) == 0
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "survivor", "qpu_error", "qpu_skip", "end"]
    assert "the solver is offline" in events[2]["message"] and "RuntimeError" in capsys.readouterr().err
    assert [message["program"] for message in out.messages()] == ["gpu-256"]
    assert regime_io.read_spend(tmp_path / "out") == (RESERVATION_US, 1)


def write_spec(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps({"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]],
                                "allowed_h_milli": [-1000, 0, 1000], "allowed_j_milli": [-1000, 1000]}))
    return path


def lane_energies(spec, nonce, states):
    """The whole-unit energy of each row of ``states`` on the model of ``nonce``, from the definition."""
    h, j = replay.model_from_nonce(spec, nonce)
    edges = spec.dense_edges.tolist()
    return [round(sum(h[i] * s[i] for i in range(len(s))) + sum(j[k] * s[u] * s[v] for k, (u, v) in enumerate(edges)))
            for s in states.tolist()]


def spec_survivor(spec, index, first_byte=2):
    """A survivor line whose energies fit its lanes on the model of its nonce in ``spec``, as the lane check needs."""
    energies = lane_energies(spec, nonce_of(first_byte, index), STATES3)
    return survivor_line(index, first_byte, best=min(energies), states=STATES3, energies=energies)


def test_an_offline_solver_at_the_start_keeps_the_qpu_off_and_the_sidecar_answers_every_survivor(
    sidecar, tmp_path, fake_sampler, capsys
):
    write_inputs(tmp_path)
    spec_path = write_spec(tmp_path)
    fake_sampler.connect_failures = 1
    out = Output()
    lines = [ready_line(nodes=3, edges=2), survivor_line(0, 2, states=STATES3), survivor_line(1, 1, states=STATES3)]
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    flags = ["--spec", str(spec_path), "--yes", "--arms", "qpu-cold:1,qpu-seeded:1"]
    assert sidecar.main(argv(tmp_path, *flags), stdin=stdin, stdout=out) == 0
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "qpu_error", "survivor", "qpu_skip", "survivor", "qpu_skip", "end"]
    assert (events[1]["stage"], events[1]["exception"]) == ("connect", "SolverNotFoundError")
    assert [message["program"] for message in out.messages()] == ["gpu-256", "gpu-256"]
    assert not [call for call in fake_sampler if call[0] == "sample"]
    assert not (tmp_path / "out" / regime_io.SPEND_JOURNAL).exists()
    assert "QPU error at connect: SolverNotFoundError" in capsys.readouterr().err


def test_the_real_backend_opens_a_fresh_session_when_the_retry_works(sidecar, tmp_path, fake_sampler):
    pytest.importorskip("quip_msa")
    spec_path = write_spec(tmp_path)
    spec = replay.load_spec(spec_path)
    states = STATES3

    def survivor(index):
        nonce = nonce_of(2, index)
        energies = lane_energies(spec, nonce, states)
        return survivor_line(index, 2, best=min(energies), states=states, energies=energies)

    fake_sampler.connect_failures = 1
    clock, out, log = FakeClock(), Output(), EventLog()
    cfg = config(sidecar, tmp_path, arms=[("qpu-seeded", 1)])
    backend = sidecar.OceanBackend(spec, cfg)

    def feed():
        yield ready_line(nodes=3, edges=2)
        clock.now = 10.0  # the start failed at 0 s, so the retry is due at 30 s
        yield survivor(0)
        out.wait_for(1)
        clock.now = 31.0
        yield survivor(1)  # the retry starts at this survivor, which still runs on gpu-256
        log.wait_for("qpu_recovered")
        out.wait_for(2)
        clock.now = 32.0
        yield survivor(2)  # the new session takes the call
        out.wait_for(3)

    try:
        answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=backend, clock=clock, out=out, log=log,
                                      arms=[("qpu-seeded", 1)], qpu_gate_milli=10_000_000)
    finally:
        backend.close()
    assert [answer["program"] for answer in answers] == ["gpu-256", "gpu-256", "seeded-1048576-b0250"]
    kinds = [call[0] for call in fake_sampler]
    # The failed start closed its session. The retry opened a new one, and the old one was never reused.
    assert kinds == ["init", "connect", "close", "init", "connect", "topology", "sample", "close"]
    assert [e["type"] for e in events if e["type"] in ("qpu_error", "qpu_recovered", "lane_check")] == [
        "qpu_error", "qpu_recovered", "lane_check"]


def test_the_lanes_of_the_first_qpu_survivor_are_rescored_and_a_match_keeps_the_qpu_on(sidecar, tmp_path):
    gate = Gate(sidecar.MockBackend(64), hold=False, rescore=lambda job: job.survivor.energies * 1000)
    lines = [ready_line(), survivor_line(0, 1), survivor_line(1, 2)]
    answers, events = run_sidecar(sidecar, tmp_path, lines, backend=gate)
    (check,) = [e for e in events if e["type"] == "lane_check"]  # one check for the launch, not one for each survivor
    assert (check["id"], check["nonce"], check["lanes"]) == ("7:0", nonce_of(1, 0), LANES)
    assert {answer["id"]: answer["program"] for answer in answers} == {
        "7:0": "cold-1048576", "7:1": "seeded-1048576-b0250"}
    assert sorted(gate.calls) == ["7:0", "7:1"] and not any(e["type"] == "lane_order_mismatch" for e in events)


def scrambled(job):
    """The energies of the lanes of a survivor in the reverse order, in milli."""
    return job.survivor.energies[::-1] * 1000


def one_lane_off(job):
    """The right energies, with the last lane wrong."""
    rescored = job.survivor.energies * 1000
    rescored[-1] += 1000
    return rescored


@pytest.mark.parametrize("rescore, mismatched, first, expected, rescored", [
    pytest.param(scrambled, LANES, 0, -100_000, 27_000, id="the lane order is reversed"),
    pytest.param(one_lane_off, 1, LANES - 1, 27_000, 28_000, id="one lane differs, and it is the last"),
])
def test_lanes_that_do_not_fit_their_energies_switch_the_qpu_off_for_the_launch(
    sidecar, tmp_path, capsys, rescore, mismatched, first, expected, rescored
):
    gate = Gate(sidecar.MockBackend(64), hold=False, rescore=rescore)
    lines = [ready_line(), survivor_line(0, 1), survivor_line(1, 2)]
    answers, events = run_sidecar(sidecar, tmp_path, lines, backend=gate)
    (mismatch,) = [e for e in events if e["type"] == "lane_order_mismatch"]
    assert (mismatch["id"], mismatch["lanes"], mismatch["mismatched"], mismatch["first_lane"]) == (
        "7:0", LANES, mismatched, first)
    assert (mismatch["first_expected_milli"], mismatch["first_rescored_milli"]) == (expected, rescored)
    assert not any(e["type"] == "lane_check" for e in events)
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [("7:0", "lane_order"), ("7:1", "lane_order")]
    assert [answer["program"] for answer in answers] == ["gpu-256", "gpu-256"]
    assert gate.calls == [] and not (tmp_path / regime_io.SPEND_JOURNAL).exists()
    assert "lane order of the miner is not the order of --spec" in capsys.readouterr().err


def test_a_lane_check_that_fails_is_a_qpu_error_and_runs_again_after_the_qpu_comes_back(sidecar, tmp_path):
    clock, out, log = FakeClock(), Output(), EventLog()
    broken = {"on": True}

    def rescore(job):
        if broken["on"]:
            raise RuntimeError("the model cannot be drawn")
        return job.survivor.energies * 1000

    gate = Gate(sidecar.MockBackend(64), hold=False, rescore=rescore)

    def feed():
        yield ready_line()
        clock.now = 500.0
        yield survivor_line(0, 1)  # the check fails, and the QPU goes off
        out.wait_for(1)
        broken["on"] = False
        clock.now = 531.0
        yield survivor_line(1, 1)  # the retry of the connection starts here
        log.wait_for("qpu_recovered")
        out.wait_for(2)
        yield survivor_line(2, 1)  # the check works now, and the call goes
        out.wait_for(3)

    answers, events = run_sidecar(sidecar, tmp_path, feed(), backend=gate, clock=clock, out=out, log=log,
                                  arms=[("qpu-cold", 1)])
    (error,) = [e for e in events if e["type"] == "qpu_error"]
    assert (error["stage"], error["id"]) == ("lane_check", "7:0")
    assert [e["id"] for e in events if e["type"] == "lane_check"] == ["7:2"]
    assert [answer["program"] for answer in sorted(answers, key=lambda a: a["id"])] == [
        "gpu-256", "gpu-256", "cold-1048576"]


def test_a_recovery_between_the_lane_check_and_the_pump_does_not_let_a_call_skip_the_check(
    sidecar, tmp_path, monkeypatch
):
    gate = Gate(sidecar.MockBackend(64), hold=False, connect_failures=1,
                rescore=lambda job: job.survivor.energies * 1000)  # the start fails, so the QPU is off
    admit = sidecar.Sidecar._admit

    def recover_before_the_pump(self, job):
        if job.id == "7:1":  # the retry lands after the check was skipped for 7:1, and before the pump sees it
            with self._lock:
                self._error_off = False
        return admit(self, job)

    monkeypatch.setattr(sidecar.Sidecar, "_admit", recover_before_the_pump)
    lines = [ready_line(), survivor_line(0, 1), survivor_line(1, 1), survivor_line(2, 1)]
    answers, events = run_sidecar(sidecar, tmp_path, lines, backend=gate, arms=[("qpu-cold", 1)])
    kinds = [e["type"] for e in events]
    assert kinds.index("lane_check") < kinds.index("qpu")  # no call goes before the check
    assert gate.calls == ["7:2"]
    assert [(e["id"], e["reason"]) for e in events if e["type"] == "qpu_skip"] == [
        ("7:0", "error"), ("7:1", "lane_unchecked")]
    assert [answer["program"] for answer in sorted(answers, key=lambda a: a["id"])] == [
        "gpu-256", "gpu-256", "cold-1048576"]


def test_the_lanes_of_a_real_run_are_rescored_on_the_model_of_the_nonce(sidecar, tmp_path, fake_sampler):
    pytest.importorskip("quip_msa")
    write_inputs(tmp_path, cut=10_000_000)
    spec_path = write_spec(tmp_path)
    spec = replay.load_spec(spec_path)
    states = STATES3
    energies = lane_energies(spec, nonce_of(2, 0), states)
    assert energies != energies[::-1]  # the scrambled order below differs from the true one
    flags = ["--spec", str(spec_path), "--yes", "--arms", "qpu-seeded:1"]
    for given, kind in ((energies, "lane_check"), (energies[::-1], "lane_order_mismatch")):
        out_dir = tmp_path / kind
        line = survivor_line(0, 2, best=min(given), states=states, energies=given)
        stdin = io.StringIO(ready_line(nodes=3, edges=2) + "\n" + line + "\n")
        argv_flags = ["--out-dir", str(out_dir), "--programs", str(tmp_path / "programs.json"),
                      "--qpu-gates", str(tmp_path / "gates.json"), "--qpu-seconds", "10", "--spend-until",
                      "2099-01-01T00:00:00Z", *flags]
        out = Output()
        assert sidecar.main(argv_flags, stdin=stdin, stdout=out) == 0
        events = read_events(out_dir)
        assert kind in [e["type"] for e in events]
        expected = "seeded-1048576-b0250" if kind == "lane_check" else "gpu-256"
        assert [message["program"] for message in out.messages()] == [expected]
        assert bool([call for call in fake_sampler if call[0] == "sample"]) is (kind == "lane_check")
        fake_sampler.clear()


def test_a_mock_run_with_a_spec_checks_the_lanes_of_the_miner(sidecar, tmp_path):
    pytest.importorskip("quip_msa")
    write_inputs(tmp_path, cut=10_000_000)
    spec_path = write_spec(tmp_path)
    spec = replay.load_spec(spec_path)
    energies = lane_energies(spec, nonce_of(2, 0), STATES3)
    assert energies != energies[::-1]  # the scrambled order below differs from the true one
    for given, kind in ((energies, "lane_check"), (energies[::-1], "lane_order_mismatch")):
        out_dir = tmp_path / kind
        line = survivor_line(0, 2, best=min(given), states=STATES3, energies=given)
        stdin = io.StringIO(ready_line(nodes=3, edges=2) + "\n" + line + "\n")
        flags = ["--out-dir", str(out_dir), "--programs", str(tmp_path / "programs.json"),
                 "--qpu-gates", str(tmp_path / "gates.json"), "--qpu-seconds", "10", "--spend-until",
                 "2099-01-01T00:00:00Z", "--mock", "--spec", str(spec_path), "--yes", "--arms", "qpu-seeded:1"]
        out = Output()
        assert sidecar.main(flags, stdin=stdin, stdout=out) == 0
        events = read_events(out_dir)
        assert kind in [e["type"] for e in events]
        expected = "seeded-1048576-b0250" if kind == "lane_check" else "gpu-256"
        assert [message["program"] for message in out.messages()] == [expected]
        assert any(e["type"] == "qpu" for e in events) is (kind == "lane_check")  # the mock QPU ran, or it did not


@pytest.mark.parametrize("nodes, edges, named", [(4, 2, "4 nodes and 2 edges"), (3, 5, "3 nodes and 5 edges")])
def test_a_mock_run_with_a_spec_ends_the_run_when_the_model_differs_from_the_spec(
    sidecar, tmp_path, nodes, edges, named
):
    write_inputs(tmp_path)
    out = Output()
    stdin = io.StringIO(ready_line(nodes=nodes, edges=edges) + "\n" + survivor_line(0, 2) + "\n")
    flags = ["--mock", "--spec", str(write_spec(tmp_path)), "--yes", "--arms", "qpu-seeded:1"]
    assert sidecar.main(argv(tmp_path, *flags), stdin=stdin, stdout=out) == 1
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events] == ["start", "error"]
    assert named in events[1]["message"] and "3 nodes and 2 edges" in events[1]["message"]
    assert out.lines == [] and not (tmp_path / "out" / regime_io.SPEND_JOURNAL).exists()


@pytest.mark.parametrize("first, second", [(True, False), (False, True)])
def test_a_launch_with_another_mock_value_than_the_first_is_refused_before_it_spends(
    sidecar, tmp_path, fake_sampler, capsys, first, second
):
    write_inputs(tmp_path)
    spec_path = write_spec(tmp_path)

    def launch_with(mock):
        flags = ["--mock"] if mock else ["--spec", str(spec_path)]
        stdin = io.StringIO(ready_line(nodes=3, edges=2) + "\n" + survivor_line(0, 1, states=STATES3) + "\n")
        return sidecar.main(argv(tmp_path, *flags, "--yes", "--arms", "qpu-cold:1"), stdin=stdin, stdout=Output()), stdin

    assert launch_with(first)[0] == 0
    del fake_sampler[:]
    spent = regime_io.read_spend(tmp_path / "out")
    code, stdin = launch_with(second)
    assert code == 1
    events = read_events(tmp_path / "out")
    assert [e["type"] for e in events][-1] == "error" and [e["type"] for e in events].count("start") == 1
    assert f"started with mock={first}" in events[-1]["message"] and f"mock={second}" in events[-1]["message"]
    assert stdin.tell() == 0 and fake_sampler == [] and regime_io.read_spend(tmp_path / "out") == spent
    assert "Give a mock launch an out-dir of its own" in capsys.readouterr().err


def test_a_mock_run_answers_through_main_with_no_ocean_and_no_spec(sidecar, tmp_path, capsys):
    write_inputs(tmp_path)
    out = Output()
    stdin = io.StringIO("\n".join([ready_line(), survivor_line(0, 1), survivor_line(1, 0)]) + "\n")
    # The first QPU call of a run is always inside the pace. The survivor of the GPU arm never waits for it.
    assert sidecar.main(argv(tmp_path, "--mock", "--yes", "--arms", "gpu-256:1,qpu-cold:1"), stdin=stdin,
                        stdout=out) == 0
    answers = {answer["id"]: answer for answer in out.messages()}
    assert (answers["7:0"]["program"], answers["7:1"]["program"]) == ("cold-1048576", "gpu-256")
    assert capsys.readouterr().out == ""
    assert [e["type"] for e in read_events(tmp_path / "out")][-1] == "end"


@pytest.mark.parametrize("j_scale, auto_scale", [("1.0", None), ("0.5", False)])
def test_a_real_run_samples_each_survivor_through_the_sampler_and_journals_the_charge(
    sidecar, tmp_path, fake_sampler, j_scale, auto_scale
):
    pytest.importorskip("quip_msa")
    write_inputs(tmp_path, cut=10_000_000)
    spec_path = write_spec(tmp_path)
    spec = replay.load_spec(spec_path)
    energies = lane_energies(spec, nonce_of(2, 0), STATES3)  # the lanes fit their energies, as the lane check needs
    out = Output()
    line = survivor_line(0, 2, best=min(energies), states=STATES3, energies=energies)
    stdin = io.StringIO(ready_line(nodes=3, edges=2) + "\n" + line + "\n")
    flags = ["--spec", str(spec_path), "--yes", "--arms", "qpu-seeded:1", "--in-flight", "2", "--j-scale", j_scale]
    assert sidecar.main(argv(tmp_path, *flags), stdin=stdin, stdout=out) == 0

    assert fake_sampler[:3] == [("init", {"mock": False, "submit_workers": 2}), ("connect",),
                                ("topology", [0, 1, 2], [(0, 1), (1, 2)])]
    assert fake_sampler[-1] == ("close",)
    (_, h, j, kwargs), = [call for call in fake_sampler if call[0] == "sample"]
    nonce = nonce_of(2, 0)
    h_model, j_model = replay.model_from_nonce(spec, nonce)
    assert np.array_equal(h, h_model) and np.allclose(j, float(j_scale) * j_model)
    assert (kwargs["num_reads"], kwargs["anneal_time_us"]) == (64, 80) and kwargs["auto_scale"] is auto_scale
    assert kwargs["nonce_seed"] == bytes.fromhex(nonce)

    # The fake answers [-1, 1, 1] in spec order for every read, and the script scores it on the full model.
    field = sum(h_model[i] * row for i, row in enumerate([-1, 1, 1]))
    coupling = sum(j_model[k] * [-1, 1, 1][u] * [-1, 1, 1][v] for k, (u, v) in enumerate(spec.dense_edges.tolist()))
    events = read_events(tmp_path / "out")
    (qpu,) = [e for e in events if e["type"] == "qpu"]
    assert qpu["best_milli"] == round(1000 * (field + coupling)) and qpu["kept"] is True
    assert [e["id"] for e in events if e["type"] == "lane_check"] == ["7:0"]
    assert (qpu["access_us"], qpu["reads"]) == (ACCESS_US, 64)
    (answer,) = out.messages()
    assert answer["program"] == "seeded-1048576-b0250" and 1 <= answer["seeded"] <= 8
    assert unpack_lanes(answer["lanes"], 3, answer["seeded"])[0].tolist() == [-1, 1, 1]
    assert regime_io.read_spend(tmp_path / "out") == (ACCESS_US, 0)
