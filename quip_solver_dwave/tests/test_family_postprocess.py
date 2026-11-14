"""The family postprocess: raw reads kept, failures visible, billing untouched.

A fake ``quip-dwave-qpp`` stands in for the processor so the failure,
mismatch and stop paths run on any host. The success path runs the real
binary when it is built.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest

from quip_solver_core import miner_pb2, wire

from quip_solver_dwave import postprocess, qpp
from quip_solver_dwave.postprocess import (
    FAILED,
    INCOMPLETE,
    OK,
    PostprocessConfigInvalid,
    Postprocessor,
    postprocess_config_from_toml,
)
from quip_solver_dwave.tests.test_family_leases import (
    ACCESS_US,
    Coordinator,
    FakeDevice,
    admit,
    assert_closed_once,
    family_generator,
    has_done,
    lease_job,
    lease_messages,
    target,
    zephyr_topology,
)
from quip_solver_dwave.usage import UsageLedger

FAKE_QPP = """#!{python}
# A stand-in for quip-dwave-qpp. FAKE_QPP_MODE picks the behaviour:
# first (report seed 0 with its true energy), lie (report a wrong energy),
# fail (exit 1), hang (never finish).
import json, os, sys, time
import numpy as np
model, seeds, out, mode = sys.argv[1:5]
how = os.environ.get("FAKE_QPP_MODE", "first")
if how == "fail":
    sys.stderr.write("bad stages\\n"); sys.exit(1)
if how == "hang":
    time.sleep(60)
raw = open(model, "rb").read()
n, t, p, e = np.frombuffer(raw[:16], dtype="<i4")
rows = np.frombuffer(raw[16:16 + 12 * e], dtype="<i4").reshape(e, 3).astype(np.int64)
s = np.fromfile(seeds, dtype=np.int8).reshape(-1, n).astype(np.int64)[0]
best = int((s[rows[:, 0]] * s[rows[:, 1]] * rows[:, 2]).sum())
s.astype(np.int8).tofile(out)
report = {{"mode": mode, "start": best, "best": best + (1 if how == "lie" else 0),
          "trace": [[0, 0.0, best, best]], "cpu": 0.01, "args": sys.argv[5:]}}
print(json.dumps(report))
"""


@pytest.fixture
def fake_qpp(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "fake-qpp"
    path.write_text(FAKE_QPP.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(qpp.BINARY_ENV, str(path))
    monkeypatch.setenv("FAKE_QPP_MODE", "first")
    return path


def config_toml(records: Path, policy: str = "qpp-race768/1", **extra) -> str:
    lines = ["[postprocess]", f'records = "{records}"', f'"quip/twisted-ice-dimer" = "{policy}"']
    lines += [f"{k} = {v}" for k, v in extra.items()]
    return "\n".join(lines) + "\n"


# -- configuration -------------------------------------------------------------


@pytest.mark.parametrize("text", ["", 'budget = "30s"\n'])
def test_no_table_means_the_default_postprocess(text):
    config = postprocess_config_from_toml(text)
    assert config is not None and not config.explicit
    assert config.records is None and 1 <= config.threads <= postprocess.DEFAULT_MAX_THREADS
    policy = config.policies["quip/twisted-ice-dimer"]
    assert policy.name == "qpp-race2048-repair32/1"
    assert policy.params["stages"] == "8:64,2:256,1:1024" and policy.params["repair_depth"] == 32


def test_the_default_counts_the_processors_the_process_may_use(monkeypatch):
    # A container cpuset limits the affinity, not os.cpu_count().
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0}, raising=False)
    config = postprocess_config_from_toml("")
    assert config is not None and (config.threads, config.max_pending) == (1, 2)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)), raising=False)
    config = postprocess_config_from_toml("")
    assert config is not None
    assert (config.threads, config.max_pending) == (postprocess.DEFAULT_MAX_THREADS, 8)


def test_a_table_sets_no_queue_limit(tmp_path):
    config = postprocess_config_from_toml(config_toml(tmp_path))
    assert config is not None and config.max_pending is None


def test_a_full_queue_refuses_the_next_salt(monkeypatch, fake_qpp):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0}, raising=False)
    config = postprocess_config_from_toml("")
    assert config is not None
    processor = Postprocessor(config)
    gate = threading.Event()

    def hold() -> None:
        gate.wait()

    try:
        first, second = processor.submit(hold), processor.submit(hold)
        assert first is not None and second is not None
        assert processor.submit(hold) is None and processor.skipped == 1
        gate.set()
        first.result(timeout=5)
        second.result(timeout=5)
        again = None
        deadline = time.monotonic() + 5
        while again is None and time.monotonic() < deadline:
            again = processor.submit(hold)
        assert again is not None
    finally:
        gate.set()
        processor.close()


def test_false_turns_the_postprocess_off():
    assert postprocess_config_from_toml("postprocess = false\n") is None


def test_the_table_names_its_records_threads_and_policies(tmp_path):
    config = postprocess_config_from_toml(config_toml(tmp_path, threads=2))
    assert config is not None
    assert config.records == tmp_path and config.threads == 2
    assert config.policies["quip/twisted-ice-dimer"].name == "qpp-race768/1"
    race = config.policies["quip/twisted-ice-dimer"]
    assert race.mode == "race" and race.params["stages"] == "4:64,1:512"


@pytest.mark.parametrize(
    "text",
    [
        '[postprocess]\n"quip/twisted-ice-dimer" = "qpp-race768/1"\n',
        '[postprocess]\nrecords = "/x"\n',
        '[postprocess]\nrecords = "/x"\n"quip/twisted-ice-dimer" = "greedy"\n',
        '[postprocess]\nrecords = "/x"\n"quip/charged-ice" = "qpp-race768/1"\n',
        '[postprocess]\nrecords = "/x"\nthreads = 0\n"quip/twisted-ice-dimer" = "qpp-race768/1"\n',
        'postprocess = "on"\n',
    ],
)
def test_an_unusable_table_is_refused(text):
    with pytest.raises(PostprocessConfigInvalid):
        postprocess_config_from_toml(text)


def test_a_missing_processor_binary_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(qpp.BINARY_ENV, str(tmp_path / "absent"))
    config = postprocess_config_from_toml(config_toml(tmp_path / "records"))
    assert config is not None
    with pytest.raises(PostprocessConfigInvalid):
        Postprocessor(config)


# -- the dimer model from an instance -------------------------------------------


def _salt(count: int = 1):
    topo = zephyr_topology()
    admitted = admit(lease_job(family_generator(topo, count=count)), topo=topo)
    problem = admitted.problem(0)
    assert problem is not None
    return problem


def test_the_model_takes_its_tiles_and_pairs_from_the_instance_metadata():
    problem = _salt()
    model = postprocess.dimer_model(
        problem.nodes, problem.positions, problem.h_milli, problem.j_milli, problem.metadata
    )
    weight = {
        (min(u, v), max(u, v)): j for (u, v), j in zip(problem.positions.tolist(), problem.j_milli.tolist())
    }
    assert model.tiles.shape[1] == 4 and len(model.tiles)
    assert all(list(t) == sorted(t) for t in model.tiles.tolist())
    # Every tile edge is a strong bond, and every pair is a -1000 edge.
    for tile in model.tiles.tolist():
        for a in range(4):
            for b in range(a + 1, 4):
                assert abs(weight[(tile[a], tile[b])]) == 1000
    assert len(model.pairs) and all(weight[(a, b)] == -1000 for a, b in model.pairs.tolist())
    model.gauge()


def test_a_field_is_outside_the_model():
    problem = _salt()
    h = problem.h_milli.copy()
    h[0] = 5
    with pytest.raises(ValueError, match="field"):
        postprocess.dimer_model(problem.nodes, problem.positions, h, problem.j_milli, problem.metadata)


# -- one trial -------------------------------------------------------------------


def _run(processor: Postprocessor, problem, spins, stop=lambda: False):
    return processor.run(
        job_id=b"lease-1",
        salt_index=problem.index,
        family_id="quip/twisted-ice-dimer",
        nonce=problem.nonce,
        nodes=problem.nodes,
        positions=problem.positions,
        h_milli=problem.h_milli,
        j_milli=problem.j_milli,
        metadata=problem.metadata,
        raw_spins=spins,
        raw_energies=problem.energies_milli(spins),
        stop=stop,
    )


def _reads(problem, k: int = 4) -> np.ndarray:
    return np.random.default_rng(3).choice(np.array([-1, 1], dtype=np.int8), (k, len(problem.nodes)))


def _record(processor: Postprocessor, trial):
    assert processor.store is not None
    with np.load(processor.store.path(trial)) as z:
        return {k: z[k] for k in z.files}


def test_a_successful_trial_keeps_every_raw_read_and_the_final_state(tmp_path, fake_qpp):
    config = postprocess_config_from_toml(config_toml(tmp_path / "records"))
    assert config is not None
    processor = Postprocessor(config)
    problem = _salt()
    reads = _reads(problem)
    trial = _run(processor, problem, reads)
    assert trial.outcome == OK
    assert trial.processor.startswith("quip-dwave-qpp sha256:")
    assert trial.seed == int.from_bytes(problem.nonce[:8], "little")
    assert trial.host_s > 0 and trial.processor_cpu_s == 0.01
    rec = _record(processor, trial)
    assert (rec["raw_spins"] == reads).all()
    assert rec["raw_energies"].tolist() == problem.energies_milli(reads).tolist()
    assert rec["final_spins"].shape == (1, len(problem.nodes))
    assert rec["final_energies"].tolist() == problem.energies_milli(rec["final_spins"]).tolist()
    summary = json.loads(str(rec["summary"]))
    assert summary["outcome"] == OK and summary["policy"] == "qpp-race768/1"
    line = json.loads((tmp_path / "records" / "trials.jsonl").read_text().splitlines()[-1])
    assert line == summary


def test_the_default_postprocess_writes_no_record(tmp_path, fake_qpp):
    config = postprocess_config_from_toml("")
    assert config is not None
    processor = Postprocessor(config)
    problem = _salt()
    trial = _run(processor, problem, _reads(problem))
    assert trial.outcome == OK and trial.policy == "qpp-race2048-repair32/1"
    assert trial.final_spins.shape == (1, len(problem.nodes))
    assert processor.store is None
    assert [p.name for p in tmp_path.iterdir()] == [fake_qpp.name]


def test_the_policy_options_reach_the_processor(tmp_path, fake_qpp, monkeypatch):
    seen: List[dict] = []
    real = qpp.run

    def spy(model, seeds, mode, **kw):
        seen.append(dict(kw, mode=mode, seeds=len(seeds)))
        return real(model, seeds, mode, **kw)

    monkeypatch.setattr(qpp, "run", spy)
    config = postprocess_config_from_toml(config_toml(tmp_path))
    assert config is not None
    problem = _salt()
    _run(Postprocessor(config), problem, _reads(problem, 5))
    (call,) = seen
    assert call["mode"] == "race" and call["stages"] == "4:64,1:512"
    assert (call["b0"], call["b1"], call["single_every"]) == (6, 20, 4)
    assert (call["pop_first"], call["pop_count"], call["seeds"]) == (0, 5, 5)


@pytest.mark.parametrize(("mode", "match"), [("fail", "bad stages"), ("lie", "host rescore")])
def test_a_failed_processor_is_a_failed_trial_with_its_raw_reads(tmp_path, fake_qpp, monkeypatch, mode, match):
    monkeypatch.setenv("FAKE_QPP_MODE", mode)
    config = postprocess_config_from_toml(config_toml(tmp_path))
    assert config is not None
    processor = Postprocessor(config)
    problem = _salt()
    reads = _reads(problem)
    trial = _run(processor, problem, reads)
    assert trial.outcome == FAILED and match in (trial.error or "")
    rec = _record(processor, trial)
    assert (rec["raw_spins"] == reads).all()
    assert rec["final_spins"].size == 0
    assert json.loads(str(rec["summary"]))["outcome"] == FAILED


def test_a_stop_kills_the_processor_and_leaves_an_incomplete_trial(tmp_path, fake_qpp, monkeypatch):
    monkeypatch.setenv("FAKE_QPP_MODE", "hang")
    config = postprocess_config_from_toml(config_toml(tmp_path))
    assert config is not None
    processor = Postprocessor(config)
    problem = _salt()
    deadline = time.monotonic() + 0.3
    started = time.monotonic()
    trial = _run(processor, problem, _reads(problem), stop=lambda: time.monotonic() > deadline)
    assert time.monotonic() - started < 5
    assert trial.outcome == INCOMPLETE
    assert (_record(processor, trial)["raw_spins"].shape[0]) == 4


def test_a_stop_before_the_start_never_runs_the_processor(tmp_path, fake_qpp, monkeypatch):
    calls: List[int] = []
    monkeypatch.setattr(qpp, "run", lambda *a, **k: calls.append(1))
    config = postprocess_config_from_toml(config_toml(tmp_path))
    assert config is not None
    processor = Postprocessor(config)
    problem = _salt()
    trial = _run(processor, problem, _reads(problem), stop=lambda: True)
    assert trial.outcome == INCOMPLETE and not calls


@pytest.mark.skipif(
    os.environ.get(qpp.BINARY_ENV) is None and qpp.find_binary() is None,
    reason="quip-dwave-qpp is not built",
)
@pytest.mark.parametrize("policy", sorted(postprocess.POLICIES))
def test_the_real_processor_lowers_the_best_raw_read(tmp_path, policy):
    real = qpp.find_binary()
    assert real is not None and real.is_file()
    config = postprocess_config_from_toml(config_toml(tmp_path, policy))
    assert config is not None
    problem = _salt()
    reads = _reads(problem, 8)
    trial = _run(Postprocessor(config, binary=real), problem, reads)
    assert trial.outcome == OK, trial.error
    assert int(trial.final_energies[0]) <= int(problem.energies_milli(reads).min())


# -- the session ------------------------------------------------------------------


def _family_session(monkeypatch, tmp_path, device: FakeDevice, *, budget: bool = False) -> Coordinator:
    coord = Coordinator(monkeypatch, device)
    toml = config_toml(tmp_path / "records")
    if budget:
        toml = f'budget = "3000s"\nusage_db = "{tmp_path / "usage.db"}"\n' + toml
    coord.start(zephyr_topology(), target(), depth=2, toml=toml)
    if budget:
        coord.send(cancel=miner_pb2.Cancel(max_generation=1))
    return coord


def test_a_lease_salt_reports_the_final_state_beside_every_raw_read(monkeypatch, tmp_path, fake_qpp):
    device = FakeDevice()
    coord = _family_session(monkeypatch, tmp_path, device, budget=True)
    topo = zephyr_topology()
    job = lease_job(family_generator(topo, count=3))
    coord.send(job=job)
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    results, _ = lease_messages(sent)
    admitted = admit(job, topo=topo)
    assert done.salts_done == 3 and len(results) == 3
    for result in results:
        problem = admitted.problem(admitted.lease.index_of(result.salt))
        assert problem is not None
        # One final state, then the two raw reads, each scored exactly.
        assert len(result.solutions) == 3
        rows = np.array([wire.decode_spins_packed(s.spins, len(problem.nodes)) for s in result.solutions])
        assert [s.energy_milli for s in result.solutions] == problem.energies_milli(rows).tolist()
    lines = (tmp_path / "records" / "trials.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["salt_index"] for x in lines) == [0, 1, 2]
    assert all(json.loads(x)["outcome"] == OK for x in lines)
    # Billing is the device's access time alone; host time never reaches it.
    assert UsageLedger(str(tmp_path / "usage.db")).spent_us_since(0.0) == 3 * ACCESS_US


def test_a_failed_postprocess_still_reports_the_raw_reads(monkeypatch, tmp_path, fake_qpp):
    monkeypatch.setenv("FAKE_QPP_MODE", "fail")
    device = FakeDevice()
    coord = _family_session(monkeypatch, tmp_path, device)
    coord.send(job=lease_job(family_generator(zephyr_topology(), count=2)))
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    results, _ = lease_messages(sent)
    assert done.salts_done == 2
    assert all(len(r.solutions) == 2 for r in results)
    lines = (tmp_path / "records" / "trials.jsonl").read_text().splitlines()
    assert [json.loads(x)["outcome"] for x in lines] == [FAILED, FAILED]


def test_a_cancel_during_the_postprocess_closes_the_lease_and_bills_the_salt(monkeypatch, tmp_path, fake_qpp):
    monkeypatch.setenv("FAKE_QPP_MODE", "hang")
    device = FakeDevice()
    coord = _family_session(monkeypatch, tmp_path, device, budget=True)
    coord.send(job=lease_job(family_generator(zephyr_topology(), count=1)))
    records = tmp_path / "records"
    deadline = time.monotonic() + 10
    while not list(records.glob("*.npz")) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert list(records.glob("*.npz")), "the postprocess never started"
    coord.send(cancel=miner_pb2.Cancel(max_generation=7))
    coord.wait_for(has_done())
    sent = coord.finish()
    done = assert_closed_once(sent)
    assert done.salts_done == 0 and not lease_messages(sent)[0]
    (line,) = (records / "trials.jsonl").read_text().splitlines()
    assert json.loads(line)["outcome"] == INCOMPLETE
    assert UsageLedger(str(tmp_path / "usage.db")).spent_us_since(0.0) == ACCESS_US


def test_an_invalid_table_ends_the_session_at_configure(monkeypatch, tmp_path):
    monkeypatch.setenv(qpp.BINARY_ENV, str(tmp_path / "absent"))
    coord = Coordinator(monkeypatch, FakeDevice())
    coord.send(welcome=miner_pb2.Welcome(protocol_version=2))
    coord.send(
        configure=miner_pb2.Configure(queue_depth=1, idle_timeout_s=300, backend_toml=config_toml(tmp_path))
    )
    coord.thread.join(timeout=10)
    assert coord.exit_code == 64
    kinds = coord.kinds()
    assert "fatal" in kinds and "ready" not in kinds



def test_the_default_without_its_processor_reports_raw_reads(monkeypatch, tmp_path):
    monkeypatch.setenv(qpp.BINARY_ENV, str(tmp_path / "absent"))
    coord = Coordinator(monkeypatch, FakeDevice())
    topo = zephyr_topology()
    coord.start(topo, target(), depth=2, toml="")
    coord.send(job=lease_job(family_generator(topo, count=1)))
    coord.wait_for(has_done())
    sent = coord.finish()
    assert "fatal" not in [m.WhichOneof("msg") for m in sent]
    results, _ = lease_messages(sent)
    # The two raw reads of the salt, and no final state ahead of them.
    assert len(results) == 1 and len(results[0].solutions) == 2


def test_a_backlog_never_delays_results(monkeypatch, tmp_path, fake_qpp):
    # One processor that never finishes, and a queue of two. The two queued
    # salts wait; every later salt reports its raw reads at once.
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0}, raising=False)
    monkeypatch.setenv("FAKE_QPP_MODE", "hang")
    coord = Coordinator(monkeypatch, FakeDevice())
    topo = zephyr_topology()
    coord.start(topo, target(), depth=1, toml="")
    coord.send(job=lease_job(family_generator(topo, count=6)))
    seen: List[miner_pb2.MinerMsg] = []

    def four_results(sent) -> bool:
        seen[:] = sent
        return len(lease_messages(sent)[0]) == 4

    coord.wait_for(four_results)
    assert all(len(r.solutions) == 2 for r in lease_messages(seen)[0])
    coord.send(cancel=miner_pb2.Cancel(max_generation=7))
    coord.wait_for(has_done())
    assert_closed_once(coord.finish())


def test_the_postprocess_does_not_hold_the_qpu_slot(monkeypatch, tmp_path, fake_qpp):
    # Depth 1: one salt on the QPU at a time. Salt 1 must reach the device
    # while salt 0's processor still runs, or host work would idle the QPU.
    monkeypatch.setenv("FAKE_QPP_MODE", "hang")
    device = FakeDevice()
    coord = Coordinator(monkeypatch, device)
    coord.start(zephyr_topology(), target(), depth=1, toml=config_toml(tmp_path / "records"))
    coord.send(job=lease_job(family_generator(zephyr_topology(), count=2)))
    assert device.entered[1].wait(timeout=10), "salt 1 waited for salt 0's postprocess"
    coord.send(cancel=miner_pb2.Cancel(max_generation=7))
    coord.wait_for(has_done())
    assert_closed_once(coord.finish())
