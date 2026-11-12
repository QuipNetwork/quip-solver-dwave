import importlib.util
import json
import threading
from pathlib import Path

import numpy as np
import pytest

from quip_solver_dwave import regime_io, replay
from quip_solver_dwave.ocean import SampleResult

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chain_qpu_capture.py"


def load():
    spec = importlib.util.spec_from_file_location("chain_qpu_capture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def journal(path):
    return [json.loads(line) for line in (path / regime_io.SPEND_JOURNAL).read_text().splitlines()]


def jobs(count, reservation_us=60_000):
    return [{"key": f"{i:064x}-r0", "reservation_us": reservation_us} for i in range(count)]


def test_estimate_follows_the_measured_access_time_model():
    module = load()
    # 64 reads at 80 us measured 49.9 ms, and 64 reads at 2,000 us measured 172.7 ms.
    assert module.estimate_us(64, 80) == pytest.approx(49_885, abs=1)
    assert module.estimate_us(64, 2000) == pytest.approx(172_765, abs=1)


def test_capture_runs_every_job_and_journals_submit_then_charge(tmp_path):
    module = load()
    done = []

    def sample(job):
        done.append(job["key"])
        return 50_000

    status, captured = module.capture(jobs(5), sample, tmp_path, "cell", 80, cap_seconds=10.0, in_flight=2)
    assert (status, captured) == (0, 5) and sorted(done) == [job["key"] for job in jobs(5)]
    events = journal(tmp_path)
    assert [e["event"] for e in events].count(regime_io.SUBMIT) == 5
    assert sum(e["us"] for e in events if e["event"] == regime_io.CHARGE) == 250_000
    assert regime_io.read_spend(tmp_path) == (250_000, 0)


def test_capture_stops_at_the_cap_and_charges_jobs_in_flight(tmp_path):
    module = load()
    release = threading.Event()
    started = []

    def sample(job):
        started.append(job["key"])
        release.wait(timeout=5)
        return 50_000

    # Two reservations of 60 ms fit a 0.15 s cap. The third would pass it.
    worker = threading.Timer(0.2, release.set)
    worker.start()
    status, captured = module.capture(jobs(6), sample, tmp_path, "cell", 80, cap_seconds=0.15, in_flight=4)
    worker.join()
    assert status == 3 and captured == 2 and len(started) == 2
    charged, open_jobs = regime_io.read_spend(tmp_path)
    assert (charged, open_jobs) == (100_000, 0)


def test_capture_never_keeps_more_jobs_in_flight_than_the_window(tmp_path):
    module = load()
    live, peak, lock = [0], [0], threading.Lock()
    full = threading.Event()

    def sample(job):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            if live[0] == 3:
                full.set()
        # The first jobs wait for a full window, so the peak does not depend on how fast the journal writes.
        full.wait(timeout=5)
        threading.Event().wait(0.01)
        with lock:
            live[0] -= 1
        return 1_000

    assert module.capture(jobs(12, 1_000), sample, tmp_path, "cell", 80, cap_seconds=10.0, in_flight=3) == (0, 12)
    assert peak[0] == 3


def test_a_failed_job_stops_new_submissions_and_keeps_its_reservation_open(tmp_path):
    module = load()

    def sample(job):
        if job["key"].startswith("0" * 63 + "1"):
            raise RuntimeError("solver offline")
        return 50_000

    with pytest.raises(RuntimeError, match="solver offline"):
        module.capture(jobs(6), sample, tmp_path, "cell", 80, cap_seconds=10.0, in_flight=1)
    charged, open_jobs = regime_io.read_spend(tmp_path)
    # Job 0 is charged. Job 1 failed after its submit line, so its reservation still counts.
    assert open_jobs == 1 and charged == 50_000 + 60_000


def test_pending_jobs_skip_captures_that_exist(tmp_path):
    module = load()
    (tmp_path / f"{'ab' * 32}-r0.npz").write_bytes(b"x")
    pending = module.pending_jobs(["ab" * 32, "cd" * 32], repeats=2, out_dir=tmp_path, reads=64, anneal_us=80)
    assert [job["key"] for job in pending] == [f"{'ab' * 32}-r1", f"{'cd' * 32}-r0", f"{'cd' * 32}-r1"]
    assert pending[0]["reservation_us"] == int(module.estimate_us(64, 80) * module.RESERVATION_MARGIN)


def test_full_energies_are_in_milli_and_sum_every_coupling():
    module = load()
    spins = np.array([[1, -1, 1], [1, 1, 1]], dtype=np.int8)
    edges = np.array([[0, 1], [1, 2]])
    assert module.full_energies_milli(spins, edges, np.zeros(3), np.array([1.0, -1.0])).tolist() == [0, 0]
    assert module.full_energies_milli(spins, edges, np.zeros(3), np.array([1.0, 1.0])).tolist() == [-2000, 2000]


def test_the_charge_is_journaled_before_the_save_runs_and_survives_its_failure(tmp_path):
    module = load()
    seen = []

    def save(job):
        seen.append(regime_io.read_spend(tmp_path))
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        module.capture(jobs(3), lambda job: 50_000, tmp_path, "cell", 80, cap_seconds=10.0, in_flight=1, save=save)
    # The save of job 0 already found its charge in the journal, and no job stays open after the failure.
    assert seen == [(50_000, 0)]
    assert regime_io.read_spend(tmp_path) == (50_000, 0)


def test_save_runs_for_each_job_after_its_charge(tmp_path):
    module = load()
    saved = []
    result = module.capture(jobs(4), lambda job: 50_000, tmp_path, "cell", 80, cap_seconds=10.0, in_flight=2,
                            save=lambda job: saved.append(job["key"]))
    assert result == (0, 4) and sorted(saved) == [job["key"] for job in jobs(4)]


def test_a_job_with_no_access_time_is_charged_its_reservation_and_stops_the_run(tmp_path):
    module = load()
    saved = []

    def sample(job):
        return 0 if job["key"].startswith("0" * 63 + "1") else 50_000

    with pytest.raises(SystemExit) as stopped:
        module.capture(jobs(5), sample, tmp_path, "cell", 80, cap_seconds=10.0, in_flight=1,
                       save=lambda job: saved.append(job["key"]))
    # Job 1 reported no access time, so the journal charges its reservation and nobody saves its reads.
    # Jobs 2 to 4 never start.
    assert stopped.value.code not in (None, 0)
    events = journal(tmp_path)
    assert [e["nonce"] for e in events if e["event"] == regime_io.SUBMIT] == [job["key"] for job in jobs(2)]
    assert [e["us"] for e in events if e["event"] == regime_io.CHARGE] == [50_000, 60_000]
    assert regime_io.read_spend(tmp_path) == (110_000, 0)
    assert saved == [jobs(1)[0]["key"]]


# The tests below run ``main`` against a fake in place of ``OceanSampler``, so none of them reaches the QPU.


def run_main(module, tmp_path, *flags, nonces=("ab" * 32,)):
    """Run ``main`` on a three-node topology with a nonzero field. Its output goes to ``tmp_path / "out"``."""
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]],
                                "allowed_h_milli": [-1000, 0, 1000], "allowed_j_milli": [-1000, 1000]}))
    nonce_file = tmp_path / "nonces.txt"
    nonce_file.write_text("".join(f"{nonce}\n" for nonce in nonces))
    argv = ["--spec", str(spec), "--nonces", str(nonce_file), "--out-dir", str(tmp_path / "out"), *flags]
    return module.main(argv)


def reference_energy_milli(spec, nonce, row):
    """The energy of one read on the model of ``nonce``, from the definition, with plain Python."""
    h, j = replay.model_from_nonce(spec, nonce)
    field = sum(h[i] * row[i] for i in range(len(row)))
    coupling = sum(j[k] * row[u] * row[v] for k, (u, v) in enumerate(spec.dense_edges.tolist()))
    return round(1000 * (field + coupling))


@pytest.fixture
def fake_sampler(monkeypatch):
    """Put a fake in place of ``OceanSampler``. The list it returns holds every call that the script makes."""
    calls = []

    class Fake:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def ensure_connected(self):
            calls.append(("connect",))

        def set_session_topology(self, nodes, edges):
            calls.append(("topology", list(nodes), list(edges)))

        def sample(self, nodes, h, edges, j, **kwargs):
            calls.append(("sample", h, j, kwargs))
            # The nodes come back in reverse order, so the script has to put them back in the order of the spec.
            reads = np.tile(np.array([1, 1, -1], dtype=np.int8), (kwargs["num_reads"], 1))
            return SampleResult(spins=reads, variables=[2, 1, 0], energies=[0.0] * kwargs["num_reads"],
                                device_access_time_us=50_000, num_reads=kwargs["num_reads"])

        def close(self):
            calls.append(("close",))

    monkeypatch.setattr("quip_solver_dwave.ocean.OceanSampler", Fake)
    return calls


def test_a_dry_run_prints_its_estimate_and_never_connects(tmp_path, capsys, fake_sampler):
    module = load()
    assert run_main(module, tmp_path, "--cap-seconds", "1", "--repeats", "2", "--dry-run") == 0
    assert "2 jobs to run, about 0.1 s of QPU access" in capsys.readouterr().out
    assert fake_sampler == [] and not (tmp_path / "out" / regime_io.SPEND_JOURNAL).exists()


def test_without_yes_a_run_prints_its_estimate_and_never_connects(tmp_path, capsys, fake_sampler):
    module = load()
    assert run_main(module, tmp_path, "--cap-seconds", "1") == 2
    output = capsys.readouterr().out
    assert "1 jobs to run" in output and "pass --yes" in output
    assert fake_sampler == [] and not (tmp_path / "out" / regime_io.SPEND_JOURNAL).exists()


def test_open_reservations_stop_a_run_even_with_yes(tmp_path, fake_sampler):
    module = load()
    regime_io.append_spend(tmp_path / "out", regime_io.SUBMIT, "cell", "key", 80, 60_000)
    with pytest.raises(SystemExit, match="open reservations"):
        run_main(module, tmp_path, "--cap-seconds", "1", "--yes")
    assert fake_sampler == []


@pytest.mark.parametrize("flags", [[], ["--cap-seconds", "nan"], ["--cap-seconds", "inf"], ["--cap-seconds", "0"]])
def test_a_run_needs_a_cap_that_is_finite_and_positive(tmp_path, capsys, fake_sampler, flags):
    module = load()
    with pytest.raises(SystemExit) as stopped:
        run_main(module, tmp_path, *flags, "--yes")
    assert stopped.value.code == 2 and "cap-seconds" in capsys.readouterr().err
    assert fake_sampler == []


@pytest.mark.parametrize("j_scale, auto_scale", [("1.0", None), ("0.5", False)])
def test_a_run_saves_each_capture_in_spec_order_and_journals_its_charge(tmp_path, fake_sampler, j_scale, auto_scale):
    pytest.importorskip("quip_msa")
    module = load()
    nonces = ["ab" * 32, "cd" * 32]
    flags = ["--cap-seconds", "10", "--reads", "4", "--in-flight", "2", "--j-scale", j_scale, "--yes"]
    assert run_main(module, tmp_path, *flags, nonces=nonces) == 0

    out = tmp_path / "out"
    spec = replay.load_spec(tmp_path / "spec.json")
    assert fake_sampler[:3] == [
        ("init", {"mock": False, "submit_workers": 2}), ("connect",), ("topology", [0, 1, 2], [(0, 1), (1, 2)]),
    ]
    assert fake_sampler[-1] == ("close",)
    sent = {call[3]["nonce_seed"]: call for call in fake_sampler if call[0] == "sample"}
    for nonce in nonces:
        _, h, j, kwargs = sent[bytes.fromhex(nonce)]
        h_model, j_model = replay.model_from_nonce(spec, nonce)
        assert np.array_equal(h, h_model) and np.allclose(j, float(j_scale) * j_model)
        assert kwargs["auto_scale"] is auto_scale and kwargs["num_reads"] == 4 and kwargs["anneal_time_us"] == 80
        saved = np.load(out / f"{nonce}-r0.npz")
        assert set(saved.files) == {"spins", "energies_milli", "access_us", "wall_s", "reads", "anneal_us", "j_scale"}
        assert saved["spins"].dtype == np.int8 and saved["spins"].tolist() == [[-1, 1, 1]] * 4
        assert saved["energies_milli"].dtype == np.int64
        assert saved["energies_milli"].tolist() == [reference_energy_milli(spec, nonce, [-1, 1, 1])] * 4
        assert saved["access_us"].dtype == np.int64 and int(saved["access_us"]) == 50_000
        assert (int(saved["reads"]), int(saved["anneal_us"]), float(saved["j_scale"])) == (4, 80, float(j_scale))
        assert saved["wall_s"].dtype == np.float64 and float(saved["wall_s"]) >= 0
    assert regime_io.read_spend(out) == (100_000, 0)


def test_a_decode_failure_after_the_answer_still_leaves_the_charge_recorded(tmp_path, fake_sampler, monkeypatch):
    pytest.importorskip("quip_msa")
    module = load()

    def broken(spins, variables, spec):
        raise ValueError("the reads lack spec nodes")

    monkeypatch.setattr(replay, "spec_order", broken)
    with pytest.raises(ValueError, match="lack spec nodes"):
        run_main(module, tmp_path, "--cap-seconds", "10", "--reads", "4", "--in-flight", "1", "--yes")
    # The QPU answered and the decode failed. The journal holds the charge, with no reservation left open.
    assert regime_io.read_spend(tmp_path / "out") == (50_000, 0)
    assert not list((tmp_path / "out").glob("*.npz"))
