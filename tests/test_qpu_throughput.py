"""The overlapped QPU throughput capture: manifest, in-flight cap, spend journal, resume, analysis.

Fakes only. The "real" path runs against a thread-safe double injected
through ``qpu_throughput.connect``, as ``test_round2_capture.py`` does for
``round2_capture.py``, so no live sampler is ever constructed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import threading
import time

import numpy as np
import pytest

from quip_solver_dwave import regime_io, round2_io
from quip_solver_dwave.ocean import SampleResult

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "qpu_throughput.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qpu_throughput", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


qpu_throughput = _load_module()

PROPERTIES = {
    "chip_id": "mock-chip",
    "problem_timing_data": {
        "version": "1.0.0",
        "typical_programming_time": 33631.59,
        "reverse_annealing_with_reinit_prog_time_delta": 0.0,
        "reverse_annealing_without_reinit_prog_time_delta": 31.33,
        "default_programming_thermalization": 1000.0,
        "default_annealing_time": 20.0,
        "readout_time_model": "pwl_log_log",
        "readout_time_model_parameters": [
            0.0, 0.7118156236873552, 1.622919864592155, 2.5460233863267376, 3.6610550848533783,
            1.3562242666114566, 1.6328679675785214, 1.7857781469507206, 1.99055000334358, 1.9905608299940198,
        ],
        "qpu_delay_time_per_sample": 60.57,
        "reverse_annealing_with_reinit_delay_time_delta": -44.5,
        "reverse_annealing_without_reinit_delay_time_delta": -41.5,
        "default_readout_thermalization": 0.0,
        "decorrelation_max_nominal_anneal_time": 2000.0,
        "decorrelation_time_range": [0.0, 0.0],
    },
}

CELL = "native-pm1"
LABELS = [100, 101, 102]
EDGES = np.array([[0, 1], [1, 2]], dtype=np.int64)


def _nonce(i: int) -> str:
    return format(i, "064x")


def _fixture(tmp_path, count):
    """``count`` native models, each bundled and range-audited at scale 0.5."""
    nonces = [_nonce(i + 1) for i in range(count)]
    for i, nonce in enumerate(nonces):
        h = np.array([0.1, -0.2, 0.3]) * (i + 1)
        j = np.array([0.4, -0.5])
        manifest = {
            "cell": CELL, "nonce": nonce,
            "identity": {"topology_hash": "topo-fixture", "model_order_hash": "order-fixture"},
            "variable_labels": list(LABELS), "offset": 0.0,
        }
        round2_io.write_bundle(tmp_path / "bundles" / CELL / nonce, manifest, {"h": h, "edges": EDGES, "j": j})
        model_dir = tmp_path / "range-audit" / CELL / nonce
        model_dir.mkdir(parents=True)
        regime_io.atomic_write_json(model_dir / "manifest.json", {
            "cell": CELL, "nonce": nonce, "status": "ok", "ceiling": 2.0,
            "limiting_constraint": {"kind": "h", "label": 100, "value": 0.3, "bound": "upper"},
            "chain_strength": None, "chain_strength_ratio_to_logical": None,
            "scales": [{"requested": 0.5, "submitted_scale": 1.0}],
        })
        regime_io.atomic_savez(model_dir / "arrays.npz", h_050=h, j_050=j)
    (tmp_path / "properties.json").write_text(json.dumps(PROPERTIES), encoding="utf-8")
    return nonces


def _job_spec(tmp_path, nonces, arms, margin=1.2):
    return {
        "cell": CELL,
        "nonces": list(nonces),
        "requested_scale": 0.5,
        "anneal_us": 400,
        "arms": [{"reads": r, "concurrency": c} for r, c in arms],
        "order_seed": 7,
        "bundles_root": str(tmp_path / "bundles"),
        "range_audit_dir": str(tmp_path / "range-audit"),
        "properties_path": str(tmp_path / "properties.json"),
        "spec_path": None,
        "embeddings_dir": None,
        "reservation_margin": margin,
    }


def _manifest(tmp_path, count, arms, margin=1.2):
    nonces = _fixture(tmp_path, count)
    manifest = qpu_throughput.build_manifest(_job_spec(tmp_path, nonces, arms, margin))
    path = tmp_path / "manifest.json"
    regime_io.atomic_write_json(path, manifest)
    return manifest, path


def _approve(tmp_path, manifest_path, cap_seconds):
    manifest_hash = qpu_throughput.hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    path = tmp_path / "approval.json"
    path.write_text(json.dumps({"manifest_hash": manifest_hash, "cap_seconds": cap_seconds}), encoding="utf-8")
    return path


class _Sampler:
    """A thread-safe ``OceanSampler`` double that counts how many calls overlap."""

    def __init__(self, access_us=40_000, delay_s=0.05, fail_nonce=None):
        self.access_us = access_us
        self.delay_s = delay_s
        self.fail_nonce = fail_nonce
        self._lock = threading.Lock()
        self.active = 0
        self.max_active_by_reads: dict[int, int] = {}
        self.calls: list[dict] = []
        self.closed = False
        self.live_nodes = list(LABELS)
        self.live_edges = [(100, 101), (101, 102)]

    def ensure_connected(self):
        pass

    def set_session_topology(self, _nodes, _edges):
        pass

    def sample(self, nodes, _h, _edges, _j, *, num_reads, nonce_seed, auto_scale=None, **_unused):
        assert auto_scale is False
        with self._lock:
            self.active += 1
            peak = self.max_active_by_reads.get(num_reads, 0)
            self.max_active_by_reads[num_reads] = max(peak, self.active)
            self.calls.append({"nonce": nonce_seed.hex(), "reads": num_reads})
        try:
            time.sleep(self.delay_s)
            if nonce_seed.hex() == self.fail_nonce:
                raise RuntimeError("submit failed")
        finally:
            with self._lock:
                self.active -= 1
        return SampleResult(
            spins=np.ones((num_reads, len(nodes)), dtype=np.int8),
            variables=[int(q) for q in nodes],
            energies=[0.0] * num_reads,
            device_access_time_us=self.access_us,
            num_reads=num_reads,
        )

    def close(self):
        self.closed = True


def _capture(manifest_path, round_dir, *extra):
    return qpu_throughput.main(["capture", "--manifest", str(manifest_path), "--round-dir", str(round_dir), *extra])


@pytest.fixture(autouse=True)
def no_real_sampler(monkeypatch):
    """A test that forgets to patch ``connect`` for a real run fails here, before any network call."""
    real_connect = qpu_throughput.connect

    def guarded(mock, workers):
        if not mock:
            raise AssertionError("a test tried to connect a real (non-mock) sampler")
        return real_connect(mock, workers)

    monkeypatch.setattr(qpu_throughput, "connect", guarded)


def _journal(round_dir):
    path = round_dir / regime_io.SPEND_JOURNAL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# ----------------------------------------------------------------- manifest


def test_capture_keys_are_unique_across_arms_and_every_arm_shares_one_order(tmp_path):
    manifest, _ = _manifest(tmp_path, 3, [(64, 1), (64, 4), (1024, 4)])
    keys = [job["capture_key"] for job in manifest["jobs"]]
    assert len(keys) == 9 and len(set(keys)) == 9
    assert manifest["job_order"] == keys
    for arm in manifest["arms"]:
        nonces = [job["nonce"] for job in manifest["jobs"] if job["arm"] == arm["arm"]]
        assert nonces == manifest["model_order"]
    # Same reads, different concurrency: one model key, two capture keys.
    first = {(job["reads"], job["concurrency"]): job for job in manifest["jobs"] if job["nonce"] == _nonce(1)}
    assert first[(64, 1)]["model_capture_key"] == first[(64, 4)]["model_capture_key"]
    assert first[(64, 1)]["capture_key"] != first[(64, 4)]["capture_key"]
    assert manifest["total_reservation_us"] == pytest.approx(1.2 * manifest["total_estimate_us"])


def test_a_repeated_arm_refuses_to_build_a_manifest(tmp_path):
    nonces = _fixture(tmp_path, 2)
    with pytest.raises(ValueError, match="same capture key"):
        qpu_throughput.build_manifest(_job_spec(tmp_path, nonces, [(64, 4), (64, 4)]))


# ------------------------------------------------------------------ capture


def test_the_in_flight_count_never_passes_the_arm_concurrency(tmp_path, monkeypatch):
    _manifest_data, manifest_path = _manifest(tmp_path, 8, [(2, 1), (4, 3)])
    approval = _approve(tmp_path, manifest_path, 1000.0)
    sampler = _Sampler()
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: sampler)
    round_dir = tmp_path / "round"

    assert _capture(manifest_path, round_dir, "--approval", str(approval), "--yes") == 0
    assert sampler.max_active_by_reads == {2: 1, 4: 3}
    assert len(sampler.calls) == 16
    assert sampler.closed
    journal = _journal(round_dir)
    assert sum(e["event"] == regime_io.SUBMIT for e in journal) == 16
    assert sum(e["event"] == regime_io.CHARGE for e in journal) == 16
    assert regime_io.read_spend(round_dir) == (16 * 40_000, 0)


def test_reservations_stop_submitting_before_the_cap_would_pass(tmp_path, monkeypatch):
    manifest, manifest_path = _manifest(tmp_path, 6, [(8, 4)])
    reservation_us = manifest["jobs"][0]["reservation_us"]
    # Room for exactly three reservations at once; the fourth would pass the
    # cap while the first three are still in flight and uncharged.
    cap_seconds = (3 * reservation_us + 1) / 1e6
    approval = _approve(tmp_path, manifest_path, cap_seconds)
    sampler = _Sampler(access_us=int(reservation_us))
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: sampler)
    round_dir = tmp_path / "round"

    assert _capture(manifest_path, round_dir, "--approval", str(approval), "--yes") == 3
    assert len(sampler.calls) == 3
    spent, open_jobs = regime_io.read_spend(round_dir)
    assert open_jobs == 0
    assert spent / 1e6 <= cap_seconds


def test_a_real_capture_refuses_without_an_approval(tmp_path, monkeypatch):
    _manifest_data, manifest_path = _manifest(tmp_path, 1, [(8, 1)])
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: _Sampler())
    with pytest.raises(SystemExit, match="no approval"):
        _capture(manifest_path, tmp_path / "round", "--approval", str(tmp_path / "missing.json"), "--yes")


def test_a_failed_job_drains_the_others_and_leaves_every_charge_recorded(tmp_path, monkeypatch):
    manifest, manifest_path = _manifest(tmp_path, 6, [(8, 3)])
    approval = _approve(tmp_path, manifest_path, 1000.0)
    bad = manifest["model_order"][1]
    sampler = _Sampler(fail_nonce=bad)
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: sampler)
    round_dir = tmp_path / "round"

    with pytest.raises(RuntimeError, match="submit failed"):
        _capture(manifest_path, round_dir, "--approval", str(approval), "--yes")
    bad_key = next(job["capture_key"] for job in manifest["jobs"] if job["nonce"] == bad)
    failed = json.loads((round_dir / "failed_captures.json").read_text())
    assert set(failed) == {bad_key}
    # Every job but the failed one was charged; the failed one stays open.
    open_keys, charged = qpu_throughput.capture_tools.journal_capture_status(round_dir)
    assert open_keys == {bad_key}
    assert len(charged) == len(sampler.calls) - 1

    # The next run refuses before submitting anything.
    second = _Sampler()
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: second)
    with pytest.raises(SystemExit, match="previously failed"):
        _capture(manifest_path, round_dir, "--approval", str(approval), "--yes")
    assert second.calls == []


def test_an_unknown_charge_is_charged_at_the_reservation_and_marked_failed(tmp_path, monkeypatch):
    manifest, manifest_path = _manifest(tmp_path, 2, [(8, 2)])
    approval = _approve(tmp_path, manifest_path, 1000.0)
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: _Sampler(access_us=0))
    round_dir = tmp_path / "round"

    with pytest.raises(SystemExit, match="no QPU access time"):
        _capture(manifest_path, round_dir, "--approval", str(approval), "--yes")
    spent, open_jobs = regime_io.read_spend(round_dir)
    assert open_jobs == 0
    assert spent == sum(int(job["reservation_us"]) for job in manifest["jobs"])
    assert len(json.loads((round_dir / "failed_captures.json").read_text())) == 2


def test_resume_skips_every_saved_job(tmp_path, monkeypatch):
    manifest, manifest_path = _manifest(tmp_path, 3, [(8, 1), (16, 2)])
    round_dir = tmp_path / "round"
    first = _Sampler()
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: first)
    assert _capture(manifest_path, round_dir, "--mock") == 0
    assert len(first.calls) == 6

    second = _Sampler()
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: second)
    assert _capture(manifest_path, round_dir, "--mock") == 0
    assert second.calls == []

    job = manifest["jobs"][4]
    qpu_throughput.output_path(round_dir, job).unlink()
    third = _Sampler()
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: third)
    assert _capture(manifest_path, round_dir, "--mock") == 0
    assert third.calls == [{"nonce": job["nonce"], "reads": job["reads"]}]


def test_mock_and_real_never_share_a_round_directory(tmp_path, monkeypatch):
    _manifest_data, manifest_path = _manifest(tmp_path, 1, [(8, 1)])
    approval = _approve(tmp_path, manifest_path, 1000.0)
    monkeypatch.setattr(qpu_throughput, "connect", lambda *_args: _Sampler())
    round_dir = tmp_path / "round"
    assert _capture(manifest_path, round_dir, "--mock") == 0
    with pytest.raises(SystemExit, match="must not share"):
        _capture(manifest_path, round_dir, "--approval", str(approval), "--yes")


# ----------------------------------------------------------------- analysis


def test_arm_stats_arithmetic_on_synthetic_records():
    records = [
        {"run_id": "a", "submitted_s": 10.0, "resolved_s": 11.5, "wall_s": 1.5, "access_us": 100_000},
        {"run_id": "a", "submitted_s": 10.5, "resolved_s": 12.0, "wall_s": 1.5, "access_us": 100_000},
        {"run_id": "a", "submitted_s": 11.0, "resolved_s": 13.0, "wall_s": 2.0, "access_us": 200_000},
        # A resumed run: its own clock, its own span.
        {"run_id": "b", "submitted_s": 500.0, "resolved_s": 501.0, "wall_s": 1.0, "access_us": 100_000},
    ]
    stats = qpu_throughput.arm_stats(records)
    assert stats["runs"] == 2
    assert stats["makespan_s"] == pytest.approx(3.0 + 1.0)
    assert stats["models_per_s"] == pytest.approx(4 / 4.0)
    assert stats["wall_median_s"] == pytest.approx(1.5)
    assert stats["wall_p90_s"] == pytest.approx(float(np.percentile([1.5, 1.5, 2.0, 1.0], 90)))
    assert stats["access_total_s"] == pytest.approx(0.5)
    assert stats["access_median_s"] == pytest.approx(0.1)
    assert stats["access_limited_models_per_s"] == pytest.approx(8.0)
    assert stats["access_limited_over_achieved"] == pytest.approx(8.0)
    assert stats["achieved_over_access_limited"] == pytest.approx(0.125)


def test_quality_vs_reference_over_shared_models():
    result = qpu_throughput.quality_vs_reference({"a": -10.0, "b": -8.0, "c": -5.0}, {"a": -10.0, "b": -9.0})
    assert result["models"] == 2
    assert result["median_diff"] == pytest.approx(0.5)
    assert result["share_equal"] == pytest.approx(0.5)
    assert result["share_at_or_below"] == pytest.approx(0.5)


# --------------------------------------------------------------- end to end


def test_the_mock_runs_end_to_end_and_overlap_shortens_the_makespan(tmp_path, monkeypatch, capsys):
    manifest, manifest_path = _manifest(tmp_path, 3, [(8, 1), (8, 3)])

    def fast_mock(mock, workers):
        assert mock
        inner = qpu_throughput.LatencySampler(
            qpu_throughput.capture_tools.OfflineSampler(), latency_s=0.3, programming_us=1000, per_read_us=100
        )
        return qpu_throughput.RecordingOceanSampler(sampler=inner, mock=True, submit_workers=workers)

    monkeypatch.setattr(qpu_throughput, "connect", fast_mock)
    round_dir = tmp_path / "mock-round"
    assert _capture(manifest_path, round_dir, "--mock") == 0
    assert not (round_dir / regime_io.SPEND_JOURNAL).exists()
    assert json.loads((round_dir / "capture-mode.json").read_text()) == {"mock": True}

    for job in manifest["jobs"]:
        with np.load(qpu_throughput.output_path(round_dir, job)) as saved:
            assert str(saved["capture_key"]) == job["capture_key"]
            assert saved["spins"].shape == (8, 3)
            assert int(saved["qpu_access_time_us"]) == 1000 + 8 * 100
            assert json.loads(str(saved["timing_json"]))["qpu_access_time"] == 1800.0
            assert float(saved["best_energy"]) == pytest.approx(float(np.min(saved["energies"])))
            assert float(saved["wall_s"]) >= 0.3

    # The mock round stands in as its own reference: every best energy matches.
    assert qpu_throughput.main([
        "analyze", "--manifest", str(manifest_path), "--round-dir", str(round_dir),
        "--reference-round", str(round_dir / "reads-8-conc-1"),
    ]) == 0
    analysis = json.loads((round_dir / "analysis.json").read_text())
    serial, overlapped = analysis["arms"]
    assert serial["jobs"] == overlapped["jobs"] == 3
    assert serial["makespan_s"] >= 3 * 0.3
    assert overlapped["makespan_s"] < serial["makespan_s"] / 2
    assert serial["vs_reference"]["share_equal"] == 1.0
    assert "reads-8-conc-3" in capsys.readouterr().out
