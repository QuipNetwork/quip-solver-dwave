import importlib.util
import pathlib

import numpy as np
import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "capture_qpu_reads.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("capture_qpu_reads", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_qpu_reads = _load_module()


def test_atomic_savez_compressed_leaves_no_partial_file_at_final_path(tmp_path):
    path = tmp_path / "nonce.npz"
    capture_qpu_reads.atomic_savez_compressed(
        str(path), spins=np.ones((2, 3), dtype=np.int8), access_us=np.int64(5)
    )
    assert path.exists()
    loaded = np.load(path)
    assert loaded["spins"].tolist() == [[1, 1, 1], [1, 1, 1]]
    assert int(loaded["access_us"]) == 5
    # No stray temp files left behind that resume could mistake for a capture.
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_savez_compressed_cleans_up_temp_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "nonce.npz"

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(capture_qpu_reads.np, "savez_compressed", _boom)
    with pytest.raises(RuntimeError):
        capture_qpu_reads.atomic_savez_compressed(str(path), spins=np.ones((1, 1)))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_check_budget_allows_estimate_under_the_cap():
    capture_qpu_reads.check_budget(1, max_qpu_seconds=30.0)


def test_check_budget_refuses_estimate_over_the_cap():
    with pytest.raises(SystemExit):
        capture_qpu_reads.check_budget(1000, max_qpu_seconds=30.0)


class _FakeResult:
    def __init__(self, spins, variables, access_us):
        self.spins = spins
        self.variables = variables
        self.device_access_time_us = access_us


class _FakeSampler:
    """Stands in for OceanSampler: each read spends far more than the estimate."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    def ensure_connected(self):
        pass

    def set_session_topology(self, nodes, edges):
        pass

    warm_starts: list = []

    def sample(self, nodes, h, edges, j, num_reads, nonce_seed, label, warm_start=None):
        _FakeSampler.warm_starts.append(warm_start)
        spins = np.ones((num_reads, len(nodes)), dtype=np.int8)
        return _FakeResult(spins, list(nodes), access_us=200_000)

    def close(self):
        self.closed = True


def test_submit_loop_stops_at_max_qpu_seconds_and_reports_progress(tmp_path, monkeypatch, capsys):
    heavy = tmp_path / "heavy.csv"
    heavy.write_text(
        "set,nonce\nlow-energy,aa\nlow-energy,bb\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    class _Spec:
        nodes = np.array([1, 2, 3])
        edges = np.array([[1, 2], [2, 3]])
        dense_edges = np.array([[0, 1], [1, 2]])

    monkeypatch.setattr(capture_qpu_reads, "OceanSampler", _FakeSampler)
    monkeypatch.setattr(capture_qpu_reads.replay, "load_spec", lambda _path: _Spec())
    monkeypatch.setattr(
        capture_qpu_reads.replay, "model_from_nonce",
        lambda _spec, _nonce: (np.zeros(3), np.zeros(2)),
    )
    monkeypatch.setattr(
        capture_qpu_reads.replay, "spec_order",
        lambda spins, _variables, _spec: spins,
    )
    monkeypatch.setattr(
        capture_qpu_reads.sys, "argv",
        [
            "capture_qpu_reads.py",
            "--spec", "spec.json",
            "--heavy", str(heavy),
            "--out-dir", str(out_dir),
            "--per-set", "2",
            "--reads", "4",
            "--yes",
            "--max-qpu-seconds", "0.15",
        ],
    )
    rc = capture_qpu_reads.main()
    captured = capsys.readouterr()
    assert rc == 3
    assert "stopping" in captured.err
    assert "1 of 2 models done" in captured.err
    assert "0.1 s" in captured.err
    # The one model already paid for keeps its file; the rest were never submitted.
    written = sorted(p.name for p in out_dir.iterdir())
    assert len(written) == 1


def test_dry_run_never_constructs_or_connects_a_sampler(tmp_path, monkeypatch):
    heavy = tmp_path / "heavy.csv"
    heavy.write_text("set,nonce\nlow-energy,ab\n", encoding="utf-8")

    class ExplodingSampler:
        def __init__(self, *args, **kwargs):
            raise AssertionError("OceanSampler must not be constructed under --dry-run")

    monkeypatch.setattr(capture_qpu_reads, "OceanSampler", ExplodingSampler)
    monkeypatch.setattr(
        capture_qpu_reads.replay, "load_spec", lambda _path: object()
    )
    monkeypatch.setattr(
        capture_qpu_reads.sys, "argv",
        [
            "capture_qpu_reads.py",
            "--spec", "spec.json",
            "--heavy", str(heavy),
            "--out-dir", str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert capture_qpu_reads.main() == 0


def test_seed_sweeps_reverse_anneals_from_the_best_lite_state(tmp_path, monkeypatch):
    heavy = tmp_path / "heavy.csv"
    heavy.write_text("set,nonce\nlow-energy,aa\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    class _Spec:
        nodes = np.array([1, 2, 3])
        edges = np.array([[1, 2], [2, 3]])
        dense_edges = np.array([[0, 1], [1, 2]])

    class _Kernel:
        def sample(self, h, dense_edges, j, num_sweeps, num_reads, seed):
            assert num_sweeps == 512
            spins = np.array([[1, 1, 1], [-1, 1, -1]], dtype=np.int8)
            return spins, np.array([-10, -30], dtype=np.int64)

    _FakeSampler.warm_starts = []
    monkeypatch.setattr(capture_qpu_reads, "OceanSampler", _FakeSampler)
    monkeypatch.setattr(capture_qpu_reads, "load_kernel", lambda: _Kernel())
    monkeypatch.setattr(capture_qpu_reads.replay, "load_spec", lambda _path: _Spec())
    monkeypatch.setattr(
        capture_qpu_reads.replay, "model_from_nonce",
        lambda _spec, _nonce: (np.zeros(3), np.zeros(2)),
    )
    monkeypatch.setattr(
        capture_qpu_reads.replay, "spec_order", lambda spins, _variables, _spec: spins
    )
    monkeypatch.setattr(
        capture_qpu_reads.sys, "argv",
        [
            "capture_qpu_reads.py", "--spec", "spec.json", "--heavy", str(heavy),
            "--out-dir", str(out_dir), "--reads", "4", "--yes",
            "--seed-sweeps", "512", "--reversal-s", "0.7",
        ],
    )
    assert capture_qpu_reads.main() == 0
    (warm_start,) = _FakeSampler.warm_starts
    assert warm_start.state.tolist() == [-1, 1, -1]
    assert warm_start.reversal_s == 0.7
    loaded = np.load(out_dir / "aa.npz")
    assert int(loaded["seed_energy_milli"]) == -30
    assert float(loaded["reversal_s"]) == 0.7
