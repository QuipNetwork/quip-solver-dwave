"""The budget guard, the spend journal, the drift check, and the resume logic of regime_capture.py."""

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from quip_solver_dwave import regime_io, regimes
from quip_solver_dwave.defects import DefectInfo
from quip_solver_dwave.ocean import SampleResult

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "regime_capture.py"

# The working graph of the tests: a path of five qubits.
NODES = [10, 11, 12, 13, 14]
EDGES = [(10, 11), (11, 12), (12, 13), (13, 14)]
# The estimate of one job at 8 reads is 36,097 microseconds at 80 µs and 38,657 at 400 µs.
# Two models at both anneal times make 149,508 microseconds.
ESTIMATE_80_US = 36_097


def _load_module():
    spec = importlib.util.spec_from_file_location("regime_capture", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regime_capture = _load_module()


class _Sampler:
    """A double of ``OceanSampler``: it answers each job with spins of +1 and charges a set access time."""

    def __init__(self, access_us=40_000, live_edges=EDGES, rows=None, defect_info=None, during_job=None):
        self.access_us = access_us
        self.live_nodes = list(NODES)
        self.live_edges = list(live_edges)
        self.rows = rows
        self.defect_info = defect_info
        self.during_job = during_job
        self.jobs = []
        self.closed = False

    def ensure_connected(self):
        pass

    def set_session_topology(self, nodes, edges):
        pass

    def sample(self, nodes, h, edges, j, *, num_reads, anneal_time_us, nonce_seed, label):
        self.jobs.append((nonce_seed.hex(), anneal_time_us))
        if self.during_job is not None:
            self.during_job()
        rows = num_reads if self.rows is None else self.rows
        return SampleResult(
            spins=np.ones((rows, len(nodes)), dtype=np.int8),
            variables=[int(q) for q in nodes],
            energies=[0.0] * rows,
            device_access_time_us=self.access_us,
            num_reads=rows,
            defect_info=self.defect_info,
        )

    def close(self):
        self.closed = True


def _run(monkeypatch, tmp_path, sampler, *options):
    """Run main() for two models of native-pm1 at 8 reads, with ``sampler`` in place of the connection."""
    pytest.importorskip("quip_msa")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "nodes": NODES,
        "edges": [list(edge) for edge in EDGES],
        "allowed_h_milli": [0],
        "allowed_j_milli": [-1000, 1000],
    }))
    monkeypatch.setattr(regime_capture, "connect", lambda mock: sampler)
    monkeypatch.setattr(sys, "argv", [
        "regime_capture.py", "--cell", "native-pm1", "--spec", str(spec), "--round-dir", str(tmp_path / "round"),
        "--models", "2", "--reads", "8", *options,
    ])
    return regime_capture.main()


def _captures(tmp_path):
    return sorted((tmp_path / "round" / "native-pm1").glob("qpu-*/*.npz"))


def _capture(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, access_us=np.int64(1), spins=np.ones((8, 2), dtype=np.int8), energies=np.zeros(8))


def test_the_estimate_matches_the_access_time_model_of_the_specification():
    assert regime_capture.estimate_us(64, 80) == 52_393
    assert regime_capture.estimate_us(64, 400) == 72_873


def test_check_budget_counts_prior_spend_and_refuses_a_run_that_would_pass_the_cap():
    regime_capture.check_budget(60_000_000, 12_500_000, 75.0)
    with pytest.raises(SystemExit, match="62.6 s spent plus about 12.5 s planned is 75.1 s"):
        regime_capture.check_budget(62_600_000, 12_500_000, 75.0)


def test_pending_skips_a_capture_that_has_a_file(tmp_path):
    _capture(tmp_path / "qpu-80" / "aa.npz")
    todo = regime_capture.pending(tmp_path, ["aa", "bb"], [80, 400], 8)
    assert todo == [("bb", 80), ("aa", 400), ("bb", 400)]


def test_a_real_run_without_yes_submits_nothing(monkeypatch, tmp_path):
    sampler = _Sampler()
    assert _run(monkeypatch, tmp_path, sampler) == 2
    assert sampler.jobs == []
    assert regime_io.read_spend(tmp_path / "round") == (0, 0)


def test_a_dry_run_submits_nothing_even_with_yes(monkeypatch, tmp_path):
    sampler = _Sampler()
    assert _run(monkeypatch, tmp_path, sampler, "--dry-run", "--yes") == 0
    assert sampler.jobs == []


def test_a_run_refuses_to_start_when_its_estimate_would_pass_the_cap(monkeypatch, tmp_path):
    sampler = _Sampler()
    with pytest.raises(SystemExit, match="refusing to run"):
        _run(monkeypatch, tmp_path, sampler, "--yes", "--max-qpu-seconds", "0.149")
    assert sampler.jobs == []


def test_a_run_stops_before_the_job_that_would_pass_the_cap(monkeypatch, tmp_path):
    # Each job costs 40,000 microseconds, which is more than its estimate. Three jobs cost 120,000.
    # The estimate of the fourth job would bring the round to 158,657, over the cap of 150,000.
    sampler = _Sampler(access_us=40_000)
    assert _run(monkeypatch, tmp_path, sampler, "--yes", "--max-qpu-seconds", "0.15") == 3
    assert len(sampler.jobs) == 3
    assert regime_io.read_spend(tmp_path / "round") == (120_000, 0)
    assert sampler.closed


@pytest.mark.parametrize("cap", ["nan", "inf", "0", "-5"])
def test_a_cap_that_is_not_a_finite_positive_number_is_refused(monkeypatch, tmp_path, cap):
    sampler = _Sampler()
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, sampler, "--yes", "--max-qpu-seconds", cap)
    assert sampler.jobs == []


def test_the_journal_keeps_the_charge_of_a_job_whose_save_fails(monkeypatch, tmp_path):
    def full_disk(*_args, **_kwargs):
        raise OSError("no space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(regime_capture.regime_io, "atomic_savez", full_disk)
        with pytest.raises(OSError, match="no space left"):
            _run(patch, tmp_path, _Sampler(), "--yes")
    assert _captures(tmp_path) == []
    assert regime_io.read_spend(tmp_path / "round") == (40_000, 0)

    # The next run buys the lost capture again, and the round counts both jobs.
    assert _run(monkeypatch, tmp_path, _Sampler(), "--yes") == 0
    assert len(_captures(tmp_path)) == 4
    assert regime_io.read_spend(tmp_path / "round") == (200_000, 0)


def test_a_job_with_no_answer_counts_at_its_estimate_and_the_next_run_says_so(monkeypatch, tmp_path, capsys):
    def dropped():
        raise RuntimeError("the connection dropped")

    with pytest.raises(RuntimeError, match="the connection dropped"):
        _run(monkeypatch, tmp_path, _Sampler(during_job=dropped), "--yes")
    assert regime_io.read_spend(tmp_path / "round") == (ESTIMATE_80_US, 1)

    assert _run(monkeypatch, tmp_path, _Sampler(), "--dry-run") == 0
    assert "jobs with no recorded charge: 1" in capsys.readouterr().out


def test_a_mock_run_marks_its_files_and_spends_nothing(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _Sampler(), "--mock") == 0
    assert len(_captures(tmp_path)) == 4
    for path in _captures(tmp_path):
        with np.load(path) as capture:
            assert bool(capture["mock"]) and int(capture["access_us"]) == 0
    assert regime_io.read_spend(tmp_path / "round") == (0, 0)


def test_a_real_capture_file_says_that_the_mock_did_not_make_it(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _Sampler(), "--yes") == 0
    with np.load(_captures(tmp_path)[0]) as capture:
        assert not bool(capture["mock"]) and int(capture["access_us"]) == 40_000


def test_a_real_run_refuses_a_cell_that_holds_mock_captures(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _Sampler(), "--mock") == 0
    sampler = _Sampler()
    with pytest.raises(SystemExit, match="mock is True in .* and False in this run"):
        _run(monkeypatch, tmp_path, sampler, "--yes")
    assert sampler.jobs == []


def test_a_rerun_with_other_reads_is_refused(monkeypatch, tmp_path):
    # Without the manifest, a capture of 8 reads satisfied a rerun that asked for 64.
    assert _run(monkeypatch, tmp_path, _Sampler(), "--yes") == 0
    sampler = _Sampler()
    with pytest.raises(SystemExit, match="reads is 8 in .* and 64 in this run"):
        _run(monkeypatch, tmp_path, sampler, "--yes", "--reads", "64")
    assert sampler.jobs == []


def _model(edges) -> regimes.Model:
    nodes = np.array(NODES, dtype=np.int64)
    qpu_edges = np.array(edges, dtype=np.int64)
    h = np.zeros(len(nodes))
    j = np.ones(len(qpu_edges))
    dense = qpu_edges - 10
    return regimes.Model(h, dense, j, nodes, h, qpu_edges, j, dense, (j * 1000).astype(np.int64))


def test_drift_counts_the_qubits_and_the_couplers_that_the_live_chip_lacks():
    model = _model(EDGES)
    assert regime_capture.drift(model, NODES, EDGES) == (0, 0)
    # The chip lists a coupler in either direction.
    assert regime_capture.drift(model, NODES, [(11, 10), (12, 11), (13, 12), (14, 13)]) == (0, 0)
    assert regime_capture.drift(model, NODES, EDGES[:3]) == (0, 1)
    # Without qubit 12 the chip also lacks the two couplers of that qubit.
    assert regime_capture.drift(model, [10, 11, 13, 14], [(10, 11), (13, 14)]) == (1, 2)


def test_a_run_stops_before_any_spend_when_the_live_chip_lacks_a_coupler(monkeypatch, tmp_path):
    sampler = _Sampler(live_edges=EDGES[:3])
    with pytest.raises(SystemExit, match="lacks 0 of the qubits and 1 of the couplers of this cell"):
        _run(monkeypatch, tmp_path, sampler, "--yes")
    assert sampler.jobs == []
    assert regime_io.read_spend(tmp_path / "round") == (0, 0)
    assert sampler.closed


def test_allow_drift_runs_and_records_the_removed_couplers(monkeypatch, tmp_path):
    lost = DefectInfo(fixed_spins={}, energy_offset=0.0, removed_edges={(13, 14): 1.0})
    sampler = _Sampler(live_edges=EDGES[:3], defect_info=lost)
    assert _run(monkeypatch, tmp_path, sampler, "--yes", "--allow-drift") == 0
    with np.load(_captures(tmp_path)[0]) as capture:
        assert int(capture["removed_couplers"]) == 1 and int(capture["clamped"]) == 0


def test_a_short_answer_is_saved_and_then_stops_the_run(monkeypatch, tmp_path):
    sampler = _Sampler(rows=7)
    with pytest.raises(SystemExit, match="holds 7 rows, and the run asked for 8 reads"):
        _run(monkeypatch, tmp_path, sampler, "--yes")
    assert len(sampler.jobs) == 1 and len(_captures(tmp_path)) == 1
    assert regime_io.read_spend(tmp_path / "round") == (40_000, 0)


def test_a_run_counts_what_another_process_spends_meanwhile(monkeypatch, tmp_path):
    def another_process():
        regime_io.append_spend(tmp_path / "round", regime_io.SUBMIT, "diamond-pm1", "ff", 80, 52_393)
        regime_io.append_spend(tmp_path / "round", regime_io.CHARGE, "diamond-pm1", "ff", 80, 80_000_000)

    sampler = _Sampler(during_job=another_process)
    assert _run(monkeypatch, tmp_path, sampler, "--yes") == 3
    assert len(sampler.jobs) == 1


def test_a_resume_refuses_the_short_answer_that_stopped_the_previous_run(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match='holds 7 rows'):
        _run(monkeypatch, tmp_path, _Sampler(rows=7), '--yes')
    saved = _captures(tmp_path)[0].read_bytes()
    sampler = _Sampler()
    with pytest.raises(SystemExit, match='row count'):
        _run(monkeypatch, tmp_path, sampler, '--yes')
    assert sampler.jobs == []
    assert _captures(tmp_path)[0].read_bytes() == saved
    assert regime_io.read_spend(tmp_path / 'round') == (40_000, 0)


def test_a_capture_refuses_a_round_locked_by_another_process(monkeypatch, tmp_path):
    import subprocess

    round_dir = tmp_path / 'round'
    round_dir.mkdir()
    code = (
        'import fcntl,sys; '
        'f=open(sys.argv[1], "a"); fcntl.flock(f, fcntl.LOCK_EX); '
        'print("locked", flush=True); sys.stdin.readline()'
    )
    holder = subprocess.Popen(
        [sys.executable, '-c', code, str(round_dir / '.capture.lock')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == 'locked'
        sampler = _Sampler()
        with pytest.raises(SystemExit, match='another capture'):
            _run(monkeypatch, tmp_path, sampler, '--yes')
        assert sampler.jobs == []
        assert regime_io.read_spend(round_dir) == (0, 0)
    finally:
        holder.communicate('\n', timeout=10)
    assert _run(monkeypatch, tmp_path, _Sampler(), '--yes') == 0
