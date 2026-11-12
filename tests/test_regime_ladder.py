"""The kernel-path guard, the extra-rung rule, the seed arm, and the resume helpers of regime_ladder.py."""

import importlib.util
import pathlib

import numpy as np
import pytest

from quip_solver_dwave import regimes

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "regime_ladder.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("regime_ladder", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regime_ladder = _load_module()


class _Kernel:
    """A kernel double: a seeded call fails when the problem is not on the multi-spin path."""

    def __init__(self, takes_seeded_jobs: bool) -> None:
        self.takes_seeded_jobs = takes_seeded_jobs

    def sample(self, h, edges, j, **kwargs):
        if "initial_spins" in kwargs and not self.takes_seeded_jobs:
            raise ValueError("the problem needs the scalar SA fallback")
        return np.ones((1, len(h)), dtype=np.int8), np.zeros(1, dtype=np.int64)


def _model(edges: np.ndarray) -> regimes.Model:
    n = int(edges.max()) + 1
    h = np.zeros(n)
    j = np.ones(len(edges))
    nodes = np.arange(n)
    return regimes.Model(h, edges, j, nodes, h, edges, j, edges, (j * 1000).astype(np.int64))


PATH = np.array([[0, 1], [1, 2]], dtype=np.int64)


def test_the_guard_passes_when_the_observed_kernel_is_the_expected_one():
    regime_ladder.guard_kernel_path(_Kernel(True), regimes.CPU_MSA, _model(PATH))
    regime_ladder.guard_kernel_path(_Kernel(False), regimes.CPU_SA, _model(PATH))


def test_the_guard_stops_a_cell_that_would_time_the_wrong_kernel():
    with pytest.raises(SystemExit, match="serves this cell with cpu-sa, and the cell expects cpu-msa"):
        regime_ladder.guard_kernel_path(_Kernel(False), regimes.CPU_MSA, _model(PATH))


def test_the_guard_stops_a_multi_spin_cell_with_more_than_63_unit_bonds_on_a_spin():
    star = np.array([[0, 1]] * 64, dtype=np.int64)
    with pytest.raises(SystemExit, match="64 unit bonds"):
        regime_ladder.guard_kernel_path(_Kernel(True), regimes.CPU_MSA, _model(star))


def test_each_rung_gets_its_own_kernel_seed():
    nonce = "f1a4bd779e0447d02a98c52ae85cdd7358bd7fb5da9bdda76f8a1797999229f9"
    seeds = {regime_ladder.rng_seed(nonce, sweeps) for sweeps in (512, 2048, 8192)}
    assert len(seeds) == 3
    assert all(0 <= seed < 1 << 63 for seed in seeds)


def test_the_extra_rung_runs_for_a_model_that_no_rung_matches_at_one_anneal_time():
    ladder = {8: -90.0, 64: -95.0}
    # No rung reaches the QPU best energy of 80 µs. The 64-sweep rung reaches the one of 400 µs.
    assert regime_ladder.needs_extra_rung(ladder, [8, 64], {80: -100.0, 400: -94.0})
    assert not regime_ladder.needs_extra_rung(ladder, [8, 64], {80: -95.0, 400: -94.0})


def test_a_model_with_a_matching_rung_gets_no_extra_rung_even_if_the_top_rung_stays_higher():
    # The 8-sweep run reaches the QPU best energy, and the 64-sweep run does not.
    assert not regime_ladder.needs_extra_rung({8: -100.0, 64: -99.0}, [8, 64], {80: -100.0})


def test_a_finished_extra_rung_does_not_count_as_a_matching_rung():
    assert regime_ladder.needs_extra_rung({8: -90.0, 64: -95.0, 128: -101.0}, [8, 64], {80: -100.0})


def test_the_extra_rung_waits_for_every_rung_and_for_a_capture():
    assert not regime_ladder.needs_extra_rung({8: -90.0}, [8, 64], {80: -100.0})
    assert not regime_ladder.needs_extra_rung({8: -90.0, 64: -95.0}, [8, 64], {})


class _SeedKernel:
    """A kernel double for the seed arm: a cold call returns set states, and a seeded call keeps its start states.

    The real kernel returns ``num_reads`` rows also when it gets fewer start states, so the double adds filler rows.
    """

    def __init__(self, lite_states, filler) -> None:
        self.lite_states = np.array(lite_states, dtype=np.int8)
        self.filler = filler
        self.calls = []

    def sample(self, h, edges, j, **kwargs):
        self.calls.append(kwargs)
        if "initial_spins" not in kwargs:
            return self.lite_states, None
        start = [list(state) for state in kwargs["initial_spins"]]
        return np.array(start + [self.filler] * (kwargs["num_reads"] - len(start)), dtype=np.int8), None


def test_the_seed_arm_scores_the_qpu_lanes_and_the_lite_lanes_apart(tmp_path):
    # On a path of four spins with J = +1, the energy in milli is 1000 * (s0 s1 + s1 s2 + s2 s3).
    ground, low, high, top = [1, -1, 1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]      # -3000, -1000, 1000, 3000
    capture = tmp_path / "aa.npz"
    np.savez_compressed(
        capture, spins=np.array([top, high, top, high], dtype=np.int8), energies=np.array([3000, 1000, 3000, 1000])
    )
    # The two best QPU reads are one state, so the QPU gets one lane. The lite run adds one new state, "low".
    kernel = _SeedKernel(lite_states=[low, low, high, high], filler=ground)
    nonce = "ab" * 32
    path4 = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64)
    row = regime_ladder.seed_row(
        kernel, lambda h, edges, j: (0.5, 8.0), _model(path4), nonce, capture, reads=4, lite_sweeps=8, seed_sweeps=32
    )

    # The lite lane ends lower than the QPU lane. The filler rows hold the ground state and belong to no lane,
    # so a slice that is too wide shows as -1000 for the QPU lane or as -3000 for the lite lane.
    assert row[:5] == [nonce, 1, 1, "1000.0", "-1000.0"]
    lite_call, seeded_call = kernel.calls
    assert lite_call["num_sweeps"] == 8 and "start_beta" not in lite_call
    assert seeded_call["initial_spins"].tolist() == [high, low]        # the QPU lane comes first
    assert seeded_call["num_sweeps"] == 32
    assert seeded_call["start_beta"] == 2.0                            # the geometric midpoint of 0.5 and 8.0
    assert seeded_call["seed"] == regime_ladder.rng_seed(nonce, 32) + 1


def test_done_keys_reads_back_what_a_run_wrote(tmp_path):
    path = tmp_path / "ladder.csv"
    handle, writer = regime_ladder.open_csv(path, regime_ladder.LADDER_HEADER)
    writer.writerow(["aa", 512, "-1.0", "0.1", "cpu-msa", ""])
    handle.close()
    assert regime_ladder.done_keys(path, ("nonce", "sweeps")) == {("aa", "512")}
    assert regime_ladder.read_ladder(path) == {"aa": {512: -1.0}}
    # A second open appends and does not repeat the header.
    handle, writer = regime_ladder.open_csv(path, regime_ladder.LADDER_HEADER)
    writer.writerow(["aa", 2048, "-2.0", "0.2", "cpu-msa", ""])
    handle.close()
    assert path.read_text().count("nonce,") == 1


def _run_small_ladder(tmp_path, monkeypatch, *options):
    import json
    import sys

    pytest.importorskip('quip_msa')
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'nodes': [0, 1, 2], 'edges': [[0, 1], [1, 2]],
        'allowed_h_milli': [0], 'allowed_j_milli': [-1000, 1000],
    }))
    monkeypatch.setattr(sys, 'argv', [
        'regime_ladder.py', '--cell', 'native-pm1', '--spec', str(spec),
        '--round-dir', str(tmp_path / 'round'), '--models', '1', '--reads', '4',
        '--rungs', '2', '8', '--seed-sweeps', '4', '--extra-rung', '16', '--threads', '1', *options,
    ])
    return regime_ladder.main()


def test_the_ladder_refuses_reads_that_differ_from_the_capture(tmp_path, monkeypatch):
    from quip_solver_dwave import regime_io

    regime_io.check_manifest(tmp_path / 'round' / 'native-pm1', reads=8)
    with pytest.raises(SystemExit, match='reads is 8'):
        _run_small_ladder(tmp_path, monkeypatch)


@pytest.mark.parametrize('options, field', [
    (['--reads', '8'], 'reads'),
    (['--seed-sweeps', '16'], 'seed_sweeps'),
    (['--rungs', '4', '8'], 'ladder_rungs'),
    (['--threads', '2'], 'threads'),
])
def test_the_ladder_refuses_resume_with_different_run_settings(tmp_path, monkeypatch, options, field):
    assert _run_small_ladder(tmp_path, monkeypatch) == 0
    saved = (tmp_path / 'round' / 'native-pm1' / 'ladder.csv').read_bytes()
    with pytest.raises(SystemExit, match=field):
        _run_small_ladder(tmp_path, monkeypatch, *options)
    assert (tmp_path / 'round' / 'native-pm1' / 'ladder.csv').read_bytes() == saved
