"""The tables of regime_report.py, on a small round written by hand."""

import csv
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from quip_solver_dwave import regime_io

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "regime_report.py"
RUNGS = [8, 64]
EXTRA_RUNG = 128


def _load_module():
    spec = importlib.util.spec_from_file_location("regime_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regime_report = _load_module()


def _write_capture(path, best, clamped=0, removed=0, sum_max=4.0, mock=False, j_rms=1.0, j_abs_min=1.0):
    # Two reads of four spins. Read 0 holds two assets and read 1 holds three, for the clique table.
    reads = np.array([[1, 1, -1, -1], [1, 1, 1, -1]], dtype=np.int8)
    np.savez_compressed(
        path,
        spins=reads,
        energies=np.array([best, best + 5.0]),
        access_us=np.int64(0 if mock else 50_000),
        anneal_us=np.int64(80),
        mock=np.bool_(mock),
        clamped=np.int64(clamped),
        removed_couplers=np.int64(removed),
        broken_chain_share=np.float64(0.0),
        chain_strength=np.float64(0.0),
        j_min=np.float64(-1.0),
        j_max=np.float64(1.0),
        sum_min=np.float64(-4.0),
        sum_max=np.float64(sum_max),
        h_abs_max=np.float64(0.0),
        j_rms=np.float64(j_rms),
        j_abs_min=np.float64(j_abs_min),
    )


def _write_cell(round_dir, name, label, ladder, qpu_best, wall_s, k_share="", nonces=None, manifest=None, **capture):
    """``ladder[nonce][sweeps]`` is a classical best energy. ``qpu_best[nonce]`` is the QPU best at 80 µs.

    ``nonces`` are the models of the cell, and by default each one has a capture. ``capture`` goes to each file.
    """
    cell_dir = round_dir / name
    (cell_dir / "qpu-80").mkdir(parents=True)
    mock = bool(capture.get("mock", False))
    regime_io.atomic_write_json(cell_dir / "nonces.json", {"rng_seed": 1, "nonces": list(nonces or qpu_best)})
    regime_io.atomic_write_json(cell_dir / "manifest.json", {
        "topology_hash": "ab12", "reads": 2, "anneal_us": [80], "mock": mock, **(manifest or {}),
    })
    with open(cell_dir / "ladder.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nonce", "sweeps", "best_energy", "wall_s", "kernel_label", "k_share"])
        for nonce, rungs in ladder.items():
            for sweeps, best in rungs.items():
                writer.writerow([nonce, sweeps, repr(best), wall_s[sweeps], label, k_share])
    for nonce, best in qpu_best.items():
        _write_capture(
            cell_dir / "qpu-80" / f"{nonce}.npz", best,
            clamped=1 if nonce == "aa" else 0, removed=2 if nonce == "bb" else 0, **capture,
        )
        if not mock:
            regime_io.append_spend(round_dir, regime_io.SUBMIT, name, nonce, 80, 52_393)
            regime_io.append_spend(round_dir, regime_io.CHARGE, name, nonce, 80, 50_000)
    return cell_dir


@pytest.fixture
def round_dir(tmp_path):
    # The baseline: the first rung matches every model.
    _write_cell(
        tmp_path, "native-pm1", "cpu-msa",
        ladder={n: {8: -100.0 - i, 64: -110.0 - i} for i, n in enumerate(["aa", "bb", "cc", "dd"])},
        qpu_best={n: -100.0 - i for i, n in enumerate(["aa", "bb", "cc", "dd"])},
        wall_s={8: 0.1, 64: 0.8},
    )
    # The candidate: two models match at 64 sweeps, one at 8, and one stays unmatched after the extra rung.
    cell_dir = _write_cell(
        tmp_path, "diamond-pm1", "cpu-msa",
        ladder={
            "aa": {8: -90.0, 64: -100.0},
            "bb": {8: -91.0, 64: -101.0},
            "cc": {8: -92.0, 64: -99.0, EXTRA_RUNG: -99.5},
            "dd": {8: -103.0, 64: -104.0},
        },
        qpu_best={"aa": -100.0, "bb": -101.0, "cc": -102.0, "dd": -103.0},
        wall_s={8: 0.25, 64: 2.0, EXTRA_RUNG: 4.0},
        j_rms=0.5, j_abs_min=0.25,
    )
    with open(cell_dir / "seed.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["nonce", "qpu_lanes", "lite_lanes", "qpu_lane_best", "lite_lane_best", "wall_s"])
        writer.writerow(["aa", 4, 4, "-105.0", "-104.0", "1.0"])      # the QPU lanes win
        writer.writerow(["bb", 4, 4, "-105.0", "-104.0", "1.0"])      # the QPU lanes win
        writer.writerow(["cc", 3, 5, "-104.0", "-105.0", "1.0"])      # the lite lanes win
        writer.writerow(["dd", 4, 4, "-105.0", "-105.0", "1.0"])      # a tie counts for neither
        writer.writerow(["ee", 0, 8, "", "-105.0", "1.0"])            # no QPU lane, so no contest
    return tmp_path


def test_matching_counts_the_unmatched_models_and_takes_quartiles_over_the_matched_ones(round_dir):
    cell = regime_report.load_cell(round_dir / "diamond-pm1")
    found = regime_report.matching(cell, 80, RUNGS, EXTRA_RUNG)
    assert found["models"] == 4
    assert found["unmatched"] == 1                        # no rung reaches the QPU best energy of "cc"
    assert found["unmatched_share"] == 0.25
    assert found["at_rung"] == {8: 1, 64: 2}
    assert found["sweeps"] == (36.0, 64.0, 64.0)          # the quartiles of 8, 64, and 64
    assert found["time"] == (1.125, 2.0, 2.0)             # the median wall time of each matched rung
    assert found["extra"] == (0, 1)                       # the extra rung ran for one model and did not match it


def test_matching_skips_a_model_whose_ladder_is_not_complete(round_dir):
    cell = regime_report.load_cell(round_dir / "diamond-pm1")
    del cell["ladder"]["aa"][64]
    assert regime_report.matching(cell, 80, RUNGS, EXTRA_RUNG)["models"] == 3


def test_matching_counts_a_model_that_the_extra_rung_matches(round_dir):
    cell = regime_report.load_cell(round_dir / "diamond-pm1")
    cell["ladder"]["cc"][EXTRA_RUNG] = -102.0
    assert regime_report.matching(cell, 80, RUNGS, EXTRA_RUNG)["extra"] == (1, 1)


def test_the_report_has_a_row_for_each_cell_in_each_table(round_dir, monkeypatch):
    out = round_dir / "RESULTS.md"
    monkeypatch.setattr(sys, "argv", [
        "regime_report.py", "--round-dir", str(round_dir), "--out", str(out),
        "--rungs", "8", "64", "--extra-rung", str(EXTRA_RUNG), "--resamples", "50",
    ])
    assert regime_report.main() == 0
    text = out.read_text(encoding="utf-8")

    assert "QPU access time spent: 0.40 s." in text                  # the journal holds 8 charges of 50 ms
    assert "CPU kernels that the miner runs today" in text and "no GPU arm" in text
    # One capture has a clamped qubit, and another one lost 2 couplers.
    assert "| `diamond-pm1` | `cpu-msa` | 4 | 4 at 80 µs | 1 | 1 | 2 | 0 |" in text
    assert "| `native-pm1` | 80 | 4 | 0 | 0 | 0 to 0.49 | 8, 8, 8 | 0.1, 0.1, 0.1 | 0 of 0 |" in text
    assert "| `diamond-pm1` | 80 | 4 | 1 | 0.25 | 0.0456 to 0.699 | 36, 64, 64 | 1.12, 2, 2 | 0 of 1 |" in text
    # Seed lane wins. One model has 3 QPU lanes and 5 lite lanes.
    assert "| `diamond-pm1` | 4 | 2 | 1 | 1 | 4 | 1 |" in text
    # The energy scale: the RMS coupling and the smallest coupling, each as a share of the largest.
    assert "| `native-pm1` | 1 | 1 |" in text
    assert "| `diamond-pm1` | 0.5 | 0.25 |" in text
    assert "| `diamond-pm1` | 80 | QPU with deep, lite held fixed |" in text
    assert "## Clique cell" not in text
    assert "## Solver ranges" not in text


def test_the_report_gives_the_rung_shares_the_interval_and_the_condition_that_holds(round_dir, monkeypatch):
    out = round_dir / "RESULTS.md"
    monkeypatch.setattr(sys, "argv", [
        "regime_report.py", "--round-dir", str(round_dir), "--out", str(out),
        "--rungs", "8", "64", "--extra-rung", str(EXTRA_RUNG), "--resamples", "50",
    ])
    assert regime_report.main() == 0
    text = out.read_text(encoding="utf-8")

    # The share of models that match at each rung, and the unmatched share. Each row sums to 1.
    assert "| Cell | Anneal µs | Models | 8 sweeps | 64 sweeps | Unmatched |" in text
    assert "| `native-pm1` | 80 | 4 | 1 | 0 | 0 |" in text
    assert "| `diamond-pm1` | 80 | 4 | 0.25 | 0.5 | 0.25 |" in text
    # 64 sweeps against 8 is under 10, and 2.0 s against 0.1 s is over 10: the cost of the kernel made the time.
    # One unmatched model of 4 has an interval that spans 25%. The share of the baseline stands beside it.
    assert (
        "| `diamond-pm1` | 80 | 8 | 20 | kernel cost alone | 0.25 | 0.0456 to 0.699 | 0 | unmatched share, uncertain "
        "| yes |"
    ) in text
    # A sweep of the candidate costs 2.5 times a sweep of the baseline, so 8 is the least ratio that passes.
    assert "| `diamond-pm1` | 80 | 2.5 | 8 | 8 | yes |" in text


def test_the_resolution_table_shows_a_first_condition_that_cannot_hold(round_dir):
    # The baseline now matches at the top rung, so the ladder offers a sweep ratio of 1 at most.
    for index, nonce in enumerate(["aa", "bb", "cc", "dd"]):
        _write_capture(round_dir / "native-pm1" / "qpu-80" / f"{nonce}.npz", -110.0 - index)
    # A sweep of this cell costs half a sweep of the baseline, so no sweep ratio of the ladder passes.
    _write_cell(
        round_dir, "cubic-dimer-pm1", "cpu-msa",
        ladder={"aa": {8: -100.0, 64: -110.0}}, qpu_best={"aa": -100.0}, wall_s={8: 0.05, 64: 0.4},
    )
    names = ("native-pm1", "diamond-pm1", "cubic-dimer-pm1")
    cells = {name: regime_report.load_cell(round_dir / name) for name in names}
    match = {(name, 80): regime_report.matching(cell, 80, RUNGS, EXTRA_RUNG) for name, cell in cells.items()}
    table = regime_report.resolution_table(cells, match, RUNGS)
    assert "| `diamond-pm1` | 80 | 2.5 | 8 | 1 | no |" in table
    assert "| `cubic-dimer-pm1` | 80 | 0.5 | none | 1 | no |" in table


def test_the_clique_table_gives_the_share_of_reads_with_exactly_k_assets(round_dir, monkeypatch):
    _write_cell(
        round_dir, "clique-portfolio", "cpu-sa",
        ladder={"aa": {8: -1.5, 64: -1.75}, "bb": {8: -1.25, 64: -1.5}},
        qpu_best={"aa": -1.0, "bb": -1.0},
        wall_s={8: 0.01, 64: 0.08},
        k_share="0.7500",
        manifest={"clique_holdings": 2},
    )
    out = round_dir / "RESULTS.md"
    monkeypatch.setattr(sys, "argv", [
        "regime_report.py", "--round-dir", str(round_dir), "--out", str(out), "--rungs", "8", "64", "--resamples", "50",
    ])
    assert regime_report.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "| QPU, 80 µs | 2 / 4 (50%) |" in text
    # A 0.75 share of 2 reads matches no whole count, so none is invented.
    assert "| Kernel, all rungs | 0.75 mean share; counts not recoverable |" in text


def test_the_ranges_table_counts_the_models_outside_a_solver_range(round_dir, monkeypatch):
    # One baseline model has a qubit whose couplings sum to 16, over the limit of 15.
    _write_capture(round_dir / "native-pm1" / "qpu-80" / "aa.npz", -100.0, sum_max=16.0)
    properties = round_dir / "solver-properties.json"
    properties.write_text(json.dumps({"extended_j_range": [-2.0, 1.0], "per_qubit_coupling_range": [-18.0, 15.0]}))
    out = round_dir / "RESULTS.md"
    monkeypatch.setattr(sys, "argv", [
        "regime_report.py", "--round-dir", str(round_dir), "--out", str(out),
        "--rungs", "8", "64", "--resamples", "50", "--solver-properties", str(properties),
    ])
    assert regime_report.main() == 0
    text = out.read_text(encoding="utf-8")
    assert "| `native-pm1` | -1 | 1 | -4 | 16 | 0 | 1 of 4 |" in text
    assert "| `diamond-pm1` | -1 | 1 | -4 | 4 | 0 | 0 of 4 |" in text


def test_the_report_stops_when_the_round_has_no_cell(tmp_path, monkeypatch):
    argv = ["regime_report.py", "--round-dir", str(tmp_path), "--out", str(tmp_path / "RESULTS.md")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="no cell under"):
        regime_report.main()


def _report(round_dir, monkeypatch, *options):
    out = round_dir / "RESULTS.md"
    monkeypatch.setattr(sys, "argv", [
        "regime_report.py", "--round-dir", str(round_dir), "--out", str(out),
        "--rungs", "8", "64", "--resamples", "50", *options,
    ])
    assert regime_report.main() == 0
    return out.read_text(encoding="utf-8")


def _write_short_cell(round_dir):
    """A third cell of two models. Model "bb" has no capture and no run at 64 sweeps."""
    _write_cell(
        round_dir, "cubic-dimer-pm1", "cpu-msa",
        ladder={"aa": {8: -100.0, 64: -110.0}, "bb": {8: -100.0}},
        qpu_best={"aa": -100.0}, wall_s={8: 0.05, 64: 0.4}, nonces=["aa", "bb"],
    )


def test_the_report_refuses_a_round_that_lacks_a_capture_or_a_rung(round_dir, monkeypatch):
    _write_short_cell(round_dir)
    with pytest.raises(SystemExit, match="cubic-dimer-pm1: 1 of 2 captures at 80 µs, 1 of 2 runs at 64 sweeps"):
        _report(round_dir, monkeypatch)
    assert not (round_dir / "RESULTS.md").exists()


def test_the_report_counts_a_cell_that_has_captures_and_no_ladder(round_dir, monkeypatch):
    (round_dir / "diamond-pm1" / "ladder.csv").unlink()
    with pytest.raises(SystemExit, match="diamond-pm1: 4 of 4 runs at 8 sweeps, 4 of 4 runs at 64 sweeps"):
        _report(round_dir, monkeypatch)


def test_allow_incomplete_writes_the_report_and_gives_the_cell_no_verdict(round_dir, monkeypatch):
    _write_short_cell(round_dir)
    text = _report(round_dir, monkeypatch, "--allow-incomplete")
    assert "The round is not complete. `cubic-dimer-pm1` lacks 1 of 2 captures at 80 µs" in text
    rows = [line for line in text.splitlines() if line.startswith("| `cubic-dimer-pm1` | 80 |")]
    assert rows[-2].endswith("| incomplete |")                       # the screening rule comes before its resolution
    assert "| `diamond-pm1` | 80 | 8 | 20 | kernel cost alone |" in text and "| yes |" in text


def test_an_incomplete_baseline_leaves_every_cell_with_no_verdict(round_dir, monkeypatch):
    (round_dir / "native-pm1" / "qpu-80" / "dd.npz").unlink()
    text = _report(round_dir, monkeypatch, "--allow-incomplete")
    row = next(line for line in text.splitlines() if line.startswith("| `diamond-pm1` | 80 | 8 |"))
    assert row.endswith("| incomplete |")


def test_the_report_refuses_a_round_that_mixes_mock_captures_and_real_captures(round_dir, monkeypatch):
    _write_capture(round_dir / "diamond-pm1" / "qpu-80" / "aa.npz", -100.0, mock=True)
    with pytest.raises(SystemExit, match="mixes mock captures and real captures"):
        _report(round_dir, monkeypatch)


def test_a_mock_round_says_so_in_its_first_lines(tmp_path, monkeypatch):
    for name in ("native-pm1", "diamond-pm1"):
        _write_cell(
            tmp_path, name, "cpu-msa", ladder={"aa": {8: -100.0, 64: -110.0}}, qpu_best={"aa": -100.0},
            wall_s={8: 0.1, 64: 0.8}, mock=True,
        )
    text = _report(tmp_path, monkeypatch)
    assert "This is a mock round" in text and "QPU access time spent: 0.00 s." in text


def test_the_report_refuses_cells_of_different_topologies(round_dir, monkeypatch):
    _write_cell(
        round_dir, "cubic-dimer-pm1", "cpu-msa", ladder={"aa": {8: -100.0, 64: -110.0}}, qpu_best={"aa": -100.0},
        wall_s={8: 0.05, 64: 0.4}, manifest={"topology_hash": "cd34"},
    )
    with pytest.raises(SystemExit, match="different topologies"):
        _report(round_dir, monkeypatch)


def test_the_first_lines_count_a_job_with_no_recorded_charge(round_dir, monkeypatch):
    regime_io.append_spend(round_dir, regime_io.SUBMIT, "diamond-pm1", "aa", 80, 52_393)
    text = _report(round_dir, monkeypatch)
    assert "QPU access time spent: 0.45 s." in text                  # 8 charges of 50 ms and one estimate
    assert "Jobs with no recorded charge: 1." in text


def test_the_report_discloses_an_unavailable_per_qubit_range(round_dir, monkeypatch):
    properties = round_dir / 'solver-properties.json'
    properties.write_text(json.dumps({
        'extended_j_range': [-2.0, 1.0], 'per_qubit_coupling_range': None,
        'per_group_coupling_range': [-13.0, 10.0],
    }))
    text = _report(round_dir, monkeypatch, '--solver-properties', str(properties))
    assert 'Per-qubit coupling range is unavailable' in text
    assert 'Group sums were not recorded' in text
    assert '| `native-pm1` | -1 | 1 | -4 | 4 | 0 | 0 of 4 | n/a |' in text


def test_the_report_refuses_a_candidate_without_a_baseline(tmp_path, monkeypatch):
    _write_cell(tmp_path, 'diamond-pm1', 'cpu-msa', ladder={'aa': {8: -90.0, 64: -95.0}},
                qpu_best={'aa': -100.0}, wall_s={8: 0.1, 64: 0.8})
    with pytest.raises(SystemExit, match='native-pm1'):
        _report(tmp_path, monkeypatch)
    text = _report(tmp_path, monkeypatch, '--allow-incomplete')
    row = next(line for line in text.splitlines() if line.startswith('| `diamond-pm1` | 80 | n/a |'))
    assert row.endswith('| incomplete |')


def test_the_report_requires_baseline_coverage_at_every_compared_anneal(round_dir, monkeypatch):
    cell = round_dir / 'diamond-pm1'
    (cell / 'qpu-80').rename(cell / 'qpu-400')
    manifest = json.loads((cell / 'manifest.json').read_text())
    manifest['anneal_us'] = [400]
    (cell / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match='native-pm1.*400'):
        _report(round_dir, monkeypatch)
    text = _report(round_dir, monkeypatch, '--allow-incomplete')
    row = next(line for line in text.splitlines() if line.startswith('| `diamond-pm1` | 400 | n/a |'))
    assert row.endswith('| incomplete |')


@pytest.mark.parametrize('field', ['spins', 'energies'])
def test_the_report_refuses_a_saved_answer_with_the_wrong_row_count(round_dir, monkeypatch, field):
    path = round_dir / 'diamond-pm1' / 'qpu-80' / 'aa.npz'
    with np.load(path) as saved:
        arrays = dict(saved)
    arrays[field] = arrays[field][:-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(SystemExit, match='row count'):
        _report(round_dir, monkeypatch)


def test_seven_infeasible_reads_of_32000_never_render_as_perfect():
    # Round 1 clique: 7 of 500 kernel runs held k assets on 63 of 64 reads.
    cell = {"manifest": {"clique_holdings": 9, "reads": 64}, "qpu": {}, "k_share": [0.9844] * 7 + [1.0] * 493}
    table = regime_report.clique_table(cell, [])
    assert "| Kernel, all rungs | 31993 / 32000 (99.978125%) [reconstructed from rounded shares] |" in table


def test_a_share_below_one_never_renders_as_perfect():
    assert regime_report.fmt_share((7 * 63 / 64 + 493) / 500) == "0.9998"
    assert regime_report.fmt_share(1 - 1e-9) == "0.999999999"
    assert regime_report.fmt_share(1.0) == "1"
    assert regime_report.fmt_share(0.0) == "0"
    assert regime_report.fmt_share(0.75) == "0.75"
    assert regime_report.fmt_share(None) == "n/a"
