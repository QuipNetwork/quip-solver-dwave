import csv
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "seeded_sweep.py"


def _load_module():
    pytest.importorskip("quip_msa")
    spec = importlib.util.spec_from_file_location("seeded_sweep", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seeded_sweep = _load_module()

ROW = [
    "nonce", "arm", "start_fraction", "sweeps", "best_milli",
    "qpu_lanes", "qpu_lane_best_milli", "fill_lane_best_milli", "wall_ms",
    "fill_wall_ms",
]


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(seeded_sweep.HEADER)
        for row in rows:
            writer.writerow(row)


def test_done_rows_drops_a_malformed_last_line(tmp_path):
    path = tmp_path / "seeded.csv"
    _write_csv(path, [["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"]])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("n2,cold,,")  # truncated tail, no trailing newline
    done = seeded_sweep.done_rows(str(path))
    assert done == {("n1", "cold", "", 1024)}


def test_ensure_trailing_newline_appends_one_when_missing(tmp_path):
    path = tmp_path / "seeded.csv"
    path.write_bytes(b"n1,cold,,1024,-100,0,,,5,0")
    seeded_sweep.ensure_trailing_newline(str(path))
    assert path.read_bytes().endswith(b"\n")


def test_ensure_trailing_newline_is_a_no_op_when_already_present(tmp_path):
    path = tmp_path / "seeded.csv"
    path.write_bytes(b"n1,cold,,1024,-100,0,,,5,0\n")
    seeded_sweep.ensure_trailing_newline(str(path))
    assert path.read_bytes() == b"n1,cold,,1024,-100,0,,,5,0\n"


def test_summarise_skips_a_missing_cell_instead_of_raising(tmp_path, capsys):
    out = tmp_path / "seeded.csv"
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    # n1 has both cold rows and a seeded row; n2 has only a cold row (its
    # seeded row is missing, e.g. an interrupted run).
    _write_csv(out, [
        ["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"],
        ["n1", "seeded", "0.50", 1024, -110, 4, -110, -90, "6", "20"],
        ["n2", "cold", "", 1024, -80, 0, "", "", "5", "0"],
    ])
    for nonce, best in (("n1", -110), ("n2", -80)):
        np.savez(reads_dir / f"{nonce}.npz", energies_milli=np.array([best]))

    args = seeded_sweep.argparse.Namespace(out=str(out), reads_dir=str(reads_dir), ceiling_fraction="0.50")
    rc = seeded_sweep.summarise(args)
    text = capsys.readouterr().out

    assert rc == 0
    assert "skipped" in text
    assert text.count("skipped") == 1


class _Spec:
    # 6 nodes gives 2**6 = 64 distinct +-1 rows, one per QPU read, so pack_lanes'
    # de-duplication does not collapse the fixture into a single lane.
    nodes = np.array([1, 2, 3, 4, 5, 6])
    edges = np.array([[1, 2], [2, 3]])
    dense_edges = np.array([[0, 1], [1, 2]])


def _distinct_spins(count: int, width: int) -> np.ndarray:
    bits = ((np.arange(count)[:, None] >> np.arange(width)[None, :]) & 1).astype(np.int8)
    return np.where(bits == 0, -1, 1).astype(np.int8)


class _FakeKernel:
    """Records every call so a test can check which samples were run."""

    def __init__(self):
        self.calls = []

    def sample(self, h, dense_edges, j, num_sweeps, num_reads, seed, initial_spins=None, start_beta=None):
        self.calls.append(dict(num_sweeps=num_sweeps, num_reads=num_reads, initial_spins=initial_spins))
        # Offset by seed so a lite fill call does not draw the same rows as the
        # QPU reads fixture and get de-duplicated away by pack_lanes.
        spins = _distinct_spins(num_reads, len(h)) * (1 if seed % 2 == 0 else -1)
        energies = np.arange(num_reads, dtype=np.int64) * -10
        return spins, energies


def _patch_run_deps(monkeypatch, kernel):
    monkeypatch.setattr(seeded_sweep.replay, "load_spec", lambda _path: _Spec())
    monkeypatch.setattr(seeded_sweep, "load_kernel", lambda: kernel)
    monkeypatch.setattr(
        seeded_sweep.replay, "model_from_nonce",
        lambda _spec, _nonce: (np.zeros(6), np.zeros(2)),
    )
    monkeypatch.setattr(
        seeded_sweep.quip_msa, "default_beta_range", lambda h, edges, j: (1.0, 10.0)
    )


def _run_args(reads_dir, out, qpu_lanes):
    return seeded_sweep.argparse.Namespace(
        spec="spec.json",
        reads_dir=str(reads_dir),
        out=str(out),
        sweeps=[1024],
        fractions=[0.5],
        lite_sweeps=8,
        qpu_lanes=qpu_lanes,
        threads=1,
        limit=0,
    )


def test_qpu_lanes_64_skips_the_fill_run_and_records_zero_fill_wall_ms(tmp_path, monkeypatch):
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    np.savez(
        reads_dir / "aa.npz",
        spins=_distinct_spins(64, 6),
        energies_milli=np.arange(64, dtype=np.int64),
    )
    out = tmp_path / "seeded.csv"
    kernel = _FakeKernel()
    _patch_run_deps(monkeypatch, kernel)

    rc = seeded_sweep.run(_run_args(reads_dir, out, qpu_lanes=64))
    assert rc == 0

    # Only the cold and seeded samples ran; no lite fill sample (num_reads=64,
    # num_sweeps=lite_sweeps, no initial_spins) was ever taken.
    fill_calls = [c for c in kernel.calls if c["initial_spins"] is None and c["num_sweeps"] == 8]
    assert fill_calls == []

    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    seeded_rows = [r for r in rows if r["arm"] == "seeded"]
    assert seeded_rows
    assert all(r["fill_wall_ms"] == "0" for r in seeded_rows)
    assert all(r["fill_lane_best_milli"] == "" for r in seeded_rows)


def test_qpu_lanes_below_64_seeds_best_n_and_fills_the_rest_with_lite(tmp_path, monkeypatch):
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    np.savez(
        reads_dir / "aa.npz",
        spins=_distinct_spins(64, 6),
        energies_milli=np.arange(64, dtype=np.int64),
    )
    out = tmp_path / "seeded.csv"
    kernel = _FakeKernel()
    _patch_run_deps(monkeypatch, kernel)

    rc = seeded_sweep.run(_run_args(reads_dir, out, qpu_lanes=32))
    assert rc == 0

    fill_calls = [c for c in kernel.calls if c["initial_spins"] is None and c["num_sweeps"] == 8]
    assert len(fill_calls) == 1

    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    seeded_rows = [r for r in rows if r["arm"] == "seeded"]
    assert seeded_rows
    for r in seeded_rows:
        assert int(r["qpu_lanes"]) <= 32
        assert r["fill_lane_best_milli"] != ""


def test_qpu_lanes_out_of_range_is_rejected_by_argparse(monkeypatch, tmp_path):
    argv = [
        "seeded_sweep.py", "run",
        "--spec", "spec.json",
        "--reads-dir", str(tmp_path),
        "--out", str(tmp_path / "seeded.csv"),
        "--qpu-lanes", "65",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        seeded_sweep.main()


def test_ceiling_fraction_is_normalised_like_the_stored_key(tmp_path, capsys, monkeypatch):
    out = tmp_path / "seeded.csv"
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    _write_csv(out, [
        ["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"],
        ["n1", "seeded", "0.60", 1024, -110, 4, -110, -90, "6", "20"],
    ])
    np.savez(reads_dir / "n1.npz", energies_milli=np.array([-110]))

    argv = [
        "seeded_sweep.py", "summarise",
        "--reads-dir", str(reads_dir), "--out", str(out),
        "--ceiling-fraction", "0.6",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = seeded_sweep.main()
    text = capsys.readouterr().out

    assert rc == 0
    assert "skipped" not in text


def test_resume_with_a_different_qpu_lanes_is_refused(tmp_path, monkeypatch):
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    np.savez(
        reads_dir / "aa.npz",
        spins=_distinct_spins(64, 6),
        energies_milli=np.arange(64, dtype=np.int64),
    )
    out = tmp_path / "seeded.csv"
    _patch_run_deps(monkeypatch, _FakeKernel())
    assert seeded_sweep.run(_run_args(reads_dir, out, qpu_lanes=32)) == 0
    assert seeded_sweep.run(_run_args(reads_dir, out, qpu_lanes=32)) == 0
    with pytest.raises(SystemExit, match="started with --qpu-lanes 32, not 16"):
        seeded_sweep.run(_run_args(reads_dir, out, qpu_lanes=16))


def test_resume_of_a_csv_without_the_lanes_sidecar_is_refused(tmp_path):
    out = tmp_path / "seeded.csv"
    _write_csv(out, [["n1", "cold", "", 1024, -100, 0, "", "", "5", "0"]])
    with pytest.raises(SystemExit, match="--qpu-lanes is unknown"):
        seeded_sweep.check_qpu_lanes(str(out), 32)
