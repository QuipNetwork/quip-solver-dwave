import csv
import importlib.util
import pathlib
import sys

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "lite_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("lite_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lite_report = _load_module()


def _write_attempts(path, nonces_and_energies):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["nonce", "raw_best_energy_milli"])
        for nonce, energy in nonces_and_energies:
            writer.writerow([nonce, energy])


def _write_lite(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["nonce", "sweeps", "lite_best_milli"])
        for nonce, sweeps, energy in rows:
            writer.writerow([nonce, sweeps, energy])


def _run_full(monkeypatch, attempts_path, lite_path, out_dir, sweeps=128, ceiling=-100):
    monkeypatch.setattr(
        sys, "argv",
        [
            "lite_report.py",
            "--attempts", str(attempts_path),
            "--lite", str(lite_path),
            "--full",
            "--sweeps", str(sweeps),
            "--ceiling-milli", str(ceiling),
            "--out-dir", str(out_dir),
        ],
    )
    return lite_report.main()


def test_full_mode_rejects_a_data_set_with_the_right_count_but_the_wrong_nonces(tmp_path, capsys, monkeypatch):
    attempts = tmp_path / "attempts.csv"
    _write_attempts(attempts, [("aa", -200), ("bb", -50)])
    lite = tmp_path / "lite.csv"
    # Same row count as attempts (2), but "cc" is not a nonce attempts knows,
    # so the count check alone would pass while silently writing wrong lists.
    _write_lite(lite, [("aa", 128, -210), ("cc", 128, -60)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    rc = _run_full(monkeypatch, attempts, lite, out_dir)
    err = capsys.readouterr().err

    assert rc == 1
    assert "1 nonces" in err
    assert "bb" in err
    assert not (out_dir / "low-energy.nonces").exists()


def test_full_mode_accepts_a_data_set_with_matching_nonces(tmp_path, capsys, monkeypatch):
    attempts = tmp_path / "attempts.csv"
    _write_attempts(attempts, [("aa", -200), ("bb", -50)])
    lite = tmp_path / "lite.csv"
    _write_lite(lite, [("aa", 128, -210), ("bb", 128, -60)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    rc = _run_full(monkeypatch, attempts, lite, out_dir, ceiling=-100)

    assert rc == 0
    assert (out_dir / "low-energy.nonces").read_text(encoding="utf-8").strip() == "aa"
