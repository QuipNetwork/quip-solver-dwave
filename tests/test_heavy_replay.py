import csv
import importlib.util
import pathlib

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "heavy_replay.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("heavy_replay", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


heavy_replay = _load_module()


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(heavy_replay.HEADER)
        for row in rows:
            writer.writerow(row)


def test_read_energies_ignores_sentinel_rows(tmp_path):
    path = tmp_path / "heavy.csv"
    _write_csv(
        path,
        [
            ["n1", "low-energy", -14600000, 128, 131072, 10.0],
            ["n2", "low-energy", heavy_replay.NO_SOLUTION, 128, 131072, 10.0],
            ["n3", "false-positive", -14000000, 128, 131072, 10.0],
        ],
    )
    by_set, ignored = heavy_replay.read_energies(str(path))
    assert ignored == 1
    assert by_set["low-energy"] == [-14600000]
    assert by_set["false-positive"] == [-14000000]


def test_read_energies_with_no_sentinel_rows(tmp_path):
    path = tmp_path / "heavy.csv"
    _write_csv(
        path,
        [
            ["n1", "low-energy", -14600000, 128, 131072, 10.0],
            ["n3", "false-positive", -14000000, 128, 131072, 10.0],
        ],
    )
    by_set, ignored = heavy_replay.read_energies(str(path))
    assert ignored == 0
    assert by_set == {"low-energy": [-14600000], "false-positive": [-14000000]}


def test_compare_with_an_empty_set_prints_n_zero_and_skips_gap(tmp_path, capsys):
    path = tmp_path / "heavy.csv"
    _write_csv(path, [["n1", "low-energy", -14600000, 128, 131072, 10.0]])
    args = heavy_replay.argparse.Namespace(out=str(path))
    rc = heavy_replay.compare(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "false-positive" in out
    assert "skipping mean gap and KS test" in out


def test_run_writes_no_solution_rows_so_a_resume_does_not_repeat_them(tmp_path, monkeypatch):
    low = tmp_path / "low.nonces"
    low.write_text("aa\n", encoding="utf-8")
    fp = tmp_path / "fp.nonces"
    fp.write_text("", encoding="utf-8")

    def fake_run(command, **_kwargs):
        report = command[command.index("--report") + 1]
        pathlib.Path(report).write_text(
            f'{{"job_id": "aa", "best_energy_milli": {heavy_replay.NO_SOLUTION}, '
            '"reads": 128, "sweeps": 9, "wall_ms": 1}\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(heavy_replay.subprocess, "run", fake_run)
    out = tmp_path / "heavy.csv"
    args = heavy_replay.argparse.Namespace(
        low=str(low), false_positive=str(fp), out=str(out), cap=0, seed=1, chunk=10,
        coordinator="c", miner="m", spec="s", reads=128, sweeps=9, device=None,
    )
    assert heavy_replay.run(args) == 0

    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert int(rows[0]["heavy_best_milli"]) == heavy_replay.NO_SOLUTION

    run_calls = []
    monkeypatch.setattr(
        heavy_replay.subprocess, "run",
        lambda command, **_kwargs: run_calls.append(command),
    )
    assert heavy_replay.run(args) == 0
    assert run_calls == []


def test_run_asks_the_driver_to_select_every_read(tmp_path, monkeypatch):
    # The driver reports the best energy of its diverse selection. Only a
    # selection of every read makes that the true minimum.
    low = tmp_path / "low.nonces"
    low.write_text("aa\n", encoding="utf-8")
    fp = tmp_path / "fp.nonces"
    fp.write_text("", encoding="utf-8")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        report = command[command.index("--report") + 1]
        pathlib.Path(report).write_text(
            '{"job_id": "aa", "best_energy_milli": -5, "reads": 128, "sweeps": 9, "wall_ms": 1}\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(heavy_replay.subprocess, "run", fake_run)
    out = tmp_path / "heavy.csv"
    args = heavy_replay.argparse.Namespace(
        low=str(low), false_positive=str(fp), out=str(out), cap=0, seed=1, chunk=10,
        coordinator="c", miner="m", spec="s", reads=128, sweeps=9, device=None,
    )
    assert heavy_replay.run(args) == 0
    command = commands[0]
    assert command[command.index("--min-solutions") + 1] == "128"
