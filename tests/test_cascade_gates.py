import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cascade_gates.py"


def load():
    spec = importlib.util.spec_from_file_location("cascade_gates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_inputs(tmp_path, rows):
    qblocks = tmp_path / "qblocks.jsonl"
    qblocks.write_text("".join(json.dumps({"nonce": n, "energy_milli": e}) + "\n" for n, e in rows))
    trace = tmp_path / "trace.csv"
    lines = ["nonce,seed,best_0,best_1"]
    for index, (nonce, _) in enumerate(rows):
        for seed in range(2):
            lines.append(f"{nonce},{seed},{-100 - 10 * index - seed},{-200 - 10 * index - seed}")
    trace.write_text("\n".join(lines) + "\n")
    return qblocks, trace


def test_msa_orders_models_by_chain_energy_and_writes_a_screen_program(tmp_path):
    # The second model is the deeper winner, so it alone sets the gates when deepest is 1.
    qblocks, trace = write_inputs(tmp_path, [("aa" * 32, -14_700_000), ("bb" * 32, -14_740_000)])
    out = tmp_path / "programs.json"
    code = load().main(["msa", "--trace", str(trace), "--qblocks", str(qblocks), "--deepest", "1", "--margin", "10",
                        "--lanes", "128", "--legs", "8,16", "--out", str(out)])
    assert code == 0
    programs = json.loads(out.read_text())["programs"]
    screen = programs[0]
    assert screen["name"] == "screen" and screen["words"] == 2
    assert [leg["sweeps"] for leg in screen["legs"]] == [8, 8]
    assert [leg["start"] for leg in screen["legs"]] == ["fresh", "continue"]
    assert [leg["gate"] for leg in screen["legs"]] == [-100, -200]
    assert screen["legs"][1]["from_beta"] == 0.25 and "from_beta" not in screen["legs"][0]


def test_msa_refuses_a_trace_with_an_unknown_nonce(tmp_path):
    qblocks, trace = write_inputs(tmp_path, [("aa" * 32, -14_700_000)])
    trace.write_text(trace.read_text() + f"{'cc' * 32},0,-1,-2\n")
    with pytest.raises(SystemExit, match="not in the qblocks file"):
        load().main(["msa", "--trace", str(trace), "--qblocks", str(qblocks), "--deepest", "1", "--margin", "10",
                     "--lanes", "64", "--legs", "8,16", "--out", str(tmp_path / "p.json")])


def test_qpu_table_reads_best_energies_from_capture_directories(tmp_path):
    module = load()
    for name, values in (("winners", [-14_600_000, -14_580_000]), ("controls", [-14_590_000, -14_400_000])):
        (tmp_path / name).mkdir()
        for index, value in enumerate(values):
            np.savez(tmp_path / name / f"{index:064x}.npz", energies_milli=np.array([value, value + 4000]))
    out = tmp_path / "qpu.json"
    assert module.main(["qpu", "--winners", str(tmp_path / "winners"), "--controls", str(tmp_path / "controls"),
                        "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["unit"] == "milli"
    table = written["table"]
    assert table[0]["miss_allowed"] == 0.0 and table[0]["cut"] == -14_580_000
    assert table[0]["control_pass"] == 0.5
