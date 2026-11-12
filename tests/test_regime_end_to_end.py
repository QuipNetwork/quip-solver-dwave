"""The four regime-search scripts, end to end, on the offline mock sampler."""

import csv
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = ["native-pm1", "native-125", "diamond-pm1", "cubic-dimer-pm1", "clique-portfolio"]


def run(script: str, *args: str) -> str:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *args], env=env, capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, f"{script} failed:\n{done.stdout}\n{done.stderr}"
    assert "Warning" not in done.stderr, f"{script} warned:\n{done.stderr}"
    return done.stdout


def test_the_scripts_run_a_whole_round_on_the_mock(tmp_path):
    graphs = pytest.importorskip("dwave.graphs")
    pytest.importorskip("quip_msa")
    graph = graphs.zephyr_graph(3, 4)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "nodes": sorted(graph.nodes),
        "edges": [list(edge) for edge in graph.edges],
        "allowed_h_milli": [0],
        "allowed_j_milli": [-1000, 1000],
    }))
    embeddings_dir = tmp_path / "embeddings"
    round_dir = tmp_path / "round"

    out = run("embed_probe.py", "--spec", str(spec), "--out-dir", str(embeddings_dir),
              "--diamond", "4", "4", "8", "--cubic-dimer", "3", "3", "2", "--clique", "12", "--zephyr-m", "3")
    assert "0 of the 32 spins inside the graph are vacant" in out and "clique 12" in out

    common = ["--spec", str(spec), "--round-dir", str(round_dir), "--embeddings-dir", str(embeddings_dir),
              "--models", "3", "--reads", "8", "--clique-holdings", "4"]
    ladder = ["--rungs", "8", "64", "--extra-rung", "128", "--seed-sweeps", "32", "--threads", "2"]

    # A ladder before the capture writes the rungs only, and it says why the seed arm has nothing to do.
    out = run("regime_ladder.py", "--cell", "native-pm1", *ladder, *common)
    assert "3 of 3 models have no capture at 80 µs" in out
    with open(round_dir / "native-pm1" / "ladder.csv", newline="") as handle:
        assert {row["sweeps"] for row in csv.DictReader(handle)} == {"8", "64"}

    for cell in CELLS:
        out = run("regime_capture.py", "--cell", cell, "--mock", *common)
        assert "captured 6" in out
        assert len(list((round_dir / cell / "qpu-80").glob("*.npz"))) == 3
        # A second run finds nothing to do.
        assert "0 captures to run" in run("regime_capture.py", "--cell", cell, "--mock", *common)

    for cell in CELLS:
        run("regime_ladder.py", "--cell", cell, *ladder, *common)
        with open(round_dir / cell / "ladder.csv", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["sweeps"] for row in rows} >= {"8", "64"}
        expected = "cpu-msa" if cell in ("native-pm1", "diamond-pm1", "cubic-dimer-pm1") else "cpu-sa"
        assert {row["kernel_label"] for row in rows} == {expected}
        assert (round_dir / cell / "seed.csv").exists() == (expected == "cpu-msa")
        if cell == "clique-portfolio":
            assert all(row["k_share"] for row in rows)

    # Step 0 reads the ranges from the live solver. The test writes the ranges of Advantage2 by hand.
    properties = tmp_path / "solver-properties.json"
    properties.write_text(json.dumps({"extended_j_range": [-2.0, 1.0], "per_qubit_coupling_range": [-18.0, 15.0]}))
    results = tmp_path / "RESULTS.md"
    run("regime_report.py", "--round-dir", str(round_dir), "--out", str(results), "--rungs", "8", "64",
        "--extra-rung", "128", "--resamples", "50", "--solver-properties", str(properties))
    text = results.read_text()
    assert "This is a mock round" in text and "Reads with exactly 4 assets" in text
    for cell in CELLS:
        assert f"`{cell}`" in text
    headings = (
        "Cells", "Energy scale", "Matching time", "Matching rung", "Gate window", "Information", "Seed lane wins",
        "Screening rule", "Screening resolution", "Clique cell", "Solver ranges",
    )
    for heading in headings:
        assert f"## {heading}" in text

    # The manifest of a cell stops a run that would build other models than the capture did.
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    other = [value if value != "4" else "5" for value in common]
    refused = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "regime_ladder.py"), "--cell", "clique-portfolio", *ladder, *other],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert refused.returncode != 0 and "clique_holdings is 4" in refused.stderr
