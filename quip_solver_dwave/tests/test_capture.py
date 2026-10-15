"""Capturing a live working graph as a coordinator topology spec."""

import json

import pytest

from quip_solver_dwave import EXIT_CLEAN
from quip_solver_dwave.capture import (
    canonical_edges,
    capture_spec,
    compare_specs,
    format_comparison,
)


def test_both_coupler_directions_collapse_to_one_edge():
    """A solver reports (a,b) and (b,a); the spec carries each coupler once."""
    assert canonical_edges([(2, 1), (1, 2), (0, 3)]) == [[0, 3], [1, 2]]


def test_capture_sorts_and_dedups():
    spec = capture_spec([3, 1, 1], [(3, 1), (1, 3)])
    assert spec["nodes"] == [1, 3]
    assert spec["edges"] == [[1, 3]]


def test_capture_drops_edges_with_unknown_endpoints():
    """The coordinator rejects a spec whose edge names a node it does not have
    (topology_spec.rs: EdgeUnknownNode), so never emit one."""
    spec = capture_spec([1, 2], [(1, 2), (2, 99)])
    assert spec["edges"] == [[1, 2]]


def test_allowed_sets_are_inputs_not_chip_properties():
    spec = capture_spec([1, 2], [(1, 2)], allowed_h_milli=[-1000, 0], allowed_j_milli=[500])
    assert spec["allowed_h_milli"] == [-1000, 0]
    assert spec["allowed_j_milli"] == [500]


def test_compare_counts_both_directions_of_drift():
    published = {"nodes": [1, 2, 3], "edges": [[1, 2], [2, 3]]}
    captured = {"nodes": [2, 3, 4], "edges": [[2, 3], [3, 4]]}
    diff = compare_specs(published, captured)
    assert diff["missing_nodes"] == 1      # 1 is gone from the chip
    assert diff["unused_nodes"] == 1       # 4 is hardware the network ignores
    assert diff["missing_edges"] == 1      # (1,2) must be reduced away per job
    assert diff["unused_edges"] == 1


def test_identical_graphs_show_no_drift():
    spec = {"nodes": [1, 2], "edges": [[1, 2]]}
    diff = compare_specs(spec, spec)
    assert diff["missing_nodes"] == 0
    assert diff["missing_edges"] == 0


def test_comparison_names_the_per_job_cost():
    diff = compare_specs(
        {"nodes": [1, 2, 3], "edges": [[1, 2], [2, 3]]},
        {"nodes": [1, 2, 3], "edges": [[1, 2]]},
    )
    text = format_comparison(diff)
    assert "50.00%" in text
    assert "scored unoptimized" in text


class _FakeSampler:
    live_nodes = [0, 1, 2]
    live_edges = [(0, 1), (1, 2)]

    def __init__(self, *a, **kw):
        self.connected = False
        self.closed = False

    def ensure_connected(self):
        self.connected = True

    def close(self):
        self.closed = True


def test_dump_topology_writes_a_spec_the_coordinator_can_read(tmp_path, monkeypatch, capsys):
    from quip_solver_dwave.cli import main

    monkeypatch.setattr("quip_solver_dwave.cli.ocean_importable", lambda: True)
    monkeypatch.setattr("quip_solver_dwave.cli.OceanSampler", _FakeSampler)
    out = tmp_path / "captured.spec.json"
    assert main(["--dump-topology", str(out)]) == EXIT_CLEAN

    spec = json.loads(out.read_text())
    assert spec["nodes"] == [0, 1, 2]
    assert spec["edges"] == [[0, 1], [1, 2]]
    assert spec["allowed_h_milli"] == [0]
    printed = capsys.readouterr().out
    assert "seed-chain --topology" in printed


def test_dump_topology_reports_the_diff(tmp_path, monkeypatch, capsys):
    from quip_solver_dwave.cli import main

    monkeypatch.setattr("quip_solver_dwave.cli.ocean_importable", lambda: True)
    monkeypatch.setattr("quip_solver_dwave.cli.OceanSampler", _FakeSampler)
    published = tmp_path / "published.json"
    published.write_text(json.dumps({"nodes": [0, 1, 2, 9], "edges": [[0, 1], [1, 2], [1, 9]]}))
    out = tmp_path / "captured.json"
    assert main(["--dump-topology", str(out), "--compare", str(published)]) == EXIT_CLEAN
    printed = capsys.readouterr().out
    assert "missing from this chip: 1" in printed


def test_bad_allowed_values_are_rejected(tmp_path, monkeypatch):
    from quip_solver_dwave.cli import main

    monkeypatch.setattr("quip_solver_dwave.cli.ocean_importable", lambda: True)
    monkeypatch.setattr("quip_solver_dwave.cli.OceanSampler", _FakeSampler)
    with pytest.raises(SystemExit, match="allowed-h-milli"):
        main(["--dump-topology", str(tmp_path / "x.json"), "--allowed-h-milli", "abc"])
