"""Tests for scripts/round2_report.py table builders and validation gate."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round2_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round2_report", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_module()
CELL = "diamond-pm1"


def _record(kernel="cpu-sa", sweeps=512, nonce="aa", status="completed", best=-10.0, **changes):
    record = {
        "cell": CELL, "nonce": nonce, "kernel": kernel, "sweeps": sweeps, "status": status,
        "best_energy": best if status == "completed" else None, "sampling_s": 1.0,
        "caller_end_to_end_s": 1.5, "host": {"contamination": {"contaminated": False}},
        "comparison": {"available": True, "target_reached": True},
    }
    record.update(changes)
    return record


def _row(text: str, kernel: str, sweeps: int) -> list[str]:
    lines = [line for line in text if line.startswith(f"| `{kernel}` | {sweeps} |")]
    assert len(lines) == 1
    return [cell.strip() for cell in lines[0].strip("|").split("|")]


def test_an_empty_arm_reports_every_model_as_missing():
    rows = report.depth_quality_time_table(CELL, [], [])
    cells = _row(rows, "cpu-sa", 512)
    assert cells[2:8] == ["0", "100", "0", "0", "0", "0"]
    assert cells[8] == "n/a"


def test_one_missing_depth_is_counted_without_affecting_the_others():
    campaign = [_record(sweeps=sweeps) for sweeps in report.DEPTHS if sweeps != 8192]
    rows = report.depth_quality_time_table(CELL, campaign, [])
    assert _row(rows, "cpu-sa", 8192)[2:4] == ["0", "100"]
    assert _row(rows, "cpu-sa", 2048)[2:5] == ["1", "99", "1"]
    assert _row(rows, "cpu-sa", 2048)[9] == "1 of 1"


def test_loaded_host_timing_never_enters_the_clean_columns():
    loaded = _record(host={"contamination": {"contaminated": True}}, sampling_s=50.0, caller_end_to_end_s=60.0)
    rows = report.depth_quality_time_table(CELL, [_record(), loaded], [])
    cells = _row(rows, "cpu-sa", 512)
    assert cells[10] == "1 (1)"
    assert cells[11] == "1.5 (1)"
    assert cells[12] == "1"


def test_all_ties_count_as_ties_in_every_quality_column():
    best = {(CELL, kernel, report.DEEPEST, nonce): -5.0 for kernel in report.KERNELS for nonce in ("aa", "bb")}
    pilot = [
        {"cell": CELL, "nonce": nonce, "scale": 1.0, "anneal_us": 80, "best_energy": -5.0}
        for nonce in ("aa", "bb")
    ]
    rows = report.qpu_quality_table(CELL, pilot, best)
    line = next(line for line in rows if "`cpu-sa`" in line)
    assert line.count("QPU 0 / tie 2 / CPU 0 (of 2)") == 3


def test_captures_that_share_a_model_are_one_bootstrap_group():
    best = {(CELL, "cpu-sa", report.DEEPEST, "aa"): -5.0}
    pilot = [
        {"cell": CELL, "nonce": "aa", "scale": 1.0, "anneal_us": 80, "best_energy": energy}
        for energy in (-5.1, -5.2)
    ]
    line = next(line for line in report.qpu_quality_table(CELL, pilot, best) if "`cpu-sa`" in line)
    assert "1 models (descriptive)" in line


def test_unavailable_physical_metadata_is_labeled_not_invented():
    pilot = [{"cell": CELL, "nonce": "aa", "scale": 0.5, "anneal_us": 400, "best_energy": -4.0}]
    rows = report.physical_scale_table(CELL, pilot, {})
    line = next(line for line in rows if line.startswith("| 50% |"))
    assert line.count("unavailable") == 4
    assert line.endswith("| n/a |")


def test_the_routed_arm_uses_unit_where_supported_and_f64_elsewhere():
    campaign = [
        _record(kernel="cpu-msa-unit", nonce="aa", best=-1.0),
        _record(kernel="cpu-msa-f64", nonce="aa", best=-2.0),
        _record(kernel="cpu-msa-unit", nonce="bb", status="unsupported"),
        _record(kernel="cpu-msa-f64", nonce="bb", best=-3.0),
    ]
    routed = {r["nonce"]: r for r in report.derive_routed(campaign)}
    assert routed["aa"]["routed_kernel"] == "cpu-msa-unit" and routed["aa"]["best_energy"] == -1.0
    assert routed["bb"]["routed_kernel"] == "cpu-msa-f64" and routed["bb"]["best_energy"] == -3.0
    assert {r["kernel"] for r in routed.values()} == {"cpu-msa"}


def test_the_report_refuses_evidence_with_a_failed_validation(tmp_path):
    (tmp_path / "cpu-campaign").mkdir()
    (tmp_path / "combined").mkdir()
    (tmp_path / "physical-pilot-analysis").mkdir()
    (tmp_path / "cpu-campaign" / "validation.json").write_text(
        json.dumps({"invalid_records": [{"run_key": "x"}], "orphan_run_dirs": []})
    )
    (tmp_path / "combined" / "cpu-tree-validation.json").write_text(json.dumps({"invalid_records": []}))
    (tmp_path / "physical-pilot-analysis" / "reconciliation.json").write_text(json.dumps({"all_checks_passed": True}))
    with pytest.raises(SystemExit, match="cpu-campaign"):
        report.require_validated(tmp_path)


def test_the_portfolio_reference_is_not_compared_with_itself():
    fixtures = {"m": {"fixture": "n18-k6/ramp", "provenance": "synthetic-test", "n_assets": 18, "cardinality_k": 6}}
    diagnostics = [{"raw_cardinality": 2, "repaired_cardinality": 6}]
    portfolio = [
        {"model_hash": "m", "requested_kernel": kernel, "status": "completed", "deadline_status": "completed",
         "objective": 0.5, "feasible": True, "caller_end_to_end_s": seconds, "diagnostics_records": diagnostics}
        for kernel, seconds in ((report.PORTFOLIO_REFERENCE, 0.5), ("cpu-sa", 0.25))
    ]
    rows = report.portfolio_table(portfolio, fixtures)
    reference = next(line for line in rows if f"| `{report.PORTFOLIO_REFERENCE}` |" in line)
    assert reference.endswith("| reference | reference |")
    assert next(line for line in rows if "| `cpu-sa` |" in line).endswith("| strict tie, material tie | 0.50x |")


def _family(objectives: dict[str, dict[str, float]], oracle: dict[str, float]) -> dict:
    return {"arms": {name: {arm: {"objective": v} for arm, v in arms.items()} for name, arms in objectives.items()},
            "oracle": oracle}


def test_family_counts_use_the_proven_optimum_and_otherwise_the_best_found():
    family = _family({
        "d0-real": {"a": 1.0, "b": 1.004, "random": 2.0},   # proven 0.998: a within 0.5%, b 0.6% away
        # best found 1.0: a at best, b within 0.5%. The unshown arm is not a reference arm.
        "d1-real": {"a": 1.0, "b": 1.004, "random": 2.0, "unshown": 0.5},
        "d2-real": {"a": 1.0, "b": 1.01, "random": 2.0},    # b more than 0.5% away
    }, {"d0-real": 0.998})
    rows = report.family_quality_table(family, (("a", "a"), ("b", "b")), reference_arms=("a", "b", "random"))
    assert rows[0].startswith("| Method | real (3) |")
    assert "| a | 2 / 3 |" in rows[2]
    assert "| b | 0 / 1 |" in rows[3]
    proven = report.family_quality_table(family, (("a", "a"),), reference_arms=("a",), proven_only=True)
    assert proven[0].startswith("| Method | real (1) |")
    assert "| a | 0 / 1 |" in proven[2]


def test_the_spread_measure_skips_models_where_random_reads_find_the_best():
    family = _family({
        "d0-real": {"a": 1.5, "random": 3.0},
        "d1-real": {"a": 1.0, "random": 1.0},
    }, {"d0-real": 1.0, "d1-real": 1.0})
    rows = report.family_spread_table(family, ["d0-real", "d1-real"], (("a", "a"),), reference_arms=("a",))
    assert "| a | 0.25 (n=1) |" in rows[2]


def test_the_report_refuses_a_capture_that_differs_from_its_manifest(tmp_path, monkeypatch):
    capture = tmp_path / "portfolio-qpu"
    (capture / "round" / "jobs").mkdir(parents=True)
    (capture / "manifest.json").write_text(json.dumps({"jobs": [{"capture_key": "k1"}, {"capture_key": "k2"}]}))
    (capture / "round" / "jobs" / "only-20us.json").write_text(json.dumps({"capture_key": "k1"}))
    monkeypatch.setattr(report, "_family_set", lambda *_: {"arms": {}, "oracle": {}})
    with pytest.raises(SystemExit, match="exactly the manifest's jobs"):
        report.load_families(tmp_path)
