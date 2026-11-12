"""Round 3 report charts: each chart function on a small fixture, and the readers."""

from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import pytest
from matplotlib.text import Annotation

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "round3_charts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("round3_charts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


charts = _load_module()


def svg_of(fig) -> str:
    """The SVG text of a figure. The module keeps text as text, so labels are searchable."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg")
    plt.close(fig)
    return buffer.getvalue()


def _ladder_summary(n_deep: int = 100):
    rows = []
    for cell in charts.CELLS:
        rows.append({
            "cell": cell,
            "reach_target": 97,
            "vs_cpu": {
                "cpu-msa-unit@8192": {"n": 100, "strict_qpu": 90, "strict_cpu": 10},
                "cpu-msa-unit@131072": {"n": 100, "strict_qpu": 55, "strict_cpu": 45},
                "cpu-msa-unit@524288": {"n": n_deep, "strict_qpu": 20, "strict_cpu": 27},
                "cpu-sa@8192": {"n": 100, "strict_qpu": 100, "strict_cpu": 0},
            },
        })
    return rows


def test_lattice_ladder_has_one_panel_per_regime_with_direct_kernel_labels():
    fig = charts.chart_lattice_ladder(_ladder_summary())
    assert len(fig.axes) == len(charts.CELLS)
    svg = svg_of(fig)
    assert svg.count("cpu-msa-unit") == len(charts.CELLS)
    assert svg.count("cpu-sa") == len(charts.CELLS)
    for cell in charts.CELLS:
        assert cell in svg


def test_lattice_ladder_plots_the_share_of_n_when_a_rung_ran_fewer_models():
    fig = charts.chart_lattice_ladder(_ladder_summary(n_deep=47))
    line = [ln for ln in fig.axes[0].get_lines() if len(ln.get_xdata()) == 3][0]
    assert line.get_ydata()[-1] == pytest.approx(100.0 * 20 / 47)


def test_label_ends_separates_labels_that_end_at_the_same_point():
    fig, ax = plt.subplots()
    ax.set_ylim(0, 100)
    charts.label_ends(ax, [(8, 0.0, "a", charts.INK), (8, 0.0, "b", charts.GRAY), (8, 60.0, "c", charts.INK)])
    offsets = {t.get_text(): t.xyann for t in ax.texts if isinstance(t, Annotation)}
    assert offsets["a"] != offsets["b"]
    assert offsets["a"][1] >= 0 and offsets["b"][1] >= 0  # a group on the floor stacks upward
    assert offsets["c"] == (4, 0)
    plt.close(fig)


def test_read_clean_timing_skips_unsupported_records(tmp_path):
    runs = tmp_path / "runs"
    for key, record in {
        "a": {"cell": "native-pm1", "kernel": "cpu-sa", "observed_kernel": "cpu-sa", "sweeps": 8192,
              "status": "completed", "sampling_s": 1.5},
        "b": {"cell": "clique-portfolio", "kernel": "cpu-msa-unit", "observed_kernel": None, "sweeps": 8192,
              "status": "unsupported", "sampling_s": None},
    }.items():
        (runs / key).mkdir(parents=True)
        (runs / key / "result.json").write_text(json.dumps(record), encoding="utf-8")
    rows = charts.read_clean_timing(tmp_path)
    assert rows == [{"cell": "native-pm1", "kernel": "cpu-sa", "sweeps": 8192, "sampling_s": 1.5}]


def test_clean_time_draws_the_qpu_reference_lines_in_every_panel():
    rows = [{"cell": cell, "kernel": kernel, "sweeps": sweeps, "sampling_s": sweeps / 20000 * factor}
            for cell in charts.CELLS for kernel, factor in (("cpu-msa-unit", 1), ("cpu-msa-f64", 6))
            for sweeps in (8192, 32768, 131072)]
    fig = charts.chart_clean_time(rows)
    dashed = [[ln for ln in ax.get_lines() if ln.get_linestyle() == "--"] for ax in fig.axes]
    assert [len(lines) for lines in dashed] == [2] * len(charts.CELLS)
    assert sorted(float(ln.get_ydata()[0]) for ln in dashed[0]) == [0.07, 2.2]
    assert "QPU access" not in svg_of(fig)  # the caption names the lines; text in the panel crosses a data line


def _seeded_rows():
    rows = []
    for cell in charts.CELLS:
        kernel = "cpu-msa-f64" if cell == "native-125" else "cpu-msa-unit"
        for fraction in (0.25, 0.5):
            for sweeps, same, deep in ((8192, 100, 5), (32768, 71, 12), (131072, 62, 62)):
                rows.append({"cell": cell, "kernel": kernel, "sweeps": sweeps, "fraction": fraction,
                             "vs_cold_same": {"n": 100, "seeded_better": same, "cold_better": 100 - same},
                             "vs_cold_131072": {"n": 100, "seeded_better": deep, "cold_better": 100 - deep}})
    rows.append({"cell": "cubic-dimer-pm1", "kernel": "cpu-msa-f64", "sweeps": 8192, "fraction": 0.25,
                 "vs_cold_same": {"n": 100, "seeded_better": 100, "cold_better": 0},
                 "vs_cold_131072": {"n": 100, "seeded_better": 78, "cold_better": 5}})
    return rows


def test_seeded_uses_the_unit_kernel_on_sign_lattices_and_f64_on_native_125():
    fig = charts.chart_seeded(_seeded_rows())
    assert len(fig.axes) == len(charts.CELLS)
    # two fractions, two comparisons each: four data lines per panel (the 50% reference line has 2 points),
    # and the one-point f64 cubic row is not drawn
    assert all(len([ln for ln in ax.get_lines() if len(ln.get_xdata()) == 3]) == 4 for ax in fig.axes)
    svg = svg_of(fig)
    assert svg.count("f = 0.25") == len(charts.CELLS)


def test_cuda_depth_uses_the_128_read_rows_only():
    table = [
        {"sweeps": 131072, "reads": 64, "vs_qpu_best": {"qpu_better": 80, "tie": 17, "cpu_better": 3},
         "vs_qpu_best64": {"qpu_better": 66, "tie": 0, "cpu_better": 34}},
        {"sweeps": 131072, "reads": 128, "vs_qpu_best": {"qpu_better": 71, "tie": 20, "cpu_better": 9},
         "vs_qpu_best64": {"qpu_better": 55, "tie": 0, "cpu_better": 45}},
        {"sweeps": 1048576, "reads": 128, "vs_qpu_best": {"qpu_better": 43, "tie": 34, "cpu_better": 23},
         "vs_qpu_best64": {"qpu_better": 23, "tie": 0, "cpu_better": 77}},
    ]
    fig = charts.chart_cuda_depth(table)
    qpu_better = [bar.get_width() for bar in fig.axes[0].containers[0]]
    assert qpu_better == [71, 55, 43, 23]  # the 64-read row is not drawn
    svg = svg_of(fig)
    assert "CUDA 131,072 sweeps vs QPU 1,024 reads" in svg
    assert "CUDA 1,048,576 sweeps vs QPU 64 reads" in svg


def test_throughput_marks_the_achieved_rate_and_the_access_limit():
    arms = [{"reads": 64, "concurrency": 16, "models_per_s": 5.28, "access_limited_models_per_s": 14.24},
            {"reads": 1024, "concurrency": 4, "models_per_s": 0.62, "access_limited_models_per_s": 1.66}]
    svg = svg_of(charts.chart_throughput(arms))
    assert "64 reads, 16 in flight" in svg and "limit 14.2" in svg and "5.28" in svg


def test_depth_extension_accumulates_the_models_at_target():
    summary = {"base_sweeps": 131072, "base_models": 100, "base_missing": 47,
               "depths": {"524288": {"h%d" % i: {"target_reached": i < 28} for i in range(47)},
                          "2097152": {"h%d" % i: {"target_reached": i < 10} for i in range(19)}}}
    fig = charts.chart_depth_extension(summary, qpu_reach=97)
    line = fig.axes[0].get_lines()[0]
    assert list(line.get_ydata()) == [53, 81, 91]
    assert "QPU, 1,024 reads: 97" in svg_of(fig)


def test_portfolio_ac_plots_the_qpu_at_its_best_anneal_against_four_arms():
    rows = [{"variant": variant, "qpu_best_anneal_us": 80,
             "qpu": {"20": {"reach_best": 23}, "80": {"reach_best": 25}, "400": {"reach_best": 25}},
             "cpu": {"dwave-neal-reference": {"reach_best": 13}, "cpu-msa-f64": {"reach_best": 12},
                     "neal-final-30x": {"reach_best": 36}, "sa-30x-500": {"reach_best": 37}}}
            for variant in ("real", "iid", "shuffle", "rotate")]
    fig = charts.chart_portfolio_ac(rows)
    qpu_line = [ln for ln in fig.axes[0].get_lines() if ln.get_label() == "QPU, best anneal"][0]
    assert list(qpu_line.get_xdata()) == [25, 25, 25, 25]
    assert "scalar cpu-sa, 30x range" in svg_of(fig)


def test_family_b2_picks_the_anneal_with_the_most_qpu_optima_per_group():
    def row(group, anneal, qpu, models=12):
        arms = {arm: {"at_optimum": 3} for arm in ("production neal", "tuned neal 30x", "deep neal", "best CPU arm")}
        return {"group": group, "anneal_us": anneal, "exact": {"models": models, "QPU": {"at_optimum": qpu}, **arms}}
    rows = [row("B2 32:11", 20, 5), row("B2 32:11", 400, 6), row("B+B2 44:14 pooled", 80, 20, models=60),
            row("B2 32:11 unused", 20, 12)]
    fig = charts.chart_family_b2(rows)
    qpu_line = [ln for ln in fig.axes[0].get_lines() if ln.get_label() == "QPU, best anneal"][0]
    assert list(qpu_line.get_xdata()) == pytest.approx([100.0 * 6 / 12, 100.0 * 20 / 60])
    assert "B+B2 44:14 pooled (60 models)" in svg_of(fig)


def test_famb_race_orders_by_size_anneal_and_cost_and_treats_a_missing_count_as_zero():
    verdicts = {"n=56 anneal=20us cost_a": {"qpu-better": 12},
                "n=44 anneal=400us cost_b": {"tie": 5, "qpu-worse": 3, "qpu-better": 4},
                "n=44 anneal=20us cost_a": {"qpu-better": 8, "tie": 4}}
    fig = charts.chart_famb_race(verdicts)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["N=44, 20 µs, access only", "N=44, 400 µs, access + finish", "N=56, 20 µs, access only"]


def test_msa_diversity_shows_minimum_to_median_at_131072_sweeps_only():
    summary = [{"cell": "clique-portfolio", "kernel": "cpu-msa-f64", "sweeps": 131072,
                "distinct_rows_median": 1.0, "distinct_rows_min": 1},
               {"cell": "native-pm1", "kernel": "cpu-msa-unit", "sweeps": 131072,
                "distinct_rows_median": 64.0, "distinct_rows_min": 64},
               {"cell": "native-pm1", "kernel": "cpu-msa-unit", "sweeps": 8192,
                "distinct_rows_median": 64.0, "distinct_rows_min": 64}]
    fig = charts.chart_msa_diversity(summary)
    assert len(fig.axes[0].get_yticklabels()) == 2
    assert "clique-portfolio, cpu-msa-f64" in svg_of(fig)


def test_build_figures_returns_ten_charts_from_the_data_roots():
    if not (charts.ROUND2.exists() and charts.ROUND3.exists()):
        pytest.skip("data roots are not on this host")
    figures = charts.build_figures(charts.ROUND2, charts.ROUND3)
    assert len(figures) == 10
    for fig in figures.values():
        plt.close(fig)
