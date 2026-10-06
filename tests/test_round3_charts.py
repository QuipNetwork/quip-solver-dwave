"""Round 3 report charts: each chart function on a small fixture, and the readers."""

from __future__ import annotations

import importlib.util
import io
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
