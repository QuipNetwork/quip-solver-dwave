#!/usr/bin/env python3
"""Round 3 report charts: one SVG per dataset, from the analysis files on disk.

Each ``chart_*`` function takes parsed rows and returns a matplotlib ``Figure``,
so a test can call it on a small fixture. ``main`` reads the files under the
round 2 and round 3 data roots and writes every chart as SVG into ``--out``.

Design rules: one accent color for the QPU and gray for every classical arm,
no gridlines, a direct label instead of a legend where one fits, shared scales
across small multiples, and every number a chart shows is in a report table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: searchable, small, and Notion renders it

ROUND2 = Path("/home/carback1/quip-data/regimes/round2")
ROUND3 = Path("/home/carback1/quip-data/regimes/round3")
CELLS = ("cubic-dimer-pm1", "native-pm1", "native-125", "diamond-pm1")
INK = "#222222"
GRAY = "#8a8a8a"
LIGHT = "#c8c8c8"
QPU = "#1f5fbf"
KERNEL_STYLE: Dict[str, Dict[str, str]] = {
    "cpu-msa-unit": {"color": INK, "ls": "-"},
    "cpu-msa-f64": {"color": GRAY, "ls": "-"},
    "cpu-sa": {"color": GRAY, "ls": ":"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax: Axes, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Two gray axes, small labels, no frame on the top or right."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRAY)
    ax.tick_params(colors=GRAY, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, loc="left", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK)


def label_end(ax: Axes, x: float, y: float, text: str, color: str = INK) -> None:
    """A direct label at the last point of a line, in place of a legend."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7, color=color, va="center")


def label_ends(ax: Axes, entries: Sequence[Tuple[float, float, str, str]], gap_points: float = 8.0) -> None:
    """Direct labels for several lines. Labels whose end points sit within ``gap_points`` are stacked apart.

    The stack grows toward the middle of the axes, away from the nearer frame edge, so a group of lines that
    end on the floor of the axes gets its labels above the floor. Call this after the scales and limits are set.
    """
    pixels_per_point = ax.figure.dpi / 72.0
    gap = gap_points * pixels_per_point

    def display_y_of(x: float, y: float) -> float:
        return float(ax.transData.transform((x, y))[1])

    upward = min(display_y_of(e[0], e[1]) for e in entries) < (ax.bbox.y0 + ax.bbox.y1) / 2
    ordered = sorted(entries, key=lambda e: display_y_of(e[0], e[1]), reverse=not upward)
    placed: Optional[float] = None  # display y of the last label, in pixels
    for x, y, text, color in ordered:
        display_y = display_y_of(x, y)
        if placed is None:
            final_y = display_y
        elif upward:
            final_y = max(display_y, placed + gap)
        else:
            final_y = min(display_y, placed - gap)
        offset = (final_y - display_y) / pixels_per_point
        ax.annotate(text, (x, y), xytext=(4, offset), textcoords="offset points", fontsize=7, color=color,
                    va="center")
        placed = final_y


def stacked_bars(ax: Axes, labels: Sequence[str], triples: Sequence[Tuple[int, int, int]],
                 names: Tuple[str, str, str], total: int) -> None:
    """Three-part horizontal bars of counts. The first part is the QPU's, in the accent color."""
    y = list(range(len(labels)))
    left = [0] * len(labels)
    for index, (name, color) in enumerate(zip(names, (QPU, LIGHT, GRAY))):
        widths = [triple[index] for triple in triples]
        ax.barh(y, widths, left=left, color=color, height=0.6, label=name)
        for yy, start, width in zip(y, left, widths):
            if width / total >= 0.06:
                ax.text(start + width / 2, yy, str(width), ha="center", va="center", fontsize=7,
                        color=INK if color == LIGHT else "white")
        left = [start + width for start, width in zip(left, widths)]
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, total)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=3, bbox_to_anchor=(1, 1))


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def chart_lattice_ladder(summary: Sequence[Mapping[str, Any]]) -> Figure:
    """Share of models where the QPU's expected best of 64 reads beats cold CPU, by sweeps and kernel.

    A rung that ran fewer than 100 models (the deep ``cubic-dimer-pm1`` rungs) shows its share of the
    models it ran.
    """
    fig, axes = plt.subplots(1, len(CELLS), figsize=(11, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.5)  # room for the direct labels past the right edge of each panel
    rows = {row["cell"]: row for row in summary}
    for ax, cell in zip(axes, CELLS):
        series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for key, cmp in rows[cell]["vs_cpu"].items():
            kernel, sweeps = key.split("@")
            series[kernel].append((int(sweeps), 100.0 * cmp["strict_qpu"] / cmp["n"]))
        ends = []
        for kernel, points in sorted(series.items()):
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", ms=3, lw=1, **KERNEL_STYLE[kernel])
            ends.append((points[-1][0], points[-1][1], kernel, KERNEL_STYLE[kernel]["color"]))
        ax.set_xscale("log", base=2)
        ax.axhline(50, color=LIGHT, lw=0.6)
        ax.set_ylim(0, 100)
        label_ends(ax, ends)
        style(ax, title=cell, xlabel="sweeps", ylabel="QPU better, % of models" if cell == CELLS[0] else "")
    return fig


def build_figures(round2: Path, round3: Path) -> Dict[str, Figure]:
    """Every chart, keyed by file stem. Later tasks add entries."""
    eval_summary = read_json(round3 / "analysis" / "eval-summary.json")["summary"]
    return {
        "lattice-ladder": chart_lattice_ladder(eval_summary),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="directory for the SVG files")
    parser.add_argument("--round2", type=Path, default=ROUND2)
    parser.add_argument("--round3", type=Path, default=ROUND3)
    args = parser.parse_args(argv)
    for name, fig in build_figures(args.round2, args.round3).items():
        path = args.out / f"{name}.svg"
        save(fig, path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
