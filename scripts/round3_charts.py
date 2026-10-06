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


def read_clean_timing(root: Path) -> List[Dict[str, Any]]:
    """One row per completed clean run: the kernel that ran, its sweeps, and its sampling seconds."""
    rows: List[Dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/result.json")):
        record = read_json(path)
        if record["status"] == "completed":
            rows.append({"cell": record["cell"], "kernel": record["observed_kernel"],
                         "sweeps": record["sweeps"], "sampling_s": record["sampling_s"]})
    return rows


def read_seeded(round3: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in ("eval100", "f025", "n125-early"):
        rows += read_json(round3 / "seeded" / name / "summary.json")
    return rows


def chart_clean_time(rows: Sequence[Mapping[str, Any]]) -> Figure:
    """Sampling seconds of one model on one quiet core, by sweeps and kernel, against the QPU's times."""
    fig, axes = plt.subplots(1, len(CELLS), figsize=(11, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.5)
    for ax, cell in zip(axes, CELLS):
        series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for row in rows:
            if row["cell"] == cell:
                series[row["kernel"]].append((row["sweeps"], row["sampling_s"]))
        ends = []
        for kernel, points in sorted(series.items()):
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", ms=3, lw=1, **KERNEL_STYLE[kernel])
            ends.append((points[-1][0], points[-1][1], kernel, KERNEL_STYLE[kernel]["color"]))
        for seconds in (0.07, 2.2):  # QPU access for 64 reads, and QPU wall time per call; the caption names them
            ax.axhline(seconds, color=QPU, lw=0.6, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        label_ends(ax, ends)
        style(ax, title=cell, xlabel="sweeps", ylabel="seconds, one model, one core" if cell == CELLS[0] else "")
    return fig


def chart_seeded(rows: Sequence[Mapping[str, Any]]) -> Figure:
    """Models of 100 where the QPU-seeded run beats cold, at equal sweeps (solid) and against cold 131,072 (dashed).

    Sign lattices show the unit kernel. ``native-125`` has only the f64 kernel. The report caption explains the
    line styles, so the chart carries no legend.
    """
    fig, axes = plt.subplots(1, len(CELLS), figsize=(11, 2.8), sharey=True)
    fig.subplots_adjust(wspace=0.5)
    for ax, cell in zip(axes, CELLS):
        ends = []
        for fraction, color in ((0.25, INK), (0.5, GRAY)):
            picked = sorted((r for r in rows if r["cell"] == cell and r["fraction"] == fraction
                             and (r["kernel"] == "cpu-msa-unit" or cell == "native-125")),
                            key=lambda r: r["sweeps"])
            if not picked:
                continue
            x = [r["sweeps"] for r in picked]
            same = [r["vs_cold_same"]["seeded_better"] for r in picked]
            deep = [r["vs_cold_131072"]["seeded_better"] for r in picked]
            ax.plot(x, same, marker="o", ms=3, lw=1, color=color)
            ax.plot(x, deep, marker="o", ms=3, lw=1, color=color, ls="--")
            ends.append((x[-1], same[-1], f"f = {fraction}", color))
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 100)
        ax.axhline(50, color=LIGHT, lw=0.6)
        label_ends(ax, ends)
        style(ax, title=cell, xlabel="seeded sweeps", ylabel="seeded better, of 100" if cell == CELLS[0] else "")
    return fig


def chart_cuda_depth(table: Sequence[Mapping[str, Any]]) -> Figure:
    """cubic-dimer-pm1: CUDA MSA at 128 reads against the QPU's 1,024 reads and its expected best of 64."""
    labels: List[str] = []
    triples: List[Tuple[int, int, int]] = []
    for row in sorted((r for r in table if r["reads"] == 128), key=lambda r: r["sweeps"]):
        for key, qpu_reads in (("vs_qpu_best", "1,024 reads"), ("vs_qpu_best64", "64 reads")):
            labels.append(f"CUDA {row['sweeps']:,} sweeps vs QPU {qpu_reads}")
            cmp = row[key]
            triples.append((cmp["qpu_better"], cmp["tie"], cmp["cpu_better"]))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1))
    stacked_bars(ax, labels, triples, ("QPU better", "tie", "CUDA better"), total=100)
    style(ax, xlabel="models of 100")
    return fig


def chart_throughput(arms: Sequence[Mapping[str, Any]]) -> Figure:
    """cubic-dimer-pm1 models per second end to end, by reads and calls in flight, against the access limit."""
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(arms) + 1))
    y = list(range(len(arms)))
    ax.hlines(y, [0.0] * len(arms), [a["access_limited_models_per_s"] for a in arms], color=LIGHT, lw=4)
    ax.plot([a["models_per_s"] for a in arms], y, "o", color=QPU, ms=6)
    for yy, arm in zip(y, arms):
        ax.text(arm["models_per_s"], yy - 0.3, f"{arm['models_per_s']:.2f}", ha="center", fontsize=7, color=QPU)
        ax.text(arm["access_limited_models_per_s"], yy, f" limit {arm['access_limited_models_per_s']:.1f}",
                va="center", fontsize=7, color=GRAY)
    ax.set_yticks(y, [f"{a['reads']:,} reads, {a['concurrency']} in flight" for a in arms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 17)
    style(ax, xlabel="models per second")
    return fig


def chart_depth_extension(summary: Mapping[str, Any], qpu_reach: int) -> Figure:
    """cubic-dimer-pm1: models of 100 at the frozen target as cpu-msa-unit sweeps grow, against the QPU."""
    reached = summary["base_models"] - summary["base_missing"]
    points = [(summary["base_sweeps"], reached)]
    for sweeps, models in sorted(summary["depths"].items(), key=lambda kv: int(kv[0])):
        reached += sum(1 for model in models.values() if model["target_reached"])
        points.append((int(sweeps), reached))
    fig, ax = plt.subplots(figsize=(5, 2.6))
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", ms=3, lw=1, color=INK)
    for x, y in points:
        ax.annotate(str(y), (x, y), xytext=(0, 5), textcoords="offset points", fontsize=7, ha="center", color=INK)
    ax.axhline(qpu_reach, color=QPU, lw=0.8, ls="--")
    ax.annotate(f"QPU, 1,024 reads: {qpu_reach}", (points[0][0], qpu_reach), xytext=(0, -9),
                textcoords="offset points", fontsize=7, color=QPU)
    ax.set_xscale("log", base=2)
    ax.set_ylim(40, 100)
    style(ax, xlabel="cpu-msa-unit sweeps", ylabel="models at target, of 100")
    return fig


def build_figures(round2: Path, round3: Path) -> Dict[str, Figure]:
    """Every chart, keyed by file stem. Later tasks add entries."""
    eval_summary = read_json(round3 / "analysis" / "eval-summary.json")["summary"]
    qpu_reach = next(row["reach_target"] for row in eval_summary if row["cell"] == "cubic-dimer-pm1")
    return {
        "lattice-ladder": chart_lattice_ladder(eval_summary),
        "clean-time": chart_clean_time(read_clean_timing(round2 / "cpu-timing-clean")),
        "seeded": chart_seeded(read_seeded(round3)),
        "cuda-depth": chart_cuda_depth(read_json(round3 / "cubic-cuda" / "analysis.json")["table"]),
        "throughput": chart_throughput(read_json(round3 / "qpu-throughput" / "round" / "analysis.json")["arms"]),
        "depth-extension": chart_depth_extension(
            read_json(round2 / "cubic-dimer-depth-extension" / "summary.json"), qpu_reach),
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
