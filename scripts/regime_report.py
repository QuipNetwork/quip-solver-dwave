#!/usr/bin/env python3
"""Write the results of one round of the regime search.

Reads the capture files and the ladder and seed CSV files of every cell under
``--round-dir``, and writes one Markdown file with a table for each metric of
the specification: matching time, matching rung, gate window, information,
seed lane wins, the screening rule, and the resolution of that rule. Every
metric compares arms inside one cell. With ``--solver-properties`` it also
states which cells the solver API rescales.

A cell of the round is a folder with a ``nonces.json``. The report stops when
a cell lacks a capture or a run of the ladder, because a verdict from a part
of the models misleads. ``--allow-incomplete`` writes the report anyway, and
such a cell then gets no verdict. The spend comes from the spend journal.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from quip_solver_dwave import regime_io, regimes, round2_metrics
from quip_solver_dwave import regime_metrics as metrics

BASELINE = "native-pm1"
EXTREMES = ("j_min", "j_max", "sum_min", "sum_max", "h_abs_max", "j_rms", "j_abs_min")
Match = Dict[Tuple[str, int], dict]


def load_cell(cell_dir: Path) -> dict:
    ladder: Dict[str, Dict[int, float]] = {}
    walls: Dict[int, List[float]] = {}
    k_share: List[float] = []
    label = ""
    if (cell_dir / "ladder.csv").exists():
        with open(cell_dir / "ladder.csv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                sweeps = int(row["sweeps"])
                ladder.setdefault(row["nonce"], {})[sweeps] = float(row["best_energy"])
                walls.setdefault(sweeps, []).append(float(row["wall_s"]))
                label = row["kernel_label"]
                if row.get("k_share"):
                    k_share.append(float(row["k_share"]))
    qpu: Dict[int, Dict[str, dict]] = {}
    for folder in sorted(cell_dir.glob("qpu-*")):
        anneal = int(folder.name.split("-")[1])
        for path in sorted(folder.glob("*.npz")):
            if ".tmp." in path.name:
                continue
            with np.load(path) as capture:
                qpu.setdefault(anneal, {})[path.stem] = {
                    "best": float(capture["energies"].min()),
                    "mock": bool(capture["mock"]),
                    "clamped": int(capture["clamped"]),
                    "removed": int(capture["removed_couplers"]),
                    "broken": float(capture["broken_chain_share"]),
                    "spins": int(capture["spins"].shape[1]),
                    "spin_rows": len(capture["spins"]),
                    "energy_rows": len(capture["energies"]),
                    "held": (capture["spins"] > 0).sum(axis=1),
                    "extremes": {name: float(capture[name]) for name in EXTREMES},
                }
    seed_rows: List[dict] = []
    if (cell_dir / "seed.csv").exists():
        with open(cell_dir / "seed.csv", newline="", encoding="utf-8") as handle:
            seed_rows = list(csv.DictReader(handle))
    manifest_path = cell_dir / regime_io.MANIFEST_FILE
    return {
        "ladder": ladder, "walls": walls, "label": label, "qpu": qpu, "seed": seed_rows, "k_share": k_share,
        "nonces": json.loads((cell_dir / "nonces.json").read_text(encoding="utf-8"))["nonces"],
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
    }


def missing(cell: dict, rungs: List[int]) -> List[str]:
    """What a cell lacks for a complete round: a capture of each model at each anneal time, and a run at each rung."""
    nonces = cell["nonces"]
    anneals = cell["manifest"].get("anneal_us", [])
    gaps = [] if anneals else ["every capture"]
    for anneal in anneals:
        lacking = sum(nonce not in cell["qpu"].get(anneal, {}) for nonce in nonces)
        if lacking:
            gaps.append(f"{lacking} of {len(nonces)} captures at {anneal} µs")
        invalid = sum(
            capture["spin_rows"] != cell["manifest"].get("reads")
            or capture["energy_rows"] != cell["manifest"].get("reads")
            for nonce, capture in cell["qpu"].get(anneal, {}).items() if nonce in nonces
        )
        if invalid:
            gaps.append(f"{invalid} captures at {anneal} µs have a wrong row count")
    for sweeps in sorted(rungs):
        lacking = sum(sweeps not in cell["ladder"].get(nonce, {}) for nonce in nonces)
        if lacking:
            gaps.append(f"{lacking} of {len(nonces)} runs at {sweeps} sweeps")
    return gaps


def quartiles(values: List[float]) -> Tuple[float, float, float]:
    low, mid, high = np.percentile(values, [25, 50, 75])
    return float(low), float(mid), float(high)


def matching(cell: dict, anneal: int, rungs: List[int], extra_rung: int) -> dict:
    """Matching sweeps and time of every model that has a full ladder and a capture.

    A model counts as unmatched when no rung reaches its QPU best energy. ``at_rung`` counts the models that match at
    each rung. ``extra`` counts the unmatched models that ran the extra rung, and how many of them it matched.
    """
    wall = {sweeps: statistics.median(times) for sweeps, times in cell["walls"].items()}
    sweeps_found: List[float] = []
    times: List[float] = []
    unmatched = 0
    models = 0
    extra_runs = 0
    extra_matched = 0
    for nonce, capture in cell["qpu"].get(anneal, {}).items():
        ladder = {s: e for s, e in cell["ladder"].get(nonce, {}).items() if s in rungs}
        if len(ladder) != len(rungs):
            continue
        models += 1
        found = metrics.match_sweeps(ladder, capture["best"])
        if found is None:
            unmatched += 1
            extra = cell["ladder"][nonce].get(extra_rung)
            if extra is not None:
                extra_runs += 1
                extra_matched += int(extra <= capture["best"])
            continue
        sweeps_found.append(found)
        times.append(metrics.matching_time(found, wall) or 0.0)
    return {
        "models": models,
        "unmatched": unmatched,
        "unmatched_share": unmatched / models if models else float("nan"),
        "at_rung": {sweeps: sweeps_found.count(float(sweeps)) for sweeps in rungs},
        "sweeps": quartiles(sweeps_found) if sweeps_found else None,
        "time": quartiles(times) if times else None,
        "extra": (extra_matched, extra_runs),
    }


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None or value != value else f"{value:.{digits}g}"


def fmt_share(value: Optional[float]) -> str:
    """Format a share so that a value below one never prints as a perfect 1."""
    text = fmt(value)
    digits = 3
    while value is not None and value < 1.0 and text == "1" and digits < 15:
        digits += 1
        text = fmt(value, digits)
    return text


def interval(count: int, total: int) -> str:
    low, high = metrics.wilson_interval(count, total)
    return f"{fmt(low)} to {fmt(high)}"


def cells_table(cells: Dict[str, dict]) -> List[str]:
    lines = [
        "## Cells", "",
        "| Cell | Kernel | Spins | Models with a capture | Captures with a clamped qubit "
        "| Clamped qubits, worst capture | Removed couplers, worst capture | Broken chains, mean |",
        "| -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for name, cell in cells.items():
        captures = [c for caps in cell["qpu"].values() for c in caps.values()]
        counts = ", ".join(f"{len(caps)} at {anneal} µs" for anneal, caps in sorted(cell["qpu"].items()))
        lines.append(
            f"| `{name}` | `{cell['label']}` | {captures[0]['spins'] if captures else 'n/a'} | {counts or 0} | "
            f"{sum(c['clamped'] > 0 for c in captures)} | "
            f"{max((c['clamped'] for c in captures), default=0)} | "
            f"{max((c['removed'] for c in captures), default=0)} | "
            f"{fmt(float(np.mean([c['broken'] for c in captures])) if captures else None)} |"
        )
    return lines


def matching_table(match: Match) -> List[str]:
    lines = [
        "## Matching time", "",
        "Median and quartiles over the matched models. Time is core-seconds. "
        "A model counts as unmatched when no rung reaches its QPU best energy. "
        "The interval is the 95% Wilson interval of the unmatched share.", "",
        "| Cell | Anneal µs | Models | Unmatched | Unmatched share | 95% interval | Matching sweeps (25%, 50%, 75%) "
        "| Matching time (25%, 50%, 75%) | Unmatched models that the extra rung matches |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for (name, anneal), m in match.items():
        sweeps = ", ".join(fmt(v) for v in m["sweeps"]) if m["sweeps"] else "n/a"
        times = ", ".join(fmt(v) for v in m["time"]) if m["time"] else "n/a"
        lines.append(
            f"| `{name}` | {anneal} | {m['models']} | {m['unmatched']} | {fmt(m['unmatched_share'])} | "
            f"{interval(m['unmatched'], m['models'])} | {sweeps} | {times} | {m['extra'][0]} of {m['extra'][1]} |"
        )
    return lines


def rung_table(match: Match, rungs: List[int]) -> List[str]:
    """The matching sweeps of a model take the values of the rungs only, and a median hides that."""
    rungs = sorted(rungs)
    lines = [
        "## Matching rung", "", "The share of models that match at each rung. The shares of a row add up to 1.", "",
        "| Cell | Anneal µs | Models | " + " | ".join(f"{sweeps} sweeps" for sweeps in rungs) + " | Unmatched |",
        "| -- | -- | -- | " + " | ".join("--" for _ in rungs) + " | -- |",
    ]
    for (name, anneal), m in match.items():
        shares = " | ".join(fmt(m["at_rung"][sweeps] / m["models"] if m["models"] else None) for sweeps in rungs)
        lines.append(f"| `{name}` | {anneal} | {m['models']} | {shares} | {fmt(m['unmatched_share'])} |")
    return lines


def window_and_information_tables(
    cells: Dict[str, dict], anneals: List[int], lite_rung: int, deep_rung: int, resamples: int
) -> Tuple[List[str], List[str]]:
    """Both tables read the same three series of a cell: lite, deep, and QPU best energies."""
    window = [
        "## Gate window", "",
        "| Cell | Anneal µs | E lite | E deep | E QPU | Spread | Window | QPU position |",
        "| -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    information = [
        "## Information", "",
        "| Cell | Anneal µs | Rank correlation | Value | 95% interval |",
        "| -- | -- | -- | -- | -- |",
    ]
    for name, cell in cells.items():
        for anneal in anneals:
            captured = cell["qpu"].get(anneal, {})
            nonces = [n for n in captured if {lite_rung, deep_rung} <= set(cell["ladder"].get(n, {}))]
            if len(nonces) < 3:
                continue
            lite = [cell["ladder"][n][lite_rung] for n in nonces]
            deep = [cell["ladder"][n][deep_rung] for n in nonces]
            best = [cell["qpu"][anneal][n]["best"] for n in nonces]
            w = metrics.gate_window(lite, deep, best)
            window.append(
                f"| `{name}` | {anneal} | {fmt(w['e_lite'], 8)} | {fmt(w['e_deep'], 8)} | {fmt(w['e_qpu'], 8)} | "
                f"{fmt(w['spread'])} | {fmt(w['window'])} | {fmt(w['qpu_position'])} |"
            )
            rows = (
                ("lite with deep", metrics.spearman, (lite, deep)),
                ("QPU with deep", metrics.spearman, (best, deep)),
                ("QPU with deep, lite held fixed", metrics.partial_spearman, (best, deep, lite)),
            )
            for title, statistic, series in rows:
                low, high = metrics.bootstrap_interval(statistic, *series, resamples=resamples)
                information.append(
                    f"| `{name}` | {anneal} | {title} | {fmt(statistic(*series))} | {fmt(low)} to {fmt(high)} |"
                )
    return window, information


def seed_table(cells: Dict[str, dict]) -> List[str]:
    lines = [
        "## Seed lane wins", "",
        "| Cell | Models | QPU lanes win | Lite lanes win | Ties | Median QPU lanes "
        "| Models with unequal lane groups |",
        "| -- | -- | -- | -- | -- | -- | -- |",
    ]
    for name, cell in cells.items():
        rows = [r for r in cell["seed"] if r["qpu_lane_best"] and r["lite_lane_best"]]
        if not rows:
            continue
        qpu_wins = sum(float(r["qpu_lane_best"]) < float(r["lite_lane_best"]) for r in rows)
        lite_wins = sum(float(r["lite_lane_best"]) < float(r["qpu_lane_best"]) for r in rows)
        lanes = statistics.median(int(r["qpu_lanes"]) for r in rows)
        unequal = sum(int(r["qpu_lanes"]) != int(r["lite_lanes"]) for r in rows)
        lines.append(
            f"| `{name}` | {len(rows)} | {qpu_wins} | {lite_wins} | {len(rows) - qpu_wins - lite_wins} | "
            f"{fmt(lanes)} | {unequal} |"
        )
    return lines


def scale_table(cells: Dict[str, dict]) -> List[str]:
    """The energy scale of each cell, which the round does not control (specification section 7.3)."""
    lines = [
        "## Energy scale", "",
        "The cells do not share one energy scale, and the round has no control at a matched scale. "
        "A result holds for the energy scale and for the analog precision of its cell. "
        "With `--solver-properties`, the Solver ranges table checks the available limits and states any missing checks.",
        "Each value is the median over the models, as a share of the largest coupling size of the model. "
        "A coupling far under the largest one is close to the analog precision of the chip, "
        "and this report cannot measure that precision.", "",
        "| Cell | RMS coupling | Smallest coupling |",
        "| -- | -- | -- |",
    ]
    for name, cell in cells.items():
        rows = [c["extremes"] for caps in cell["qpu"].values() for c in caps.values()]
        if not rows:
            continue
        largest = [max(abs(r["j_min"]), abs(r["j_max"])) for r in rows]
        rms = statistics.median(r["j_rms"] / top for r, top in zip(rows, largest))
        smallest = statistics.median(r["j_abs_min"] / top for r, top in zip(rows, largest))
        lines.append(f"| `{name}` | {fmt(rms)} | {fmt(smallest)} |")
    return lines


def screening_table(match: Match, incomplete: List[str]) -> List[str]:
    """``incomplete`` names the cells that lack data. Such a cell gets no verdict, and no cell gets one without a
    complete baseline."""
    lines = [
        "## Screening rule", "",
        "A cell is of interest when one of two conditions holds. "
        "The first condition is a median matching time of 10 times the baseline or more. "
        "The second condition is an unmatched share of 25% or more.",
        "A condition that holds shows its name. The first condition names what made the time. "
        "It is `matching sweeps` when the sweep ratio is also 10 or more, and `kernel cost alone` when it is not. "
        "The sweep ratio shows the change in hardness. The time ratio adds the cost of the kernel. "
        "The ladder measures wall times with every worker thread loaded, and that load can cost one kernel more "
        "than another.",
        "The second condition is uncertain when the 95% Wilson interval of the unmatched share spans 25%. "
        "The baseline share is the unmatched share of the baseline at the same anneal time.",
        "A result for `cubic-dimer-pm1` holds against single-spin kernels only. "
        "No classical arm of this round flips a dimer as one unit.", "",
        "| Cell | Anneal µs | Sweep ratio to baseline | Time ratio to baseline | First condition | Unmatched share "
        "| 95% interval | Baseline share | Second condition | Of interest |",
        "| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |",
    ]
    for (name, anneal), m in match.items():
        if name == BASELINE:
            continue
        base = match.get((BASELINE, anneal))
        base_time = base["time"][1] if base and base["time"] else None
        base_sweeps = base["sweeps"][1] if base and base["sweeps"] else None
        time_mid = m["time"][1] if m["time"] else None
        sweeps_mid = m["sweeps"][1] if m["sweeps"] else None
        first = metrics.time_condition(time_mid, base_time, sweeps_mid, base_sweeps)
        second, uncertain = metrics.share_condition(m["unmatched"], m["models"])
        verdict = "incomplete" if {name, BASELINE} & set(incomplete) else "yes" if first or second else "no"
        lines.append(
            f"| `{name}` | {anneal} | {fmt(sweeps_mid / base_sweeps if sweeps_mid and base_sweeps else None)} | "
            f"{fmt(time_mid / base_time if time_mid and base_time else None)} | {first or 'no'} | "
            f"{fmt(m['unmatched_share'])} | {interval(m['unmatched'], m['models'])} | "
            f"{fmt(base['unmatched_share'] if base else None)} | "
            f"{second or 'no'}{', uncertain' if uncertain else ''} | {verdict} |"
        )
    return lines


def resolution_table(cells: Dict[str, dict], match: Match, rungs: List[int]) -> List[str]:
    """The resolution of the first condition of the screening rule (specification section 9.5).

    A sweep ratio takes only the values that two rungs give. The least ratio that passes depends on the cost of a
    sweep in the cell. The largest ratio depends on the rung of the baseline median.
    """
    top = max(rungs)
    offered = sorted({high / low for high in rungs for low in rungs})
    base_walls = cells[BASELINE]["walls"].get(top) if BASELINE in cells else None
    base_wall = statistics.median(base_walls) if base_walls else None
    lines = [
        "## Screening resolution", "",
        "The first condition of the screening rule is coarse, because a sweep ratio takes the values of the rungs "
        "only. The sweep cost is the median wall time of the top rung against the baseline. "
        "A sweep ratio passes when its product with the sweep cost is 10 or more. "
        "The largest ratio is the top rung against the median matching sweeps of the baseline.",
        "When the least ratio that passes is over the largest ratio, only the second condition can select the cell.",
        "",
        "| Cell | Anneal µs | Sweep cost against baseline | Least sweep ratio that passes "
        "| Largest sweep ratio of the ladder | First condition can hold |",
        "| -- | -- | -- | -- | -- | -- |",
    ]
    for name, anneal in match:
        if name == BASELINE:
            continue
        walls = cells[name]["walls"].get(top)
        cost = statistics.median(walls) / base_wall if walls and base_wall else None
        least = next((ratio for ratio in offered if cost is not None and cost * ratio >= metrics.TIME_FACTOR), None)
        base = match.get((BASELINE, anneal))
        largest = top / base["sweeps"][1] if base and base["sweeps"] else None
        can_hold = least is not None and largest is not None and least <= largest
        least_text = "none" if cost is not None and least is None else fmt(least)
        lines.append(
            f"| `{name}` | {anneal} | {fmt(cost)} | {least_text} | {fmt(largest)} | {'yes' if can_hold else 'no'} |"
        )
    return lines


def ranges_table(cells: Dict[str, dict], properties: dict) -> List[str]:
    """Coupling extremes against available solver ranges, without inferring missing group sums."""
    j_low, j_high = properties["extended_j_range"]
    sum_range = properties.get("per_qubit_coupling_range")
    if sum_range is None:
        range_note = (
            f"`extended_j_range` is {j_low} to {j_high}. Per-qubit coupling range is unavailable. "
            f"`per_group_coupling_range` is {properties.get('per_group_coupling_range', 'unavailable')}. "
            "Group sums were not recorded, so this table cannot determine which models the solver rescales."
        )
    else:
        sum_low, sum_high = sum_range
        range_note = (
            f"`extended_j_range` is {j_low} to {j_high}. `per_qubit_coupling_range` is {sum_low} to {sum_high}. "
            "The solver API scales a model down when a value is outside a range. "
            "It scales a model up when the largest coefficient is less than 1."
        )
    lines = [
        "## Solver ranges", "",
        range_note, "",
        "| Cell | Lowest J | Highest J | Lowest coupling sum | Highest coupling sum | Largest field "
        + ("| Models outside J range | Coupling-sum check |" if sum_range is None
           else "| Models outside a range |"),
        "| -- | -- | -- | -- | -- | -- | -- |" + (" -- |" if sum_range is None else ""),
    ]
    for name, cell in cells.items():
        rows = [c["extremes"] for caps in cell["qpu"].values() for c in caps.values()]
        if not rows:
            continue
        outside_j = sum(r["j_min"] < j_low or r["j_max"] > j_high for r in rows)
        outside_text = f"{outside_j} of {len(rows)} | n/a"
        if sum_range is not None:
            outside = sum(
                r["j_min"] < j_low or r["j_max"] > j_high
                or r["sum_min"] < sum_range[0] or r["sum_max"] > sum_range[1]
                for r in rows
            )
            outside_text = f"{outside} of {len(rows)}"
        lines.append(
            f"| `{name}` | {fmt(min(r['j_min'] for r in rows))} | {fmt(max(r['j_max'] for r in rows))} | "
            f"{fmt(min(r['sum_min'] for r in rows))} | {fmt(max(r['sum_max'] for r in rows))} | "
            f"{fmt(max(r['h_abs_max'] for r in rows))} | {outside_text} |"
        )
    return lines


def clique_table(cell: dict, anneals: List[int]) -> List[str]:
    holdings = cell["manifest"]["clique_holdings"]
    lines = ["## Clique cell", "", f"| Arm | Reads with exactly {holdings} assets |", "| -- | -- |"]
    for anneal in anneals:
        held = np.concatenate([c["held"] for c in cell["qpu"].get(anneal, {}).values()] or [np.array([])])
        count = int((held == holdings).sum())
        lines.append(f"| QPU, {anneal} µs | {round2_metrics.format_feasibility(count, len(held)) if len(held) else 'n/a'} |")
    # The ladder CSV keeps only rounded shares. The manifest pins the reads of every run.
    reads = int(cell["manifest"]["reads"])
    ranges = [round2_metrics.reconstructed_feasible_range(share, reads) for share in cell["k_share"]]
    if not ranges:
        kernel = "n/a"
    elif all(low == high for low, high in ranges):
        total = sum(low for low, _ in ranges)
        kernel = f"{round2_metrics.format_feasibility(total, reads * len(ranges))} [reconstructed from rounded shares]"
    else:
        kernel = f"{fmt_share(float(np.mean(cell['k_share'])))} mean share; counts not recoverable"
    lines.append(f"| Kernel, all rungs | {kernel} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--round-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rungs", type=int, nargs="+", default=[512, 2048, 8192, 32768, 131072])
    parser.add_argument("--extra-rung", type=int, default=524288)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--solver-properties", help="solver-properties.json of step 0, for the solver ranges table")
    parser.add_argument("--allow-incomplete", action="store_true", help="write the report for a round that lacks data")
    args = parser.parse_args()

    round_dir = Path(args.round_dir)
    cells = {
        name: load_cell(round_dir / name)
        for name in regimes.CELL_NAMES
        if (round_dir / name / "nonces.json").exists()
    }
    if not cells:
        raise SystemExit(f"no cell under {round_dir} has a nonces.json")
    kinds = {c["mock"] for cell in cells.values() for caps in cell["qpu"].values() for c in caps.values()}
    if len(kinds) > 1:
        raise SystemExit("the round mixes mock captures and real captures")
    if len({cell["manifest"].get("topology_hash") for cell in cells.values()}) > 1:
        raise SystemExit("the cells of the round come from different topologies")
    gaps = {name: missing(cell, args.rungs) for name, cell in cells.items()}
    if BASELINE not in cells:
        gaps[BASELINE] = ["every baseline capture and run"]
    else:
        baseline = cells[BASELINE]
        compared = {anneal for cell in cells.values() for anneal in cell["manifest"].get("anneal_us", [])}
        for anneal in sorted(compared - set(baseline["manifest"].get("anneal_us", []))):
            gaps[BASELINE].append(f"baseline coverage at {anneal} µs")
    incomplete = [name for name in gaps if gaps[name]]
    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            "the round is not complete:\n"
            + "\n".join(f"  {name}: {', '.join(gaps[name])}" for name in incomplete)
            + "\nFinish the round, or pass --allow-incomplete for a report with no verdict for these cells"
        )
    anneals = sorted({anneal for cell in cells.values() for anneal in cell["qpu"]})
    match = {
        (name, anneal): matching(cell, anneal, args.rungs, args.extra_rung)
        for name, cell in cells.items()
        for anneal in anneals
    }
    window, information = window_and_information_tables(
        cells, anneals, min(args.rungs), max(args.rungs), args.resamples
    )

    spent, open_jobs = regime_io.read_spend(round_dir)
    lines = [
        "# Regime search results", "",
        f"QPU access time spent: {spent / 1e6:.2f} s. "
        + (f"Jobs with no recorded charge: {open_jobs}. Each one counts at its estimate. " if open_jobs else "")
        + "Energies are lower-is-better and compare only inside a cell.",
        "The classical arm is the CPU kernels that the miner runs today, `cpu-msa` and `cpu-sa`. "
        "The round has no GPU arm.",
    ]
    if kinds == {True}:
        lines.append("This is a mock round. The offline sampler made the reads, so no number here describes the QPU.")
    lines += [f"The round is not complete. `{name}` lacks {', '.join(gaps[name])}." for name in incomplete]
    lines.append("")
    tables = [
        cells_table(cells), scale_table(cells), matching_table(match), rung_table(match, args.rungs), window,
        information, seed_table(cells), screening_table(match, incomplete),
        resolution_table(cells, match, args.rungs),
    ]
    if "clique-portfolio" in cells:
        tables.append(clique_table(cells["clique-portfolio"], anneals))
    if args.solver_properties:
        properties = json.loads(Path(args.solver_properties).read_text(encoding="utf-8"))
        tables.append(ranges_table(cells, properties))
    for table in tables:
        lines += table + [""]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
