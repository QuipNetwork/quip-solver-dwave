"""``--solve``: one Ising problem as JSON on stdin, every read on stdout.

SPEC section 2 of quip-solver-core defines the input, the output and the exit
codes. This mode never enters the session loop, so no coordinator, budget,
ledger or round strategy is involved: it connects, samples once, and exits.

The problem's node ``i`` is qubit ``i``. Without ``--embed``, a model has to
sit on the chip's working graph already, and one that does not is refused as
capacity, as SPEC section 2 asks. With it, Ocean embeds the model onto the
chip and every read reports how its chains held.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import IO, Any

import numpy as np
from quip_solver_core import scoring

from quip_solver_dwave import (
    ALGORITHM,
    BACKEND,
    EXIT_CAPACITY,
    EXIT_CLEAN,
    EXIT_CONFIG_INVALID,
    EXIT_ENV_INCOMPATIBLE,
    EXIT_INTERNAL_FATAL,
    MAX_EDGES,
    MAX_NODES,
)
from quip_solver_dwave.job import spins_in_node_order
from quip_solver_dwave.ocean import (
    OceanSampler,
    is_solver_unavailable,
    mock_mode_enabled,
    ocean_importable,
)
from quip_solver_dwave.qp import OffChipError
from quip_solver_dwave.schedule import ScheduleError

logger = logging.getLogger(__name__)

# Every key but these two is required, and no other key is accepted.
REQUIRED_KEYS = frozenset(
    {"h", "j", "edges", "num_reads", "num_sweeps", "sweeps_per_beta", "beta_range", "seed"}
)
OPTIONAL_KEYS = frozenset({"beta_schedule_type", "anneal_time_us"})

_U64_MAX = 2**64 - 1


class Malformed(ValueError):
    """The input is not a valid problem: exit 64."""


class Capacity(Exception):
    """The model exceeds what this solver solves: exit 65."""


@dataclass(frozen=True)
class Problem:
    """One parsed ``--solve`` problem, coefficients in units."""

    h: list[float]
    j: list[float]
    edges: list[tuple[int, int]]
    num_reads: int
    num_sweeps: int
    sweeps_per_beta: int
    beta_range: list[float] | None
    seed: int
    anneal_time_us: int | None


def _number(where: str, value: Any) -> float:
    # bool is an int in Python and NaN parses from Python's json; neither is a
    # JSON number a coefficient can be.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Malformed(f"{where} is not a number")
    if not math.isfinite(value):
        raise Malformed(f"{where} is not finite")
    return float(value)


def _integer(key: str, value: Any, minimum: int, maximum: int = _U64_MAX) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Malformed(f"{key} is not an integer")
    if not minimum <= value <= maximum:
        raise Malformed(f"{key} must be from {minimum} to {maximum}")
    return value


def _numbers(key: str, value: Any) -> list[float]:
    if not isinstance(value, list):
        raise Malformed(f"{key} is not an array")
    return [_number(f"{key}[{i}]", v) for i, v in enumerate(value)]


def _edges(value: Any, num_nodes: int) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        raise Malformed("edges is not an array")
    edges: list[tuple[int, int]] = []
    for index, edge in enumerate(value):
        if not isinstance(edge, list) or len(edge) != 2:
            raise Malformed(f"edges[{index}] is not a [u, v] pair")
        u = _integer(f"edges[{index}][0]", edge[0], 0)
        v = _integer(f"edges[{index}][1]", edge[1], 0)
        if u >= num_nodes or v >= num_nodes:
            raise Malformed(f"edges[{index}] has an endpoint outside h")
        if u == v:
            raise Malformed(f"edges[{index}] joins node {u} to itself")
        edges.append((u, v))
    return edges


def parse_problem(raw: bytes) -> Problem:
    """Parse and validate one problem.

    Raises:
        Malformed: Anything SPEC section 2 lists as a malformed input.
    """
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Malformed(f"not JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise Malformed("the problem is not a JSON object")
    missing = REQUIRED_KEYS - doc.keys()
    if missing:
        raise Malformed(f"missing keys: {sorted(missing)}")
    unknown = doc.keys() - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise Malformed(f"unknown keys: {sorted(unknown)}")

    schedule = doc.get("beta_schedule_type", "geometric")
    if schedule not in ("geometric", "linear"):
        raise Malformed(f"unknown beta_schedule_type {schedule!r}")
    if schedule == "linear":
        # An annealer has no beta ladder to shape, so it does not list
        # linear-schedule, and SPEC section 2 refuses the request.
        raise Malformed("beta_schedule_type 'linear' needs a solver that lists linear-schedule")

    h = _numbers("h", doc["h"])
    j = _numbers("j", doc["j"])
    edges = _edges(doc["edges"], len(h))
    if len(j) != len(edges):
        raise Malformed(f"{len(j)} couplings for {len(edges)} edges")

    beta_range = doc["beta_range"]
    if beta_range is not None:
        if not isinstance(beta_range, list) or len(beta_range) != 2:
            raise Malformed("beta_range is not null or a [hot, cold] pair")
        beta_range = [_number(f"beta_range[{i}]", b) for i, b in enumerate(beta_range)]

    anneal_time_us = doc.get("anneal_time_us")
    if anneal_time_us is not None:
        anneal_time_us = _integer("anneal_time_us", anneal_time_us, 1)

    return Problem(
        h=h,
        j=j,
        edges=edges,
        num_reads=_integer("num_reads", doc["num_reads"], 1),
        num_sweeps=_integer("num_sweeps", doc["num_sweeps"], 0),
        sweeps_per_beta=_integer("sweeps_per_beta", doc["sweeps_per_beta"], 1),
        beta_range=beta_range,
        seed=_integer("seed", doc["seed"], 0),
        anneal_time_us=anneal_time_us,
    )


def check_size(problem: Problem) -> None:
    """Refuse a model past the advertised node and edge limits.

    Raises:
        Capacity: More nodes than ``MAX_NODES`` or edges than ``MAX_EDGES``.
    """
    if len(problem.h) > MAX_NODES:
        raise Capacity(f"{len(problem.h)} nodes exceed the maximum of {MAX_NODES}")
    if len(problem.edges) > MAX_EDGES:
        raise Capacity(f"{len(problem.edges)} edges exceed the maximum of {MAX_EDGES}")


def _sampler_echo(problem: Problem) -> dict[str, Any]:
    # The sweep, ladder and seed keys steer nothing on a QPU. They are echoed
    # as sent so one input shape reaches every solver (SPEC section 2).
    echo: dict[str, Any] = {
        "backend": BACKEND,
        "algorithm": ALGORITHM,
        "num_reads": problem.num_reads,
        "num_sweeps": problem.num_sweeps,
        "sweeps_per_beta": problem.sweeps_per_beta,
        "beta_range": problem.beta_range,
        "beta_schedule_type": "geometric",
        "seed": problem.seed,
    }
    if problem.anneal_time_us is not None:
        echo["anneal_time_us"] = problem.anneal_time_us
    return echo


def solve(problem: Problem, sampler: OceanSampler, *, started_ns: int) -> list[dict[str, Any]]:
    """Sample ``problem`` once and render one output object per read.

    Raises:
        Capacity: Too many reads for the solver, a node or edge off the
            working graph, or an anneal time outside the solver's range.
        RuntimeError: The sampler answered with a read count other than the
            one asked for.
    """
    max_reads = sampler.max_reads
    if max_reads is not None and problem.num_reads > max_reads:
        raise Capacity(f"num_reads {problem.num_reads} exceeds the solver's maximum of {max_reads}")

    nodes = np.arange(len(problem.h), dtype=np.int64)
    sample_started = time.monotonic_ns()
    try:
        result = sampler.sample(
            nodes,
            np.asarray(problem.h, dtype=np.float64),
            np.asarray(problem.edges, dtype=np.int64).reshape(-1, 2),
            np.asarray(problem.j, dtype=np.float64),
            num_reads=problem.num_reads,
            anneal_time_us=problem.anneal_time_us,
            label="quip-solve",
        )
    except (OffChipError, ScheduleError) as exc:
        raise Capacity(str(exc)) from None
    sample_us = (time.monotonic_ns() - sample_started) // 1000

    # The cloud folds identical reads into one row; SPEC section 2 wants one
    # object per read, so each row repeats once per anneal that returned it.
    rows = spins_in_node_order(nodes, result)
    occurrences = result.occurrences or [1] * len(rows)
    if sum(occurrences) != problem.num_reads:
        raise RuntimeError(
            f"the sampler returned {sum(occurrences)} reads for num_reads {problem.num_reads}"
        )
    timing = {
        # D-Wave's billed QPU time when it reports one, else the wall clock
        # around the call, so sample_us never exceeds total_us.
        "sample_us": min(result.device_access_time_us or sample_us, sample_us),
        "total_us": (time.monotonic_ns() - started_ns) // 1000,
    }
    echo = _sampler_echo(problem)
    reads: list[dict[str, Any]] = []
    for index, (row, count) in enumerate(zip(rows, occurrences)):
        spins = [int(s) for s in row]
        # Rescored with the network's scorer rather than rounded from the
        # QPU's float, so the energy is exact for the spins reported.
        energy = scoring.energy_milli(spins, problem.h, problem.j, problem.edges)
        read: dict[str, Any] = {
            "spins": spins,
            "energy_milli": energy,
            "timing": timing,
            "sampler": echo,
        }
        if result.embedding is not None:
            read["embedding"] = {
                "embedded": True,
                "chain_break_fraction": result.embedding.chain_break_fraction[index],
                "max_chain_length": result.embedding.max_chain_length,
            }
        reads.extend([read] * count)
    return reads


def run_solve(
    *,
    solver_name: str | None,
    mock: bool,
    embed: bool = False,
    stdin: IO[bytes] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """``--solve``: read one problem, write its reads, return the exit code."""
    started_ns = time.monotonic_ns()
    source: IO[bytes] = stdin if stdin is not None else sys.stdin.buffer
    sink: IO[str] = stdout if stdout is not None else sys.stdout
    try:
        problem = parse_problem(source.read())
        check_size(problem)
    except Malformed as exc:
        print(f"error: malformed problem: {exc}", file=sys.stderr)
        return EXIT_CONFIG_INVALID
    except Capacity as exc:
        print(f"error: model exceeds this solver: {exc}", file=sys.stderr)
        return EXIT_CAPACITY

    use_mock = mock or mock_mode_enabled()
    if not use_mock and not ocean_importable():
        print("error: dwave-ocean-sdk is not importable", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE
    try:
        sampler = OceanSampler(solver_name=solver_name, mock=use_mock, embed=embed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_INVALID
    try:
        sampler.ensure_connected()
    except Exception as exc:  # noqa: BLE001 - operator-facing message, not a trace
        print(f"error: could not open the solver: {exc}", file=sys.stderr)
        return EXIT_ENV_INCOMPATIBLE

    try:
        reads = solve(problem, sampler, started_ns=started_ns)
    except Capacity as exc:
        print(f"error: model exceeds this solver: {exc}", file=sys.stderr)
        return EXIT_CAPACITY
    except Exception as exc:  # every sampler failure maps to an exit code
        if is_solver_unavailable(exc):
            print(f"error: the solver is unavailable: {exc}", file=sys.stderr)
            return EXIT_ENV_INCOMPATIBLE
        logger.exception("sampling failed")
        return EXIT_INTERNAL_FATAL
    finally:
        sampler.close()

    try:
        json.dump(reads, sink, separators=(",", ":"))
        sink.write("\n")
        sink.flush()
    except BrokenPipeError:
        # The reader went away; the solve itself succeeded.
        pass
    return EXIT_CLEAN
