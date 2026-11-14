"""Uniform config-override discipline for the dwave miner.

Python mirror of the Rust ``quip_solver_core::config`` helpers so every miner
type behaves the same: config (from the coordinator's ``Configure``) overrides
CLI/env with a warning, and unrecognized keys warn. Credentials are not carried
here — the dwave miner uses D-Wave's own config (dwave.conf + env).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar

from quip_solver_dwave.tiling import DEFAULT_TILE_PASSES, TILE_MODES

logger = logging.getLogger(__name__)

# Keys consumed by the shared session layer, not any single backend; never
# warned about as unknown. Mirrors miner-core's SESSION_KEYS.
SESSION_KEYS = frozenset({"num_sweeps"})

T = TypeVar("T")


def config_override(name: str, cli: T, from_config: Optional[T]) -> T:
    """Resolve one setting with ``config > CLI`` precedence.

    Warns and returns the config value only on an effective change; otherwise
    returns ``cli`` silently.
    """
    if from_config is not None and from_config != cli:
        logger.warning(
            "config overrides %s: %s -> %s (from coordinator)", name, cli, from_config
        )
        return from_config
    return cli


def warn_unknown_fields(
    backend: str, present: Iterable[str], known: Iterable[str]
) -> None:
    """Warn once per unrecognized key. Session keys are never flagged."""
    known_all = set(known) | SESSION_KEYS
    for key in present:
        if key not in known_all:
            logger.warning("config: unknown field '%s' for %s (ignored)", key, backend)


@dataclass(frozen=True)
class SamplingDefaults:
    """Session-wide sampling parameters from ``Configure.backend_toml``.

    These are *defaults*, the lowest rung of the per-job precedence ladder in
    :func:`quip_solver_dwave.job.sampling_params`: the job's own
    ``IsingProblem`` wins, then the session ``SetTarget``, then these, then the
    hard-coded fallback. Zero means "not set" on the wire, so it means the same
    here and leaves the next rung down in charge.

    ``anneal_time_us`` at 0 leaves D-Wave's own default anneal in place.
    ``reversal_s_milli`` and ``reversal_pause_us`` apply only to a job that
    carries a start state, and at 0 leave the built-in reverse anneal in place.
    """

    num_reads: int = 0
    anneal_time_us: int = 0
    reversal_s_milli: int = 0
    reversal_pause_us: int = 0


def sampling_defaults_from_toml(toml_text: str) -> SamplingDefaults:
    """Read the sampling keys from ``backend_toml``.

    A malformed document yields the empty defaults rather than raising: the
    budget parser is the one that refuses to run on unparseable config (an
    unmetered miner is a spending risk, a default anneal time is not), so
    reporting the same failure twice would only obscure it.
    """
    if not toml_text or not toml_text.strip():
        return SamplingDefaults()
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return SamplingDefaults()

    return SamplingDefaults(
        num_reads=_non_negative(data, "num_reads"),
        anneal_time_us=_non_negative(data, "anneal_time_us"),
        reversal_s_milli=_reversal_s_milli(data),
        reversal_pause_us=_non_negative(data, "reversal_pause_us"),
    )


def _reversal_s_milli(data: dict) -> int:
    """Read ``reversal_s_milli``, or 0 when unset or outside 1 to 999.

    The reversal point is a fraction of the anneal strictly inside (0, 1). The
    wire rejects 1000 or more as malformed, so an operator value that large is
    a misconfiguration to report, not a default to apply to every seeded job.
    """
    value = _non_negative(data, "reversal_s_milli")
    if value >= 1000:
        logger.warning(
            "ignoring reversal_s_milli=%d from config: expected 1 to 999", value
        )
        return 0
    return value


def _non_negative(data: dict, key: str) -> int:
    """Read a non-negative int key, or 0 when unset or nonsense.

    ``bool`` is excluded on purpose: it is an ``int`` subclass in Python, so
    ``queue_depth = true`` would otherwise resolve to a one-deep pipeline
    instead of being reported as the misconfiguration it is.
    """
    raw = data.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        if raw is not None:
            logger.warning(
                "ignoring %s=%r from config: expected a non-negative integer",
                key,
                raw,
            )
        return 0
    return raw


def queue_depth_from_toml(toml_text: str) -> int:
    """Read ``queue_depth`` from ``backend_toml``; 0 when the operator is silent.

    How many submissions this backend keeps in flight is a property of the
    device and its cloud round trip, which the coordinator cannot know, so the
    operator gets to override what it sends. Same lenient parse as the
    sampling defaults: a document that will not parse is the budget parser's
    problem to report, not this one's.
    """
    if not toml_text or not toml_text.strip():
        return 0
    try:
        data = tomllib.loads(toml_text)
    except Exception:
        return 0
    return _non_negative(data, "queue_depth")


# Measured on Advantage2 on 2026-11-14 at 48 reads of a 160 us anneal: 40 slots
# kept 36 or more problems at SAPI at the chip's 15 problems per second, and
# answered 0.4 s sooner than 48 slots.
DEFAULT_SAPI_SLOTS = 40


def _backend_table(toml_text: str) -> dict:
    """The parsed ``backend_toml``, or an empty table when it is blank or will not parse."""
    if not toml_text or not toml_text.strip():
        return {}
    try:
        return tomllib.loads(toml_text)
    except Exception:
        return {}


def sapi_slots_from_toml(toml_text: str) -> int:
    """Read ``sapi_slots`` from ``backend_toml``; :data:`DEFAULT_SAPI_SLOTS` when unset.

    A positive value sends each problem through
    :class:`quip_solver_dwave.direct_sapi.DirectTransport` with that many
    problems at SAPI at once; the rest of ``queue_depth`` waits locally, where
    a Cancel costs nothing. An explicit 0 keeps the cloud client's own
    transport. Same lenient parse as :func:`queue_depth_from_toml`.
    """
    data = _backend_table(toml_text)
    if data.get("sapi_slots") is None:
        return DEFAULT_SAPI_SLOTS
    return _non_negative(data, "sapi_slots")


def solver_choice_from_toml(toml_text: str) -> "tuple[Optional[str], Optional[str]]":
    """``(solver, region)`` from ``backend_toml``; None for each key it does not set.

    A coordinator picks the Advantage2 of each miner here, for example
    ``solver = "Advantage2_system4"`` with ``region = "na-east-1"``. An unset
    key leaves the choice to the SDK (``DWAVE_API_SOLVER``,
    ``DWAVE_API_REGION``, ``dwave.conf``). A value that is not a non-empty
    string is warned about and ignored.
    """
    data = _backend_table(toml_text)
    chosen = []
    for key in ("solver", "region"):
        value = data.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            logger.warning("%s must be a solver or region name, got %r; ignoring it", key, value)
            value = None
        chosen.append(value.strip() if isinstance(value, str) else None)
    return chosen[0], chosen[1]


def tile_passes_from_toml(toml_text: str) -> int:
    """Read ``tile_passes`` from ``backend_toml``; :data:`DEFAULT_TILE_PASSES` when unset or 0.

    Only a chip smaller than the session's Zephyr reads it: each job anneals
    every patch this many times (see :mod:`quip_solver_dwave.tiling`).
    """
    return _non_negative(_backend_table(toml_text), "tile_passes") or DEFAULT_TILE_PASSES


def tile_mode_from_toml(toml_text: str) -> str:
    """Read ``tile_mode`` from ``backend_toml``; ``sequential`` when unset or unknown.

    ``batch-seam`` anneals the core patches at once and then the seam patches
    across their cuts (see :mod:`quip_solver_dwave.tiling`).
    """
    value = _backend_table(toml_text).get("tile_mode", "sequential")
    if value not in TILE_MODES:
        logger.warning("tile_mode %r is not one of %s; using sequential", value, TILE_MODES)
        return "sequential"
    return value


def programming_thermalization_us_from_toml(toml_text: str) -> int:
    """Read ``programming_thermalization_us`` from ``backend_toml``; 0 when unset.

    The QPU waits this long after it programs a problem and before the first
    read, and bills the wait. On Advantage2 on 2026-11-14 the solver's own
    1,000 us and 0 gave the same best energy on 120 models (difference
    0.38 +/- 0.29 units), so the default is 0.
    """
    return _non_negative(_backend_table(toml_text), "programming_thermalization_us")
