"""Offline reproduction of P (qportfolio)'s portfolio scoring pipeline.

Every portfolio solver arm (QPU, dwave-neal, scalar SA, MSA) will pass its raw
reads through the adapter here (:func:`score_reads`) before a Round 2 result is
trusted. It reproduces, without modifying P, the shared finish every provider
in P's own race goes through: ``qpo.encoders.encode_select`` -> a dimod
``BinaryQuadraticModel`` -> ``qpo.solvers.sampling.select_solution`` -> the true
portfolio objective and ``qpo.solvers.feasibility.check_feasibility``.

P (``qpo``, ``dimod``, ``scipy``) lives only in its own pinned environment, so
every P import in this module is LAZY (inside a function body). Importing this
module never requires P; calling most of its functions does.

Fixtures built here are always ``provenance = "synthetic-test"``: Task 1 of
the Round 2 plan found no historical portfolio market capture, so historical
replication stays blocked and no live market data is ever fetched.

A fixture's ``diversification_beta`` is P's cardinality-mode frustration term
(``PortfolioProblem.frustration_beta``, the ``beta`` in ``qpo.problem``'s
module docstring). It is NOT an annealing inverse temperature; nothing here
touches an annealer's schedule. Two ways to arrive at that value are kept
distinct via ``beta_mode``:

    "manuscript-ramp"         :func:`manuscript_ramp_beta` -- the ramp derived
                               directly from portfolio-paper-source.tex's
                               eq:beta-ramp (lines 267-275).
    "repository-default-ramp" :func:`repository_default_beta` -- P's shipped
                               ramp, as it stands in its pinned ``qpo.config``
                               today (``qpo.encoders.resolve_frustration_beta``).

The two ramps are DIFFERENT formulas (different floor and slope) that happen
to agree only at n=28, where both reach their peak fraction of 0.4; nowhere
else should they be expected to match.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

PathLike = Union[str, Path]

#: Historical provenance is blocked (Task 1); every fixture built here carries this.
SYNTHETIC_PROVENANCE = "synthetic-test"

BETA_MODES = ("manuscript-ramp", "repository-default-ramp")

#: Basket-size bounds of the manuscript's beta ramp (portfolio-paper-source.tex,
#: eq:beta-ramp) -- deliberately the same N_MIN/N_FULL P's own ramp uses.
_MANUSCRIPT_BETA_N_MIN = 15
_MANUSCRIPT_BETA_N_FULL = 28


def make_problem(
    mu: npt.ArrayLike,
    Sigma: npt.ArrayLike,
    gamma: float,
    w_min: float,
    w_max: float,
    cardinality_k: Optional[int],
    frustration_beta: float = 0.0,
    asset_tickers: Optional[List[str]] = None,
):
    """Build a ``qpo.problem.PortfolioProblem`` from plain arrays. Imports qpo lazily."""
    from qpo.problem import PortfolioProblem  # pyright: ignore[reportMissingImports]

    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(Sigma, dtype=np.float64)
    n = mu_arr.shape[0]
    tickers = list(asset_tickers) if asset_tickers is not None else [f"A{i}" for i in range(n)]
    return PortfolioProblem(
        mu=mu_arr,
        Sigma=sigma_arr,
        gamma=float(gamma),
        w_max=float(w_max),
        w_min=float(w_min),
        asset_tickers=tickers,
        cardinality_k=int(cardinality_k) if cardinality_k is not None else None,
        frustration_beta=float(frustration_beta),
    )


def synthetic_one_factor_problem(
    n: int, k: int, seed: int, gamma: float = 5.0, w_max: float = 0.5
):
    """Deterministic one-factor PSD instance, the shared generator for tests and fixtures.

    A single shared risk factor plus idiosyncratic noise is the minimum
    structure that gives a dense, full-rank, positively-correlated Sigma --
    the same shape as P's own ``tests/conftest.py::problem_factory``. Kept
    here once so ``scripts/portfolio_replication.py`` and
    ``tests/test_portfolio_replication.py`` share one definition instead of
    two copies drifting apart. ``frustration_beta`` is left at 0; callers set
    it from :func:`manuscript_ramp_beta` or :func:`repository_default_beta`.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.5, 1.5, size=n)
    idiosyncratic = rng.uniform(0.0002, 0.0020, size=n)
    Sigma = 0.3 * 0.0004 * np.outer(loadings, loadings) + np.diag(idiosyncratic)
    mu = rng.normal(0.0, 0.0006, size=n)
    return make_problem(
        mu=mu,
        Sigma=Sigma,
        gamma=gamma,
        w_min=0.25 / k,
        w_max=w_max,
        cardinality_k=k,
        asset_tickers=[f"A{i}" for i in range(n)],
    )


def manuscript_beta_fraction(n: int) -> float:
    """The manuscript's beta(n) fraction.

    ``portfolio-paper-source.tex``, eq:beta-ramp (lines 267-275)::

        beta(n) = 0.4 * [0.3 + 0.7 * (n - 15) / 13],   15 <= n <= 28
                = 0,                                    n < 15

    The equation is defined only on ``[15, 28]``; ``n > 28`` clamps to the
    n=28 value (0.4), the manuscript's largest basket, mirroring how P's own
    repository ramp clamps above its ``N_FULL``. This is DIFFERENT from
    :func:`repository_default_beta` (different floor, 0.3 vs 0, and a
    different slope): the two happen to agree only at n=28, where both reach
    a fraction of 0.4.
    """
    if n < _MANUSCRIPT_BETA_N_MIN:
        return 0.0
    n_clamped = min(n, _MANUSCRIPT_BETA_N_FULL)
    span = _MANUSCRIPT_BETA_N_FULL - _MANUSCRIPT_BETA_N_MIN
    return 0.4 * (0.3 + 0.7 * (n_clamped - _MANUSCRIPT_BETA_N_MIN) / span)


def manuscript_ramp_beta(problem) -> float:
    """Absolute beta from the manuscript's ramp: ``manuscript_beta_fraction(N) * select_objective_scale(problem)``."""
    from qpo.encoders import select_objective_scale  # pyright: ignore[reportMissingImports]

    return manuscript_beta_fraction(problem.N) * float(select_objective_scale(problem))


def repository_default_beta(problem) -> float:
    """P's shipped ramp: ``qpo.encoders.resolve_frustration_beta(problem)``.

    Zero below ``qpo.config.CARDINALITY_BETA_N_MIN`` assets; the caller decides
    whether to apply the result to ``problem.frustration_beta``.
    """
    from qpo.encoders import resolve_frustration_beta  # pyright: ignore[reportMissingImports]

    return float(resolve_frustration_beta(problem))


def encode_to_ising(problem) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]:
    """``encode_select(problem)``, converted to Ising via dimod, offset retained.

    Returns ``(qubo, h, edges, j, offset)``: ``h`` is float64 shape ``(n,)``,
    ``edges`` is int64 shape ``(m, 2)`` with every pair ordered ``(min, max)``,
    ``j`` is float64 shape ``(m,)`` aligned with ``edges``, and ``offset`` is a
    plain float. This is the Round 2 bundle array contract
    (:mod:`quip_solver_dwave.round2_io`).
    """
    import dimod
    from qpo.encoders import encode_select  # pyright: ignore[reportMissingImports]

    qubo = encode_select(problem)
    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
    h_dict, j_dict, offset = bqm.to_ising()

    n = qubo.n
    h = np.array([float(h_dict.get(i, 0.0)) for i in range(n)], dtype=np.float64)

    pairs = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in j_dict})
    if pairs:
        edges = np.array(pairs, dtype=np.int64)
        j = np.array([_pair_value(j_dict, u, v) for u, v in pairs], dtype=np.float64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        j = np.zeros(0, dtype=np.float64)

    return qubo, h, edges, j, float(offset)


def _pair_value(j_dict: Dict[Tuple[int, int], Any], u: int, v: int) -> float:
    """dimod stores each undirected pair under exactly one of ``(u, v)``/``(v, u)``."""
    if (u, v) in j_dict:
        return float(j_dict[(u, v)])
    return float(j_dict[(v, u)])


def _git_facts(repo: Path) -> Tuple[str, bool]:
    """HEAD and dirty state of ``repo``. Raises if either git call fails; never returns None.

    "Dirty" means a TRACKED file differs from ``HEAD`` (staged or unstaged) --
    ``--untracked-files=no``, so a stray scratch file sitting in the working
    tree does not trip this. Untracked files are never part of any commit, so
    they cannot make ``d_head`` misidentify the code that wrote a bundle,
    which is the only thing this check protects.
    """
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if head.returncode != 0:
        raise RuntimeError(f"could not determine {repo}'s HEAD commit: {head.stderr.strip()}")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        raise RuntimeError(f"could not determine {repo}'s dirty state: {status.stderr.strip()}")
    return head.stdout.strip(), bool(status.stdout.strip())


def _package_versions() -> Dict[str, str]:
    """Installed version of every package the portfolio pipeline depends on. Raises if one is missing."""
    import importlib.metadata as metadata

    versions: Dict[str, str] = {}
    for name in ("numpy", "scipy", "dimod", "qpo"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"could not determine the installed version of {name}") from exc
    return versions


def export_provenance(*, allow_dirty: bool = False) -> Dict[str, Any]:
    """D and P git state, interpreter, and installed package versions.

    Recorded in every fixture bundle's manifest, the same way Task 2's model
    bundles carry ``source_commits``/``package_versions``, so a later reader
    can tell exactly what code and environment produced this fixture.

    D's tree must be clean unless ``allow_dirty=True``: a dirty D tree means
    ``d_head`` would not actually identify the code that wrote the bundle
    (mirrors ``scripts/round2_export.py``'s own dirty-tree refusal). P's tree
    is recorded but never gated -- P is pinned reference code this repo does
    not own, and a stray untracked file there (e.g. its own lockfile) is not
    this repo's problem to enforce.

    This is NOT Task 1's ``environment.json`` snapshot, which is a fixed
    historical record and is never regenerated here (controller ruling).
    """
    import qpo  # pyright: ignore[reportMissingImports]  # only to locate P's checkout

    d_root = Path(__file__).resolve().parents[1]
    d_head, d_dirty = _git_facts(d_root)
    if d_dirty and not allow_dirty:
        raise RuntimeError(
            f"{d_root} has uncommitted changes; refusing to record export provenance "
            "from a dirty tree (d_head would not identify the code that wrote this "
            "bundle). Commit or stash first, or pass allow_dirty=True."
        )
    qpo_file = qpo.__file__
    if qpo_file is None:
        raise RuntimeError("qpo module has no __file__; cannot locate P's checkout")
    p_root = Path(qpo_file).resolve().parents[2]
    p_head, p_dirty = _git_facts(p_root)

    return {
        "d_head": d_head,
        "d_dirty": d_dirty,
        "p_head": p_head,
        "p_dirty": p_dirty,
        "interpreter": {"executable": sys.executable, "version": sys.version},
        "package_versions": _package_versions(),
    }


def portfolio_fixture_manifest(
    problem,
    beta_mode: str,
    provenance: str = SYNTHETIC_PROVENANCE,
) -> Dict[str, Any]:
    """Manifest fields for a portfolio fixture bundle.

    ``diversification_beta`` records the full derivation, not just the
    absolute value actually used by encoding and objective scoring
    (``effective``): which ramp produced it (``mode``), the fraction that
    ramp returned for this basket size (``fraction``), the per-problem scale
    it was multiplied against (``objective_scale``), and the ramp's identity
    (``formula``). ``effective`` is cross-checked against an independent
    recomputation from ``beta_mode`` (raises if ``problem.frustration_beta``
    was set some other way), so the recorded value is provably what
    :func:`manuscript_ramp_beta`/:func:`repository_default_beta` would
    produce, not merely whatever the caller happened to set. This is the
    select-encoding DIVERSIFICATION term
    (``PortfolioProblem.frustration_beta``), never an annealing/simulated-
    annealing inverse temperature -- ``note`` says so explicitly.
    """
    if beta_mode not in BETA_MODES:
        raise ValueError(f"beta_mode must be one of {BETA_MODES}, got {beta_mode!r}")

    from qpo.encoders import select_objective_scale  # pyright: ignore[reportMissingImports]

    scale = float(select_objective_scale(problem))
    if beta_mode == "manuscript-ramp":
        fraction = manuscript_beta_fraction(problem.N)
        formula = (
            "beta(n) = 0.4*[0.3 + 0.7*(n-15)/13] for 15<=n<=28, else 0 "
            "(portfolio-paper-source.tex, eq:beta-ramp, lines 267-275)"
        )
        expected = fraction * scale
    else:
        expected = repository_default_beta(problem)
        fraction = expected / scale if scale else 0.0
        formula = (
            "qpo.encoders.resolve_frustration_beta: fraction ramps from "
            "CARDINALITY_BETA_FLOOR=0.3 at CARDINALITY_BETA_N_MIN=15 to "
            "CARDINALITY_FRUSTRATION_BETA=0.4 at CARDINALITY_BETA_N_FULL=28, "
            "else 0 below N_MIN (qpo/config.py:265-283)"
        )

    if not np.isclose(problem.frustration_beta, expected, atol=1e-12, rtol=1e-10):
        raise ValueError(
            f"problem.frustration_beta ({problem.frustration_beta!r}) does not match "
            f"beta_mode {beta_mode!r}'s derivation ({expected!r}); set it from "
            "manuscript_ramp_beta()/repository_default_beta() before building the fixture"
        )

    return {
        "provenance": provenance,
        "n_assets": int(problem.N),
        "mu": problem.mu.tolist(),
        "sigma": problem.Sigma.tolist(),
        "gamma": float(problem.gamma),
        "cardinality_k": problem.cardinality_k,
        "w_min": float(problem.w_min),
        "w_max": float(problem.w_max),
        "asset_tickers": list(problem.asset_tickers),
        "diversification_beta": {
            "effective": float(problem.frustration_beta),
            "mode": beta_mode,
            "fraction": float(fraction),
            "objective_scale": scale,
            "formula": formula,
            "note": (
                "select-encoding diversification term (PortfolioProblem.frustration_beta); "
                "NOT an annealing/simulated-annealing inverse temperature"
            ),
        },
        "encoding_scheme": "select",
    }


def write_portfolio_fixture(
    path: PathLike,
    problem,
    beta_mode: str,
    provenance: str = SYNTHETIC_PROVENANCE,
    allow_dirty: bool = False,
) -> str:
    """Encode ``problem`` to Ising and write it as a Round 2 bundle.

    Uses :func:`quip_solver_dwave.round2_io.write_bundle`, so the on-disk
    contract (``h``/``edges``/``j``/``offset`` arrays, hashed manifest) is
    identical to the Task 2 model bundles; only the manifest's portfolio
    fields (mu, Sigma, K, gamma, bounds, diversification beta, export
    provenance) are specific to this fixture. ``allow_dirty`` is forwarded to
    :func:`export_provenance`; leave it ``False`` for a real deliverable
    export, set it ``True`` only for a throwaway ``tmp_path`` fixture built
    while the working tree may legitimately be mid-edit.
    """
    from quip_solver_dwave.round2_io import write_bundle

    qubo, h, edges, j, offset = encode_to_ising(problem)
    manifest = portfolio_fixture_manifest(problem, beta_mode=beta_mode, provenance=provenance)
    manifest["offset"] = offset
    manifest["export_provenance"] = export_provenance(allow_dirty=allow_dirty)
    arrays = {"h": h, "edges": edges, "j": j, "offset": np.asarray(offset, dtype=np.float64)}
    return write_bundle(path, manifest, arrays)


def load_problem(manifest: Dict[str, Any], arrays: Dict[str, np.ndarray]):
    """Reconstruct a ``qpo.problem.PortfolioProblem`` from a written fixture bundle.

    Reads ``mu``, ``sigma``, ``gamma``, ``cardinality_k``, ``w_min``,
    ``w_max``, ``asset_tickers`` and the recorded effective diversification
    beta from ``manifest`` -- everything :func:`portfolio_fixture_manifest`
    wrote. ``arrays`` (the bundle's ``h``/``edges``/``j``/``offset``, as
    :func:`quip_solver_dwave.round2_io.read_bundle` returns them) is accepted
    for signature symmetry with that return shape but is not needed to
    rebuild the problem: mu/Sigma live in the manifest, not in the Ising
    arrays, which encode the QUBO derived FROM the problem, not the problem
    itself.
    """
    del arrays  # not needed for reconstruction; see docstring
    return make_problem(
        mu=manifest["mu"],
        Sigma=manifest["sigma"],
        gamma=manifest["gamma"],
        w_min=manifest["w_min"],
        w_max=manifest["w_max"],
        cardinality_k=manifest["cardinality_k"],
        frustration_beta=manifest["diversification_beta"]["effective"],
        asset_tickers=manifest["asset_tickers"],
    )


def _bits_from_spins(spins: Any, n_vars: Optional[int] = None) -> np.ndarray:
    """Validate and convert +-1 spin reads to {0,1} bits.

    Accepts a single read, shape ``(n_vars,)``, or many, shape
    ``(n_reads, n_vars)``. Raises on empty, nonfinite, or non-+-1 input: this
    is the "validate nonempty finite input before scoring" gate from the task
    brief, run before anything is handed to P.
    """
    arr = np.asarray(spins, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("spins must be a nonempty (n_reads, n_vars) array")
    if not np.isfinite(arr).all():
        raise ValueError("spins must be finite")
    if not np.isin(arr, (-1.0, 1.0)).all():
        raise ValueError("spins must be +-1 valued")
    if n_vars is not None and arr.shape[1] != n_vars:
        raise ValueError(f"spins carry {arr.shape[1]} variables, expected {n_vars}")
    return ((arr.astype(np.int16) + 1) // 2).astype(np.int8)


def score_reads(problem, qubo, spins: Any) -> Dict[str, Any]:
    """Reproduce P's scoring adapter for one batch of raw reads.

    Matches the task brief's ``score_reads`` sketch against P's real API:
    ``select_solution`` returns ``(weights, bits)`` where ``bits`` is the RAW
    read that produced the best final portfolio, not a post-repair support
    indicator (P repairs and weights every read internally; only the winning
    read's original bits come back -- the returned ``selected_bits`` here is
    that same raw winning read, never the repaired support). See
    :func:`per_read_diagnostics` for the actual post-repair support of every
    read. ``check_feasibility`` takes ``(weights, w_max, w_min, ...,
    cardinality_k=...)`` positionally, matching the sketch.

    P's own feasibility gate does not reject NaN (``nan > eps`` is False in
    NumPy, so a NaN budget residual silently reads as "ok"). This adapter
    overrides the gate to infeasible whenever the resulting weights or
    objective are nonfinite, per the task brief's explicit requirement, even
    though P's gate alone would have called it feasible.
    """
    import dimod
    from qpo.solvers.feasibility import check_feasibility  # pyright: ignore[reportMissingImports]
    from qpo.solvers.sampling import select_solution  # pyright: ignore[reportMissingImports]

    bits = _bits_from_spins(spins, n_vars=qubo.n)

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
    response = dimod.SampleSet.from_samples_bqm((bits, list(range(qubo.n))), bqm)
    weights, selected_bits = select_solution(response, qubo, problem)
    objective = problem.objective(weights)
    gate = check_feasibility(
        weights,
        problem.w_max,
        problem.w_min,
        cardinality_k=problem.cardinality_k,
    )
    finite = bool(np.isfinite(weights).all() and np.isfinite(objective))
    raw_feasible_count = (
        int(np.count_nonzero(bits.sum(axis=1) == problem.cardinality_k))
        if problem.cardinality_k is not None
        else None
    )
    return {
        "weights": weights,
        "selected_bits": selected_bits,
        "objective": objective,
        "feasible": bool(gate.feasible) and finite,
        "raw_feasible_count": raw_feasible_count,
        "returned_reads": int(bits.shape[0]),
    }


def per_read_diagnostics(problem, spins: Any) -> Dict[str, Any]:
    """Per-read repair/weighting diagnostics, cardinality (select-encoding) problems only.

    Calls the same reference ``qpo.projection.greedy_project`` and
    ``qpo.projection.weights_for_support`` every provider's finish uses. For
    every read this records the pre-repair support and cardinality (K before
    repair), the post-repair support and cardinality (always exactly K after
    repair), what the repair added/removed, the FULL valid weights vector
    ``weights_for_support`` returned (not just its objective), the final
    objective, and post-repair feasibility -- each read scored independently
    of whichever single read :func:`score_reads` picks as the winner (whose
    ``selected_bits`` is that raw winning read, never a repaired support; see
    that function's docstring).

    ``weighting_failed`` is a tri-state, not a heuristic guess:

    * ``True`` -- ``weights_for_support`` raised, or returned nonfinite
      weights/objective. Both are real, observed failures of P's reference
      weighting step (verified directly: inverted bounds, ``w_min > w_max``
      set on the problem after construction bypasses P's own
      ``validate()``, make ``qpo.weighting.optimal_weights`` raise
      ``ValueError`` from scipy).
    * ``None`` -- no failure was observed, but P's own fallback path (an
      exact equal-weight vector substituted when SLSQP's raw result violates
      the budget or box) exposes no flag, and an exact equal-weight result
      can equally be a genuine optimum. Absent an instrumented signal this is
      honestly unknown, not asserted either way.

    A per-support multiplicity count (``support_multiplicities``) is also
    returned, since P's own per-support caching (``qpo.solvers.sampling``)
    means many reads can repair to the identical support.
    """
    from qpo.projection import greedy_project, weights_for_support  # pyright: ignore[reportMissingImports]
    from qpo.solvers.feasibility import check_feasibility  # pyright: ignore[reportMissingImports]

    if problem.cardinality_k is None:
        raise ValueError("per_read_diagnostics requires a cardinality problem")

    bits = _bits_from_spins(spins, n_vars=problem.N)
    n = problem.N
    rho = problem.rho if problem.frustration_beta else None

    records: List[Dict[str, Any]] = []
    multiplicities: Dict[Tuple[int, ...], int] = {}
    for row in bits:
        raw_support = frozenset(int(i) for i in np.nonzero(row[:n])[0])
        support = greedy_project(row, problem, rho)
        multiplicities[support] = multiplicities.get(support, 0) + 1

        weighting_error: Optional[str] = None
        try:
            weights = weights_for_support(support, problem)
        except Exception as exc:  # a real, observed failure of P's reference weighting
            weighting_error = repr(exc)
            weights = np.full(n, np.nan)

        objective = problem.objective(weights)
        finite = bool(np.isfinite(weights).all() and np.isfinite(objective))
        if weighting_error is not None or not finite:
            weighting_failed: Optional[bool] = True
        else:
            # P's own equal-weight fallback fires silently on the inside of
            # weights_for_support with no externally observable flag, so
            # "no failure observed" cannot be promoted to "no failure
            # happened" -- recorded as unknown rather than guessed.
            weighting_failed = None

        if weighting_error is None:
            gate = check_feasibility(
                weights, problem.w_max, problem.w_min, cardinality_k=problem.cardinality_k
            )
            post_repair_feasible = bool(gate.feasible) and finite
            post_repair_reason = gate.reason if finite else "nonfinite weights or objective"
        else:
            post_repair_feasible = False
            post_repair_reason = f"weighting raised: {weighting_error}"

        support_set = frozenset(support)
        records.append(
            {
                "raw_support": sorted(raw_support),
                "raw_cardinality": len(raw_support),
                "repaired_support": list(support),
                "repaired_cardinality": len(support),
                "added": sorted(support_set - raw_support),
                "removed": sorted(raw_support - support_set),
                "weights": weights,
                "objective": objective,
                "post_repair_feasible": post_repair_feasible,
                "post_repair_reason": post_repair_reason,
                "weighting_failed": weighting_failed,
                "weighting_error": weighting_error,
            }
        )

    return {"records": records, "support_multiplicities": dict(multiplicities)}
