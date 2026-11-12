"""Tests for the offline reproduction of P (qportfolio)'s scoring pipeline.

Task 3 of docs/superpowers/plans/2026-09-22-regime-search-round2.md: this is
the shared scoring adapter every portfolio solver arm (QPU, dwave-neal,
scalar SA, MSA) will pass its raw reads through. Every test here calls the
pinned P implementation directly (``qpo``, ``dimod``), so the whole module
skips cleanly when P is not on the path -- P lives only in its own pinned
environment (see the task-3 report for the exact invocation).

Tolerance: coefficients and objectives are compared at 1e-12 absolute /
1e-10 relative, the software tolerance the task brief declares for this
comparison.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from quip_solver_dwave import portfolio_replication as pr
from quip_solver_dwave import round2_io

if TYPE_CHECKING:  # pyright needs a real class, not a pytest.importorskip() variable, for annotations
    from qpo.problem import PortfolioProblem  # pyright: ignore[reportMissingImports]

qpo = pytest.importorskip("qpo")
dimod = pytest.importorskip("dimod")

encode_select = qpo.encoders.encode_select
resolve_frustration_beta = qpo.encoders.resolve_frustration_beta
select_objective_scale = qpo.encoders.select_objective_scale

ABS_TOL = 1e-12
REL_TOL = 1e-10


def assert_close(actual, expected, *, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    np.testing.assert_allclose(actual, expected, atol=abs_tol, rtol=rel_tol)


# --------------------------------------------------------------------- fixtures


def four_asset_problem(beta: float = 0.0) -> PortfolioProblem:
    """4-asset cardinality problem, small enough to enumerate exhaustively."""
    mu = np.array([0.02, 0.015, 0.01, 0.005])
    Sigma = np.array(
        [
            [0.04, 0.02, 0.01, 0.005],
            [0.02, 0.03, 0.015, 0.005],
            [0.01, 0.015, 0.02, 0.002],
            [0.005, 0.005, 0.002, 0.01],
        ]
    )
    return pr.make_problem(
        mu=mu,
        Sigma=Sigma,
        gamma=3.0,
        w_min=0.1,
        w_max=0.7,
        cardinality_k=2,
        frustration_beta=beta,
        asset_tickers=["A", "B", "C", "D"],
    )


# -------------------------------------------------------- step 2: exhaustive 16-state check


class TestExhaustiveSmallStates:
    """4-asset problem: 2**4 = 16 states, checked against direct binary energy."""

    @pytest.mark.parametrize("beta_fraction", [0.0, 0.5])
    def test_qubo_dict_matches_direct_binary_energy(self, beta_fraction):
        problem = four_asset_problem()
        if beta_fraction:
            problem.frustration_beta = beta_fraction * select_objective_scale(problem)
            problem.invalidate_cache()
        qubo = encode_select(problem)
        qdict = qubo.to_dict()

        for bits_tuple in itertools.product((0, 1), repeat=4):
            x = np.array(bits_tuple, dtype=float)
            direct = float(x @ qubo.Q @ x)
            from_dict = sum(
                coeff * (x[i] if i == j else x[i] * x[j]) for (i, j), coeff in qdict.items()
            )
            assert_close(from_dict, direct)

    @pytest.mark.parametrize("beta_fraction", [0.0, 0.5])
    def test_ising_conversion_retains_offset_and_energy(self, beta_fraction):
        problem = four_asset_problem()
        if beta_fraction:
            problem.frustration_beta = beta_fraction * select_objective_scale(problem)
            problem.invalidate_cache()
        qubo, h, edges, j, offset = pr.encode_to_ising(problem)

        # Independently recompute the Ising conversion via dimod to check our
        # (h, edges, j, offset) extraction round-trips, then compare Ising
        # energy against the direct QUBO energy for every one of the 16 states.
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())
        for bits_tuple in itertools.product((0, 1), repeat=4):
            bits = np.array(bits_tuple, dtype=float)
            qubo_energy = float(bits @ qubo.Q @ bits)

            spins = 2.0 * bits - 1.0
            ising_energy = float(h @ spins) + offset
            for idx, (u, v) in enumerate(edges):
                ising_energy += j[idx] * spins[u] * spins[v]
            assert_close(ising_energy, qubo_energy)

            dimod_energy = bqm.energy({i: int(b) for i, b in enumerate(bits_tuple)})
            assert_close(qubo_energy, dimod_energy)


# ------------------------------------------------------------- score_reads adapter


class TestScoreReads:
    def test_rejects_empty_response(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        with pytest.raises(ValueError):
            pr.score_reads(problem, qubo, np.zeros((0, qubo.n)))

    def test_rejects_nonfinite_input(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array([[1.0, -1.0, np.nan, 1.0]])
        with pytest.raises(ValueError):
            pr.score_reads(problem, qubo, spins)

    def test_rejects_non_pm1_input(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array([[1.0, 0.0, -1.0, 1.0]])
        with pytest.raises(ValueError):
            pr.score_reads(problem, qubo, spins)

    def test_accepts_a_single_1d_read(self):
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array([-1.0, -1.0, -1.0, -1.0])
        result = pr.score_reads(problem, qubo, spins)
        assert result["returned_reads"] == 1

    def test_minimum_raw_energy_read_can_differ_from_best_final_portfolio(self):
        """[1,1,0,0] has lower raw select-QUBO energy than [0,1,1,1], but its
        post-repair portfolio is worse: this is P's documented select-encoding
        caveat (energy is the equal-weight surrogate, not the true objective)."""
        problem = four_asset_problem(beta=0.0)
        problem.frustration_beta = 0.5 * select_objective_scale(problem)
        problem.invalidate_cache()
        qubo = encode_select(problem)
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo.to_dict())

        low_energy_worse = (1, 1, 0, 0)
        high_energy_better = (0, 1, 1, 1)
        e_low = bqm.energy({i: b for i, b in enumerate(low_energy_worse)})
        e_high = bqm.energy({i: b for i, b in enumerate(high_energy_better)})
        assert e_low < e_high  # confirms the raw-energy ordering used below

        spins = np.array(
            [
                [2 * b - 1 for b in low_energy_worse],
                [2 * b - 1 for b in high_energy_better],
            ],
            dtype=float,
        )
        result = pr.score_reads(problem, qubo, spins)
        selected_bits = tuple(int((s + 1) // 2) for s in result["selected_bits"])
        assert selected_bits == high_energy_better
        assert selected_bits != low_energy_worse

    def test_raw_feasible_count_before_repair(self):
        """raw_feasible_count checks K on the RAW bits, before any repair."""
        problem = four_asset_problem()
        qubo = encode_select(problem)
        spins = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],  # raw K=0
                [1.0, 1.0, -1.0, -1.0],  # raw K=2 == cardinality_k
                [1.0, 1.0, 1.0, 1.0],  # raw K=4
            ]
        )
        result = pr.score_reads(problem, qubo, spins)
        assert result["raw_feasible_count"] == 1

    def test_final_score_is_identical_across_hypothetical_adapters(self):
        """Same raw reads through 'QPU', 'neal', 'scalar', 'MSA' must score identically:
        score_reads is a pure function of (problem, qubo, spins)."""
        problem = four_asset_problem()
        problem.frustration_beta = 0.3 * select_objective_scale(problem)
        problem.invalidate_cache()
        qubo = encode_select(problem)
        rng = np.random.default_rng(0)
        spins = rng.choice([-1.0, 1.0], size=(12, qubo.n))

        adapters = ["qpu", "neal", "scalar", "msa"]
        results = [pr.score_reads(problem, qubo, spins) for _ in adapters]
        base = results[0]
        for other in results[1:]:
            assert_close(other["weights"], base["weights"])
            assert_close(other["objective"], base["objective"])
            assert other["feasible"] == base["feasible"]

    def test_post_repair_feasibility_holds_even_when_gate_alone_would_pass_nan(self):
        """P's own check_feasibility does not reject NaN (nan > eps is False in
        NumPy). A NaN objective must still flip score_reads' feasible flag to
        False even though P's gate reports ok. The QUBO is built from a clean
        problem first (encode_select itself rejects a NaN coefficient via its
        symmetry check), then mu is corrupted in place afterwards so only the
        objective step, not the encoding, sees the NaN."""
        problem = four_asset_problem()
        qubo = encode_select(problem)
        problem.mu[1] = np.nan
        spins = np.array([-1.0, -1.0, 1.0, 1.0])
        result = pr.score_reads(problem, qubo, spins)
        assert not np.isfinite(result["objective"])
        assert result["feasible"] is False


# --------------------------------------------------------- per-read diagnostics


class TestPerReadDiagnostics:
    def test_requires_a_cardinality_problem(self):
        problem = pr.make_problem(
            mu=[0.01, 0.02],
            Sigma=[[1e-4, 0.0], [0.0, 1e-4]],
            gamma=1.0,
            w_min=0.0,
            w_max=1.0,
            cardinality_k=None,
        )
        with pytest.raises(ValueError):
            pr.per_read_diagnostics(problem, np.array([-1.0, 1.0]))

    def test_rejects_spin_width_mismatch(self):
        """Same width check score_reads applies, against problem.N."""
        problem = four_asset_problem()
        with pytest.raises(ValueError):
            pr.per_read_diagnostics(problem, np.array([-1.0, 1.0, -1.0]))  # 3, not 4

    def test_k_before_and_after_repair(self):
        problem = four_asset_problem()
        spins = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],  # raw K=0, must be repaired up to K=2
                [1.0, 1.0, 1.0, -1.0],  # raw K=3, must be repaired down to K=2
                [1.0, 1.0, -1.0, -1.0],  # raw K=2 already
            ]
        )
        diag = pr.per_read_diagnostics(problem, spins)
        records = diag["records"]
        assert [r["raw_cardinality"] for r in records] == [0, 3, 2]
        assert [r["repaired_cardinality"] for r in records] == [2, 2, 2]
        assert all(len(r["repaired_support"]) == problem.cardinality_k for r in records)
        assert len(records[0]["added"]) == 2 and not records[0]["removed"]
        assert len(records[1]["removed"]) == 1 and not records[1]["added"]
        assert not records[2]["added"] and not records[2]["removed"]

    def test_records_the_full_valid_weights_and_final_objective(self):
        """weights_for_support's actual weight vector is kept, not dropped down
        to just its objective -- it is the only place a caller can see WHERE
        the weight went, not just how good it was."""
        problem = four_asset_problem()
        spins = np.array([1.0, 1.0, -1.0, -1.0])  # already exactly K=2
        diag = pr.per_read_diagnostics(problem, spins)
        record = diag["records"][0]
        assert record["weights"].shape == (problem.N,)
        assert np.isclose(record["weights"].sum(), 1.0)
        for i in record["repaired_support"]:
            assert record["weights"][i] > 0.0
        assert record["objective"] == pytest.approx(problem.objective(record["weights"]))

    def test_support_multiplicities_count_repaired_supports(self):
        problem = four_asset_problem()
        spins = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0, -1.0],
            ]
        )
        diag = pr.per_read_diagnostics(problem, spins)
        total = sum(diag["support_multiplicities"].values())
        assert total == len(spins)
        for support in diag["support_multiplicities"]:
            assert len(support) == problem.cardinality_k

    def test_post_repair_feasibility_is_recorded(self):
        problem = four_asset_problem()
        spins = np.array([1.0, 1.0, -1.0, -1.0])
        diag = pr.per_read_diagnostics(problem, spins)
        assert diag["records"][0]["post_repair_feasible"] is True
        assert diag["records"][0]["weighting_failed"] is None  # no failure observed
        assert diag["records"][0]["weighting_error"] is None

    def test_weighting_failure_is_flagged_on_a_real_raised_result(self):
        """P's qpo.weighting.optimal_weights passes its bounds straight to
        scipy.optimize.minimize; an inverted (w_min > w_max) bound makes scipy
        itself raise ValueError. This is a genuine failure of P's reference
        weighting step (verified directly against qpo.weighting.optimal_weights
        beforehand), reached by mutating the problem after construction
        (bypassing P's own validate(), the same way a NaN mu is used elsewhere
        in this file) -- not a monkeypatched stub standing in for P."""
        problem = four_asset_problem()
        problem.w_min, problem.w_max = 0.9, 0.1  # inverted: scipy raises on this
        spins = np.array([1.0, 1.0, -1.0, -1.0])
        diag = pr.per_read_diagnostics(problem, spins)
        record = diag["records"][0]
        assert record["weighting_failed"] is True
        assert record["weighting_error"] is not None
        assert not record["post_repair_feasible"]

    def test_weighting_failure_is_flagged_on_nonfinite_output(self):
        """A NaN mu (mutated in after construction, as elsewhere in this file)
        makes the resulting objective nonfinite; that is also a real,
        observed failure signal, distinct from an exception."""
        problem = four_asset_problem()
        problem.mu[0] = np.nan
        spins = np.array([1.0, 1.0, -1.0, -1.0])
        diag = pr.per_read_diagnostics(problem, spins)
        record = diag["records"][0]
        assert record["weighting_failed"] is True
        assert record["weighting_error"] is None  # no exception, just a NaN result
        assert not record["post_repair_feasible"]


# --------------------------------------------------------------------- beta handling


class TestBeta:
    def test_zero_beta_matches_encoding_and_objective(self):
        problem = four_asset_problem(beta=0.0)
        assert problem.frustration_beta == 0.0
        qubo = encode_select(problem)
        # No frustration coupling: off-diagonal is exactly (gamma/K^2) Sigma_ij.
        k, gamma = problem.cardinality_k, problem.gamma
        coef = gamma / (2.0 * k * k)
        expected_offdiag = coef * problem.Sigma[0, 1]
        assert_close(qubo.Q[0, 1], expected_offdiag)

        weights = np.array([0.6, 0.0, 0.4, 0.0])
        objective = problem.objective(weights)
        expected = 0.5 * gamma * weights @ problem.Sigma @ weights - problem.mu @ weights
        assert_close(objective, expected)

    def test_positive_beta_is_used_by_both_encoding_and_objective(self):
        problem = four_asset_problem()
        beta = 0.4 * select_objective_scale(problem)
        problem.frustration_beta = beta
        problem.invalidate_cache()
        qubo = encode_select(problem)

        # Encoding: Q[i, j] and Q[j, i] each hold coef*Sigma_ij already (full
        # symmetric matrix, not a triangle), plus half of beta*rho_ij; to_dict's
        # pair coefficient is their sum, i.e. 2*coef*Sigma_ij + beta*rho_ij.
        k, gamma = problem.cardinality_k, problem.gamma
        coef = gamma / (2.0 * k * k)
        expected_offdiag = 2.0 * coef * problem.Sigma[0, 1] + beta * problem.rho[0, 1]
        assert_close(qubo.Q[0, 1] + qubo.Q[1, 0], expected_offdiag)

        # Objective: the same recorded beta drives the held-pair penalty.
        weights = np.array([0.6, 0.0, 0.4, 0.0])
        held = (weights > 1e-9).astype(float)
        pair_corr = 0.5 * (held @ problem.rho @ held - held.sum())
        expected = (
            0.5 * gamma * weights @ problem.Sigma @ weights
            - problem.mu @ weights
            + beta * pair_corr
        )
        assert_close(problem.objective(weights), expected)

    @pytest.mark.parametrize(
        "n,k,expected_manuscript_fraction,expected_repository_fraction",
        [
            (4, 2, 0.0, 0.0),  # below both ramps' N_MIN=15: both are 0
            (18, 6, 0.4 * (0.3 + 0.7 * 3 / 13), 0.3 + 0.1 * 3 / 13),  # differ mid-ramp
            (28, 9, 0.4, 0.4),  # both reach their peak fraction at N_FULL=28
        ],
    )
    def test_each_ramp_matches_its_own_formula(
        self, n, k, expected_manuscript_fraction, expected_repository_fraction
    ):
        """The manuscript ramp (portfolio-paper-source.tex eq:beta-ramp) and the
        repository ramp (qpo.encoders.resolve_frustration_beta) are DIFFERENT
        formulas (different floor, different slope) that happen to agree only
        at n=28. They must not be asserted to differ everywhere -- only to
        each match their own definition."""
        problem = pr.synthetic_one_factor_problem(n, k, seed=n)
        assert_close(pr.manuscript_beta_fraction(n), expected_manuscript_fraction)

        scale = select_objective_scale(problem)
        repository_fraction = resolve_frustration_beta(problem) / scale if scale else 0.0
        assert_close(repository_fraction, expected_repository_fraction)

    def test_manuscript_ramp_and_repository_default_ramp_fixtures_stay_separate(self, tmp_path):
        # N=18 sits strictly between both ramps' N_MIN and N_FULL, where the
        # two formulas give different fractions (see the parametrized test
        # above), so their manifests are a real, not coincidental, contrast.
        n, k = 18, 6
        manuscript_problem = pr.synthetic_one_factor_problem(n, k, seed=1)
        manuscript_problem.frustration_beta = pr.manuscript_ramp_beta(manuscript_problem)
        manuscript_problem.invalidate_cache()

        default_problem = pr.synthetic_one_factor_problem(n, k, seed=1)
        default_problem.frustration_beta = pr.repository_default_beta(default_problem)
        default_problem.invalidate_cache()

        assert default_problem.frustration_beta > 0.0
        assert manuscript_problem.frustration_beta != default_problem.frustration_beta

        manuscript_path = tmp_path / "manuscript-ramp"
        default_path = tmp_path / "repository-default-ramp"
        # allow_dirty: this is a throwaway tmp_path fixture testing manifest
        # structure, not the deliverable export (scripts/portfolio_replication.py
        # requires a clean tree by default; see TestExportProvenance below).
        pr.write_portfolio_fixture(
            manuscript_path, manuscript_problem, beta_mode="manuscript-ramp", allow_dirty=True
        )
        pr.write_portfolio_fixture(
            default_path, default_problem, beta_mode="repository-default-ramp", allow_dirty=True
        )

        manuscript_manifest, _ = round2_io.read_bundle(manuscript_path)
        default_manifest, _ = round2_io.read_bundle(default_path)
        assert manuscript_manifest["diversification_beta"]["mode"] == "manuscript-ramp"
        assert default_manifest["diversification_beta"]["mode"] == "repository-default-ramp"
        assert manuscript_manifest["diversification_beta"]["effective"] == pytest.approx(
            manuscript_problem.frustration_beta
        )
        assert default_manifest["diversification_beta"]["effective"] == pytest.approx(
            default_problem.frustration_beta
        )
        assert (
            manuscript_manifest["diversification_beta"]["effective"]
            != default_manifest["diversification_beta"]["effective"]
        )
        for manifest in (manuscript_manifest, default_manifest):
            beta = manifest["diversification_beta"]
            assert set(beta) >= {"effective", "mode", "fraction", "objective_scale", "formula", "note"}
            assert "annealing" in beta["note"] or "inverse temperature" in beta["note"]

    def test_manifest_rejects_a_beta_that_does_not_match_its_declared_mode(self):
        problem = pr.synthetic_one_factor_problem(18, 6, seed=1)
        problem.frustration_beta = 0.999  # matches neither ramp at N=18
        with pytest.raises(ValueError):
            pr.portfolio_fixture_manifest(problem, beta_mode="repository-default-ramp")


# ------------------------------------------------------------- fixture round-trip


class TestFixtureBundles:
    @pytest.mark.parametrize("n,k", [(4, 2), (18, 6), (28, 9)])
    def test_fixture_round_trips_through_write_bundle_and_matches_direct_energy(
        self, tmp_path, n, k
    ):
        problem = pr.synthetic_one_factor_problem(n, k, seed=n)
        problem.frustration_beta = pr.repository_default_beta(problem)
        problem.invalidate_cache()

        path = tmp_path / f"n{n}"
        # allow_dirty: a throwaway tmp_path fixture, not the deliverable export.
        pr.write_portfolio_fixture(
            path, problem, beta_mode="repository-default-ramp", allow_dirty=True
        )
        manifest, arrays = round2_io.read_bundle(path)

        assert manifest["provenance"] == "synthetic-test"
        assert manifest["n_assets"] == n
        assert manifest["cardinality_k"] == k

        # Rebuild the BQM from the bundle's h/edges/j/offset and check its
        # energy against encode_select's own Q for a batch of random reads.
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {i: float(v) for i, v in enumerate(arrays["h"])},
            {
                (int(u), int(v)): float(w)
                for (u, v), w in zip(arrays["edges"], arrays["j"])
            },
            float(arrays["offset"]),
        )
        qubo = encode_select(problem)
        rng = np.random.default_rng(n)
        for _ in range(5):
            bits = rng.integers(0, 2, size=n)
            spins = {i: (2 * int(b) - 1) for i, b in enumerate(bits)}
            ising_energy = bqm.energy(spins)
            qubo_energy = float(bits.astype(float) @ qubo.Q @ bits.astype(float))
            assert_close(ising_energy, qubo_energy)

    @pytest.mark.parametrize("n,k", [(4, 2), (18, 6), (28, 9)])
    def test_score_reads_on_the_written_fixture(self, tmp_path, n, k):
        problem = pr.synthetic_one_factor_problem(n, k, seed=100 + n)
        problem.frustration_beta = pr.repository_default_beta(problem)
        problem.invalidate_cache()

        path = tmp_path / f"n{n}-score"
        pr.write_portfolio_fixture(
            path, problem, beta_mode="repository-default-ramp", allow_dirty=True
        )
        round2_io.read_bundle(path)  # exercises the hash/round-trip contract

        qubo = encode_select(problem)
        rng = np.random.default_rng(n + 1)
        spins = rng.choice([-1.0, 1.0], size=(8, n))
        result = pr.score_reads(problem, qubo, spins)
        assert result["returned_reads"] == 8
        assert np.isfinite(result["weights"]).all()
        assert np.isfinite(result["objective"])
        assert result["weights"].shape == (n,)


# ------------------------------------------------------------------- load_problem


class TestLoadProblem:
    @pytest.mark.parametrize("n,k", [(4, 2), (18, 6), (28, 9)])
    def test_round_trips_into_a_problem_with_a_matching_objective(self, tmp_path, n, k):
        original = pr.synthetic_one_factor_problem(n, k, seed=n)
        original.frustration_beta = pr.repository_default_beta(original)
        original.invalidate_cache()

        path = tmp_path / f"n{n}-load"
        pr.write_portfolio_fixture(
            path, original, beta_mode="repository-default-ramp", allow_dirty=True
        )
        manifest, arrays = round2_io.read_bundle(path)
        reloaded = pr.load_problem(manifest, arrays)

        assert reloaded.N == original.N
        assert reloaded.cardinality_k == original.cardinality_k
        assert reloaded.frustration_beta == pytest.approx(original.frustration_beta)

        rng = np.random.default_rng(n + 7)
        for _ in range(4):
            support = rng.choice(n, size=k, replace=False)
            weights = np.zeros(n)
            weights[support] = 1.0 / k
            assert_close(reloaded.objective(weights), original.objective(weights))


# --------------------------------------------------------------- export provenance


class TestExportProvenance:
    def test_records_d_head_p_head_interpreter_and_package_versions(self):
        provenance = pr.export_provenance(allow_dirty=True)
        assert len(provenance["d_head"]) == 40  # a git SHA
        assert len(provenance["p_head"]) == 40
        assert provenance["interpreter"]["executable"]
        for name in ("numpy", "scipy", "dimod", "qpo"):
            assert provenance["package_versions"][name]

    def test_written_fixture_manifest_carries_export_provenance(self, tmp_path):
        problem = four_asset_problem()
        path = tmp_path / "provenance"
        pr.write_portfolio_fixture(
            path, problem, beta_mode="repository-default-ramp", allow_dirty=True
        )
        manifest, _ = round2_io.read_bundle(path)
        assert "export_provenance" in manifest
        assert manifest["export_provenance"]["d_dirty"] in (True, False)
        assert len(manifest["export_provenance"]["p_head"]) == 40

    def test_refuses_a_dirty_d_tree_unless_allowed(self, monkeypatch):
        monkeypatch.setattr(pr, "_git_facts", lambda repo: ("deadbeef", True))
        with pytest.raises(RuntimeError):
            pr.export_provenance(allow_dirty=False)
        # allow_dirty=True still succeeds against the same (patched) dirty state.
        pr.export_provenance(allow_dirty=True)
