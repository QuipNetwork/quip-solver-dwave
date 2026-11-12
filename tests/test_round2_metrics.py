"""Tests for round2_metrics: pairwise quality, feasibility, arm summaries, and bootstrap gaps.

Ported from quip-miner-cpu ``research/round2/tests/test_round2_report.py``. The
arm-summary tests use this repository's ``round2-cpu-run-v1`` records.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from quip_solver_dwave import round2_metrics as metrics


def test_strict_and_material_quality_are_different():
    # The task brief's own step-1 test, verbatim.
    assert metrics.quality_outcome(-1.001, -1.0, 0.0) == "qpu"
    assert metrics.quality_outcome(-1.001, -1.0, 0.005) == "tie"


def test_quality_outcome_rejects_nonfinite_or_negative_tolerance():
    with pytest.raises(ValueError):
        metrics.quality_outcome(float("nan"), -1.0, 0.0)
    with pytest.raises(ValueError):
        metrics.quality_outcome(-1.0, -1.0, -0.1)


def test_quality_outcome_picks_the_lower_energy_as_the_winner():
    assert metrics.quality_outcome(-2.0, -1.0, 0.0) == "qpu"
    assert metrics.quality_outcome(-1.0, -2.0, 0.0) == "cpu"


def test_quality_columns_keep_strict_numeric_and_material_separate():
    # A gap of 1e-10 is numeric noise: a tie under the numeric tolerance and under
    # material, but a real (if tiny) loss under strict zero-tolerance equality.
    columns = metrics.quality_columns(-1.0000000001, -1.0)
    assert columns == {"strict": "qpu", "numeric_tolerance": "tie", "material": "tie"}
    # A gap of 0.2% is real but immaterial: numeric tolerance sees a difference,
    # 0.5% materiality calls it a tie. Zero materiality is never used outside "strict".
    columns = metrics.quality_columns(-1.002, -1.0)
    assert columns["strict"] == "qpu"
    assert columns["numeric_tolerance"] == "qpu"
    assert columns["material"] == "tie"


def test_tally_quality_outcomes_reports_every_key_even_at_zero():
    assert metrics.tally_quality_outcomes(["qpu", "qpu", "tie"]) == {"qpu": 2, "tie": 1, "cpu": 0}


def test_tally_quality_outcomes_all_ties():
    assert metrics.tally_quality_outcomes(["tie"] * 5) == {"qpu": 0, "tie": 5, "cpu": 0}


def test_tally_quality_outcomes_rejects_an_unknown_outcome():
    with pytest.raises(ValueError):
        metrics.tally_quality_outcomes(["qpu", "bogus"])


def test_format_feasibility_never_rounds_a_near_one_share_to_perfect():
    # The task brief's own regression case: seven infeasible reads out of 32,000.
    assert metrics.format_feasibility(31993, 32000) == "31993 / 32000 (99.978125%)"


def test_format_feasibility_shows_a_true_perfect_share_as_perfect():
    assert metrics.format_feasibility(32000, 32000) == "32000 / 32000 (100%)"


def test_format_feasibility_shows_a_zero_share():
    assert metrics.format_feasibility(0, 500) == "0 / 500 (0%)"


def test_format_feasibility_rejects_an_impossible_count():
    with pytest.raises(ValueError):
        metrics.format_feasibility(33, 32)
    with pytest.raises(ValueError):
        metrics.format_feasibility(0, 0)


def test_format_feasibility_never_prints_100_percent_when_a_read_is_infeasible_at_huge_scale():
    # 1 bad read in 2e8: six decimal places round 99.9999995% up to "100.000000",
    # which would falsely claim a perfect share (review, Important/Also-fix item).
    text = metrics.format_feasibility(199_999_999, 200_000_000)
    assert text == "199999999 / 200000000 (<100%)"


def test_reconstructed_feasible_range_is_exact_when_only_one_numerator_fits():
    # 0.7500 at 4 reads: only 3/4 rounds to 0.7500 at 4 decimals.
    assert metrics.reconstructed_feasible_range(0.75, 4) == (3, 3)


def test_reconstructed_feasible_range_is_ambiguous_when_rounding_allows_more_than_one():
    # A share of 0.5 rounded to only 1 decimal, out of 200 reads: every numerator
    # from 90 to 110 (share 0.45 to 0.55) rounds to the same recorded 0.5.
    low, high = metrics.reconstructed_feasible_range(0.5, 200, decimals=1)
    assert (low, high) == (90, 110)


def test_format_reconstructed_feasibility_labels_an_ambiguous_numerator():
    text = metrics.format_reconstructed_feasibility(0.5, 200, decimals=1)
    assert "reconstructed" in text
    assert "ambiguous" in text
    assert "/ 200 " in text


def test_format_reconstructed_feasibility_does_not_call_an_ambiguous_share_exact():
    text = metrics.format_reconstructed_feasibility(0.75, 4)
    assert "ambiguous" not in text
    assert text.startswith("3 / 4 (75%)")


def test_lane_diversity_reports_unique_states_and_pairwise_hamming_distances():
    spins = np.asarray([[1, 1, 1], [1, 1, -1], [-1, -1, -1]], dtype=np.int8)
    assert metrics.lane_diversity(spins) == {
        "unique_states": 3,
        "mean_pairwise_hamming": 2.0,
        "minimum_pairwise_hamming": 1,
    }


# ----------------------------------------------------------------- arm summary


def test_denominator_text_always_states_what_a_share_is_of():
    assert metrics.denominator_text(3, 4) == "3 of 4"


def _run(
    status: str = "completed", best: Optional[float] = -1.0, wall: Optional[float] = 1.0,
    contaminated: bool = False,
) -> dict:
    return {
        "status": status, "best_energy": best, "caller_end_to_end_s": wall,
        "host": {"contamination": {"contaminated": contaminated}},
    }


def test_summarize_arm_handles_an_empty_arm_without_crashing():
    summary = metrics.summarize_arm([])
    assert summary["observed"] == 0
    assert summary["completed"] == 0
    assert summary["missing"] is None
    assert summary["median_best_energy"] is None
    assert summary["median_wall_s"] is None


def test_summarize_arm_reports_a_missing_depth_against_the_expected_count():
    summary = metrics.summarize_arm([_run()] * 4, expected=5)
    assert summary["observed"] == 4
    assert summary["missing"] == 1
    assert summary["completed"] == 4


def test_summarize_arm_keeps_failed_unsupported_and_nonfinite_out_of_the_median():
    records = [
        _run(best=-3.0, wall=1.0), _run(best=-1.0, wall=3.0),
        _run(status="timeout", best=None, wall=None),
        _run(status="unsupported", best=None, wall=None),
        _run(best=float("nan"), wall=9.0),
    ]
    summary = metrics.summarize_arm(records)
    assert summary["observed"] == 5
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["unsupported"] == 1
    assert summary["nonfinite"] == 1
    assert summary["median_best_energy"] == -2.0
    assert summary["median_wall_s"] == 2.0


def test_summarize_arm_uses_every_completed_record_for_quality_but_splits_wall_time_by_label():
    parallel = _run(best=-3.0, wall=None, contaminated=True)
    summary = metrics.summarize_arm([_run(best=-1.0, wall=1.0), _run(best=-2.0, wall=100.0, contaminated=True), parallel])
    assert summary["completed"] == 3
    assert summary["median_best_energy"] == -2.0
    assert summary["median_wall_s"] == 1.0
    assert summary["clean_n"] == 1
    assert summary["timing_withheld"] == 1
    assert summary["timing_by_label"] == {
        "clean serial": {"n": 1, "median_wall_s": 1.0},
        "contaminated": {"n": 1, "median_wall_s": 100.0},
    }


def test_summarize_arm_treats_a_missing_contamination_verdict_as_contaminated():
    record = {"status": "completed", "best_energy": -1.0, "caller_end_to_end_s": 2.0, "host": {}}
    summary = metrics.summarize_arm([record])
    assert summary["clean_n"] == 0
    assert summary["timing_by_label"] == {"contaminated": {"n": 1, "median_wall_s": 2.0}}


# --------------------------------------------------------------- bootstrap gaps


def test_bootstrap_paired_gap_is_descriptive_with_too_few_groups():
    # Two portfolio baskets (n=18, n=28) each contribute two beta-label gaps: four
    # gaps, but only two independent snapshots (task brief, step 6).
    gaps = [0.01, 0.012, 0.03, 0.028]
    groups = ["n18", "n18", "n28", "n28"]
    result = metrics.bootstrap_paired_gap(gaps, groups, seed=0, resamples=200)
    assert result["group_count"] == 2
    assert result["descriptive"] is True


def test_bootstrap_paired_gap_never_treats_a_shared_snapshot_as_two_independent_points():
    # All four gaps share one snapshot: resampling must draw the whole group at
    # once, so the group count is 1, not 4, and no real interval is possible.
    gaps = [0.01, 0.012, 0.03, 0.028]
    groups = ["only-snapshot"] * 4
    result = metrics.bootstrap_paired_gap(gaps, groups, seed=0, resamples=200)
    assert result["group_count"] == 1
    assert result["descriptive"] is True
    assert math.isnan(result["low"]) and math.isnan(result["high"])


def test_bootstrap_paired_gap_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(1)
    gaps = list(rng.normal(0.01, 0.002, size=20))
    groups = [f"model-{i}" for i in range(20)]
    first = metrics.bootstrap_paired_gap(gaps, groups, seed=42, resamples=500)
    second = metrics.bootstrap_paired_gap(gaps, groups, seed=42, resamples=500)
    assert first == second
    assert first["group_count"] == 20
    assert first["descriptive"] is False


def test_bootstrap_paired_gap_handles_no_data():
    result = metrics.bootstrap_paired_gap([], [], seed=0)
    assert result["group_count"] == 0
    assert result["descriptive"] is True


def test_bootstrap_paired_gap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        metrics.bootstrap_paired_gap([1.0], ["a", "b"], seed=0)


def test_bootstrap_paired_gap_records_its_own_seed_and_resamples_on_every_path():
    # A "fixed recorded seed" (task brief, step 6) means the seed and resample
    # count travel with the result, on every return path -- including the two
    # early-return paths (no data, too few groups) the review found missing it.
    empty = metrics.bootstrap_paired_gap([], [], seed=7, resamples=123)
    assert empty["seed"] == 7 and empty["resamples"] == 123
    one_group = metrics.bootstrap_paired_gap([1.0, 2.0], ["only", "only"], seed=7, resamples=123)
    assert one_group["seed"] == 7 and one_group["resamples"] == 123
    real = metrics.bootstrap_paired_gap([1.0, 2.0], ["a", "b"], seed=7, resamples=123)
    assert real["seed"] == 7 and real["resamples"] == 123


def test_bootstrap_paired_gap_defaults_to_the_named_fixed_seed_constant():
    result = metrics.bootstrap_paired_gap([1.0, 2.0], ["a", "b"], resamples=50)
    assert result["seed"] == metrics.BOOTSTRAP_SEED
