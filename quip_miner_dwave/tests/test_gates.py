import numpy as np
import pytest

from quip_miner_dwave import gates


def test_derive_gates_takes_the_shallowest_best_of_the_deepest_models_plus_margin():
    # trace[seed, model, leg]; model 0 is the deepest winner.
    trace = np.array([[[-100, -200], [-90, -210], [-10, -20]],
                      [[-104, -190], [-96, -205], [-12, -22]]])
    assert gates.derive_gates(trace, deepest=2, margin=10) == [-80, -180]
    assert gates.derive_gates(trace, deepest=3, margin=0) == [-10, -20]


def test_derive_gates_refuses_more_models_than_the_trace_holds():
    with pytest.raises(ValueError, match="cannot set gates"):
        gates.derive_gates(np.zeros((2, 3, 4)), deepest=4, margin=10)


def test_kept_runs_never_revive_a_run_a_gate_dropped():
    trace = np.array([[[-100, -50, -300]]])
    kept = gates.kept_runs(trace, [-90, -60, -200])
    assert kept.tolist() == [[[True, False, False]]]


def test_a_run_at_the_gate_is_kept():
    assert gates.kept_runs(np.array([[[-90]]]), [-90]).tolist() == [[[True]]]


def test_cut_for_miss_loses_at_most_the_allowed_share():
    best = np.array([-10, -20, -30, -40, -50, -60, -70, -80, -90, -100])
    assert gates.cut_for_miss(best, 0.0) == -10
    assert gates.cut_for_miss(best, 0.2) == -30
    assert (best > gates.cut_for_miss(best, 0.2)).mean() <= 0.2
    assert gates.cut_for_miss(best, 1.0) == -100


def test_gate_table_reports_loss_and_control_pass_share():
    rows = gates.gate_table(np.array([-10, -20, -30, -40]), np.array([0, -15, -25, -35, -45]), [0.0, 0.25])
    assert rows[0] == {"miss_allowed": 0.0, "cut": -10, "winners_lost": 0, "winner_miss": 0.0, "control_pass": 0.8}
    assert rows[1]["cut"] == -20 and rows[1]["winners_lost"] == 1 and rows[1]["control_pass"] == 0.6
