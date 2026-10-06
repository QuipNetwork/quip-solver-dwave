import pytest

from quip_miner_dwave import cascade_programs as cp

SCREEN = {"name": "screen", "words": 2, "observe_every": 0, "legs": [
    {"start": "fresh", "sweeps": 8, "gate": -13312},
    {"start": "continue", "sweeps": 8, "gate": -13852, "from_beta": 0.25},
    {"start": "continue", "sweeps": 48, "gate": -14306, "from_beta": 0.25},
    {"start": "continue", "sweeps": 192, "gate": -14498, "from_beta": 0.25},
]}


def test_the_screen_stops_at_the_handoff_and_its_last_gate_is_loosened():
    programs = cp.pipeline_programs(SCREEN, handoff_legs=3, handoff_margin=20, depths=[1048576],
                                    start_betas=[0.25], observe_every=16384)
    by_name = {p["name"]: p for p in programs}
    assert [leg["gate"] for leg in by_name["screen"]["legs"]] == [-13312, -13852, -14286]
    assert SCREEN["legs"][2]["gate"] == -14306, "the input is not changed"
    leg = by_name["leg-256"]["legs"]
    assert leg == [{"start": "seeded", "from_beta": 0.25, "sweeps": 192, "gate": -14498}]
    assert by_name["cold-1048576"]["legs"] == [{"start": "fresh", "sweeps": 1048576}]
    assert by_name["cold-1048576"]["observe_every"] == 16384
    assert by_name["seeded-1048576-b0250"]["legs"] == [{"start": "seeded", "from_beta": 0.25, "sweeps": 1048576}]
    assert all(p["words"] == 2 for p in programs)


def test_a_handoff_after_the_last_leg_leaves_no_continue_program():
    programs = cp.pipeline_programs(SCREEN, handoff_legs=4, handoff_margin=0, depths=[], start_betas=[],
                                    observe_every=0)
    assert [p["name"] for p in programs] == ["screen"]
    with pytest.raises(ValueError, match="handoff_legs"):
        cp.pipeline_programs(SCREEN, handoff_legs=5, handoff_margin=0, depths=[], start_betas=[], observe_every=0)


def test_the_gpu_program_runs_the_gated_leg_256_and_then_a_fresh_deep_leg():
    programs = cp.pipeline_programs(SCREEN, handoff_legs=3, handoff_margin=0, depths=[1048576], start_betas=[0.25],
                                    observe_every=16384)
    by_name = {p["name"]: p for p in programs}
    assert programs[-1] is by_name["gpu-256"]  # the sidecar and the miner refuse a programs file without it
    assert by_name["gpu-256"] == {
        "name": "gpu-256", "words": 2, "observe_every": 16384,
        "legs": [{"start": "seeded", "from_beta": 0.25, "sweeps": 192, "gate": -14498},
                 {"start": "fresh", "sweeps": 1048576}]}
    # A copy of the legs: a change to the program of the GPU arm leaves ``leg-256`` alone.
    by_name["gpu-256"]["legs"][0]["gate"] = 0
    assert by_name["leg-256"]["legs"][0]["gate"] == -14498


def test_the_gpu_program_takes_the_deep_leg_and_the_observe_interval_that_it_is_given():
    leg256 = {"name": "leg-256", "words": 1, "observe_every": 0, "legs": [{"start": "seeded", "sweeps": 64, "gate": -1}]}
    assert cp.gpu_256_program(leg256, deep_sweeps=2097152, observe_every=0) == {
        "name": "gpu-256", "words": 1, "observe_every": 0,
        "legs": [{"start": "seeded", "sweeps": 64, "gate": -1}, {"start": "fresh", "sweeps": 2097152}]}
    assert cp.gpu_256_program(leg256)["legs"][-1] == {"start": "fresh", "sweeps": cp.GPU_DEEP_SWEEPS}


def test_start_beta_is_the_log_spaced_point_of_the_ladder():
    hot, cold = 0.0173286795, 6.5169846
    assert cp.start_beta(hot, cold, 0.0) == pytest.approx(hot)
    assert cp.start_beta(hot, cold, 0.5) == pytest.approx(0.33605, abs=1e-5)
    assert cp.start_beta(hot, cold, 0.45) == pytest.approx(0.2498, abs=1e-3)
    assert cp.seeded_name(131072, 0.33605) == "seeded-131072-b0336"
