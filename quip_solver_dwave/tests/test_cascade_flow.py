import numpy as np

from quip_solver_dwave import cascade_flow as flow

PROGRAMS = {"seeded": "seeded-1m", "cold": "cold-1m"}


def survivor(nonce: str, energy: int, lanes: int = 4, nodes: int = 6, seed: int = 0) -> flow.Survivor:
    rng = np.random.default_rng(seed)
    states = (1 - 2 * rng.integers(0, 2, size=(lanes, nodes))).astype(np.int8)
    return flow.Survivor(nonce, energy, states, np.arange(lanes) - 100)


def test_qpu_gate_drops_a_model_whose_best_read_is_above_it():
    s = survivor("1" * 64, -14_300)
    spins = np.ones((3, 6), dtype=np.int8)
    energies = np.array([-14_500_000, -14_480_000, -14_490_000])
    kept = flow.qpu_plans(s, spins, energies, qpu_gate_milli=-14_500_000, arms=["qpu-seeded", "qpu-cold"],
                          programs=PROGRAMS, lanes=4)
    assert [p.arm for p in kept] == ["qpu-seeded", "cold"]
    assert flow.qpu_plans(s, spins, energies, qpu_gate_milli=-14_500_001, arms=["qpu-seeded"],
                          programs=PROGRAMS, lanes=4) == []


def test_seeded_plan_puts_distinct_qpu_reads_first_and_fills_with_screen_states():
    s = survivor("2" * 64, -14_300, lanes=4, nodes=6, seed=3)
    one = np.ones(6, dtype=np.int8)
    two = -one
    # Three QPU reads, two of them identical: one lane would be wasted.
    spins = np.stack([one, one, two])
    energies = np.array([-3_000, -2_000, -1_000])
    (plan,) = flow.qpu_plans(s, spins, energies, qpu_gate_milli=0, arms=["qpu-seeded"], programs=PROGRAMS, lanes=4)
    assert plan.program == "seeded-1m" and plan.priority == -3
    assert plan.states is not None
    # Half of the lanes may come from the QPU: the two best reads are identical, so one lane holds a QPU read.
    assert plan.qpu_lanes == 1
    assert plan.states.shape == (4, 6)
    assert len({row.tobytes() for row in plan.states}) == 4
    assert (plan.states[0] == one).all()


def test_cold_plans_share_one_unit_id_per_model_program_and_repeat():
    a = flow.cold_plan("3" * 64, PROGRAMS, -14_500)
    b = flow.cold_plan("3" * 64, PROGRAMS, -14_400)
    assert a.unit_id == b.unit_id == "3" * 64 + ":cold-1m:0" and a.states is None and a.seed == b.seed
    again = flow.cold_plan("3" * 64, PROGRAMS, -14_500, rep=1)
    assert again.unit_id != a.unit_id and again.seed != a.seed
    deeper = flow.cold_plan("3" * 64, {**PROGRAMS, "cold": "cold-2m"}, -14_500)
    assert deeper.unit_id != a.unit_id


def test_unit_seed_is_stable_and_differs_by_salt():
    assert flow.unit_seed("ab" * 32, 1) == flow.unit_seed("ab" * 32, 1)
    assert flow.unit_seed("ab" * 32, 1) != flow.unit_seed("ab" * 32, 2)
    assert 0 <= flow.unit_seed("ff" * 32, 3) < 2**63
