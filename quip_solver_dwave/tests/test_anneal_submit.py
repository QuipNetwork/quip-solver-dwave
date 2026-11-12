"""What reaches SAPI: an anneal schedule on every job, and never a time.

SAPI refuses a problem that carries both ``annealing_time`` and
``anneal_schedule``, and a reverse anneal can only be written as a schedule.
These tests pin the submitted parameters on the real (non-mock) submit path,
with only the hand-off to the cloud client stubbed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from quip_solver_dwave.ocean import MockSampler, OceanSampler, descend_from
from quip_solver_dwave.schedule import ScheduleError
from quip_solver_dwave.warm import WarmStart

_PARAMETERS = {
    "num_reads": None,
    "anneal_schedule": None,
    "initial_state": None,
    "reinitialize_state": None,
    "label": None,
}


class _Done:
    """A finished cloud problem, shaped like dwave.cloud.computation.Future."""

    samples = np.array([[1, -1]], dtype=np.int8)
    energies = [-1.5]
    num_occurrences = [1]
    variables = [0, 1]
    timing = {"qpu_programming_time": 100, "qpu_sampling_time": 50}

    def done(self):
        return True


def _cloud_sampler(properties=None):
    """A sampler on the real submit path whose solver publishes ``properties``."""
    s = OceanSampler(mock=False)
    s._connected = True
    s._is_mock = False

    published = (
        {
            "default_annealing_time": 20.0,
            "annealing_time_range": [0.5, 2000.0],
            # Labels have holes on a real chip: more physical qubits than
            # the two this solver exposes.
            "num_qubits": 4,
        }
        if properties is None
        else properties
    )

    class _Identity:
        def dict(self):
            return {"name": "FakeSolver", "version": {"graph_id": "x"}}

    class _Solver:
        _encoding_qubits = [0, 1]
        _encoding_couplers = [(0, 1)]
        _params: dict = {}
        parameters = dict(_PARAMETERS)
        properties = published
        identity = _Identity()

        def _format_params(self, type_, params):
            pass

    class _Sampler:
        solver = _Solver()

    s.sampler = _Sampler()
    bodies = []

    def _submit_encoded(solver, body, cancel_key):
        bodies.append(json.loads(body))
        return _Done()

    s._submit_encoded = _submit_encoded  # type: ignore[method-assign]
    return s, bodies


def _submit(s, **kwargs):
    return s.sample(
        np.array([0, 1]),
        np.array([1.0, -1.0]),
        np.array([(0, 1)]),
        np.array([0.5]),
        num_reads=4,
        **kwargs,
    )


def test_a_job_with_no_anneal_time_sends_the_solver_default_as_a_schedule():
    s, bodies = _cloud_sampler()
    _submit(s)

    params = bodies[0]["params"]
    assert params["anneal_schedule"] == [[0.0, 0.0], [20.0, 1.0]]
    assert "annealing_time" not in params
    s.close()


def test_an_anneal_time_override_becomes_the_end_of_the_ramp():
    s, bodies = _cloud_sampler()
    _submit(s, anneal_time_us=250)

    params = bodies[0]["params"]
    assert params["anneal_schedule"] == [[0.0, 0.0], [250.0, 1.0]]
    assert "annealing_time" not in params
    s.close()


def test_a_cold_job_sends_no_reverse_anneal_parameters():
    s, bodies = _cloud_sampler()
    _submit(s)

    assert "initial_state" not in bodies[0]["params"]
    assert "reinitialize_state" not in bodies[0]["params"]
    s.close()


def test_a_solver_that_publishes_no_default_gets_no_schedule_at_all():
    # Writing a guessed default onto the wire would change the anneal. With
    # nothing sent, the solver's own default applies, as it did before.
    s, bodies = _cloud_sampler(properties={})
    _submit(s)

    assert "anneal_schedule" not in bodies[0]["params"]
    assert "annealing_time" not in bodies[0]["params"]
    s.close()


def test_an_anneal_the_chip_cannot_run_raises_before_anything_is_submitted():
    s, bodies = _cloud_sampler()
    with pytest.raises(ScheduleError):
        _submit(s, anneal_time_us=5000)

    assert bodies == []
    s.close()


def test_a_warm_start_sends_a_reverse_schedule_and_one_entry_per_physical_qubit():
    s, bodies = _cloud_sampler()
    warm = WarmStart(
        state=np.array([-1, 1], dtype=np.int8), reversal_s=0.5, reversal_pause_us=25
    )
    _submit(s, warm_start=warm)

    params = bodies[0]["params"]
    assert params["anneal_schedule"] == [
        [0.0, 1.0],
        [10.0, 0.5],
        [35.0, 0.5],
        [45.0, 1.0],
    ]
    # Indexed by qubit label; 3 is SAPI's marker for "not in the problem".
    assert params["initial_state"] == [-1, 1, 3, 3]
    assert params["reinitialize_state"] is True
    assert "annealing_time" not in params
    s.close()


def test_a_warm_start_ramps_at_the_jobs_own_anneal_time():
    s, bodies = _cloud_sampler()
    warm = WarmStart(
        state=np.array([-1, 1], dtype=np.int8), reversal_s=0.9, reversal_pause_us=10
    )
    _submit(s, anneal_time_us=100, warm_start=warm)

    assert bodies[0]["params"]["anneal_schedule"] == [
        [0.0, 1.0],
        [10.0, 0.9],
        [20.0, 0.9],
        [30.0, 1.0],
    ]
    s.close()


def test_ocean_leaves_a_per_qubit_list_exactly_as_it_was_built():
    # build_submission_body hands every parameter to Ocean's formatter. Pinned
    # against the installed SDK: the list form passes through untouched, and
    # Ocean's own expansion of a mapping produces the same list.
    from dwave.cloud.solver import StructuredSolver

    as_list = StructuredSolver.reformat_parameters(
        "ising", {"initial_state": [-1, 3, 1, 3]}, {"num_qubits": 4}
    )
    as_mapping = StructuredSolver.reformat_parameters(
        "ising", {"initial_state": {0: -1, 2: 1}}, {"num_qubits": 4}
    )

    assert as_list["initial_state"] == [-1, 3, 1, 3]
    assert as_mapping["initial_state"] == as_list["initial_state"]


# --- the offline mock ------------------------------------------------------


def _ring(n: int):
    """The conformance driver's proof ring: every bond satisfied by ``planted``."""

    def spin(i: int) -> int:
        return 1 if ((i * 2_654_435_761) >> 7) & 1 else -1

    planted = {i: spin(i) for i in range(n)}
    h = {i: 0.0 for i in range(n)}
    j = {(i, (i + 1) % n): -1.0 * spin(i) * spin((i + 1) % n) for i in range(n)}
    return h, j, planted


def test_descent_from_a_ground_state_returns_it_unchanged():
    h, j, planted = _ring(4096)
    state, energy = descend_from(h, j, planted)

    assert state == planted
    assert energy == -4096.0


def test_descent_never_raises_the_energy():
    h, j, planted = _ring(64)
    knocked = dict(planted)
    for i in (3, 4, 20):
        knocked[i] = -knocked[i]
    start_energy = sum(c * knocked[u] * knocked[v] for (u, v), c in j.items())
    _, energy = descend_from(h, j, knocked)

    assert energy <= start_energy


def test_the_exact_mock_answers_a_seeded_job_too_large_to_enumerate():
    # ExactSolver walks every state, so without the seeded path this call
    # never returns: the ring has 2**4096 of them.
    h, j, planted = _ring(4096)
    ss = MockSampler(backend="exact").sample_ising(
        h, j, num_reads=3, initial_state=planted
    )

    assert float(ss.record.energy[0]) == -4096.0
    assert int(ss.record.num_occurrences[0]) == 3
    assert ss.info["timing"]["qpu_sampling_time"] >= 1


def test_a_warm_start_reaches_the_mock_as_a_state_mapping():
    s = OceanSampler(mock=True)
    warm = WarmStart(
        state=np.array([-1, 1], dtype=np.int8), reversal_s=0.5, reversal_pause_us=25
    )
    # h = [1, -1], J = 0.5: the ground state is [-1, +1] at energy -2.5.
    result = _submit(s, warm_start=warm)

    assert result.energies == [-2.5]
    assert result.num_reads == 4
    s.close()


# --- a chip that has lost a qubit ------------------------------------------


class _Recorder:
    """Stands in for a QPU: records what it is handed, returns one read."""

    def __init__(self):
        self.calls = []

    def sample_ising(self, h, j, **kwargs):
        import dimod

        self.calls.append((dict(h), dict(j), dict(kwargs)))
        variables = sorted({v for e in j for v in e} | set(h))
        return dimod.SampleSet.from_samples(
            [{v: 1 for v in variables}], vartype="SPIN", energy=[0.0]
        )


def test_a_missing_qubit_is_clamped_to_the_start_states_own_spin():
    rec = _Recorder()
    s = OceanSampler(sampler=rec, mock=False)
    s._live_nodes = [0, 1]
    s._live_edges = [(0, 1)]
    s.set_session_topology([0, 1, 2], [(0, 1), (1, 2)])
    assert s._defective_qubits == [2]

    warm = WarmStart(
        state=np.array([1, -1, -1], dtype=np.int8), reversal_s=0.5, reversal_pause_us=25
    )
    result = s.sample(
        np.array([0, 1, 2]),
        np.zeros(3),
        np.array([(0, 1), (1, 2)]),
        np.array([1.0, 1.0]),
        num_reads=1,
        # A seed that would draw its own spin for qubit 2 if it were used.
        nonce_seed=b"\x07",
        warm_start=warm,
    )

    h_sent, _, kwargs = rec.calls[0]
    # J(1,2) = 1 folds the clamped spin -1 into qubit 1's bias.
    assert h_sent == {0: 0.0, 1: -1.0}
    # SAPI wants a spin for each submitted qubit and for no other.
    assert kwargs["initial_state"] == {0: 1, 1: -1}
    assert result.defect_info is not None
    assert result.defect_info.fixed_spins == {2: -1}
    s.close()
