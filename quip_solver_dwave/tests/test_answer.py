"""Read an answer without building a dimod.SampleSet.

OceanSampler needs five things off a finished job: spins, variable labels,
energies, occurrence counts and timing. A cloud Future carries all five. Going
through .sampleset instead costs about 38.8 ms per job at production size, because it
turns the decoded numpy arrays into Python lists, walks them with a nested
comprehension, and hands them to dimod to turn back into numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from quip_solver_dwave.answer import AnswerView, answer_view


class FakeFuture:
    """The five attributes answer_view reads off a cloud Future."""

    def __init__(self, samples, variables, energies, num_occurrences, timing):
        self.samples = samples
        self.variables = variables
        self.energies = energies
        self.num_occurrences = num_occurrences
        self.timing = timing
        self.sampleset_touched = False

    @property
    def sampleset(self):
        self.sampleset_touched = True
        raise AssertionError("answer_view must not build a SampleSet")


def _future(**kw):
    base = dict(
        samples=np.array([[1, -1], [-1, -1]], dtype=np.int8),
        variables=[10, 20],
        energies=[-3.0, -1.0],
        num_occurrences=[3, 5],
        timing={"qpu_programming_time": 33_000, "qpu_sampling_time": 10_200},
    )
    base.update(kw)
    return FakeFuture(**base)


def test_a_cloud_future_is_read_without_building_a_sampleset():
    view = answer_view(_future())

    assert isinstance(view, AnswerView)
    assert view.spins.tolist() == [[1, -1], [-1, -1]]
    assert view.variables == [10, 20]
    assert view.energies == [-3.0, -1.0]


def test_reads_count_anneals_performed_not_distinct_solutions():
    # The cloud client folds identical reads into one row carrying
    # num_occurrences. Reporting the row count would under-report the anneals
    # the QPU actually ran, which is what SamplerMeta.reads means.
    view = answer_view(_future(num_occurrences=[3, 5]))

    assert view.reads == 8


def test_access_time_is_programming_plus_sampling():
    view = answer_view(_future())

    assert view.access_time_us == 43_200


def test_missing_timing_bills_nothing_rather_than_guessing():
    view = answer_view(_future(timing={}))

    assert view.access_time_us == 0


def test_spins_are_normalised_to_plus_or_minus_one():
    # Offline samplers can hand back 0/1. The wire format is one signed byte
    # per spin, so a 0 would reach the coordinator as a third value.
    view = answer_view(_future(samples=np.array([[0, 1]], dtype=np.int8)))

    assert view.spins.tolist() == [[-1, 1]]
    assert view.spins.dtype == np.int8


def test_padding_columns_are_dropped_and_labels_stay_aligned():
    # dwave.cloud.coders.decode_qp_numpy pads the solution matrix out to the
    # solver's full physical qubit count and writes the decoded bits into only
    # the active columns: solutions[:, active_variables] = bits. Future.samples
    # returns that padded matrix, while Future.variables returns the active
    # labels alone. Pairing them 1:1 misattributes every real qubit at or after
    # the first gap, and the padding value rides along as a reading.
    padded = np.full((2, 6), 7, dtype=np.int8)  # 7 marks a padding column
    active = [0, 2, 5]
    padded[:, active] = np.array([[1, -1, 1], [-1, 1, -1]], dtype=np.int8)

    view = answer_view(_future(samples=padded, variables=active))

    assert view.variables == active
    assert view.spins.shape == (2, len(view.variables))
    # The values must be the ones belonging to those labels, not the first
    # three columns of the padded matrix.
    assert view.spins.tolist() == [[1, -1, 1], [-1, 1, -1]]


def test_an_already_aligned_answer_is_left_alone():
    # Fixtures and future SDK versions may hand back a matrix that is already
    # restricted to the active columns.
    view = answer_view(_future())

    assert view.spins.tolist() == [[1, -1], [-1, -1]]
    assert view.variables == [10, 20]


def test_a_dimod_sampleset_reads_the_same_way():
    # The mock and injected-sampler paths still return a SampleSet.
    dimod = pytest.importorskip("dimod")
    ss = dimod.SampleSet.from_samples(
        ([[1, -1], [-1, -1]], [10, 20]), vartype="SPIN", energy=[-3.0, -1.0]
    )
    ss = ss.copy()
    ss.info.update(
        timing={"qpu_programming_time": 33_000, "qpu_sampling_time": 10_200}
    )

    view = answer_view(ss)

    assert view.spins.tolist() == [[1, -1], [-1, -1]]
    assert view.variables == [10, 20]
    assert view.energies == [-3.0, -1.0]
    assert view.access_time_us == 43_200


def test_an_empty_answer_does_not_crash():
    view = answer_view(
        _future(
            samples=np.zeros((0, 2), dtype=np.int8),
            energies=[],
            num_occurrences=[],
        )
    )

    assert view.spins.shape == (0, 2)
    assert view.reads == 0


def test_a_1d_empty_answer_is_reshaped_to_zero_rows():
    # With return_matrix=False and zero solutions, result['solutions'] is [],
    # so np.asarray gives a 1-D (0,) array. The ndim == 2 alignment guard
    # skips a 1-D array, so this must be reshaped separately or the caller
    # sees a rank-1 empty array where it expects (reads, len(variables)).
    view = answer_view(
        _future(
            samples=np.array([], dtype=np.int8),
            variables=[10, 20],
            energies=[],
            num_occurrences=[],
        )
    )

    assert view.spins.shape == (0, 2)
    assert view.reads == 0


def test_the_sampler_never_asks_the_future_for_a_sampleset():
    # The regression this plan exists to prevent. FakeFuture.sampleset raises,
    # so any path that reaches for it fails loudly rather than quietly costing
    # 38.8 ms per job.
    from quip_solver_dwave.ocean import OceanSampler

    s = OceanSampler(mock=False)
    s._connected = True
    s._is_mock = False

    fut = _future()
    result = s._decode_and_view(fut)

    assert fut.sampleset_touched is False
    assert result.spins.tolist() == [[1, -1], [-1, -1]]
    assert result.access_time_us == 43_200


def test_the_cloud_future_is_asked_for_numpy_not_lists():
    # return_matrix=False makes the decoder call .tolist() on a
    # (reads x qubits) array, which is 220k Python objects per job at
    # production size.
    from quip_solver_dwave.ocean import OceanSampler

    captured = {}

    class _Solver:
        # What the SDK defaults to. _submit_encoded never reads this — it
        # pins return_matrix=True unconditionally — so this is a deliberate
        # trap: the assertion below fails if that pin is ever wired back to
        # read this attribute instead.
        return_matrix = False

        class client:
            @staticmethod
            def _submit(body, computation):
                captured["computation"] = computation

    s = OceanSampler(mock=False)
    s._connected = True
    s._is_mock = False

    import dwave.cloud.computation as comp

    real_future = comp.Future

    def spy(solver, id_, return_matrix=False):
        captured["return_matrix"] = return_matrix
        return real_future(solver, id_, return_matrix=return_matrix)

    comp.Future = spy
    try:
        s._submit_encoded(_Solver(), b"{}", None)
    finally:
        comp.Future = real_future

    assert captured["return_matrix"] is True
