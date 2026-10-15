"""Operator-set sampling defaults from ``Configure.backend_toml``.

``anneal_time_us`` and ``num_reads`` were accepted by the dwave config schema
but read by nothing: the only consumers were the per-job ``IsingProblem`` and
the session ``SetTarget``. These pin them as the lowest rung of that ladder.
"""

from quip_solver_core import miner_pb2

from quip_solver_dwave.config import SamplingDefaults, sampling_defaults_from_toml
from quip_solver_dwave.job import _sampling_params

SWEEPS = 64


def _ising(**kw) -> miner_pb2.IsingProblem:
    return miner_pb2.IsingProblem(**kw)


def test_reads_and_anneal_are_parsed_from_config():
    got = sampling_defaults_from_toml("anneal_time_us = 80\nnum_reads = 32\n")
    assert got == SamplingDefaults(num_reads=32, anneal_time_us=80)


def test_absent_keys_leave_the_defaults_unset():
    assert sampling_defaults_from_toml('budget = "250m"\n') == SamplingDefaults()


def test_empty_and_malformed_config_yield_unset_defaults():
    assert sampling_defaults_from_toml("") == SamplingDefaults()
    assert sampling_defaults_from_toml("num_reads = 32m\n") == SamplingDefaults()


def test_non_integer_and_negative_values_are_ignored():
    assert sampling_defaults_from_toml('num_reads = "32"\n') == SamplingDefaults()
    assert sampling_defaults_from_toml("anneal_time_us = -5\n") == SamplingDefaults()
    # A bool is an int subclass in Python; it must not pass as a read count.
    assert sampling_defaults_from_toml("num_reads = true\n") == SamplingDefaults()


def test_config_defaults_apply_when_job_and_target_are_silent():
    reads, anneal, sweeps = _sampling_params(
        _ising(), None, SWEEPS, SamplingDefaults(num_reads=32, anneal_time_us=80)
    )
    assert (reads, anneal, sweeps) == (32, 80, SWEEPS)


def test_set_target_outranks_the_config_default():
    target = miner_pb2.SetTarget(num_reads=7, anneal_time_us=99)
    reads, anneal, _ = _sampling_params(
        _ising(), target, SWEEPS, SamplingDefaults(num_reads=32, anneal_time_us=80)
    )
    assert (reads, anneal) == (7, 99)


def test_per_job_values_outrank_everything():
    target = miner_pb2.SetTarget(num_reads=7, anneal_time_us=99)
    reads, anneal, _ = _sampling_params(
        _ising(num_reads=3, anneal_time_us=11),
        target,
        SWEEPS,
        SamplingDefaults(num_reads=32, anneal_time_us=80),
    )
    assert (reads, anneal) == (3, 11)


def test_without_config_defaults_the_old_fallbacks_still_hold():
    """No config: one read, and 0 anneal so the QPU picks its own."""
    assert _sampling_params(_ising(), None, SWEEPS)[:2] == (1, 0)


def test_each_dimension_falls_back_independently():
    """Setting only one key must not drag the other off its own ladder."""
    reads, anneal, _ = _sampling_params(
        _ising(), None, SWEEPS, SamplingDefaults(anneal_time_us=80)
    )
    assert (reads, anneal) == (1, 80)


def test_the_reverse_anneal_keys_are_read_from_the_config():
    got = sampling_defaults_from_toml("reversal_s_milli = 450\nreversal_pause_us = 80\n")
    assert got == SamplingDefaults(reversal_s_milli=450, reversal_pause_us=80)


def test_a_reversal_point_of_one_thousand_milli_or_more_is_ignored():
    # The wire rejects it as malformed, so applying it as a default would
    # turn every seeded job into a reject.
    got = sampling_defaults_from_toml("reversal_s_milli = 1000\n")
    assert got == SamplingDefaults()


def test_the_reverse_anneal_keys_are_known_to_the_dwave_schema():
    from quip_solver_dwave.budget import DWAVE_CONFIG_KEYS

    assert {"reversal_s_milli", "reversal_pause_us"} <= DWAVE_CONFIG_KEYS
