"""How deep the QPU pipeline runs, and who gets to say.

Depth is a property of the device and its cloud round trip, not of the
coordinator's generic flow control, so the backend sets it rather than taking
the SDK's default of 3 — a conservative number chosen for miners in general.

The depth is sized by Little's Law so a QPU that becomes free can be fed at
its own ceiling, not sized to match whatever contention the device shows
today. Measured on Advantage2_system1 with a production-sized problem: 23.2
jobs/s chip ceiling against a 1.57 s round trip needs 36 in flight, and the
3.05 s round trip seen under load needs 71. The default carries slack past
both so a connectivity blip costs throughput rather than stalling the chip.
"""

from __future__ import annotations

import logging

from quip_solver_dwave.config import queue_depth_from_toml
from quip_solver_dwave.session_loop import DEFAULT_QUEUE_DEPTH, resolve_queue_depth


def test_the_default_saturates_the_chip_with_slack_for_a_slow_round_trip():
    # 23.2 jobs/s chip ceiling x 1.57s round trip = 36 needed; the measured
    # 3.05s round trip under load needs 71. The default covers a 4.14s round
    # trip, and probing to 128 raised no error, so the cap is not near it.
    assert DEFAULT_QUEUE_DEPTH == 96
    chip_jobs_per_s = 1 / 0.0432
    assert DEFAULT_QUEUE_DEPTH / chip_jobs_per_s > 3.05  # covers observed RTT


def test_an_unset_toml_key_reads_as_zero():
    assert queue_depth_from_toml("") == 0
    assert queue_depth_from_toml("num_reads = 48\n") == 0


def test_the_operator_can_set_the_depth_in_backend_toml():
    assert queue_depth_from_toml("queue_depth = 16\n") == 16


def test_a_malformed_document_yields_no_depth(caplog):
    # The budget parser is the one that refuses to run on unreadable config;
    # repeating that failure here would only obscure it.
    assert queue_depth_from_toml("queue_depth = [unclosed\n") == 0


def test_a_nonsense_depth_is_ignored_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert queue_depth_from_toml("queue_depth = -4\n") == 0
    assert "queue_depth" in caplog.text


def test_a_boolean_is_not_a_depth(caplog):
    # TOML booleans are ints in Python; letting True mean depth 1 would be a
    # silent misconfiguration.
    with caplog.at_level(logging.WARNING):
        assert queue_depth_from_toml("queue_depth = true\n") == 0


def test_the_operator_setting_beats_the_coordinator():
    # The backend config is where QPU-specific knowledge lives, the same way
    # num_reads and anneal_time_us already override what the coordinator sends.
    assert resolve_queue_depth(coordinator=3, configured=16) == 16


def test_the_coordinator_is_used_when_the_operator_says_nothing():
    assert resolve_queue_depth(coordinator=8, configured=0) == 8


def test_an_unset_coordinator_depth_falls_back_to_the_measured_default():
    # Zero is "not set" on the wire. This is the case in production today, and
    # it is why the pipeline ran three deep.
    assert resolve_queue_depth(coordinator=0, configured=0) == DEFAULT_QUEUE_DEPTH


def test_depth_is_never_below_one():
    # A zero-depth pipeline grants no credits and mines nothing.
    assert resolve_queue_depth(coordinator=-1, configured=0) >= 1
