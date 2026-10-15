"""Strategy knobs arrive in Configure.backend_toml like the budget's."""

from __future__ import annotations

import logging

from quip_solver_dwave.budget import DWAVE_CONFIG_KEYS, warn_unknown_backend_keys
from quip_solver_dwave.strategy import StrategyConfig, strategy_config_from_toml


def test_defaults_keep_todays_behaviour():
    cfg = StrategyConfig()
    assert cfg.min_throughput_advantage == 0.25
    assert cfg.participation_chance == 0.10
    assert strategy_config_from_toml("") == cfg
    assert strategy_config_from_toml("budget = 250m") == cfg  # unparsable: the budget parser reports it


def test_the_two_keys_are_read():
    cfg = strategy_config_from_toml("min_throughput_advantage = 1\nparticipation_chance = 0\n")
    assert cfg == StrategyConfig(min_throughput_advantage=1.0, participation_chance=0.0)


def test_nonsense_values_warn_and_fall_back(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = strategy_config_from_toml("min_throughput_advantage = -1\nparticipation_chance = true\n")
    assert cfg == StrategyConfig()
    messages = [r.getMessage() for r in caplog.records]
    assert any("min_throughput_advantage" in m for m in messages)
    assert any("participation_chance" in m for m in messages)


def test_the_keys_are_known_to_the_dwave_schema(caplog):
    assert {"min_throughput_advantage", "participation_chance"} <= DWAVE_CONFIG_KEYS
    with caplog.at_level(logging.WARNING):
        warn_unknown_backend_keys("min_throughput_advantage = 0.5\nparticipation_chance = 0.2\n")
    assert not [r for r in caplog.records if "unknown field" in r.getMessage()]


def test_the_retired_keys_are_unknown(caplog):
    # The win model is gone with its knob, and the other two were renamed.
    # An operator config still naming the old keys gets the unknown-field
    # warning rather than silence.
    retired = {"min_win_probability", "slot_advantage", "explore_fraction"}
    assert not retired & DWAVE_CONFIG_KEYS
    with caplog.at_level(logging.WARNING):
        warn_unknown_backend_keys(
            "min_win_probability = 0.1\nslot_advantage = 0.5\nexplore_fraction = 0.2\n"
        )
    unknown = [r.getMessage() for r in caplog.records if "unknown field" in r.getMessage()]
    assert len(unknown) == 3
