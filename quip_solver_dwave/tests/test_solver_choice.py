"""A coordinator chooses the Advantage2 of each miner in ``backend_toml``.

The miner connects at Configure, so the ``solver`` and ``region`` keys reach
the sampler before the first SAPI call, and one coordinator config can point
a miner at Advantage2_system1 or the smaller Advantage2_system4.
"""

from __future__ import annotations

from typing import List

from quip_solver_core import miner_pb2

from quip_solver_dwave.budget import DWAVE_CONFIG_KEYS
from quip_solver_dwave.config import solver_choice_from_toml
from quip_solver_dwave.ocean import OceanSampler
from quip_solver_dwave.tests.test_solver_offline import (
    QUEUE_DEPTH,
    FlakySampler,
    _run,
    _wait_for,
    usage_db,  # noqa: F401  (pytest fixture)
)


def test_solver_and_region_come_from_backend_toml():
    toml = 'solver = "Advantage2_system4"\nregion = "na-east-1"\n'
    assert solver_choice_from_toml(toml) == ("Advantage2_system4", "na-east-1")
    assert solver_choice_from_toml("") == (None, None)
    assert solver_choice_from_toml('solver = "Advantage2_system1"') == ("Advantage2_system1", None)
    assert {"solver", "region"} <= DWAVE_CONFIG_KEYS


def test_a_solver_that_is_not_a_name_is_ignored():
    assert solver_choice_from_toml('solver = 4\nregion = ""') == (None, None)


class _Recorder:
    def __init__(self):
        self.calls: List[tuple] = []


def _sampler_recording_connects(monkeypatch):
    s = OceanSampler(mock=False)
    rec = _Recorder()

    def connect(solver_name, region, token):
        rec.calls.append((solver_name, region))
        return object()

    monkeypatch.setattr(s, "_connect_real", connect)
    return s, rec


def test_the_chosen_solver_and_region_reach_the_connect(monkeypatch):
    s, rec = _sampler_recording_connects(monkeypatch)
    s.use_solver("Advantage2_system4", "na-east-1")
    s.ensure_connected()
    assert rec.calls == [("Advantage2_system4", "na-east-1")]


def test_an_unset_key_keeps_the_choice_already_made(monkeypatch):
    s, rec = _sampler_recording_connects(monkeypatch)
    s._solver_name = "Advantage2_system1"
    s.use_solver(None, "na-west-1")
    s.ensure_connected()
    assert rec.calls == [("Advantage2_system1", "na-west-1")]


def test_a_connected_sampler_keeps_its_solver(monkeypatch):
    s, rec = _sampler_recording_connects(monkeypatch)
    s.use_solver("Advantage2_system1", None)
    s.ensure_connected()
    s.use_solver("Advantage2_system4", "na-east-1")
    assert (s._solver_name, s._region) == ("Advantage2_system1", None)
    assert rec.calls == [("Advantage2_system1", None)]


class ChoosingSampler(FlakySampler):
    """Records the solver choice and whether it came before the connect."""

    def __init__(self):
        super().__init__(offline_jobs=0)
        self.events: List[tuple] = []

    def use_solver(self, solver, region) -> None:
        self.events.append(("use_solver", solver, region))

    def ensure_connected(self) -> None:
        self.events.append(("connect",))


def test_configure_hands_the_solver_choice_over_before_it_connects(monkeypatch, usage_db):  # noqa: F811
    sampler = ChoosingSampler()

    def script(sent):
        yield miner_pb2.CoordMsg(welcome=miner_pb2.Welcome(protocol_version=2))
        yield miner_pb2.CoordMsg(
            configure=miner_pb2.Configure(
                queue_depth=QUEUE_DEPTH,
                idle_timeout_s=300,
                backend_toml=(
                    f'budget = "3000s"\nusage_db = "{usage_db}"\n'
                    'solver = "Advantage2_system4"\nregion = "na-east-1"\n'
                ),
            )
        )
        _wait_for(sent, "ready", 1)
        yield miner_pb2.CoordMsg(shutdown=miner_pb2.Shutdown(grace_ms=100))

    _run(monkeypatch, sampler, usage_db, script)
    assert sampler.events[:2] == [("use_solver", "Advantage2_system4", "na-east-1"), ("connect",)]
