"""Per-job timing the session loop needs beyond what D-Wave bills.

SAPI stamps every problem with submitted_on and solved_on. The Ocean SDK keeps
the raw problem JSON on Future._message and never parses those two fields, so
this backend reads them there. Their difference is D-Wave's own definition of
service time, queues included.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from quip_solver_dwave.ocean import OceanSampler, server_timestamps


class FakeFuture:
    def __init__(self, message=None):
        self.samples = np.array([[1, -1], [-1, -1]], dtype=np.int8)
        self.variables = [10, 20]
        self.energies = [-3.0, -1.0]
        self.num_occurrences = [1, 1]
        self.timing = {"qpu_programming_time": 33_000, "qpu_sampling_time": 10_200}
        if message is not None:
            self._message = message


SUBMITTED = "2026-09-11T14:48:36.052206Z"
SOLVED = "2026-09-11T14:48:39.087126Z"


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def test_server_timestamps_are_parsed_from_the_raw_message():
    fut = FakeFuture({"submitted_on": SUBMITTED, "solved_on": SOLVED})
    submitted, solved = server_timestamps(fut)
    assert submitted is not None
    assert solved is not None
    assert submitted == pytest.approx(_epoch(SUBMITTED))
    assert solved == pytest.approx(_epoch(SOLVED))
    assert solved - submitted == pytest.approx(3.03492, abs=1e-5)


def test_a_message_without_timestamps_yields_none():
    assert server_timestamps(FakeFuture({})) == (None, None)
    assert server_timestamps(FakeFuture()) == (None, None)
    assert server_timestamps(FakeFuture({"submitted_on": "not a date"})) == (None, None)


def test_sample_carries_server_timestamps_and_the_inflight_count(monkeypatch):
    s = OceanSampler(mock=False)
    fut = FakeFuture({"submitted_on": SUBMITTED, "solved_on": SOLVED})
    monkeypatch.setattr(s, "_submit_sync", lambda *a, **k: fut)
    # One problem already on the QPU when this one is handed over.
    s._inflight[b"other"] = object()

    result = s.sample(
        np.array([10, 20]),
        np.zeros(2),
        np.array([[10, 20]]),
        np.array([1.0]),
        num_reads=2,
    )

    assert result.submitted_on_s == pytest.approx(_epoch(SUBMITTED))
    assert result.solved_on_s == pytest.approx(_epoch(SOLVED))
    assert result.inflight_at_submit == 1
    # The session loop only sees SamplerMeta, so the same two facts ride in
    # its free-form extra map.
    assert result.extra["inflight"] == "1"
    assert result.extra["sapi_ms"] == "3035"


def test_extra_omits_sapi_time_when_the_server_gave_none(monkeypatch):
    s = OceanSampler(mock=False)
    monkeypatch.setattr(s, "_submit_sync", lambda *a, **k: FakeFuture({}))
    result = s.sample(
        np.array([10, 20]), np.zeros(2), np.array([[10, 20]]), np.array([1.0])
    )
    assert "sapi_ms" not in result.extra
    assert result.extra["inflight"] == "0"
