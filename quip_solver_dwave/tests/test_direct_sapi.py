"""The direct SAPI transport against a local HTTP stand-in for SAPI.

A real ``dwave.cloud.Client`` provides the sessions, so headers, media-type
negotiation and response validation are the SDK's own. Only the server is
fake: it records every request and answers from a per-test script.
"""

from __future__ import annotations

import base64
import json
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip("dwave.cloud")

from dwave.cloud import Client  # noqa: E402
from dwave.cloud.exceptions import CanceledFutureError, SolverOfflineError  # noqa: E402

from quip_solver_dwave.answer import answer_view  # noqa: E402
from quip_solver_dwave.direct_sapi import DirectTransport  # noqa: E402
from quip_solver_dwave.ocean import is_solver_offline, server_timestamps  # noqa: E402

ACTIVE = np.array([0, 2, 3], dtype=np.int32)
SPINS = np.array([[1, -1, 1], [-1, -1, 1]], dtype=np.int8)
TIMING = {"qpu_access_time": 36540.0, "qpu_programming_time": 34631.6, "qpu_sampling_time": 1908.4}


def _answer() -> dict:
    bits = (SPINS > 0).astype(np.uint8)
    return {
        "format": "qp",
        "num_variables": 5,
        "solutions": base64.b64encode(np.packbits(bits, axis=1).tobytes()).decode(),
        "energies": base64.b64encode(np.array([-2.0, -1.0], dtype="<f8").tobytes()).decode(),
        "num_occurrences": base64.b64encode(np.ones(2, dtype="<i4").tobytes()).decode(),
        "active_variables": base64.b64encode(ACTIVE.astype("<i4").tobytes()).decode(),
        "timing": TIMING,
    }


class _Sapi:
    """Script: what POST answers, and how many GETs return PENDING before the answer."""

    def __init__(self, post_status="PENDING", pending_gets=1, failure=None, dropped_gets=0):
        self.post_status = post_status
        self.pending_gets = pending_gets
        # The first GETs that the server closes without an answer, as a broken connection does.
        self.dropped_gets = dropped_gets
        self.drops = 0
        self.failure = failure
        self.posts: list = []
        self.gets: list = []
        self.deletes: list = []
        self.gate = threading.Event()
        self.gate.set()
        self.lock = threading.Lock()

    def status(self, done: bool) -> dict:
        msg = {"id": "p-1", "type": "ising", "status": "PENDING",
               "submitted_on": "2026-11-14T00:00:00.000000Z", "solver": {"name": "x"}}
        if self.deletes:
            msg.update(status="CANCELLED")
        elif self.failure is not None:
            msg.update(status="FAILED", error_message=self.failure)
        elif done:
            msg.update(status="COMPLETED", solved_on="2026-11-14T00:00:00.250000Z", answer=_answer())
        return msg


def _serve(sapi: _Sapi):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - the base class names it
            pass

        def _send(self, obj, media):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", f"{media}; version=3.0.0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            sapi.gate.wait()
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding") == "deflate":
                raw = zlib.decompress(raw)
            with sapi.lock:
                sapi.posts.append(json.loads(raw))
            done = sapi.post_status == "COMPLETED"
            self._send([sapi.status(done)], "application/vnd.dwave.sapi.problems+json")

        def do_GET(self):
            with sapi.lock:
                drop = sapi.drops < sapi.dropped_gets
                sapi.drops += drop
            if drop:
                self.close_connection = True
                self.connection.close()
                return
            with sapi.lock:
                sapi.gets.append(self.path)
                done = len(sapi.gets) > sapi.pending_gets
            self._send(sapi.status(done), "application/vnd.dwave.sapi.problem+json")

        def do_DELETE(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with sapi.lock:
                sapi.deletes.append(json.loads(raw))
            self._send([], "application/vnd.dwave.sapi.problems+json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def sapi_factory():
    servers, clients, transports = [], [], []

    def make(slots=1, **script):
        sapi = _Sapi(**script)
        server = _serve(sapi)
        client = Client(endpoint=f"http://127.0.0.1:{server.server_address[1]}/sapi/v2/",
                        token="test-token", cache_enabled=False)
        transport = DirectTransport(client, slots=slots)
        servers.append(server)
        clients.append(client)
        transports.append(transport)
        return sapi, transport

    yield make
    for t in transports:
        t.close()
    for c in clients:
        c.close()
    for s in servers:
        s.shutdown()


def test_a_pending_problem_is_answered_by_the_blocking_get(sapi_factory):
    sapi, transport = sapi_factory(pending_gets=1)
    problem = transport.submit(b'{"type": "ising"}')

    view = answer_view(problem)

    assert sapi.posts == [[{"type": "ising"}]]
    assert len(sapi.gets) == 2
    assert all(g.startswith("/sapi/v2/problems/p-1/?timeout=30") for g in sapi.gets)
    assert view.variables == [0, 2, 3]
    assert view.spins.tolist() == SPINS.tolist()
    assert view.energies == [-2.0, -1.0]
    assert view.reads == 2
    assert view.access_time_us == int(TIMING["qpu_programming_time"]) + int(TIMING["qpu_sampling_time"])
    submitted, solved = server_timestamps(problem)
    assert solved is not None and submitted is not None
    assert solved - submitted == pytest.approx(0.25)


def test_a_broken_answer_fetch_is_retried_and_the_problem_is_not_lost(sapi_factory):
    # The problem is already billed once SAPI has it, so a dropped connection on the fetch must not fail it.
    # The retries are the cloud client's own (``http_retry_total``), carried by ``create_session``.
    sapi, transport = sapi_factory(pending_gets=0, dropped_gets=3)

    view = answer_view(transport.submit(b"{}"))

    assert sapi.drops == 3
    assert len(sapi.posts) == 1
    assert view.spins.tolist() == SPINS.tolist()


def test_an_answer_on_the_post_needs_no_get(sapi_factory):
    sapi, transport = sapi_factory(post_status="COMPLETED")
    problem = transport.submit(b"{}")

    assert answer_view(problem).reads == 2
    assert sapi.gets == []


def test_an_offline_solver_raises_what_the_cloud_client_raises(sapi_factory):
    _, transport = sapi_factory(failure="Solver is offline for maintenance")
    problem = transport.submit(b"{}")

    with pytest.raises(SolverOfflineError) as info:
        answer_view(problem)
    assert is_solver_offline(info.value)


def test_a_cancel_before_the_post_never_reaches_sapi(sapi_factory):
    sapi, transport = sapi_factory(slots=1, pending_gets=0)
    sapi.gate.clear()  # hold the one slot inside its POST
    first = transport.submit(b'{"n": 1}')
    second = transport.submit(b'{"n": 2}')

    assert second.cancel()
    sapi.gate.set()

    assert answer_view(first).reads == 2
    with pytest.raises(CanceledFutureError):
        answer_view(second)
    assert sapi.posts == [[{"n": 1}]]
    assert sapi.deletes == []


def test_a_cancel_after_the_post_asks_sapi_to_drop_it(sapi_factory):
    sapi, transport = sapi_factory(pending_gets=10**6)
    problem = transport.submit(b"{}")
    while not sapi.gets:  # the problem has an id once its first GET is out
        threading.Event().wait(0.01)

    assert problem.cancel()
    for _ in range(500):
        if sapi.deletes:
            break
        threading.Event().wait(0.01)

    assert sapi.deletes == [["p-1"]]
    with pytest.raises(CanceledFutureError):
        answer_view(problem)


def test_the_sampler_routes_cloud_problems_through_the_slots(sapi_factory):
    from quip_solver_dwave.ocean import OceanSampler

    sapi, transport = sapi_factory(pending_gets=10**6)

    class _Solver:
        client = None

    sampler = OceanSampler(sampler=object(), mock=False)
    sampler.use_direct_transport(1)
    sampler._direct = transport  # the transport the fixture built on its client
    problem = sampler._submit_encoded(_Solver(), b"{}", b"key-1")
    while not sapi.gets:
        threading.Event().wait(0.01)

    assert sampler.cancel_inflight([b"key-1"]) == 1
    with pytest.raises(CanceledFutureError):
        answer_view(problem)
    assert sapi.deletes == [["p-1"]]


def test_sapi_slots_is_read_from_the_backend_toml():
    from quip_solver_dwave.config import DEFAULT_SAPI_SLOTS, sapi_slots_from_toml

    assert sapi_slots_from_toml("sapi_slots = 12\n") == 12
    assert sapi_slots_from_toml("sapi_slots = 0\n") == 0
    assert sapi_slots_from_toml("sapi_slots = true\n") == 0
    assert sapi_slots_from_toml("queue_depth = 96\n") == DEFAULT_SAPI_SLOTS
    assert sapi_slots_from_toml("") == DEFAULT_SAPI_SLOTS


def test_programming_thermalization_defaults_to_zero():
    from quip_solver_dwave.config import programming_thermalization_us_from_toml

    assert programming_thermalization_us_from_toml("") == 0
    assert programming_thermalization_us_from_toml("programming_thermalization_us = 1000\n") == 1000
