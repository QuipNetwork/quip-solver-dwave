"""A direct SAPI transport: one keep-alive session per slot, answer fetched by a blocking GET.

The cloud client turns one problem into three requests on three thread pools: a batched POST, a status
long-poll that a poll thread shares between up to 100 problems, and then a separate GET for the answer. The
answer therefore lands one full round trip after the problem is solved, and a new problem cannot start its
poll while all five poll threads are parked in long-polls for older ones.

SAPI also answers ``GET problems/<id>/?timeout=N`` (N up to 30 s) as soon as that problem finishes, with the
answer inline. A slot here POSTs one problem, then holds that GET open, so the answer arrives half a round
trip after the solve. At equal throughput this needs about a third of the problems in flight at SAPI (8
instead of 24 on a fake SAPI with an 80 ms round trip), which shortens each problem's queue wait and shrinks
the access time a Cancel can no longer stop.

A slot whose answer lands posts the next queued problem at once; the caller's thread has already
deflated it. The problems SAPI holds therefore stay near ``slots`` minus those in transit: about
rate x (POST time + half a round trip), 3 to 4 at 16 per second and an 80 ms round trip. To keep at
least 30 problems queued or running at SAPI, use 48 slots (a fake SAPI kept 42 or more). SAPI rejects problems beyond
1,000 per member in a solver queue.

Problems wait in a local queue until a slot is free. A Cancel of one that is still local costs nothing and
needs no request. Sessions come from the cloud client's ``create_session``, so the endpoint, token, proxy,
certificate and retry settings are the operator's D-Wave configuration, unchanged.

:class:`DirectProblem` exposes what :func:`quip_solver_dwave.answer.answer_view` and
:func:`quip_solver_dwave.ocean.server_timestamps` read off a cloud ``Future`` (``samples``, ``energies``,
``variables``, ``num_occurrences``, ``timing``, ``_message``), plus ``cancel`` and ``done``. Failures are the
cloud client's own exception classes, so ``is_solver_offline`` keeps working.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import zlib
from typing import Any, List, Optional

import numpy as np
import orjson

logger = logging.getLogger(__name__)

PROBLEMS_MEDIA = "application/vnd.dwave.sapi.problems+json"
PROBLEM_MEDIA = "application/vnd.dwave.sapi.problem+json"

# SAPI caps a blocking status request at 30 s.
POLL_TIMEOUT_S = 30


class DirectProblem:
    """One problem on the direct transport. Reads block until SAPI has answered."""

    def __init__(self, transport: "DirectTransport", body: bytes):
        self._transport = transport
        # The deflated one-element list that the POST sends.
        self.body: Optional[bytes] = body
        self.id: Optional[str] = None
        self._message: Optional[dict] = None
        self._exc: Optional[BaseException] = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._cancel_requested = False
        self._decoded: Optional[dict] = None

    # -- state set by the transport ------------------------------------------
    def _set_id(self, problem_id: str) -> bool:
        """Record the SAPI id; True when a Cancel arrived first and must be sent now."""
        with self._lock:
            self.id = problem_id
            return self._cancel_requested

    def _resolve(self, message: Optional[dict], exc: Optional[BaseException]) -> None:
        self._message, self._exc = message, exc
        self.body = None
        self._done.set()

    # -- the Future surface ---------------------------------------------------
    def done(self) -> bool:
        return self._done.is_set()

    def cancel(self) -> bool:
        """Drop the problem: locally if it was never posted, else ask SAPI to cancel it."""
        with self._lock:
            if self._done.is_set() or self._cancel_requested:
                return False
            self._cancel_requested = True
            problem_id = self.id
        if problem_id is not None:
            self._transport.cancel_remote(problem_id)
        return True

    def _wait(self) -> dict:
        self._done.wait()
        if self._exc is not None:
            raise self._exc
        assert self._message is not None
        return self._message

    def _answer(self) -> dict:
        # Wait outside the lock: cancel() takes it, and must not queue behind a reader.
        message = self._wait()
        with self._lock:
            if self._decoded is None:
                self._decoded = decode_answer(message)
            return self._decoded

    @property
    def samples(self) -> np.ndarray:
        return self._answer()["solutions"]

    @property
    def energies(self) -> np.ndarray:
        return self._answer()["energies"]

    @property
    def num_occurrences(self) -> Optional[np.ndarray]:
        return self._answer().get("num_occurrences")

    @property
    def variables(self) -> List[int]:
        return [int(v) for v in self._answer()["active_variables"]]

    @property
    def timing(self) -> dict:
        return dict(self._wait().get("answer", {}).get("timing") or {})


def decode_answer(message: dict) -> dict:
    """A ``qp`` answer as arrays, in the cloud decoder's shape: solutions padded to every qubit label.

    ``solutions`` is (reads, num_variables) int8 of +/-1 in the active columns and 0 elsewhere, which is
    what ``answer_view`` selects from with the active labels.
    """
    answer = message.get("answer") or {}
    if answer.get("format") != "qp":
        raise ValueError(f"unsupported answer format {answer.get('format')!r}")
    energies = np.frombuffer(base64.b64decode(answer["energies"]), dtype="<f8")
    active = np.frombuffer(base64.b64decode(answer["active_variables"]), dtype="<i4")
    reads = len(energies)
    bits = np.unpackbits(np.frombuffer(base64.b64decode(answer["solutions"]), dtype=np.uint8))
    bits = bits.reshape(reads, -1)[:, : len(active)] if reads else bits.reshape(0, len(active))
    solutions = np.zeros((reads, int(answer["num_variables"])), dtype=np.int8)
    solutions[:, active] = bits.astype(np.int8) * 2 - 1
    out = {"solutions": solutions, "energies": energies + float(answer.get("offset", 0) or 0),
           "active_variables": active}
    if "num_occurrences" in answer:
        out["num_occurrences"] = np.frombuffer(base64.b64decode(answer["num_occurrences"]), dtype="<i4")
    return out


class DirectTransport:
    """``slots`` worker threads, each with its own session and at most one problem at SAPI."""

    def __init__(self, client: Any, *, slots: int, compress_level: int = 6):
        self._client = client
        self._compress_level = compress_level
        self._queue: "queue.Queue[Optional[DirectProblem]]" = queue.Queue()
        self._cancel_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._threads = [
            threading.Thread(target=self._slot, name=f"sapi-direct-{i}", daemon=True)
            for i in range(max(1, slots))
        ]
        self._threads.append(threading.Thread(target=self._canceller, name="sapi-direct-cancel", daemon=True))
        for t in self._threads:
            t.start()

    def submit(self, body: bytes) -> DirectProblem:
        """Queue a problem. The caller's thread deflates it, so a slot that frees up posts at once."""
        problem = DirectProblem(self, zlib.compress(b"[" + body + b"]", self._compress_level))
        self._queue.put(problem)
        return problem

    def cancel_remote(self, problem_id: str) -> None:
        self._cancel_q.put(problem_id)

    def close(self) -> None:
        """Stop the threads once the problems already queued have finished."""
        for _ in self._threads[:-1]:
            self._queue.put(None)
        self._cancel_q.put(None)

    def _session(self, media: str):
        session = self._client.create_session()
        session.set_accept(media_type=media, accept_version="~=3.0", ask_version="3.0.0")
        return session

    def _slot(self) -> None:
        from dwave.cloud.exceptions import CanceledFutureError

        post = self._session(PROBLEMS_MEDIA)
        get = self._session(PROBLEM_MEDIA)
        try:
            while True:
                problem = self._queue.get()
                if problem is None:
                    return
                if problem._cancel_requested:
                    problem._resolve(None, CanceledFutureError())
                    continue
                try:
                    problem._resolve(self._run(post, get, problem), None)
                except Exception as exc:  # noqa: BLE001 - every failure belongs to its problem
                    problem._resolve(None, exc)
        finally:
            post.close()
            get.close()

    def _run(self, post, get, problem: DirectProblem) -> dict:
        from dwave.cloud.client import Client

        assert problem.body is not None
        message = Client._sapi_request(
            post.post, "problems/", data=problem.body, headers={"Content-Encoding": "deflate"}
        )
        status = message[0] if isinstance(message, list) else message
        while True:
            self._check(status)
            if problem.id is None and problem._set_id(status["id"]):
                self.cancel_remote(status["id"])
            if status.get("status") == "COMPLETED" and "answer" in status:
                return status
            status = Client._sapi_request(get.get, f"problems/{problem.id}/", params={"timeout": POLL_TIMEOUT_S})

    @staticmethod
    def _check(message: Any) -> None:
        """Raise what the cloud client raises for an error, a failure or a cancel."""
        from dwave.cloud.exceptions import (
            CanceledFutureError,
            InvalidAPIResponseError,
            SolverFailureError,
            SolverOfflineError,
        )

        if not isinstance(message, dict):
            raise InvalidAPIResponseError("Unexpected format of problem description response")
        if "error_code" in message and "error_msg" in message:
            raise SolverFailureError(message["error_msg"])
        status = message.get("status")
        if status is None or "id" not in message:
            raise InvalidAPIResponseError("'status' or 'id' missing in problem description response")
        if status == "CANCELLED":
            raise CanceledFutureError()
        if status == "FAILED":
            errmsg = message.get("error_message", "An unknown error has occurred.")
            if "solver is offline" in errmsg.lower():
                raise SolverOfflineError(errmsg)
            raise SolverFailureError(errmsg)

    def _canceller(self) -> None:
        from dwave.cloud.client import Client

        session = self._session(PROBLEMS_MEDIA)
        try:
            while True:
                problem_id = self._cancel_q.get()
                if problem_id is None:
                    return
                try:
                    Client._sapi_request(session.delete, "problems/", data=orjson.dumps([problem_id]))
                except Exception as exc:  # noqa: BLE001 - a lost cancel costs access time, not correctness
                    logger.warning("[QPU] cancel of %s failed: %s", problem_id, exc)
        finally:
            session.close()
