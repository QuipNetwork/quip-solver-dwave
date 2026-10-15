"""Build a SAPI ``qp`` submission payload from numpy, skipping the dicts.

Ocean's submit path costs ~27.5 ms of GIL-bound CPU per job at production size
(4577 nodes, 41514 couplers). About 13 ms goes into turning the job's wire
arrays into ``h``/``J`` dicts, and the rest goes into
:func:`dwave.cloud.coders.encode_problem_as_qp` turning those dicts straight
back into two dense float64 arrays, one dict lookup per solver coupler.

Neither dict is wanted for itself. The payload is positional: biases in the
solver's own qubit order, couplings in the solver's own coupler order. Both
orders are fixed for the session, and so is the job graph, so the mapping
between them is planned once (:meth:`QpEncoder.plan`) and every later job is
two scatters and a base64 of the raw buffer.

``encode_problem_as_qp`` is the specification for what this produces, and
``test_qp_encoding`` asserts byte equality against it.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Union

import numpy as np

EncodedQP = Dict[str, Union[str, float]]


def build_submission_body(
    solver, data: "EncodedQP", params: Dict[str, object], label: object = None
) -> bytes:
    """Wrap an encoded problem in the envelope SAPI expects.

    Mirrors ``dwave.cloud.solver.StructuredSolver._sample`` for everything
    except the encoding itself: the solver's own defaults sit under the
    caller's parameters, the solver validates and formats them, and the result
    is the JSON body its transport already knows how to submit.

    Parameter validation stays here rather than being skipped for speed. An
    unknown parameter costs a coordinator credit and a cloud round trip to
    discover otherwise, and it is a dictionary lookup per key on a dict of 15.

    Raises:
        KeyError: a parameter the solver does not accept.
    """
    import orjson

    combined = dict(solver._params)
    combined.update(params)
    for key in combined:
        if key not in solver.parameters and not str(key).startswith("x_"):
            raise KeyError(f"{key} is not a parameter of this solver")
    # Ocean mutates the dict in place; deferring keeps one source of truth for
    # per-type parameter transforms.
    solver._format_params("ising", combined)

    body = {
        "solver": solver.identity.dict(),
        "data": data,
        "type": "ising",
        "params": combined,
    }
    if label is not None:
        body["label"] = label
    return orjson.dumps(body, option=orjson.OPT_SERIALIZE_NUMPY)


@dataclass(frozen=True)
class GraphPlan:
    """Where one job graph's biases land in the solver's payload arrays.

    Valid for as long as the job graph and the solver do, which for a session
    is every job. Holding it is what makes :meth:`QpEncoder.encode` cheap.
    """

    lin_template: np.ndarray
    lin_slots: np.ndarray
    quad_slots: np.ndarray
    quad_mask: np.ndarray
    quad_width: int
    has_duplicate_couplers: bool


class QpEncoder:
    """Encodes jobs for one solver's fixed qubit and coupler ordering."""

    def __init__(
        self,
        encoding_qubits: Sequence[int],
        encoding_couplers: Sequence[Tuple[int, int]],
    ):
        self._qubits = np.asarray(encoding_qubits, dtype=np.int64)
        couplers = np.asarray(
            [tuple(c) for c in encoding_couplers], dtype=np.int64
        ).reshape(-1, 2)
        self._couplers = couplers

        # Qubit label -> slot in the linear array. A dense table beats a dict:
        # labels are small dense integers on every solver this backend
        # supports, and it is built once.
        width = int(max(self._qubits.max(initial=-1), couplers.max(initial=-1))) + 1
        self._width = width
        self._slot_of_qubit = np.full(width, -1, dtype=np.int64)
        self._slot_of_qubit[self._qubits] = np.arange(len(self._qubits))

        # Undirected coupler -> slot in the quadratic array, keyed on the
        # ordered pair so a job may present either orientation.
        lo = np.minimum(couplers[:, 0], couplers[:, 1])
        hi = np.maximum(couplers[:, 0], couplers[:, 1])
        keys = lo * width + hi
        self._key_order = np.argsort(keys)
        self._sorted_keys = keys[self._key_order]

    def plan(self, nodes: np.ndarray, edges: np.ndarray) -> GraphPlan:
        """Resolve a job graph against the solver's ordering, once.

        Raises:
            ValueError: an edge names a coupler the solver does not have.
                Submitting it would make SAPI reject the whole problem, so
                naming the coupler here beats a ProblemStructureError later.
        """
        nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

        # A real chip's qubit labels have holes: Advantage2_system1 reports
        # 4577 qubits with a maximum label of 4799. A label in a hole has no
        # slot, and _slot_of_qubit answers -1 for it, which numpy would
        # happily write to the *last* slot and silently corrupt that qubit's
        # bias. Reject it here instead, where the label can be named.
        mentioned = np.concatenate(
            [a for a in (nodes, edges.reshape(-1)) if a.size]
        ) if (nodes.size or edges.size) else np.empty(0, dtype=np.int64)
        if mentioned.size:
            out_of_range = (mentioned < 0) | (mentioned >= self._width)
            unknown = out_of_range.copy()
            in_range = ~out_of_range
            unknown[in_range] = self._slot_of_qubit[mentioned[in_range]] < 0
            if np.any(unknown):
                bad = int(mentioned[np.nonzero(unknown)[0][0]])
                raise ValueError(
                    f"qubit {bad} is not on the solver; submitting it would "
                    "make SAPI reject the whole problem"
                )

        # Active means "carries a bias or a coupling", which decides whether
        # an unmentioned qubit encodes as 0 or as NaN.
        active = np.zeros(self._width, dtype=bool)
        if nodes.size:
            active[nodes] = True
        if edges.size:
            active[edges.reshape(-1)] = True

        lin_template = np.where(active[self._qubits], 0.0, np.nan)
        lin_slots = (
            self._slot_of_qubit[nodes] if nodes.size else np.empty(0, dtype=np.int64)
        )

        if edges.size:
            lo = np.minimum(edges[:, 0], edges[:, 1])
            hi = np.maximum(edges[:, 0], edges[:, 1])
            keys = lo * self._width + hi
            pos = np.searchsorted(self._sorted_keys, keys)
            # searchsorted gives an insertion point: len() past the end, or a
            # wrong neighbour for a key that is simply absent. Both mean the
            # chip does not have this coupler.
            pos = np.clip(pos, 0, max(len(self._sorted_keys) - 1, 0))
            missing = (
                np.ones(len(keys), dtype=bool)
                if len(self._sorted_keys) == 0
                else self._sorted_keys[pos] != keys
            )
            if np.any(missing):
                bad = edges[np.nonzero(missing)[0][0]]
                raise ValueError(
                    f"coupler ({int(bad[0])}, {int(bad[1])}) is not on the "
                    "solver; submitting it would make SAPI reject the whole "
                    "problem"
                )
            quad_slots = self._key_order[pos]
        else:
            quad_slots = np.empty(0, dtype=np.int64)

        quad_mask = (
            active[self._couplers[:, 0]] & active[self._couplers[:, 1]]
            if len(self._couplers)
            else np.zeros(0, dtype=bool)
        )
        return GraphPlan(
            lin_template=lin_template,
            lin_slots=lin_slots,
            quad_slots=quad_slots,
            quad_mask=quad_mask,
            quad_width=len(self._couplers),
            has_duplicate_couplers=quad_slots.size
            != np.unique(quad_slots).size,
        )

    def encode(self, plan: GraphPlan, h: np.ndarray, j: np.ndarray) -> EncodedQP:
        """Encode one job's biases against a plan built for its graph."""
        lin = plan.lin_template.copy()
        if plan.lin_slots.size:
            lin[plan.lin_slots] = h

        quad_full = np.zeros(plan.quad_width, dtype=np.float64)
        if plan.quad_slots.size:
            if plan.has_duplicate_couplers:
                # The reference sums a coupler given in both orientations
                # rather than letting one overwrite the other. add.at is the
                # slow path, so it only runs when a graph actually needs it.
                np.add.at(quad_full, plan.quad_slots, j)
            else:
                quad_full[plan.quad_slots] = j
        quad = quad_full[plan.quad_mask]

        return {
            "format": "qp",
            "lin": base64.b64encode(lin.astype("<f8").tobytes()).decode("utf-8"),
            "quad": base64.b64encode(quad.astype("<f8").tobytes()).decode("utf-8"),
            "offset": 0.0,
        }
