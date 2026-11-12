"""Explicit family jobs: an ``ISING_SAMPLE`` job whose ``IsingProblem.family`` is set.

For a ``(family_id, revision)`` in :data:`ACCEPTED_FAMILIES` the miner is an
accepted-family solver (FAM-65). It checks the binding through the canonical
code in the ``quip_solver_core`` wheel, in this order, first failure wins:

1. the job uses the ``edges`` graph form, or code 15 (FAM-64);
2. the descriptor, codes 7 to 9 (code 10 never applies: no entry restricts
   its parameters);
3. the FAM-37 bounds, code 13, then the rebuilt instance digest, code 15,
   then the metadata, code 16 (``families.validate_instance``).

A pair the list does not hold makes the miner a generic solver for the job:
it solves the plain model and ignores the binding (FAM-69).

A bound job is sampled on the native qubits that ``FamilyBinding.node_ids``
names, in instance output order, so ``h``, ``j`` and every spin row keep the
canonical order. Energies are the plain-model energy, without the instance
offset (FAM-64); the coordinator adds the offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from quip_solver_core import AdmissionError, InstanceInvalid, miner_pb2
from quip_solver_core import families as wheel

#: The revisions this miner accepts as explicit family jobs, as
#: ``(family_id, revision)``. ``test_family_jobs`` and the ``--family`` gate
#: cover each one.
ACCEPTED_FAMILIES: Tuple[Tuple[str, int], ...] = (("quip/twisted-ice-dimer", 1),)


def accepted_family_support() -> List[miner_pb2.FamilySupport]:
    """``Capabilities.accepted_families``: every valid parameter set of each revision."""
    revisions: dict[str, List[int]] = {}
    for family_id, revision in ACCEPTED_FAMILIES:
        revisions.setdefault(family_id, []).append(revision)
    return [
        miner_pb2.FamilySupport(family_id=family_id, revisions=revs)
        for family_id, revs in revisions.items()
    ]


class FamilyJobRejected(Exception):
    """The binding failed a FAM-65 check: reject with ``reason`` and refund."""

    def __init__(self, reason: "miner_pb2.RejectReason", detail: str):
        super().__init__(detail)
        self.reason = reason


def explicit_reason(code: int) -> "miner_pb2.RejectReason":
    """The wire reason of a family code on an explicit job (FAM-42)."""
    if code == 10:
        return miner_pb2.FAMILY_UNSUPPORTED
    if code == 13:
        return miner_pb2.TOO_LARGE
    return miner_pb2.MALFORMED


@dataclass(frozen=True, eq=False)
class BoundInstance:
    """A validated explicit family instance, in canonical order."""

    family_id: str
    revision: int
    nonce: bytes
    #: Native qubit ids, in instance output order: the sampling nodes.
    node_ids: np.ndarray
    #: ``(E, 2)`` positions into ``node_ids``, in canonical edge order.
    positions: np.ndarray
    h_milli: np.ndarray
    j_milli: np.ndarray
    offset_milli: int
    metadata: bytes
    digest: bytes

    @property
    def edges(self) -> np.ndarray:
        """``(E, 2)`` native qubit ids."""
        return self.node_ids[self.positions]

    def energies_milli(self, spins: np.ndarray) -> np.ndarray:
        """The plain-model energy of each ±1 row, without the offset (FAM-64)."""
        s = np.asarray(spins, dtype=np.int64)
        u, v = self.positions[:, 0], self.positions[:, 1]
        return s @ self.h_milli + (s[:, u] * s[:, v]) @ self.j_milli


def descriptor_digest(descriptor: miner_pb2.FamilyDescriptor) -> bytes:
    """The FAM-18 descriptor digest, computed by the wheel (QUI-2267).

    Raises:
        AdmissionError: The descriptor fails codes 3 to 9.
    """
    return bytes(
        wheel.descriptor_digest(  # pyright: ignore[reportAttributeAccessIssue] - QUI-2267
            descriptor.family_id,
            descriptor.revision,
            bytes(descriptor.params),
            bytes(descriptor.params_digest),
        )
    )


def bind(
    ising: miner_pb2.IsingProblem,
    h_milli: Sequence[int],
    j_milli: Sequence[int],
    *,
    accepted: Sequence[Tuple[str, int]] = ACCEPTED_FAMILIES,
) -> Optional[BoundInstance]:
    """The validated instance of an explicit family job, or ``None``.

    ``None`` means a generic job: no binding, or a pair this miner does not
    accept. ``h_milli`` and ``j_milli`` are the decoded plain model.

    Raises:
        FamilyJobRejected: The binding fails a FAM-65 check.
    """
    if not ising.HasField("family"):
        return None
    binding = ising.family
    descriptor = binding.family
    if (descriptor.family_id, descriptor.revision) not in accepted:
        return None
    if ising.WhichOneof("graph") != "edges":
        raise FamilyJobRejected(
            miner_pb2.MALFORMED, "code 15: a family job must use the edges graph form"
        )
    edges = list(zip(ising.edges.u, ising.edges.v))
    try:
        own = descriptor_digest(descriptor)
        instance = wheel.validate_instance(
            descriptor.family_id,
            descriptor.revision,
            bytes(descriptor.params),
            bytes(descriptor.params_digest),
            own,
            bytes(binding.topology_order_digest),
            bytes(binding.nonce),
            list(binding.node_ids),
            edges,
            [int(v) for v in h_milli],
            [int(v) for v in j_milli],
            int(binding.offset_milli),
            bytes(binding.metadata),
            bytes(binding.instance_digest),
        )
    except (AdmissionError, InstanceInvalid) as exc:
        raise FamilyJobRejected(explicit_reason(exc.code), f"code {exc.code} {exc.name}") from None
    except (ValueError, OverflowError) as exc:
        # A value the binding cannot convert, such as a node id past u32.
        raise FamilyJobRejected(miner_pb2.MALFORMED, f"code 15: {exc}") from None
    return BoundInstance(
        family_id=descriptor.family_id,
        revision=int(descriptor.revision),
        nonce=bytes(binding.nonce),
        node_ids=np.asarray(instance.node_ids, dtype=np.int64),
        positions=np.asarray(instance.edges, dtype=np.intp).reshape(-1, 2),
        h_milli=np.asarray(instance.h, dtype=np.int64),
        j_milli=np.asarray(instance.j, dtype=np.int64),
        offset_milli=int(instance.offset_milli),
        metadata=bytes(instance.metadata),
        digest=bytes(instance.digest),
    )
