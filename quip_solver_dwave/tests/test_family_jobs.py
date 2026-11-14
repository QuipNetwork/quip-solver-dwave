"""Explicit family jobs (DW-1): FAM-65 checks, native order, exact energies.

The instances come from the wheel's own generator on a Zephyr z(2, 2) graph
whose qubit ids start at 100, so a job that sampled on dense positions in
place of the native ids would put every spin on the wrong qubit.
"""

from __future__ import annotations

import os
import stat
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from dwave.graphs import zephyr_graph

from quip_solver_core import AdmissionError, miner_pb2, wire
from quip_solver_core import families as wheel

from quip_solver_dwave import families, qpp
from quip_solver_dwave.families import BoundInstance, FamilyJobRejected, bind
from quip_solver_dwave.job import handle_job
from quip_solver_dwave.ocean import OceanSampler, SampleResult
from quip_solver_dwave.session_loop import capabilities_json, capabilities_message

TWISTED = "quip/twisted-ice-dimer"
P_ADJ = bytes(
    [1, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0x2D, 0, 4, 0, 4, 0, 0, 0, 6, 0, 0, 0]
)
P_W1000 = P_ADJ[:22] + bytes([0xE8, 0x03]) + P_ADJ[24:]
BASE = 100
NONCE = bytes(range(32))


@lru_cache(maxsize=None)
def topology():
    g = zephyr_graph(2, 2)
    nodes = [BASE + n for n in sorted(g.nodes)]
    edges = sorted((BASE + min(u, v), BASE + max(u, v)) for u, v in g.edges)
    return nodes, edges


@lru_cache(maxsize=None)
def instance(params: bytes = P_ADJ, nonce: bytes = NONCE):
    nodes, edges = topology()
    return wheel.generate(TWISTED, 1, params, wheel.params_digest(params), nodes, edges, nonce)


def binding(inst=None, **kw: Any) -> miner_pb2.FamilyBinding:
    inst = inst or instance()
    params = kw.pop("params", P_ADJ)
    out = miner_pb2.FamilyBinding(
        family=miner_pb2.FamilyDescriptor(
            family_id=TWISTED, revision=1, params=params, params_digest=wheel.params_digest(params)
        ),
        nonce=inst.nonce,
        instance_digest=inst.digest,
        metadata=inst.metadata,
        topology_order_digest=inst.topology_order_digest,
        node_ids=list(inst.node_ids),
        offset_milli=inst.offset_milli,
    )
    for key, value in kw.items():
        if key in ("family_id", "revision", "params_digest"):
            setattr(out.family, key, value)
        else:
            setattr(out, key, value)
    return out


def ising(inst=None, *, encoding: str = "i32", **kw: Any) -> miner_pb2.IsingProblem:
    inst = inst or instance()
    h, j = list(inst.h), list(inst.j)
    coeffs: Dict[str, Any]
    if encoding == "i32":
        coeffs = dict(encoding=miner_pb2.COEFFICIENT_ENCODING_I32, scale=1000,
                      h=wire.encode_i32_le(h), j=wire.encode_i32_le(j))
    elif encoding == "i16":
        coeffs = dict(encoding=miner_pb2.COEFFICIENT_ENCODING_I16, scale=1000,
                      h=np.asarray(h, "<i2").tobytes(), j=np.asarray(j, "<i2").tobytes())
    else:
        coeffs = dict(encoding=miner_pb2.COEFFICIENT_ENCODING_F64, scale=0,
                      h=(np.asarray(h) / 1000).astype("<f8").tobytes(),
                      j=(np.asarray(j) / 1000).astype("<f8").tobytes())
    edges = list(inst.edges)
    problem = miner_pb2.IsingProblem(
        edges=miner_pb2.EdgeList(u=[u for u, _ in edges], v=[v for _, v in edges]),
        family=kw.pop("family", binding(inst)),
        num_reads=3,
        **coeffs,
    )
    for key, value in kw.items():
        setattr(problem, key, value)
    return problem


def decoded(problem: miner_pb2.IsingProblem):
    from quip_solver_dwave.job import decode_coefficients_milli

    return decode_coefficients_milli(problem)


def bind_problem(problem: miner_pb2.IsingProblem) -> Optional[BoundInstance]:
    h, j = decoded(problem)
    return bind(problem, h, j)


# -- capabilities ----------------------------------------------------------------


def test_capabilities_list_the_accepted_revision_apart_from_generation():
    caps = capabilities_message()
    assert [(f.family_id, list(f.revisions)) for f in caps.accepted_families] == [(TWISTED, [1])]
    assert "acceptedFamilies" in capabilities_json(caps)
    assert '"acceptedFamilies":[{"familyId":"quip/twisted-ice-dimer","revisions":[1]}]' in capabilities_json(caps)


# -- binding golden checks -----------------------------------------------------------


def test_a_job_without_a_binding_is_a_plain_job():
    problem = ising()
    problem.ClearField("family")
    assert bind_problem(problem) is None


def test_a_pair_this_miner_does_not_accept_is_solved_as_a_plain_model():
    # FAM-69: a generic solver never rejects a job for its family field.
    problem = ising(family=binding(family_id="quip/ice-glass"))
    assert bind_problem(problem) is None
    assert bind_problem(ising(family=binding(revision=2))) is None


@pytest.mark.parametrize("encoding", ["i32", "i16", "f64"])
def test_the_bound_instance_keeps_the_canonical_order_and_native_ids(encoding):
    inst = instance()
    bound = bind_problem(ising(encoding=encoding))
    assert bound is not None
    assert bound.node_ids.tolist() == list(inst.node_ids)
    assert min(bound.node_ids) >= BASE
    assert bound.positions.tolist() == [list(e) for e in inst.edges]
    assert bound.h_milli.tolist() == list(inst.h) and bound.j_milli.tolist() == list(inst.j)
    assert bound.edges.tolist() == [[inst.node_ids[u], inst.node_ids[v]] for u, v in inst.edges]
    assert (bound.offset_milli, bound.metadata, bound.digest) == (inst.offset_milli, inst.metadata, inst.digest)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (dict(family=binding(params_digest=b"\0" * 32)), miner_pb2.MALFORMED),  # code 8
        (dict(family=binding(params=P_W1000)), miner_pb2.MALFORMED),  # code 9
        (dict(family=binding(instance_digest=b"\0" * 32)), miner_pb2.MALFORMED),  # code 15
        (dict(family=binding(offset_milli=1)), miner_pb2.MALFORMED),  # code 15
        (dict(family=binding(nonce=b"")), miner_pb2.MALFORMED),  # code 15
        (dict(family=binding(topology_order_digest=b"\0" * 31)), miner_pb2.MALFORMED),  # code 15
        (dict(family=binding(metadata=instance().metadata[:-1])), miner_pb2.MALFORMED),  # code 15
    ],
)
def test_a_binding_that_fails_fam_65_is_rejected(change, reason):
    with pytest.raises(FamilyJobRejected) as info:
        bind_problem(ising(**change))
    assert info.value.reason == reason


def test_a_changed_coupling_breaks_the_digest():
    problem = ising()
    j = list(instance().j)
    j[0] += 1
    problem.j = wire.encode_i32_le(j)
    with pytest.raises(FamilyJobRejected) as info:
        bind_problem(problem)
    assert info.value.reason == miner_pb2.MALFORMED


def test_the_descriptor_check_runs_through_the_wheel(monkeypatch):
    calls: List[tuple] = []
    real = wheel.descriptor_digest

    def spy(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(wheel, "descriptor_digest", spy)
    assert bind_problem(ising()) is not None
    assert calls == [(TWISTED, 1, P_ADJ, wheel.params_digest(P_ADJ))]
    assert families.descriptor_digest(ising().family.family) == bytes(instance().descriptor_digest)
    with pytest.raises(AdmissionError):
        families.descriptor_digest(binding(params=P_W1000).family)


# -- handle_job with a fake device -----------------------------------------------------


class Device:
    """Records what it was asked to anneal and answers with chosen spins."""

    def __init__(self, missing=(0, 0)):
        self.calls: List[dict] = []
        self.missing = missing

    def unplaceable(self, nodes, edges):
        return self.missing

    def sample(self, nodes, h, edges, j, **kwargs) -> SampleResult:
        self.calls.append(dict(nodes=np.asarray(nodes).tolist(), h=np.asarray(h).tolist(),
                               edges=np.asarray(edges).tolist(), j=np.asarray(j).tolist(), **kwargs))
        n = len(nodes)
        rng = np.random.default_rng(1)
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), (int(kwargs["num_reads"]), n))
        # Columns reversed: the result must reorder them into instance order.
        order = list(reversed(range(n)))
        return SampleResult(
            spins=spins[:, order],
            variables=[int(nodes[i]) for i in order],
            energies=[0.0] * spins.shape[0],
            device_access_time_us=10,
            num_reads=spins.shape[0],
        )


def job(problem: miner_pb2.IsingProblem) -> miner_pb2.Job:
    return miner_pb2.Job(job_id=b"family-1", kind=miner_pb2.ISING_SAMPLE, ising=problem)


def solve(problem, device, **kw):
    return handle_job(job(problem), device, session_nodes=[], session_edges=[], **kw)


def test_a_bound_job_is_sampled_on_the_native_qubits_in_canonical_order():
    inst = instance()
    device = Device()
    replies = solve(ising(), device)
    (call,) = device.calls
    assert call["nodes"] == list(inst.node_ids)
    assert call["edges"] == [[inst.node_ids[u], inst.node_ids[v]] for u, v in inst.edges]
    assert call["h"] == [v / 1000 for v in inst.h] and call["j"] == [v / 1000 for v in inst.j]
    assert [r.WhichOneof("msg") for r in replies] == ["result", "job_request"]


def test_a_bound_job_reports_exact_plain_model_energies_without_the_offset():
    inst = instance()
    bound = bind_problem(ising())
    assert bound is not None
    result = solve(ising(), Device())[0].result
    assert len(result.solutions) == 3
    for solution in result.solutions:
        spins = np.array([wire.decode_spins_packed(solution.spins, len(inst.node_ids))])
        assert solution.energy_milli == int(bound.energies_milli(spins)[0])
        # FAM-64: the plain model has no offset in it.
        assert solution.energy_milli == int(spins[0] @ np.asarray(inst.h)) + sum(
            int(w) * int(spins[0][u]) * int(spins[0][v]) for (u, v), w in zip(inst.edges, inst.j)
        )


def test_an_instance_off_the_solver_graph_is_refused_not_truncated():
    device = Device(missing=(1, 3))
    replies = solve(ising(), device)
    assert not device.calls
    assert replies[0].reject.reason == miner_pb2.TOPOLOGY_MISMATCH
    assert replies[1].job_request.credits == 1


def test_the_topology_hash_form_is_code_15_even_with_no_topology_cached():
    problem = ising()
    problem.topology_hash = b"\x01" * 32
    replies = solve(problem, Device())
    assert replies[0].reject.reason == miner_pb2.MALFORMED


def test_a_failed_binding_rejects_with_one_refund_and_no_sampling():
    device = Device()
    replies = solve(ising(family=binding(instance_digest=b"\0" * 32)), device)
    assert not device.calls
    assert [r.WhichOneof("msg") for r in replies] == ["reject", "job_request"]


def test_an_unaccepted_family_job_runs_on_dense_positions_as_a_plain_model():
    device = Device()
    replies = solve(ising(family=binding(family_id="quip/ice-glass")), device)
    (call,) = device.calls
    assert call["nodes"] == list(range(len(instance().node_ids)))
    assert replies[0].WhichOneof("msg") == "result"


def test_the_postprocess_state_comes_first_and_is_scored_exactly():
    seen: List[BoundInstance] = []

    def postprocess(bound, spins):
        seen.append(bound)
        return -spins[:1]

    result = solve(ising(), Device(), postprocess=postprocess)[0].result
    (bound,) = seen
    assert len(result.solutions) == 4
    rows = np.array([wire.decode_spins_packed(s.spins, len(bound.node_ids)) for s in result.solutions])
    assert (rows[0] == -rows[1]).all()
    assert [s.energy_milli for s in result.solutions] == bound.energies_milli(rows).tolist()


# -- the sampler's fit check ----------------------------------------------------------


def test_the_sampler_counts_the_parts_neither_live_nor_clamped():
    sampler = OceanSampler(mock=True)
    assert sampler.unplaceable([1, 2], np.array([[1, 2]])) == (0, 0)  # no live graph: no claim
    sampler.set_session_topology([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])
    assert sampler.unplaceable([1, 2, 3], np.array([[1, 2], [2, 3]])) == (0, 0)
    assert sampler.unplaceable([1, 9], np.array([[1, 9], [1, 3]])) == (1, 2)


# -- end to end through the session ----------------------------------------------------

FAKE_QPP = """#!{python}
import json, sys
import numpy as np
model, seeds, out, mode = sys.argv[1:5]
raw = open(model, "rb").read()
n, t, p, e = np.frombuffer(raw[:16], dtype="<i4")
rows = np.frombuffer(raw[16:16 + 12 * e], dtype="<i4").reshape(e, 3).astype(np.int64)
s = -np.fromfile(seeds, dtype=np.int8).reshape(-1, n).astype(np.int64)[0]
best = int((s[rows[:, 0]] * s[rows[:, 1]] * rows[:, 2]).sum())
s.astype(np.int8).tofile(out)
print(json.dumps({{"mode": mode, "start": best, "best": best, "trace": [[0, 0.0, best, best]], "cpu": 0.0}}))
"""


def test_an_explicit_job_goes_through_the_configured_postprocess(monkeypatch, tmp_path):
    from quip_solver_dwave.tests.test_family_leases import Coordinator, FakeDevice

    fake = tmp_path / "fake-qpp"
    fake.write_text(FAKE_QPP.format(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv(qpp.BINARY_ENV, str(fake))
    records = tmp_path / "records"
    toml = f'[postprocess]\nrecords = "{records}"\n"{TWISTED}" = "qpp-race768/1"\n'
    class SessionDevice(FakeDevice):
        """The session's sampler surface, answering any job like ``Device``."""

        def sample(self, nodes, h, edges, j, **kwargs):
            return Device().sample(nodes, h, edges, j, **kwargs)

    nodes, edges = topology()
    coord = Coordinator(monkeypatch, SessionDevice())
    coord.send(welcome=miner_pb2.Welcome(protocol_version=2))
    coord.send(configure=miner_pb2.Configure(queue_depth=1, idle_timeout_s=300, backend_toml=toml))
    coord.send(topology=miner_pb2.Topology(
        hash=b"\x5a" * 32, nodes=nodes,
        edges=miner_pb2.EdgeList(u=[u for u, _ in edges], v=[v for _, v in edges])))
    coord.send(job=miner_pb2.Job(job_id=b"family-1", kind=miner_pb2.ISING_SAMPLE, ising=ising()))
    coord.wait_for(lambda sent: any(m.WhichOneof("msg") == "result" for m in sent))
    sent = coord.finish()
    (result,) = [m.result for m in sent if m.WhichOneof("msg") == "result"]
    bound = bind_problem(ising())
    assert bound is not None
    rows = np.array([wire.decode_spins_packed(s.spins, len(bound.node_ids)) for s in result.solutions])
    # The fake processor returns the flip of the first read, ahead of the raw reads.
    assert (rows[0] == -rows[1]).all() and len(rows) == 4
    assert [s.energy_milli for s in result.solutions] == bound.energies_milli(rows).tolist()
    assert list(records.glob(f"*/{b'family-1'.hex()}.npz"))
    assert all(os.path.getsize(f) > 0 for f in records.glob("*/trials.jsonl"))
    assert list(records.glob("*/trials.jsonl"))
