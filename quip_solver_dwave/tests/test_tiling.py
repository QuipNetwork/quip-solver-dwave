"""A Zephyr chip smaller than the session's topology anneals it in patches.

The fixtures shrink the real case: a Z(2,2) chip with dead qubits stands in
for Advantage2_system4, and a Z(3,2) session graph for the network's
Advantage2_system1 topology. The chip double answers each patch by descent
from random states, which is enough to test the stitching without a QPU.
"""

from __future__ import annotations

import numpy as np
import pytest

dnx = pytest.importorskip("dwave_networkx")
dimod = pytest.importorskip("dimod")

from quip_solver_dwave.config import tile_passes_from_toml  # noqa: E402
from quip_solver_dwave.ocean import OceanSampler, descend_from  # noqa: E402
from quip_solver_dwave.tiling import (  # noqa: E402
    DEFAULT_TILE_PASSES,
    ZephyrTiling,
    infer_zephyr_rows,
    clip_fields,
    ising_energy,
)

CHIP_ROWS, SESSION_ROWS, TILE = 2, 3, 2
PROGRAMMING_US = 100
READ_US = 10


def _graph(rows, dead=()):
    g = dnx.zephyr_graph(rows, TILE)
    g.remove_nodes_from(dead)
    nodes = sorted(int(n) for n in g.nodes)
    edges = [(int(u), int(v)) for u, v in g.edges]
    return nodes, edges


class ChipDouble:
    """A Z(2,2) chip: descent from random states, billed per call."""

    def __init__(self, nodes, edges, fail_on_call=None):
        self.nodelist = nodes
        self.edgelist = edges
        self.properties = {"topology": {"type": "zephyr", "shape": [CHIP_ROWS, TILE]}}
        self.calls = []
        self._fail_on_call = fail_on_call
        self._rng = np.random.default_rng(7)

    def sample_ising(self, h, j, num_reads=1, **_):
        self.calls.append((dict(h), dict(j)))
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise RuntimeError("SAPI refused the problem")
        live = set(self.nodelist)
        live_edges = {(min(u, v), max(u, v)) for u, v in self.edgelist}
        assert set(h) <= live, "a patch named a qubit the chip does not have"
        assert {(min(u, v), max(u, v)) for u, v in j} <= live_edges, "a patch named a missing coupler"
        rows, energies = [], []
        for _ in range(num_reads):
            start = {v: int(self._rng.choice([-1, 1])) for v in h}
            spins, energy = descend_from(h, j, start)
            rows.append(spins)
            energies.append(energy)
        info = {"timing": {"qpu_programming_time": PROGRAMMING_US, "qpu_sampling_time": READ_US * num_reads}}
        ss = dimod.SampleSet.from_samples(rows, dimod.SPIN, energy=energies)
        return dimod.SampleSet(ss.record, ss.variables, info, ss.vartype)


def _problem(seed=1):
    nodes, edges = _graph(SESSION_ROWS)
    rng = np.random.default_rng(seed)
    j = rng.choice([-1.0, 1.0], size=len(edges))
    return np.asarray(nodes), np.zeros(len(nodes)), np.asarray(edges), j


def _tiled_sampler(dead=(3, 17, 40), fail_on_call=None):
    chip_nodes, chip_edges = _graph(CHIP_ROWS, dead)
    chip = ChipDouble(chip_nodes, chip_edges, fail_on_call)
    s = OceanSampler(sampler=chip, mock=True)
    s._live_nodes = chip_nodes
    s._live_edges = chip_edges
    s._live_zephyr = (CHIP_ROWS, TILE)
    nodes, _, edges, _ = _problem()
    s.set_session_topology(nodes.tolist(), [tuple(e) for e in edges.tolist()])
    return s, chip


def test_the_session_rows_are_read_off_the_couplers():
    nodes, _, edges, _ = _problem()
    assert infer_zephyr_rows(nodes, edges, TILE, CHIP_ROWS + 1) == SESSION_ROWS
    chip_nodes, chip_edges = _graph(CHIP_ROWS)
    # A Z(2) graph is not a Z(3) graph under Z(3)'s numbering.
    assert infer_zephyr_rows(np.asarray(chip_nodes), np.asarray(chip_edges), TILE, CHIP_ROWS + 1) is None


def test_every_patch_edge_is_a_chip_coupler_between_the_same_two_nodes():
    chip_nodes, chip_edges = _graph(CHIP_ROWS, (3, 17))
    nodes, _, edges, _ = _problem()
    plan = ZephyrTiling(CHIP_ROWS, TILE, chip_nodes, chip_edges).plan(nodes, edges)
    assert plan is not None and len(plan.patches) > 1
    live_edges = {(min(u, v), max(u, v)) for u, v in chip_edges}
    for patch in plan.patches:
        live_of = dict(zip(patch.pos.tolist(), patch.live.tolist()))
        for e, (a, b) in zip(patch.edge_pos.tolist(), patch.live_edges.tolist()):
            pa, pb = plan.ends[e]
            assert {live_of[int(pa)], live_of[int(pb)]} == {a, b}
            assert (min(a, b), max(a, b)) in live_edges


def test_the_cover_reaches_every_node_a_perfect_chip_can():
    chip_nodes, chip_edges = _graph(CHIP_ROWS)
    nodes, _, edges, _ = _problem()
    plan = ZephyrTiling(CHIP_ROWS, TILE, chip_nodes, chip_edges).plan(nodes, edges)
    assert plan is not None
    assert len(plan.uncovered) == 0


def test_a_patch_anneal_is_the_exact_restriction_of_the_whole_problem():
    """Folding the boundary into h must leave energy differences unchanged."""
    chip_nodes, chip_edges = _graph(CHIP_ROWS)
    nodes, h, edges, j = _problem()
    plan = ZephyrTiling(CHIP_ROWS, TILE, chip_nodes, chip_edges).plan(nodes, edges)
    assert plan is not None
    rng = np.random.default_rng(3)
    state = rng.choice([-1, 1], size=len(nodes)).astype(np.int8)
    for patch in plan.patches:
        field = h[patch.pos] + np.bincount(
            patch.fold_dst, weights=j[patch.fold_edge] * state[patch.fold_src], minlength=len(patch.pos)
        )
        j_in = j[patch.edge_pos]

        def local(x):
            return float(x @ field + (x[patch.edge_local[:, 0]] * x[patch.edge_local[:, 1]]) @ j_in)

        other = state.copy()
        other[patch.pos] = rng.choice([-1, 1], size=len(patch.pos))
        whole = ising_energy(np.stack([state, other]), h, plan.ends, j)
        assert whole[1] - whole[0] == pytest.approx(
            local(other[patch.pos].astype(float)) - local(state[patch.pos].astype(float))
        )


def test_a_tiled_job_returns_whole_states_with_exact_energies():
    s, chip = _tiled_sampler()
    nodes, h, edges, j = _problem()
    result = s.sample(nodes, h, edges, j, num_reads=8, cancel_key=b"job")
    assert result.spins.shape == (8, len(nodes))
    assert result.variables == nodes.tolist()
    exact = ising_energy(result.spins, h, np.searchsorted(nodes, edges), j)
    assert result.energies == pytest.approx(exact.tolist())
    assert result.energies == sorted(result.energies)
    calls = len(chip.calls)
    assert calls == int(result.extra["tile_calls"])
    assert result.device_access_time_us == calls * (PROGRAMMING_US + 8 * READ_US)
    # Descent on each patch puts the whole well below a random state's energy,
    # which averages zero for a ±1 glass.
    assert result.energies[0] < -0.5 * len(edges) * 0.5


def test_a_second_pass_never_raises_the_energy():
    s, _ = _tiled_sampler()
    nodes, h, edges, j = _problem()
    s.use_tile_passes(1)
    one = s.sample(nodes, h, edges, j, num_reads=4)
    s2, _ = _tiled_sampler()
    s2.use_tile_passes(3)
    three = s2.sample(nodes, h, edges, j, num_reads=4)
    assert three.energies[0] <= one.energies[0]


def test_a_warm_start_is_never_made_worse():
    s, _ = _tiled_sampler()
    nodes, h, edges, j = _problem()
    first = s.sample(nodes, h, edges, j, num_reads=4)
    from quip_solver_dwave.warm import WarmStart

    warm = WarmStart(state=first.spins[0].copy(), reversal_s=0.5, reversal_pause_us=0)
    again = s.sample(nodes, h, edges, j, num_reads=4, warm_start=warm)
    assert again.energies[0] <= first.energies[0]


def test_patches_that_finished_are_billed_when_a_later_one_fails():
    s, _ = _tiled_sampler(fail_on_call=3)
    nodes, h, edges, j = _problem()
    s._max_access_us = 0
    with pytest.raises(RuntimeError):
        s.sample(nodes, h, edges, j, num_reads=4)
    # Two patches annealed and were charged before the third failed.
    assert s.drain_unobserved_access_us() == 2 * (PROGRAMMING_US + 4 * READ_US)


def test_a_family_instance_on_the_session_graph_is_placeable_on_a_tiled_chip():
    """The network's jobs are family instances on its Zephyr: tiling must take them."""
    s, _ = _tiled_sampler()
    nodes, _, edges, _ = _problem()
    assert s.unplaceable(nodes, edges) == (0, 0)


def test_a_graph_that_is_no_larger_zephyr_is_unplaceable_on_a_tiled_chip():
    s, _ = _tiled_sampler()
    # Qubits 0 and 2 lie on one line, two places apart: no Zephyr couples them.
    assert s.unplaceable(np.asarray([0, 2]), np.asarray([[0, 2]])) == (2, 1)


def test_a_session_the_size_of_the_chip_is_not_tiled():
    chip_nodes, chip_edges = _graph(CHIP_ROWS, (3,))
    s = OceanSampler(sampler=ChipDouble(chip_nodes, chip_edges), mock=True)
    s._live_nodes = chip_nodes
    s._live_edges = chip_edges
    s._live_zephyr = (CHIP_ROWS, TILE)
    full_nodes, full_edges = _graph(CHIP_ROWS)
    s.set_session_topology(full_nodes, full_edges)
    assert s._tiling is None
    assert s._defective_qubits == [3]


def test_tile_passes_come_from_backend_toml():
    assert tile_passes_from_toml("") == DEFAULT_TILE_PASSES
    assert tile_passes_from_toml("tile_passes = 0") == DEFAULT_TILE_PASSES
    assert tile_passes_from_toml("tile_passes = 3") == 3


def test_a_twisted_ice_dimer_job_on_the_session_graph_is_solved_not_rejected():
    """The network's own job form, end to end through handle_job on a tiled chip."""
    from quip_solver_core import families as wheel
    from quip_solver_core import miner_pb2, wire

    from quip_solver_dwave.job import handle_job

    family = "quip/twisted-ice-dimer"
    # Adjacency mode, weak 0.045, 6 pairs: the layout of test_family_jobs.P_ADJ.
    params = bytes(
        [1, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0x2D, 0, 4, 0, 4, 0, 0, 0, 6, 0, 0, 0]
    )
    nodes, _, edges, _ = _problem()
    inst = wheel.generate(
        family, 1, params, wheel.params_digest(params), nodes.tolist(),
        [tuple(e) for e in edges.tolist()], bytes(range(32)),
    )
    problem = miner_pb2.IsingProblem(
        edges=miner_pb2.EdgeList(u=[u for u, _ in inst.edges], v=[v for _, v in inst.edges]),
        family=miner_pb2.FamilyBinding(
            family=miner_pb2.FamilyDescriptor(
                family_id=family, revision=1, params=params, params_digest=wheel.params_digest(params)
            ),
            nonce=inst.nonce,
            instance_digest=inst.digest,
            metadata=inst.metadata,
            topology_order_digest=inst.topology_order_digest,
            node_ids=list(inst.node_ids),
            offset_milli=inst.offset_milli,
        ),
        num_reads=4,
        encoding=miner_pb2.COEFFICIENT_ENCODING_I32,
        scale=1000,
        h=wire.encode_i32_le(list(inst.h)),
        j=wire.encode_i32_le(list(inst.j)),
    )
    s, chip = _tiled_sampler()
    out = handle_job(
        miner_pb2.Job(job_id=b"dimer-1", kind=miner_pb2.ISING_SAMPLE, ising=problem),
        s, session_nodes=[], session_edges=[],
    )
    assert out[0].WhichOneof("msg") == "result", out[0]
    assert len(chip.calls) > 1
    ids = np.asarray(inst.node_ids)
    ends = np.asarray(inst.edges, dtype=np.int64).reshape(-1, 2)
    j = np.asarray(inst.j, dtype=np.float64)
    for solution in out[0].result.solutions:
        bits = np.unpackbits(np.frombuffer(solution.spins, dtype=np.uint8), bitorder="little")[: len(ids)]
        spins = np.where(bits == 1, 1, -1)
        assert solution.energy_milli == int((spins[ends[:, 0]] * spins[ends[:, 1]]) @ j)


def test_the_cuts_come_from_the_topology_and_bind_to_any_edge_order():
    """Cuts are chosen once from the session Topology; a job only re-indexes them."""
    s, _ = _tiled_sampler()
    nodes, _, edges, _ = _problem()
    tiling = s._tiling
    assert tiling is not None
    flipped = edges[::-1][:, ::-1].copy()
    a = tiling.plan(nodes, edges)
    b = tiling.plan(nodes, flipped)
    cuts = tiling._cuts
    assert cuts is not None and cuts.rows == SESSION_ROWS
    assert a is not None and b is not None
    assert [p.live.tolist() for p in a.patches] == [p.live.tolist() for p in b.patches]
    assert [sorted(map(tuple, p.live_edges.tolist())) for p in a.patches] == [
        sorted(map(tuple, p.live_edges.tolist())) for p in b.patches
    ]


def test_a_field_past_its_in_patch_coupling_is_clipped_without_moving_the_optimum():
    edge_local = np.asarray([[0, 1], [1, 2]])
    j_in = np.asarray([1.0, 0.5])
    field = np.asarray([7.0, -0.2, -9.0])
    clipped = clip_fields(field, edge_local, j_in)
    assert clipped.tolist() == [2.0, -0.2, -1.5]
    states = np.asarray([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    best = states[np.argmin(ising_energy(states, field, edge_local, j_in))]
    best_clipped = states[np.argmin(ising_energy(states, clipped, edge_local, j_in))]
    assert best.tolist() == best_clipped.tolist()


def test_batch_seam_anneals_the_core_at_once_then_the_seams_across_its_cuts():
    s, chip = _tiled_sampler()
    s.use_tile_mode("batch-seam")
    s.use_tile_passes(1)
    nodes, h, edges, j = _problem()
    tiling = s._tiling
    assert tiling is not None
    plan = tiling.plan(nodes, edges)
    assert plan is not None and len(plan.seams) > 0
    result = s.sample(nodes, h, edges, j, num_reads=8, cancel_key=b"job")
    assert int(result.extra["tile_calls"]) == len(plan.patches) + len(plan.seams) == len(chip.calls)
    assert np.all(result.spins != 0)
    exact = ising_energy(result.spins, h, np.searchsorted(nodes, edges), j)
    assert result.energies == pytest.approx(exact.tolist())


def test_the_seams_anneal_couplers_the_core_cuts():
    s, _ = _tiled_sampler()
    nodes, _, edges, _ = _problem()
    tiling = s._tiling
    assert tiling is not None
    plan = tiling.plan(nodes, edges)
    assert plan is not None and plan.seams
    core = set(np.concatenate([p.edge_pos for p in plan.patches]).tolist())
    seam = set(np.concatenate([p.edge_pos for p in plan.seams]).tolist())
    assert seam - core


def test_a_failure_inside_the_batch_still_bills_the_patches_that_finished():
    s, _ = _tiled_sampler(fail_on_call=3)
    s.use_tile_mode("batch-seam")
    nodes, h, edges, j = _problem()
    with pytest.raises(RuntimeError):
        s.sample(nodes, h, edges, j, num_reads=4)
    tiling = s._tiling
    assert tiling is not None
    plan = tiling.plan(nodes, edges)
    assert plan is not None
    # Every core patch is in flight at once; all but the failed one finished.
    assert s.drain_unobserved_access_us() == (len(plan.patches) - 1) * (PROGRAMMING_US + 4 * READ_US)


def test_tile_mode_comes_from_backend_toml():
    from quip_solver_dwave.config import tile_mode_from_toml

    assert tile_mode_from_toml("") == "sequential"
    assert tile_mode_from_toml('tile_mode = "batch-seam"') == "batch-seam"
    assert tile_mode_from_toml('tile_mode = "zigzag"') == "sequential"
