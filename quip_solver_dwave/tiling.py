"""Anneal a Zephyr problem larger than the chip, one sub-lattice at a time.

The network's topology is a Zephyr Z(m, t) working graph, and a smaller
Advantage2 is a Z(m', t) chip with m' < m. Its linear labels are not the
network's: Zephyr numbers a qubit from its coordinates and ``m``, so qubit 5
of Z(6, 4) and qubit 5 of Z(12, 4) are different places. Clamping the labels
the chip lacks, which is what ``defects`` does for a few dead qubits, would
anneal a fraction of the problem at the wrong positions.

Instead the chip is placed on the network graph at the offsets that
``dwave_networkx.zephyr_sublattice_mappings`` enumerates. Each placement is a
patch: the network nodes whose chip qubit works, and the network couplers
whose chip coupler works. A greedy cover picks patches until every node the
chip can reach is in one.

:func:`anneal_tiled` is block coordinate descent over that cover. With the
spins outside a patch held fixed, a coupler that leaves the patch is a field
on the qubit inside it: ``J * s_i * s_out`` is ``(J * s_out) * s_i``. So each
patch anneal is the exact restriction of the whole problem, and its best read
can only lower the whole energy. A coupler inside a patch that the chip lacks
is folded the same way, from the other end's current spin; that is the one
approximation, and the acceptance test below scores the exact energy, so it
can cost a step but never accept a worse state.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Patch anneals per job: one pass sets every spin, a second re-anneals each
#: patch against the boundary the first pass left.
DEFAULT_TILE_PASSES = 2

#: The cover stops at the first patch that would reach fewer than this share
#: of the nodes. On the network graph and Advantage2_system4 the fifth patch
#: reaches 97% of nodes; the next nine add 125 nodes between them, each the
#: neighbour of a dead qubit, for 8 ms of programming apiece. Those nodes are
#: aligned with their field instead (``_settle``). 0 covers every node.
DEFAULT_MIN_NEW_SHARE = 0.01

#: Most seam placements chosen after the core cover. On the live topology the
#: core cuts 2,649 couplers and misses 125 nodes; eight seams leave 162 and 3.
MAX_SEAMS = 8

#: ``anneal_tiled`` modes. ``sequential`` anneals the core patches in turn,
#: each against the ones before it. ``batch-seam`` anneals the core patches at
#: once with no boundary, then the seam patches in turn against the result.
TILE_MODES = ("sequential", "batch-seam")


@dataclass(frozen=True, eq=False)
class Patch:
    """One placement of the chip on the problem graph, as index arrays."""

    #: Positions into the problem's nodes, in chip-label order.
    pos: np.ndarray
    #: Chip labels of those nodes, sorted.
    live: np.ndarray
    #: Positions into the problem's edges that the chip anneals jointly.
    edge_pos: np.ndarray
    #: ``(k, 2)`` chip labels of those edges.
    live_edges: np.ndarray
    #: ``(k, 2)`` patch-local indices of those edges.
    edge_local: np.ndarray
    #: Every other edge with an end in the patch becomes a field on that end:
    #: the patch-local index of the end inside ...
    fold_dst: np.ndarray
    #: ... the problem position of the spin that sets the field ...
    fold_src: np.ndarray
    #: ... and the problem position of the coupler.
    fold_edge: np.ndarray


@dataclass(frozen=True, eq=False)
class TilePlan:
    """A patch cover of one problem graph."""

    nodes: np.ndarray
    edges: np.ndarray
    patches: Tuple[Patch, ...]
    #: ``(E, 2)`` node positions of each edge's ends.
    ends: np.ndarray
    #: Problem positions that no core patch reaches: dead at every placement,
    #: or left out by ``min_new_share``.
    uncovered: np.ndarray
    #: Placements across the cuts of the core cover (``TileCuts.seam_images``).
    seams: Tuple[Patch, ...] = ()
    #: Problem positions that neither a core nor a seam patch reaches.
    seam_uncovered: np.ndarray = dc_field(default_factory=lambda: np.empty(0, dtype=np.int64))


def zephyr_shape(properties: dict) -> Optional[Tuple[int, int]]:
    """``(m, t)`` of a solver whose ``topology`` property is Zephyr, else None."""
    topology = properties.get("topology") or {}
    shape = topology.get("shape") or []
    if topology.get("type") != "zephyr" or len(shape) != 2:
        return None
    return int(shape[0]), int(shape[1])


def _zephyr_size(m: int, t: int) -> int:
    return 4 * t * m * (2 * m + 1)


def infer_zephyr_rows(
    nodes: np.ndarray, edges: np.ndarray, t: int, min_rows: int, max_rows: int = 32
) -> Optional[int]:
    """The smallest ``m >= min_rows`` for which the graph is a subgraph of Z(m, t).

    Labels alone cannot say which Zephyr a graph came from, since every m
    numbers 0, 1, 2, ... The couplers can: the graph must be a subgraph of
    Z(m, t) under that m's numbering. The largest label fixes the smallest m
    that can hold the graph; a working graph from a chip is a subgraph of
    that Zephyr or of one a size or two larger, so the search stops there
    rather than building every lattice up to ``max_rows``.
    """
    import dwave_networkx as dnx

    if not len(nodes):
        return None
    top = int(np.max(nodes))
    pairs = _edge_keys(edges)
    first = next((m for m in range(min_rows, max_rows + 1) if top < _zephyr_size(m, t)), None)
    if first is None:
        return None
    for m in range(first, min(first + 2, max_rows) + 1):
        z = dnx.zephyr_graph(m, t)
        z_edges = np.asarray(list(z.edges()), dtype=np.int64).reshape(-1, 2)
        z_keys = np.sort(_edge_keys(z_edges))
        hit = np.searchsorted(z_keys, pairs)
        hit[hit == len(z_keys)] = 0
        if len(z_keys) and np.all(z_keys[hit] == pairs):
            return m
    return None


def _edge_keys(edges: np.ndarray) -> np.ndarray:
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    lo = np.minimum(e[:, 0], e[:, 1])
    hi = np.maximum(e[:, 0], e[:, 1])
    return lo * (1 << 32) + hi


@dataclass(frozen=True, eq=False)
class TileCuts:
    """Where the chip sits on one session topology: the cuts, before any job.

    Choosing them maps every live qubit through each of the
    ``(2(m - m') + 1)^2`` offsets, about a second at Z(6) on Z(12). That
    depends on the topology and the chip alone, so it runs once per session
    Topology and every job only binds the result to its own arrays.
    """

    rows: int
    #: The topology's node labels and coupler keys, sorted, for the subgraph test.
    nodes: np.ndarray
    edge_keys: np.ndarray
    #: For each chosen placement, the topology label of every live qubit, in
    #: the tiler's sorted live order; -1 where the placement leaves the topology.
    images: Tuple[np.ndarray, ...]
    #: Placements that anneal the couplers and nodes the core images miss,
    #: in the same form.
    seam_images: Tuple[np.ndarray, ...] = ()


def _lookup(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Index of each value in ``sorted_values``, or -1 when absent."""
    if not len(sorted_values):
        return np.full(len(values), -1, dtype=np.int64)
    at = np.searchsorted(sorted_values, values)
    at[at == len(sorted_values)] = 0
    return np.where(sorted_values[at] == values, at, -1)


class ZephyrTiling:
    """The chip's geometry: cuts a larger Zephyr topology into chip patches.

    :meth:`prepare` chooses the cuts from the session topology, and
    ``OceanSampler`` runs it on its own thread when the Topology arrives.
    :meth:`plan` binds them to one job's node and edge order, which costs a few
    vectorised lookups, and caches the result on the arrays as
    ``OceanSampler._graph_plan`` caches its own.
    """

    def __init__(
        self,
        live_rows: int,
        tile: int,
        live_nodes: Sequence[int],
        live_edges: Sequence[Tuple[int, int]],
        min_new_share: float = DEFAULT_MIN_NEW_SHARE,
    ):
        self.live_rows = live_rows
        self.min_new_share = min_new_share
        self.tile = tile
        self._live_nodes = np.sort(np.asarray(live_nodes, dtype=np.int64))
        live_edges_arr = np.asarray(live_edges, dtype=np.int64).reshape(-1, 2)
        index = {int(q): i for i, q in enumerate(self._live_nodes)}
        a = np.fromiter((index.get(int(u), -1) for u in live_edges_arr[:, 0]), np.int64)
        b = np.fromiter((index.get(int(v), -1) for v in live_edges_arr[:, 1]), np.int64)
        on_chip = (a >= 0) & (b >= 0)
        # Live couplers as positions into the sorted live nodes, and as labels.
        self._a, self._b = a[on_chip], b[on_chip]
        self._live_edge_labels = live_edges_arr[on_chip]
        self._lock = threading.Lock()
        self._prepared = False
        self._cuts: Optional[TileCuts] = None
        self._cached: Optional[Tuple[np.ndarray, np.ndarray, Optional[TilePlan]]] = None

    def prepare(
        self,
        nodes: "Sequence[int] | np.ndarray",
        edges: "Sequence[Tuple[int, int]] | np.ndarray",
    ) -> Optional[TileCuts]:
        """Choose the cuts of the session topology; None when it is no larger Zephyr."""
        with self._lock:
            if not self._prepared:
                self._cuts = self._choose(
                    np.asarray(nodes, dtype=np.int64),
                    np.asarray(edges, dtype=np.int64).reshape(-1, 2),
                )
                self._prepared = True
            return self._cuts

    def plan(self, nodes: np.ndarray, edges: np.ndarray) -> Optional[TilePlan]:
        """The bound cover of a job graph, or None when the cuts cannot hold it.

        A job graph must be a subgraph of the topology the cuts came from.
        Before :meth:`prepare` has run, the job graph is taken as that topology.
        """
        if not self._prepared:
            self.prepare(nodes, edges)
        with self._lock:
            cached = self._cached
            if (
                cached is not None
                and np.array_equal(cached[0], nodes)
                and np.array_equal(cached[1], edges)
            ):
                return cached[2]
            cuts = self._cuts
        plan = None
        if cuts is not None:
            plan = self._bind(
                cuts,
                np.asarray(nodes, dtype=np.int64),
                np.asarray(edges, dtype=np.int64).reshape(-1, 2),
            )
        with self._lock:
            self._cached = (np.array(nodes, copy=True), np.array(edges, copy=True), plan)
        return plan

    def _choose(self, nodes: np.ndarray, edges: np.ndarray) -> Optional[TileCuts]:
        """Greedy cover of the topology by chip placements."""
        import dwave_networkx as dnx

        rows = infer_zephyr_rows(nodes, edges, self.tile, self.live_rows + 1)
        if rows is None:
            return None
        source = dnx.zephyr_graph(self.live_rows, self.tile)
        target = dnx.zephyr_graph(rows, self.tile)
        sorted_nodes = np.sort(nodes)
        live = self._live_nodes
        images, reach = [], []
        for mapping in dnx.zephyr_sublattice_mappings(source, target):
            image = np.fromiter((mapping(int(q)) for q in live), np.int64, count=len(live))
            image = np.where(_lookup(sorted_nodes, image) >= 0, image, -1)
            images.append(image)
            reach.append(image[image >= 0])
        covered: set = set()
        chosen = []
        while True:
            gains = [len(set(r.tolist()) - covered) for r in reach]
            best = int(np.argmax(gains))
            if gains[best] == 0 or (chosen and gains[best] < self.min_new_share * len(nodes)):
                break
            chosen.append(images[best])
            covered.update(reach[best].tolist())
        topo_keys = np.sort(_edge_keys(edges))
        joint = [self._joint_keys(image, topo_keys) for image in images]
        core_joint: set = set()
        for image in chosen:
            core_joint |= self._joint_keys(image, topo_keys)
        cut_left = set(topo_keys.tolist()) - core_joint
        missed = set(nodes.tolist()) - covered
        seams = []
        floor = max(1.0, self.min_new_share * len(cut_left))
        while len(seams) < MAX_SEAMS and (cut_left or missed):
            gains = [len(jk & cut_left) + len(set(r.tolist()) & missed) for jk, r in zip(joint, reach)]
            best = int(np.argmax(gains))
            if gains[best] < floor:
                break
            seams.append(images[best])
            cut_left -= joint[best]
            missed -= set(reach[best].tolist())
        logger.info(
            "[QPU] cut the Z(%d,%d) topology into %d Z(%d,%d) patches reaching %d of %d nodes",
            rows,
            self.tile,
            len(chosen),
            self.live_rows,
            self.tile,
            len(covered),
            len(nodes),
        )
        logger.info(
            "[QPU] %d seam patches leave %d couplers and %d nodes unannealed",
            len(seams),
            len(cut_left),
            len(missed),
        )
        return TileCuts(
            rows=rows,
            nodes=sorted_nodes,
            edge_keys=topo_keys,
            images=tuple(chosen),
            seam_images=tuple(seams),
        )

    def _joint_keys(self, image: np.ndarray, topo_keys: np.ndarray) -> set:
        """Topology couplers that this placement anneals jointly, as keys."""
        a, b = self._a, self._b
        both = (image[a] >= 0) & (image[b] >= 0)
        keys = _edge_keys(np.stack([image[a][both], image[b][both]], axis=1))
        return set(keys[_lookup(topo_keys, keys) >= 0].tolist())

    def _bind(self, cuts: TileCuts, nodes: np.ndarray, edges: np.ndarray) -> Optional[TilePlan]:
        """The cuts as index arrays into this job's nodes and edges."""
        keys = _edge_keys(edges)
        if np.any(_lookup(cuts.nodes, nodes) < 0) or np.any(_lookup(cuts.edge_keys, keys) < 0):
            return None
        order = np.argsort(nodes)
        sorted_nodes = nodes[order]
        edge_order = np.argsort(keys)
        sorted_keys = keys[edge_order]
        ends_at = _lookup(sorted_nodes, edges.reshape(-1))
        if np.any(ends_at < 0):
            return None
        ends = order[ends_at].reshape(-1, 2)
        covered = np.zeros(len(nodes), dtype=bool)
        core = tuple(
            self._place(image, nodes, ends, order, sorted_nodes, edge_order, sorted_keys, covered)
            for image in cuts.images
        )
        uncovered = np.flatnonzero(~covered)
        seams = tuple(
            self._place(image, nodes, ends, order, sorted_nodes, edge_order, sorted_keys, covered)
            for image in cuts.seam_images
        )
        return TilePlan(
            nodes=nodes,
            edges=edges,
            patches=core,
            ends=ends,
            uncovered=uncovered,
            seams=seams,
            seam_uncovered=np.flatnonzero(~covered),
        )

    def _place(self, image, nodes, ends, order, sorted_nodes, edge_order, sorted_keys, covered) -> Patch:
        """One placement bound to a job graph; marks the nodes it reaches."""
        a, b = self._a, self._b
        at = _lookup(sorted_nodes, image)
        pos_of_live = np.where(at >= 0, order[np.maximum(at, 0)], -1)
        members = np.flatnonzero(pos_of_live >= 0)
        ends_in = (pos_of_live[a] >= 0) & (pos_of_live[b] >= 0)
        e_at = _lookup(sorted_keys, _edge_keys(np.stack([image[a], image[b]], axis=1)))
        joint = ends_in & (e_at >= 0)
        covered[pos_of_live[members]] = True
        return self._patch(
            len(nodes),
            ends,
            pos_of_live[members],
            self._live_nodes[members],
            edge_order[e_at[joint]],
            self._live_edge_labels[joint],
        )

    @staticmethod
    def _patch(
        n_nodes: int,
        ends: np.ndarray,
        pos: np.ndarray,
        live: np.ndarray,
        edge_pos: np.ndarray,
        live_edges: np.ndarray,
    ) -> Patch:
        """Index arrays of one placement; ``ends`` is ``(E, 2)`` node positions."""
        local = np.full(n_nodes, -1, dtype=np.int64)
        local[pos] = np.arange(len(pos))
        joint = np.zeros(len(ends), dtype=bool)
        joint[edge_pos] = True
        lu, lv = local[ends[:, 0]], local[ends[:, 1]]
        # An edge with one end in the patch, or both ends but no chip coupler,
        # is a field on each end inside the patch.
        folded_u = (lu >= 0) & ~joint
        folded_v = (lv >= 0) & ~joint
        fold_edge = np.concatenate([np.flatnonzero(folded_u), np.flatnonzero(folded_v)])
        fold_dst = np.concatenate([lu[folded_u], lv[folded_v]])
        fold_src = np.concatenate([ends[folded_u, 1], ends[folded_v, 0]])
        return Patch(
            pos=pos,
            live=live,
            edge_pos=edge_pos,
            live_edges=live_edges,
            edge_local=np.stack([lu[edge_pos], lv[edge_pos]], axis=1).reshape(-1, 2),
            fold_dst=fold_dst,
            fold_src=fold_src,
            fold_edge=fold_edge,
        )


#: ``submit(live_nodes, h, live_edges, j)`` anneals one patch on the chip and
#: returns its reads as ``(reads, len(live_nodes))`` int8 in ``live_nodes``
#: order, with the access time D-Wave charged for them.
SubmitPatch = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, int]]


@dataclass(frozen=True)
class TiledAnswer:
    """Whole-problem states from one tiled anneal, best first."""

    #: ``(rows, N)`` int8 in the plan's node order.
    spins: np.ndarray
    #: Exact energy of each row.
    energies: np.ndarray
    #: QPU access time of every patch call, summed.
    access_us: int
    #: Patch calls made.
    calls: int


def ising_energy(spins: np.ndarray, h: np.ndarray, ends: np.ndarray, j: np.ndarray) -> np.ndarray:
    """``sum(h s) + sum(J s_u s_v)`` of each row of ``spins``.

    Elementwise sums, not ``@``: a ``(rows, E) @ (E,)`` product goes through
    BLAS, whose thread start-up made one 41,205-coupler row cost 28 ms on a
    loaded 32-core host, against 0.3 ms for the same sums done here.
    """
    s = np.atleast_2d(spins).astype(np.float64)
    return (s * h).sum(axis=1) + (s[:, ends[:, 0]] * s[:, ends[:, 1]] * j).sum(axis=1)


def clip_fields(field: np.ndarray, edge_local: np.ndarray, j_in: np.ndarray) -> np.ndarray:
    """Each field cut to its qubit's in-patch coupling, plus 1.

    A boundary folds up to 7.1 units onto one qubit of the network's family
    on Advantage2_system4, past its h range of 6, and SAPI's auto-scale
    would then shrink every coupler of the patch to fit. A qubit whose field
    is larger than the sum of its in-patch ``|J|`` takes the field's sign in
    every ground state, so any field past that sum gives the same optimum;
    the 1 unit of margin keeps the spin pinned under the anneal's noise.
    """
    reach = np.bincount(edge_local[:, 0], weights=np.abs(j_in), minlength=len(field))
    reach += np.bincount(edge_local[:, 1], weights=np.abs(j_in), minlength=len(field))
    cap = reach + 1.0
    return np.clip(field, -cap, cap)


def _field(patch: Patch, h: np.ndarray, j: np.ndarray, state: np.ndarray) -> np.ndarray:
    """The patch's fields: its own h plus every folded coupler at the current state."""
    return h[patch.pos] + np.bincount(
        patch.fold_dst,
        weights=j[patch.fold_edge] * state[patch.fold_src],
        minlength=len(patch.pos),
    )


def anneal_tiled(
    plan: TilePlan,
    h: np.ndarray,
    j: np.ndarray,
    *,
    passes: int,
    submit: SubmitPatch,
    initial: Optional[np.ndarray] = None,
    mode: str = "sequential",
) -> TiledAnswer:
    """Block coordinate descent over the plan's patches, ``passes`` times.

    ``sequential``: each pass anneals the core patches in turn. On a cold
    start the opening patch sees no field and every later one sees the
    patches before it.

    ``batch-seam``: the first pass anneals every core patch at once, with only
    the spins already set as boundary, then the seam patches in turn against
    that result. The seams anneal the couplers the core cuts. Later passes
    anneal core and seam patches in turn.

    A patch result replaces the state only if the exact whole energy does not
    rise, or if the patch holds unset spins. An ``initial`` state (a warm
    start) therefore never gets worse. The answer is up to one row per read:
    the final state and the best variants that swap in one read of one patch
    of the last stage, so the race sees reads from every part of the graph.
    """
    if mode not in TILE_MODES:
        raise ValueError(f"tile mode {mode!r} is not one of {TILE_MODES}")
    h = np.asarray(h, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    ends = plan.ends
    if initial is None:
        state = np.zeros(len(plan.nodes), dtype=np.int8)
        energy = 0.0
    else:
        state = np.where(np.asarray(initial) < 0, -1, 1).astype(np.int8)
        energy = float(ising_energy(state, h, ends, j)[0])
    access_us = 0
    calls = 0
    seams = mode == "batch-seam" and len(plan.seams) > 0
    unreached = plan.seam_uncovered if seams else plan.uncovered
    last_stage: List[Tuple[Patch, np.ndarray]] = []
    # The batch stage calls from several threads at once.
    tally = threading.Lock()

    def call(patch: Patch, field_now: np.ndarray) -> np.ndarray:
        j_in = j[patch.edge_pos]
        reads, charged = submit(
            patch.live, clip_fields(field_now, patch.edge_local, j_in), patch.live_edges, j_in
        )
        nonlocal access_us, calls
        with tally:
            access_us += int(charged)
            calls += 1
        return np.asarray(reads, dtype=np.int8).reshape(-1, len(patch.pos))

    def accept(patch: Patch, field_now: np.ndarray, reads: np.ndarray) -> None:
        nonlocal state, energy
        local = ising_energy(reads, field_now, patch.edge_local, j[patch.edge_pos])
        candidate = state.copy()
        candidate[patch.pos] = reads[int(np.argmin(local))]
        candidate_energy = float(ising_energy(candidate, h, ends, j)[0])
        if candidate_energy <= energy or not np.all(state[patch.pos]):
            state, energy = candidate, candidate_energy
        last_stage.append((patch, reads))

    def in_turn(patches: Sequence[Patch]) -> None:
        for patch in patches:
            field_now = _field(patch, h, j, state)
            accept(patch, field_now, call(patch, field_now))

    for n in range(max(1, passes)):
        last_stage = []
        if seams and n == 0:
            from concurrent.futures import ThreadPoolExecutor

            fields = [_field(p, h, j, state) for p in plan.patches]
            with ThreadPoolExecutor(max_workers=len(plan.patches)) as pool:
                batch = list(pool.map(call, plan.patches, fields))
            for patch, field_now, reads in zip(plan.patches, fields, batch):
                accept(patch, field_now, reads)
            last_stage = []
            in_turn(plan.seams)
        elif seams:
            in_turn(plan.patches)
            last_stage = []
            in_turn(plan.seams)
        else:
            in_turn(plan.patches)
        if len(unreached):
            state = _settle(state, unreached, h, ends, j)
            energy = float(ising_energy(state, h, ends, j)[0])

    rows = max((len(r) for _, r in last_stage), default=1)
    states = _best_variants(state, last_stage, h, j, ends, rows)
    energies = ising_energy(states, h, ends, j)
    keep = np.argsort(energies, kind="stable")[:rows]
    return TiledAnswer(
        spins=states[keep],
        energies=energies[keep],
        access_us=access_us,
        calls=calls,
    )


def _best_variants(
    state: np.ndarray,
    stage: Sequence[Tuple[Patch, np.ndarray]],
    h: np.ndarray,
    j: np.ndarray,
    ends: np.ndarray,
    rows: int,
) -> np.ndarray:
    """``state`` and the ``rows`` best one-patch variants of it, distinct.

    Each variant swaps one read of one patch into the final state. They are
    ranked by the patch's local energy against the final state, which is
    exact apart from couplers inside a patch that the chip lacks. The caller
    scores the survivors exactly.
    """
    deltas, variants = [], []
    for patch, reads in stage:
        field_now = _field(patch, h, j, state)
        j_in = j[patch.edge_pos]
        base = ising_energy(state[patch.pos], field_now, patch.edge_local, j_in)[0]
        delta = ising_energy(reads, field_now, patch.edge_local, j_in) - base
        for d, read in zip(delta.tolist(), reads):
            deltas.append(d)
            variants.append((patch, read))
    picked = [state]
    for k in np.argsort(np.asarray(deltas), kind="stable")[: 2 * rows]:
        patch, read = variants[int(k)]
        v = state.copy()
        v[patch.pos] = read
        picked.append(v)
    return np.unique(np.stack(picked), axis=0)


def _settle(
    state: np.ndarray, positions: np.ndarray, h: np.ndarray, ends: np.ndarray, j: np.ndarray
) -> np.ndarray:
    """Align each unreached spin with its local field; +1 where the field is 0.

    No anneal sets these spins. One alignment step is their exact optimum
    given the rest, when no two of them are neighbours; when some are, it is
    one step of descent.
    """
    state = state.copy()
    s = state.astype(np.float64)
    field = h.copy()
    field += np.bincount(ends[:, 0], weights=j * s[ends[:, 1]], minlength=len(h))
    field += np.bincount(ends[:, 1], weights=j * s[ends[:, 0]], minlength=len(h))
    state[positions] = np.where(field[positions] > 0, -1, 1).astype(np.int8)
    return state
