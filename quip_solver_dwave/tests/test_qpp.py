"""The QPU-aware postprocessor: the model file, the host rescore, and the binary round trip."""

from __future__ import annotations

import numpy as np
import pytest

from quip_solver_dwave import qpp


def _model(tiles: int = 4) -> qpp.DimerModel:
    """Tiles of four spins with ice-like couplings, linked in a ring by pairs."""
    rng = np.random.default_rng(20261111)
    n = 4 * tiles
    tile = np.arange(n).reshape(tiles, 4)
    edges, j = [], []
    for t in tile:
        signs = rng.choice([-1, 1], 4)
        for a in range(4):
            for b in range(a + 1, 4):
                edges.append((t[a], t[b]))
                j.append(int(1000 * signs[a] * signs[b]))
    pairs = np.array([(tile[k, 3], tile[(k + 1) % tiles, 0]) for k in range(tiles)])
    for u, v in pairs:
        edges.append((u, v))
        j.append(-1000)
    for _ in range(n):
        u, v = rng.choice(n, 2, replace=False)
        edges.append((u, v))
        j.append(int(rng.choice([-250, 250])))
    return qpp.DimerModel(
        n=n,
        edges=np.array(edges, dtype=np.int64),
        j_milli=np.array(j, dtype=np.int64),
        tiles=tile,
        pairs=pairs,
    )


def _states(model: qpp.DimerModel, k: int) -> np.ndarray:
    return np.random.default_rng(7).choice(np.array([-1, 1], dtype=np.int8), (k, model.n))


def test_the_gauge_follows_the_sign_of_each_coupling_to_the_first_tile_spin():
    model = _model()
    gauge = model.gauge()
    w = {(min(u, v), max(u, v)): j for (u, v), j in zip(model.edges.tolist(), model.j_milli.tolist())}
    assert (gauge[:, 0] == 1).all()
    for t, nodes in enumerate(model.tiles):
        for k in range(1, 4):
            a, b = sorted((int(nodes[0]), int(nodes[k])))
            assert gauge[t, k] == (1 if w[(a, b)] > 0 else -1)


def test_the_model_file_has_the_research_layout(tmp_path):
    model = _model(tiles=3)
    path = tmp_path / "m.bin"
    qpp.write_model(path, model)
    raw = path.read_bytes()
    t, p, e = len(model.tiles), len(model.pairs), len(model.edges)
    assert len(raw) == 4 * 4 + 12 * e + 16 * t + 4 * t + 8 * p
    head = np.frombuffer(raw[:16], dtype="<i4")
    assert head.tolist() == [model.n, t, p, e]
    rows = np.frombuffer(raw[16 : 16 + 12 * e], dtype="<i4").reshape(e, 3)
    assert rows[:, :2].tolist() == model.edges.tolist()
    assert rows[:, 2].tolist() == model.j_milli.tolist()
    gauge = np.frombuffer(raw[16 + 12 * e + 16 * t : 16 + 12 * e + 20 * t], dtype=np.int8)
    assert gauge.reshape(t, 4).tolist() == model.gauge().tolist()


def test_energies_are_milli_and_match_a_direct_sum():
    model = _model()
    states = _states(model, 3)
    direct = [
        sum(j * s[u] * s[v] for (u, v), j in zip(model.edges.tolist(), model.j_milli.tolist()))
        for s in states.astype(np.int64)
    ]
    assert qpp.energies_milli(model, states).tolist() == direct


def test_a_model_with_an_out_of_range_spin_is_refused():
    model = _model()
    bad = qpp.DimerModel(
        n=model.n, edges=model.edges, j_milli=model.j_milli, tiles=model.tiles, pairs=model.pairs + model.n
    )
    with pytest.raises(ValueError, match="outside"):
        bad.gauge()


binary = pytest.mark.skipif(qpp.find_binary() is None, reason="quip-dwave-qpp is not built")


@binary
@pytest.mark.parametrize(
    ("mode", "params"),
    [
        ("chain", {"sweeps": 64, "b0": 6}),
        ("race", {"stages": "4:16,1:32", "pop_first": 0, "pop_count": 6}),
        ("ratchet", {"rounds": 2, "children": 2, "sweeps": 8}),
        ("pa", {"steps": 3, "sweeps": 4, "pop_first": 0, "pop_count": 6}),
    ],
)
def test_every_mode_returns_a_state_the_host_rescores_to_its_best(tmp_path, mode, params):
    model = _model()
    seeds = _states(model, 6)
    result = qpp.run(model, seeds, mode, workdir=tmp_path, **params)
    assert result.mode == mode
    assert result.state.shape == (model.n,) and result.state.dtype == np.int8
    assert int(qpp.energies_milli(model, result.state[None])[0]) == result.best_milli
    assert result.best_milli <= result.start_milli or mode in ("race", "pa")
    assert result.trace and all(len(point) == 4 for point in result.trace)
    assert result.cpu_s >= 0


@binary
def test_the_same_seed_gives_the_same_state(tmp_path):
    model = _model()
    seeds = _states(model, 6)
    a = qpp.run(model, seeds, "chain", workdir=tmp_path, sweeps=64, seed=3)
    b = qpp.run(model, seeds, "chain", workdir=tmp_path, sweeps=64, seed=3)
    assert a.best_milli == b.best_milli and (a.state == b.state).all()


@binary
def test_a_frozen_mask_keeps_the_frozen_spins(tmp_path):
    model = _model()
    seeds = _states(model, 1)
    mask = np.zeros(model.n, dtype=bool)
    mask[: model.n // 2] = True
    result = qpp.run(model, seeds, "chain", workdir=tmp_path, mask=mask, sweeps=64)
    # The anneal and the final polish visit free spins only.
    frozen = ~mask
    assert (result.state[frozen] == seeds[0][frozen]).all()


@binary
def test_a_failed_run_raises_with_the_binary_message(tmp_path):
    model = _model()
    with pytest.raises(qpp.QppError, match="seed_index"):
        qpp.run(model, _states(model, 2), "chain", workdir=tmp_path, seed_index=5)
