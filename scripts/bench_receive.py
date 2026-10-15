#!/usr/bin/env python3
"""Time the receive path against the SampleSet route, on real hardware.

Two configurations are compared, one job each:

  OLD  Future built with return_matrix=False, answer read via .sampleset.
       The decoder calls .tolist() on a (reads x qubits) array, wait_sampleset
       walks the lists with a nested comprehension, and dimod converts the
       result back into numpy.
  NEW  Future built with return_matrix=True, answer read via answer_view.
       The decode stays in numpy and no SampleSet is built.

A Future caches its decode, so one job cannot measure both configurations
from cold. Hence two submits. Each timed region starts at an undecoded Future
and ends with the arrays the miner uses. Only the network wait is excluded.

Spends about 86 ms of QPU access time across the two jobs.
"""

import random
import sys
import time

import numpy as np
from dwave.system import DWaveSampler

from quip_solver_dwave.answer import answer_view

N_READS = 48
N_COUPLERS = 41514


def build_problem(base):
    """A production-shaped Ising problem, reproducible from a fixed seed."""
    rng = random.Random(7)
    n_couplers = min(N_COUPLERS, len(base.edgelist))
    edges = rng.sample([(int(u), int(v)) for u, v in base.edgelist], n_couplers)
    nodes = sorted({q for e in edges for q in e})
    h = {int(n): 0.0 for n in nodes}
    j = {e: rng.choice((-1.0, 1.0)) for e in edges}
    return h, j, nodes, n_couplers


def submit(solver, h, j, *, return_matrix: bool):
    """Submit one job with return_matrix pinned.

    Solver._sample reads self.return_matrix when it constructs the Future,
    which is the same flag OceanSampler._submit_encoded passes explicitly.
    Setting it on the solver is the only way to steer sample_ising.
    """
    solver.return_matrix = return_matrix
    return solver.sample_ising(h, j, num_reads=N_READS, label="quip-receive-bench")


def main() -> int:
    base = DWaveSampler(request_timeout=(60, 300))
    solver = base.solver
    h, j, nodes, n_couplers = build_problem(base)
    print(f"{solver.name}: {len(nodes)} nodes, {n_couplers} couplers")

    # Submit both jobs up front so they queue back to back.
    fut_old = submit(solver, h, j, return_matrix=False)
    fut_new = submit(solver, h, j, return_matrix=True)

    # The network wait is not part of either measurement.
    fut_old.wait()
    fut_new.wait()

    # OLD: undecoded Future -> SampleSet. The sampleset property hands back a
    # SampleSet built from_future, which does no work until something resolves
    # it, so .record has to be inside the timed region.
    t0 = time.perf_counter()
    ss_old = fut_old.sampleset
    ss_old.record
    old = time.perf_counter() - t0

    # NEW: undecoded Future -> arrays.
    t0 = time.perf_counter()
    view = answer_view(fut_new)
    new = time.perf_counter() - t0

    print(f"\nOLD  return_matrix=False, .sampleset : {old * 1000:7.2f} ms")
    print(f"NEW  return_matrix=True,  answer_view: {new * 1000:7.2f} ms")
    print(f"     speedup                         : {old / new:.1f}x")
    print(f"\nspins        : {view.spins.shape} {view.spins.dtype}")
    print(f"reads        : {view.reads}")
    print(f"access time  : {view.access_time_us / 1000:.1f} ms")

    # Equivalence is checked within each job, not across them: two jobs are two
    # sets of anneals and their energies differ. Both Futures are decoded by
    # now, so reading each one the other way is cheap.
    cross_old = answer_view(fut_old)
    ss_new = fut_new.sampleset
    ok = True
    for name, v, ss in (("OLD", cross_old, ss_old), ("NEW", view, ss_new)):
        same_shape = v.spins.shape == ss.record.sample.shape
        same_energy = np.allclose(sorted(v.energies), sorted(ss.record.energy))
        ok = ok and same_shape and same_energy
        print(f"\n{name} shape matches the SampleSet  : {same_shape}")
        print(f"{name} energies match the SampleSet : {same_energy}")

    base.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
