# Mining on Advantage2_system4

Status as of November 14, 2026.

The miner mines the network's topology on `Advantage2_system4`, the smaller
Advantage2 in the `na-east-1` region. The chip is a Zephyr Z(6,4) with 1,200
working qubits. The network topology is the Z(12,4) working graph of
`Advantage2_system1`, with 4,560 nodes and 41,205 couplers. The miner cuts
that graph into chip-sized patches and anneals them on the QPU.

On the instances of 17 winning qblocks, one tiled job plus the production
race ends a median 33.7 units above the gate. No job reached a gate. The QPU
does real work in the job: the same race from random states ends 70.6 units
above the gate.

## Run it

Name the solver in the `[dwave]` section of the coordinator config. The
coordinator sends the section to the miner as `backend_toml`, and the miner
reads it before it connects:

```toml
[dwave]
solver = "Advantage2_system4"   # or "Advantage2_system1"
region = "na-east-1"            # system1 is in "na-west-1"
tile_mode = "batch-seam"        # default "sequential"
tile_passes = 1                 # default 2
```

Without `solver` and `region`, the SDK chooses from `DWAVE_API_SOLVER`,
`DWAVE_API_REGION`, and `dwave.conf`. A miner keeps the solver it connected
to for the whole session.

On system4 the miner logs `session topology (4560 nodes) is larger than this
Z(6,4) chip` when the Topology arrives. `batch-seam` with one pass is the
measured best setting. Every other setting (budget, reads, anneal time,
postprocess) works as on system1. The budget ledger bills the access time of
every patch call.

The coordinator launches one D-Wave miner, from `[dwave]`, for each config.
Two Advantage2 chips at once need two coordinators, or a change in
`quip-solver` that accepts indexed `[dwave.N]` sections as `[cuda.N]` does.

## What the network sends

The chain has one mineable topology, `2d130ce6…a5345c`. Its family descriptor
is `quip/twisted-ice-dimer` revision 1 with `zephyr_m` 12, `zephyr_t` 4, a
weak coupling of 0.045, and 1,350 ferromagnetic pairs. Every h is 0. The tile
and pair couplers are ±1, and the weak layer is ±0.045. Qblocks 355 to 371
were won a median 0.3 units below their gates.

## How the tiling works

`quip_solver_dwave/tiling.py` holds the code.

1. When the Topology arrives, a background thread places the chip at each of
   the 169 sub-lattice offsets of Z(6) in Z(12). It picks core placements
   greedily until a placement would reach fewer than 1% of the nodes. The
   live topology gets five core patches. They cut 2,649 couplers and miss 125
   nodes, which are neighbours of the chip's dead qubits.
2. The same thread then picks up to eight seam placements that cover what the
   core misses. Eight seams leave 162 couplers and 3 nodes unannealed.
3. Each job binds those cuts to its own node and coupler order.
4. For each patch the miner holds the other spins fixed. A coupler that
   leaves the patch becomes a field on the qubit inside it, so the patch
   anneal is the exact restriction of the whole problem.
5. The miner keeps the patch read with the lowest energy, and only when the
   exact energy of the whole state does not rise.

The two modes order the patches differently:

- `sequential` anneals the five core patches one after another, each
  against the patches before it, once for each pass.
- `batch-seam` anneals the five core patches at once with no boundary, then
  the eight seam patches one after another against that result. The idea follows the
  block-virtualized anneals of riff-vm. The exact separator recombination of
  riff-vm does not apply, because a Z(12) cut between two Z(6) patches holds
  hundreds of qubits.

A job returns 48 whole states: the final state, and the best variants that
swap in one read of one patch of the last stage.

## Live results

The run used qblocks 355 to 371. Each instance was rebuilt from its on-chain
nonce with the wheel's `families.generate`, so every instance is the one the
winner solved. Every arm used 48 reads of a plain 160 µs anneal and went
through the production race, `qpp-race2048-repair32/1`.

| Arm | Calls | Access | Median energy | Median above gate | Range above gate |
|---|---|---|---|---|---|
| Sequential, 2 passes, raw | 10 | 196 ms | −4,979.8 | 102.9 | 85.1 to 125.2 |
| Batch-seam, 1 pass, raw | 13 | 255 ms | −5,003.8 | 79.8 | 59.5 to 106.4 |
| Sequential, 2 passes, raced | 10 | 196 ms | −5,044.5 | 39.0 | 30.3 to 49.1 |
| Batch-seam, 1 pass, raced | 13 | 255 ms | −5,049.9 | 33.7 | 21.6 to 41.4 |
| Race from 48 random states | 0 | 0 | −5,012.9 | 70.6 | 59.9 to 81.2 |
| Chain winner | | | −5,083.2 | −0.3 | −3.9 to 0.0 |

- Batch-seam beats sequential by 5.0 units after the race, standard error
  1.8, on 13 of 17 instances.
- Batch-seam beats the race from random states by 36.7 units on 17 of 17.
- Each patch call cost 19.6 ms of access. Wall time was 2.6 to 6.9 s for each
  job, mostly SAPI round trips.
- Total spend for the smoke test and this run was 7.98 s of system4 access.

A system1 job on the same instances was not run, so this table does not
compare the two chips.

## Other measured facts

| Quantity | Value |
|---|---|
| Strong couplers (tile edges and pairs) that the core cuts | 6.3%, over 20 instances |
| Largest folded field, before and after the clip | 5.1 and 4.2, against an h range of 6 |
| Cut choice, once for each Topology | 1.2 s on its own thread |
| Binding a job graph | 12 ms for the first job, 0.2 ms after |

The clip limits each folded field to the in-patch coupling of its qubit plus
1 unit. A larger field does not change the optimum, but it would make SAPI
scale down every coupler of the patch.

| Call, 48 reads at 160 µs | system1, whole graph | system4, one patch |
|---|---|---|
| Programming | 33.6 ms | 8.0 ms |
| Access for each call | 48.9 ms, from published timing | 19.6 ms, measured |

## Performance options

### Now

**A. Batch-seam, one pass.** This is the measured best mode: 33.7 units above
the gate after the race, at 255 ms of access for each job. It does not win
alone. Confidence label: measured, 17 instances.

### Next

**C. Patch loop from the batch-seam state.** This run used qblocks 366 to 371
and 2.48 s of access. Each instance started from batch-seam and the race.
Eight patch calls followed. Each call re-annealed one core or seam placement at random with
the rest held, and raced the 48 swapped-in reads at 768 sweeps. A lower result
replaced the state.

| qblock | Above gate before the loop | Above gate after 8 calls | Calls that lowered the state |
|---|---|---|---|
| 366 | 32.1 | 30.2 | 4 |
| 367 | 33.9 | 29.0 | 4 |
| 368 | 33.2 | 31.6 | 4 |
| 369 | 34.9 | 27.5 | 5 |
| 370 | 30.9 | 29.1 | 3 |
| 371 | 35.5 | 30.5 | 3 |

The loop gains a mean 3.7 units in eight calls, the same as Run J measured
for a 25% patch on system1. The gain has not levelled off at eight calls, but
a gap near 30 units needs more than a longer loop. A faster postprocess for
each call is the next lever. Confidence label: measured, 6 instances.

**D. Pair system4 with system1.** System1 makes the deep state with one
whole-graph call and the race. System4 then runs option C on that state at
under half the access cost of a system1 call. Confidence label: directional.

**E. The anneal window.** Run G found that an anneal that slows across
s = 0.26 to 0.32 gains 5 to 6 units after the race at the same access cost.
The window applies to patch calls without code changes. Confidence label: directional, measured on system1 whole-graph calls.

**F. A system1 baseline on the same instances.** One whole-graph call and the
race on the 17 instances costs about 0.8 s of system1 access. It turns the
table above into a chip comparison.

### Later

**G. A native system4 topology on the chain.** Root registers a Z(6,4)
twisted-ice-dimer topology on the system4 working graph, sets its difficulty,
and adds it as the one non-default mineable topology. The coordinator also needs a new option. Today `fetch_mining_snapshot`
passes `None` and so always mines the default topology. One call would then be one job at
19.6 ms of access. This is a governance change and a change to
`quip-solver`, not to this repository. Confidence label: exploratory.
