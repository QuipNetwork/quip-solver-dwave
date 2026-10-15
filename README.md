# quip-solver-dwave

D-Wave (Ocean) QPU Ising solver for the [quip.network](https://gitlab.com/quip.network)
solver contract, protocol version 2. Ships as a pip-installable Python package
with a `quip-dwave-qa` console entry point.

Unlike the classical miners (`quip-solver-cpu`/`-cuda`/`-metal`), there is no
SA/Gibbs binary split: this backend samples on a D-Wave QPU via Ocean
(`dwave.system.DWaveSampler`), and every job goes to the QPU.

For tests and offline runs, `QUIP_DWAVE_MOCK=1` or `--mock` swaps in a dimod
sampler. The default is `ExactSolver`, which enumerates every state and so
handles only the smallest problems. Set `QUIP_DWAVE_MOCK_BACKEND=sa` for
`SimulatedAnnealingSampler`, which scales to realistic topology sizes. You
select this path explicitly; the miner never switches to it on its own.

## Install

### Released executable (macOS, Apple Silicon)

Each release publishes `quip-dwave-qa-darwin-arm64`, a self-contained
executable that bundles Python, the Ocean SDK, and the compiled
`quip_solver_core` extension. Drop it in and run it. The host needs only the
file.

```sh
TAG=v0.3.0-rc3   # or a later release
BASE=https://gitlab.com/api/v4/projects/84792347/packages/generic/quip-solver-dwave
curl -fLO "$BASE/$TAG/quip-dwave-qa-darwin-arm64"
chmod +x quip-dwave-qa-darwin-arm64
mv quip-dwave-qa-darwin-arm64 /usr/local/bin/quip-dwave-qa
```

Rename it, because the coordinator spawns whatever `[dwave] binary` names and
its default is `quip-dwave-qa`. The CPU, CUDA, and Metal miners install the
same way.

### From source

```sh
pip install "quip-solver-dwave @ git+https://gitlab.com/quip.network/quip-solver-dwave.git"
```

The `quip_solver_core` SDK dependency installs from a published PyPI wheel on
Linux (amd64 and aarch64) and macOS (Apple Silicon). Elsewhere pip falls back
to building it from source through maturin, which needs **Rust** on the
machine.

Reaching real QPU hardware takes a D-Wave Leap token, set through
`DWAVE_API_TOKEN`. `DWAVE_API_KEY`, the name the v0.2 stack used, still works:
the miner copies it to `DWAVE_API_TOKEN` at startup and logs a deprecation
warning. Offline and classical sampling take no token.

### Building the executable yourself

```sh
pip install ".[dev]" pyinstaller
pyinstaller --clean --noconfirm pyinstaller/quip-dwave-qa.spec
QUIP_DWAVE_MOCK=1 ./dist/quip-dwave-qa --check
```

`pyinstaller/quip-dwave-qa.spec` names each collected package and gives the
reason. Ocean keeps compiled modules inside PEP 420 namespace packages that
PyInstaller does not find on its own, and PyInstaller reports a successful
build for a binary that cannot import `dimod`. Run the result before trusting
it.

## Running

**Connect to a coordinator** (production):

```sh
quip-dwave-qa --quip-coordinator unix:///run/quip/coord.sock
# or: python -m quip_solver_dwave --quip-coordinator ...
```

**Driver / fixed-input (run in isolation, no chain).** Use the coordinator's
`drive` harness pointed at the `quip-dwave-qa` entry point — `--source random`
for golden-drawn problems, `--source list <jsonl>` for a fixed replay.

**One-shot solve** (no coordinator). `--solve` reads one problem as JSON on
stdin and writes every read as JSON on stdout, as section 2 of
quip-solver-core's SPEC.md defines it. It never starts a session, so no budget,
ledger or round strategy runs.

```sh
quip-dwave-qa --solve < problem.json
```

Node `i` of the problem is qubit `i`. The model must already sit on the chip's
working graph; a node or coupler the chip lacks exits 65. The optional
`anneal_time_us` key sets the anneal time. The SA keys `num_sweeps`,
`sweeps_per_beta`, `beta_range` and `seed` are required by the shared schema,
steer nothing on a QPU, and are echoed back.

**Introspection:**

```sh
quip-dwave-qa --capabilities
quip-dwave-qa --check
```

## Pipeline depth

The miner keeps up to `queue_depth` submissions on the QPU at once. A
cloud-attached QPU spends most of each job's wall time on the round trip, so a
shallow pipeline leaves the device idle between jobs.

The default is 96. It comes from Little's Law:

    depth = chip throughput * round trip

Measured on `Advantage2_system1` with a production-sized problem (4577 nodes,
41514 couplers, `num_reads=48`): 43.2 ms of access time per job gives a chip
ceiling of 23.2 jobs/s, and an uncontended round trip is 1.57 s. That needs 36
in flight. A round trip of 3.05 s, which session logs show under load, needs
71. The default covers a round trip of 4.14 s, so a connectivity blip reduces
throughput instead of stalling the QPU.

Set `queue_depth` in the coordinator's `[dwave]` backend config to override
it:

```toml
queue_depth = 64
```

The operator value wins. If unset, the miner uses the depth the coordinator
sent. If that is also unset, the miner uses 96.

Depth does not raise jobs per hour under a budget. The budget paces spend
across the quota period, and it funds fewer jobs per second than even a
shallow pipeline delivers. Depth sets the burst rate and the duty cycle.

Depth does set how much work a reseed can strand. When the coordinator cancels
a generation, the miner asks D-Wave to drop every job of that generation still
on the QPU. D-Wave refunds only the jobs it has not started annealing, so the
quota pays for the rest. Each session log reports the measured hit rate:

```
[QPU] cancel gen<=339: asked D-Wave to drop 3 in-flight job(s), 3 still
running | session: 12 cancelled, 4 annealed anyway (33% missed)
```

The same config holds other `[dwave]` keys: `budget`, `budget_reset_day`,
`usage_db`, `num_reads`, and `anneal_time_us`. This README does not document
them yet.

## Anneal schedule and warm starts

The miner states every anneal as an `anneal_schedule`. It never sends
`annealing_time`, because SAPI refuses a problem that carries both. A forward
anneal of `T` microseconds is `[[0, 0], [T, 1]]`. `T` is `anneal_time_us`, or
the solver's `default_annealing_time` when `anneal_time_us` is 0.

A job that carries `IsingProblem.initial_spins` runs as a reverse anneal from
its first state. The miner lists the `initial-spins` feature for that reason.
`anneal_time_us` sets the slope of both ramps, so a ramp to the reversal point
`s` takes `(1 - s) * anneal_time_us`. Two `[dwave]` keys set the defaults for a
job that leaves fields 11 and 12 at 0:

```toml
reversal_s_milli = 500   # reversal point s, 1 to 999, in milli-units
reversal_pause_us = 25   # hold at s, in microseconds
```

The built-in defaults are 500 and 25, which is D-Wave's documented example.
`scripts/ab_anneal_schedule.py` checks on the live QPU that the schedule form
costs and performs the same as the old form.

## Measurement harness (QUI-1387 steps 5 to 7)

Six scripts measure the hybrid pipeline against the recorded QPU attempts of a
node. No production code path of the miner imports them. They need the
`quip_msa` wheel from quip-solver-cpu. Keep the data outside the repository, in
one directory such as `~/quip-data/qui-1387/`. That directory holds the
topology spec, `attempts.csv`, and every output.

Run them in this order:

1. `scripts/fetch_attempts.py` reads the attempts logs of a node over ssh and
   writes `attempts.csv`.
2. `scripts/lite_dataset.py` rebuilds each model from its nonce and records the
   MSA-lite energy and the rate at each sweep count.
3. `scripts/lite_report.py` prints recall and false positives for each sweep
   count and pass share, then writes `low-energy.nonces` and
   `false-positive.nonces`.
4. `scripts/heavy_replay.py` runs MSA at high sweeps on both lists through
   `quip-coordinator drive` and quip-solver-cuda, then compares the two sets.
5. `scripts/capture_qpu_reads.py` submits sampled models to the QPU and keeps
   every read.
6. `scripts/seeded_sweep.py` starts the MSA kernel from those reads at a range
   of start betas, and compares each run with a cold run.

Only `capture_qpu_reads.py` spends QPU time. It prints the estimate with
`--dry-run`, refuses to submit without `--yes`, and stops at
`--max-qpu-seconds`.

## History and the profile report

The miner keeps three tables in the usage database, beside the budget
ledger. One tracks throughput per UTC hour, covering job completions, QPU
busy time, round trips, and the D-Wave service time from SAPI's
`submitted_on` and `solved_on`. Another holds one row per qblock round. The
last holds a histogram of each job's best energy relative to the round's
target. The round strategy reads them. Nothing about mining depends on
them. A database that fails to open disables the history and logs one
warning. Every write from the session thread and the job workers goes
through the recorder's one worker thread. The seed and outcome-pickup
threads below write through the database's own lock instead.

At start the miner seeds past rounds from the coordinator's attempts files,
`<data_dir>/<qblock_id>/attempts.jsonl`. The default location is the
`attempts` directory beside the usage database, which is where the node
manager renders it. `--attempts-dir PATH` overrides it. Seeding and outcome
pickup keep only the lines for `--miner-id`, so a node running more than
one QPU miner does not absorb a sibling miner's history.

`quip-dwave-qa --profile [--usage-db PATH]` prints a grid of jobs per
second by day of the month and hour of the day, the hour factors and the
day factors behind it, a grid of D-Wave queue wait, the round totals, and the last 20
rounds with the jobs the strategy expected beside what happened. It needs
no QPU and no token. `--usage-db` applies to `--profile` only. A session
reads the coordinator's `usage_db` key instead.

## The round strategy

At every qblock boundary the budget decides first whether the miner can
afford the round. When it can, one function decides whether this round is
worth joining or whether the funds should wait for a faster hour. It reads
a snapshot of the history: throughput per slot, the round length, and the
QPU time per job. A joined round runs to its end. The budget line is a
target for the month and is consulted at boundaries only. The one thing
that stops a round in progress is the period's whole allotment being spent.

Winning a round depends on the round's difficulty, which the protocol sets
and the miner cannot see ahead, and on how many models the QPU evaluates
while the round is open. Only the second varies with time, so the strategy
compares slots by deliverable jobs and nothing else. More jobs per round is
more of the search space covered.

A slot is an hour of the day on a day of the month. D-Wave accounts run on
monthly contracts, so the shared queue follows the month as well as the
working day. Months of 28 to 31 days map onto 28 bins, so every month fills
every bin. The estimate for each of the 672 cells is the global rate times
an hour-of-day factor times a day-of-month factor. A cell with enough jobs
of its own shrinks toward that prediction instead of replacing it. This is
the multiplicative main-effects model that call-center forecasting uses for
arrival rates by day and time of day.

A join costs a whole round at this slot's rate, and the funds the rest of
the period will have, the headroom now plus the accrual until the reset,
cover only some of the rounds left in it. The best use of a fixed allotment
across hours of varying rate is to spend it in the fastest ones. The
decision ranks the remaining rounds by the jobs each would deliver, walks
down until their cost exhausts the funds, and calls that slot's jobs the
bar. It joins when this round's jobs clear the bar, within a tolerance. The
verdict and its numbers are one log line per boundary.

Two `[dwave]` keys tune it. The defaults change nothing until history
exists.

```toml
min_throughput_advantage = 0.25  # skip a round only when the bar sits more than 25% above its jobs
participation_chance = 0.10      # join this share of rounds regardless, so every slot keeps getting measured
```

The `reason` column in `--profile` and the strategy's log line name the verdict with one of these strings:

- `no-data`: no history yet, so every round the budget allows is joined.
- `explore`: the participation chance drew this round regardless of the verdict below.
- `saturated`: the funds cover every round left in the period, so the round is joined.
- `fast-slot`: this round clears the bar, so it is joined.
- `slow-slot`: this round is under the bar, so it is skipped and the funds wait for a faster hour.

Three reasons predate the strategy and still appear where it plays no part: `budget` (the budget allowed the round and no strategy is attached yet), `budget-sat-out` (spend is past the budget line, so the miner sits the round out before any strategy is consulted), and `unbudgeted` (no budget is configured, so every round is joined).

## Tests

```sh
pip install -e ".[dev]"
pytest quip_solver_dwave/tests -v
```

`test_conformance.py` runs quip-solver-core's session and `--solve` gates when
`quip-solver-drive` is on `PATH` or named by `QUIP_SOLVER_DRIVE`, and skips
otherwise.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
