# AGENTS.md

`quip-solver-dwave` is a D-Wave QPU Ising miner for the quip.network v0.3
mining protocol. It speaks the miner gRPC protocol to a coordinator over a
Unix socket and samples every job on a real QPU through the Ocean SDK.

## Commands

```sh
pip install -e ".[dev]"                      # dev install

pytest quip_solver_dwave/tests -q             # full suite (~2s, no QPU needed)
pytest quip_solver_dwave/tests/test_qp_encoding.py -q          # one file
pytest quip_solver_dwave/tests -q -k "cancel and not accounting"  # by name
pytest quip_solver_dwave/tests -q --durations=10               # find slow tests

ruff check quip_solver_dwave/                 # linter (CI does not run it; keep it clean anyway)
pyright quip_solver_dwave/                    # type checker (same)

QUIP_DWAVE_MOCK=1 quip-dwave-qa --check      # offline self-test
quip-dwave-qa --capabilities                 # print the advertised Capabilities
quip-dwave-qa --profile --usage-db /data/qpu-usage.db   # day-of-month by hour history, no QPU needed
quip-dwave-qa --quip-coordinator unix:///run/quip/coord.sock
```

CI (`.gitlab-ci.yml`) runs `pytest quip_solver_dwave/tests -v`, a wheel build,
and a conformance job that needs the external `quip-solver-drive` harness:

```sh
QUIP_SOLVER_DRIVE=/path/to/quip-solver-drive pytest quip_solver_dwave/tests/test_conformance.py -v
```

`ruff format` is not this project's baseline — several files predate it and
reformatting them is churn. Run `ruff check` and `pyright`, not the formatter.

Tests must not touch a real QPU. `QUIP_DWAVE_MOCK=1` (or `--mock`) swaps in a
dimod sampler; `QUIP_DWAVE_MOCK_BACKEND=sa` scales to realistic topologies,
while the default `ExactSolver` enumerates every state and only handles tiny
problems.

## Architecture

### The session is one thread, and credits are the throttle

`session_loop.run_session` is the whole protocol. Hello, then Welcome, then
Configure, then a credit/job cycle, all on one thread reading the coordinator's gRPC stream.
Jobs are handed to a `ThreadPoolExecutor` sized to the pipeline depth; every
other message is handled inline.

**The coordinator dispatches only against credits.** Granting them is how this
miner says "I am participating" and withholding them is how it sits a round
out. `Ready` says the session is established. Credits say the QPU is working.
They are deliberately separate messages.

Because everything funnels through that one thread, **nothing on it may block**.
Two bugs of exactly that shape have been fixed (see `test_ledger_contention`,
`test_budget_cache`), and a sibling miner lost up to 87 seconds per round to the same
class of problem (`quip-solver#33`). Do not add IO under `state_lock`.

### A Cancel is the only round boundary the miner can see

`Cancel(max_generation=N)` is the only monotone round counter the coordinator
sends, and it arrives every round even while the miner holds no credits. That
makes it the point where participation is decided, where the reseed watermark
is raised, and where in-flight work is cancelled. Jobs at or below the watermark are
"abandoned": `_is_abandoned` decides, generation 0 (mempool) never is.

### The budget is a gate, not a rate limiter

`BudgetPacer.decide` answers one question. Is cumulative spend under the flat
allowance line for this point in the quota period? If it is, the miner
participates in the **whole next qblock** at full speed. Nothing throttles
inside a round. The long-run average is bounded by arithmetic, quota divided
by access time per job, rather than by pacing.

Spend lives in `usage.UsageLedger`, a SQLite file that survives restarts.
`usage_db` defaults to a shared path, so two miners on one D-Wave account may
share it — the pacer caches the live period in memory but re-reads on a short
interval so a sibling's spend is not invisible.

**Two different ledgers, never conflate them.** Coordinator credits are
protocol flow control. D-Wave access time is money. A job the coordinator
throws away still costs quota.

### Billing rules that are easy to get wrong

- Bill **before** the abandoned check. D-Wave charged for the anneal whatever
  the coordinator decided to do with the answer.
- A cancelled submission raises with no timing attached. The sampler books a
  conservative estimate rather than nothing, because under-counting hands the
  pacer headroom the QPU has already spent.
- A submit that died in this process is billed nothing — D-Wave never saw it.
  `OceanSampler.sample` distinguishes the two by *which* future failed.
- An offline solver fails the problem after SAPI accepts it, and D-Wave
  charges nothing. The sampler bills nothing for it, and `handle_job` raises
  `SolverUnavailable` instead of a refunded reject. The session loop parks the
  credit until the next qblock boundary, so the miner probes the solver once
  per round rather than spinning on rejects at SAPI round-trip speed.
- An offline solver at connect time is not fatal either. `_connect_solver`
  logs it, the session sends Ready with no credits, and the connect is
  retried at each qblock boundary. The round is recorded as `solver-offline`.
  Exiting instead trips the supervisor's six-restart limit and leaves the
  miner dead after the solver returns. Other connect errors still exit.

### Arrays from the wire to SAPI and back

Jobs carry dense positional arrays, and so does the SAPI payload, so nothing
in between becomes a dict:

```
Job proto -> _resolve_problem -> (nodes, h, edges, j) numpy arrays
          -> QpEncoder.plan/encode -> base64 qp payload
          -> build_submission_body -> client._submit
```

`dwave.cloud.coders.encode_problem_as_qp` is the **specification** for that
payload, rather than a starting point. `test_qp_encoding` asserts byte equality against
it, including at production scale (4577 qubits, 41514 couplers). If you change
the encoder, that equality is the contract.

Coming back, spins stay in the sampler's `(reads, qubits)` int8 array. The wire
format is one signed byte per spin, so `row.tobytes()` is already the payload.
`test_spin_encoding` pins that against `wire.encode_spins`.

`answer.answer_view` reads the answer straight off the `Future`, never
through `Future.sampleset`. That property turns the decoder's numpy arrays
into Python lists. It then walks those lists with a nested comprehension over
reads times variables. dimod converts the result back into numpy, to arrive
at the arrays the decoder already had. `OceanSampler._submit_encoded` passes
`return_matrix=True` for the same reason. That flag stays safe only because
nothing here builds a SampleSet.

`scripts/bench_receive.py` times the two configurations on the live QPU, one
job each. The old configuration builds a `Future` with `return_matrix=False`
and reads it through `.sampleset`. The new one builds a `Future` with
`return_matrix=True` and reads it through `answer_view`. At production size,
4577 qubits and 41514 couplers and 48 reads, the old configuration costs
38.8 ms a job. The new one costs 1.6 ms, a factor of 24.

### One spelling for the anneal

`OceanSampler._anneal_params` is the only place that turns `anneal_time_us`
and a warm start into SAPI parameters, and it never emits `annealing_time`.
SAPI refuses a problem that carries both forms, and a reverse anneal has no
`annealing_time` form. `schedule.py` holds the rules, and `warm.py` decodes
`IsingProblem.initial_spins`.

Two traps. `initial_state` must reach `build_submission_body` as SAPI's
per-qubit list, not as a label-to-spin mapping: orjson refuses integer keys.
And the offline mock answers a seeded job by descent from the seed
(`descend_from`), because the conformance driver's seeded ring has 4096 spins
and `ExactSolver` enumerates every state.

One trap sits on this path. `Future.samples` returns a matrix padded out to
the solver's full physical qubit count, while `Future.variables` returns only
the active labels. The reader must pick the columns by label. That
bug reached a live QPU before anyone caught it.

Every Ocean internal lives in `OceanSampler._submit_encoded` (`Future`,
`Present`, `client._submit`). Keep it that way: an SDK change should have a
one-function blast radius. Polling, auth, retries and `Future.cancel` are still
the SDK's job.

### The history is a second ledger, and it never blocks the session

`history.HistoryStore` keeps three tables beside `qpu_usage_hourly`:
operational sums per UTC hour, one row per qblock round, and a histogram of
job-best energy margins to the round target. Every throughput number is
derived at query time (`profile.slot_stats`), so the estimator can change
without a migration. The design and the queueing-theory background are in
`docs/superpowers/specs/2026-09-11-qpu-time-of-week-strategy-design.md`.

`HistoryRecorder` is the only thing the session loop talks to, and it never
raises. Every write from the session thread and the job workers goes
through the recorder's one worker thread, so `Cancel` and `SetTarget`
handling never touch SQLite. Job workers record after billing and outside
`state_lock`, for the same reason billing does. The seed and pickup threads
are the exception: they write through the store directly, serialized by
the store's own lock rather than the recorder's queue.

Two rules are easy to get wrong. `Cancel(max_generation=N)` names the dead
generation, so the round it opens is keyed as generation `N + 1`, which is
what the coordinator writes to `attempts.jsonl`. And the per-job timings
reach the session loop through `SamplerMeta.extra` (`inflight`, `sapi_ms`),
because that map is the one channel that already crosses `handle_job`.

Past rounds are seeded from the coordinator's attempts files
(`attempts.seed_from_attempts`). A directory the miner was live for — a
generation the live recorder already opened, or an hour of the directory's
own span that a live hourly row covers — only contributes outcomes and any
round row a live Cancel never opened. Its margins and hourly sums would
double count. The coordinator's generations restart independently, so a
round means one directory paired with one generation number, not a
generation number alone.

The round strategy (`strategy.decide_round`) runs inside
`ParticipationGate.on_qblock_boundary`, after the budget said yes and from
a memory snapshot only (`profile.SnapshotRefresher` rebuilds it on its own
thread). It compares deliverable jobs, never win rates: difficulty is the
protocol's and unpredictable, throughput is the QPU's and varies with the
hour of the day and the day of the month. `profile.slot_stats` estimates
each (day bin, hour) cell as the global rate times an hour factor times a
day factor, and shrinks the cell's own evidence toward that prediction.
A joined round runs to its end, so a join costs a whole round at that
slot's rate. The gate consults the pacing line at boundaries only, and
the one mid-round stop is a spent allotment. The decision is a
throughput bar: rank the rounds left in the period by the jobs each would
deliver, walk down until their cost exhausts the funds the period will have
(headroom plus accrual to the reset), and join when this round clears the
slot where the funds ran out. `strategy.rounds_by_slot` counts rounds hour
by hour, not round by round, because this runs under the dispatch lock.
The verdict order is explore, saturated, fast slot, slow slot. Mutation
tests in `test_decide_round.py` pin it.

`scripts/probe_timestamps.py` runs one two-qubit job on the live QPU and
prints the SAPI timestamps that the history's queue-wait split depends on.
It exits 1 when they are missing.

### Precedence ladders

Two settings resolve through the same shape — job, then session, then operator
config, then a built-in default:

- sampling (`job._sampling_params`): the job's `IsingProblem`, then `SetTarget`,
  then `backend_toml`, then a hard-coded fallback.
- pipeline depth (`session_loop.resolve_queue_depth`): `backend_toml`, then what
  the coordinator sent, then `DEFAULT_QUEUE_DEPTH`. Read the coordinator's value
  off the wire, not off `SessionConfig`, which substitutes the SDK's own default
  for an unset field.

Operator settings arrive in `Configure.backend_toml`; `budget.DWAVE_CONFIG_KEYS`
is the accepted set. Anything else gets warned about.

### Defect clamping

When the live chip is missing qubits or couplers the session topology names,
`defects.prepare_problem` clamps them out before submit and
`defects.reconstruct_samples` puts them back afterwards. Submitting a coupler
the QPU does not have makes SAPI reject the whole problem. This path still
speaks dicts; the normal case (live graph matches) passes arrays straight
through.

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

## Conventions

- Comments explain why, especially why an obvious-looking simplification is
  wrong. Match that density.
- Tests are named as sentences describing the behaviour under test.
- Changes to billing, cancellation, or the encoder should be mutation-tested:
  revert the fix, confirm a specific test goes red, restore.
- No LLM attribution trailers in commits.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:970c3bf2 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
