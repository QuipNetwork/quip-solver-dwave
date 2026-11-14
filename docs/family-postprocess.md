# Family postprocess

The miner can postprocess the QPU reads of a family salt before it reports them.
The postprocess runs on the host, after the salt's QPU call, and adds its final states to the salt's `Result`.
Every raw read stays in that `Result` and in a record on disk.
The miner records the reads of every salt, with or without a postprocess.
Only a salt whose raw reads leave the target in reach goes to the processor.

The same path serves family leases and explicit family jobs (DW-1, QUI-2042).
An explicit job reports the energy of the plain model, without the offset of the instance (FAM-64).
Its trial record holds the same energies.

## Configuration

The postprocess runs by default.
When `Configure.backend_toml` has no `[postprocess]` table, the miner uses these settings:

- `quip/twisted-ice-dimer` gets the policy `qpp-race2048-repair32/1`.
- The records go to the directory `qpu-reads` beside the usage database.
- The number of processors that run at once is the number of processors the miner may use minus 1, with a minimum of 1 and a maximum of 4. A container `cpuset` limits that number.
- At most 2 salts for each processor wait or run at once. A salt that arrives when that queue is full reports its raw reads at once, without a postprocess. The postprocess never delays results when the host is slower than the QPU.
- If the miner does not find the processor binary, it logs a warning and reports raw reads alone.

The line `postprocess = false` turns the postprocess off. The miner still records the reads.

The top-level key `records_max_gb` sets the most disk that the records use. The default is 20.

A `[postprocess]` table replaces the default settings. A table sets no queue limit: every salt waits for its postprocess, so the host must keep up with the QPU.

```toml
[postprocess]
records = "/var/lib/quip/qpu-reads"     # default: qpu-reads beside the usage database
threads = 1                             # processors that run at once, default 1
"quip/twisted-ice-dimer" = "qpp-race768/1"
```

Each key other than `records` and `threads` is a family id, and its value is a policy name.
A family that the table does not name gets no postprocess.

The session sends `Fatal` with exit code 64 at `Configure` when any of these conditions is true:

- `records` is not a directory name.
- `threads` is not a positive integer.
- A policy name is unknown, or the policy is for another family.
- The table names no family.
- The processor binary is not found.

The miner never replaces a postprocess that a table configures with raw reads alone.

## Policies

| Policy | Family | Processor | Options |
|---|---|---|---|
| `qpp-race768/1` | `quip/twisted-ice-dimer` | `quip-dwave-qpp` in `crates/qpp` | `race`, stages `4:64,1:512`, `b0=6`, `b1=20`, `single_every=4`, every raw read in the population |
| `qpp-race2048-repair32/1` | `quip/twisted-ice-dimer` | `quip-dwave-qpp` in `crates/qpp` | `race`, stages `8:64,2:256,1:1024`, `b0=6`, `b1=20`, `single_every=4`, every raw read in the population, then `repair_depth=32` |

`qpp-race768/1` is the read race of the study in `~/quip-data/regimes/qpu-aware-pp-20261111/`.
The race polishes every raw read with a greedy descent and keeps the 4 lowest.
It runs each of the 4 for 64 cluster sweeps, then runs the lowest for 512 sweeps, 768 sweeps in all.
Run B of that study used 30 models of `ice-twisted-w62-dimer1200`.
There the race ended 0.5 to 2.1 units below an Otsuka run that started from the best raw read.

`qpp-race2048-repair32/1` is the default policy.
Its race keeps 8 reads for 64 sweeps, then 2 for 256 sweeps, then 1 for 1,024 sweeps, 2,048 sweeps in all.
The defect repair then runs on the best state of the race.
A repair move is a chain of up to 32 flips.
The first flip can be any spin.
Each later flip repairs a pair or a tile that the chain broke.
The repair applies every chain that lowers the energy, and it stops when it finds none.
The repair draws no random number.

A test used the reads of one 48-read QPU call for each of 40 chain nonces, from qblocks 141 to 180.
The mean best energy was –5,058.3 after `qpp-race768/1` and –5,061.4 after `qpp-race2048-repair32/1`.
The mean wall time for each salt was 0.16 s and 0.60 s, on one processor of a host that ran other work.
The test used one race seed.

The processor model holds couplings only.
An instance with a field is outside the model, and its trial fails.
The tiles and the pairs come from metadata sections 1 and 3 of the instance.

A new policy gets a new name.
The trial record names the policy, so results of two versions never mix.

The other families use the postprocess binaries of quip-solver-cpu through a session that the miner hosts.
That work is bead `quip-miner-dwave-a43.4`.
The miner generates no other family yet.

## The reach gate

The best raw energy of a salt predicts its energy after the postprocess.
In a test of 300 QPU calls on 10 model cells, the correlation was 0.85.
The gain of the default policy was 12 to 23 units for each cell, with a standard deviation of about 2.5 units.

The gain of a trial is its best raw energy minus its best final energy.
The reach is the largest of the last 256 gains plus 3 units.
A salt goes to the processor when its best raw energy minus the reach is at or below the target.
Any other salt reports its raw reads at once and leaves a `raw` record.

- Each family has its own gate.
- The gate admits every salt until it knows 16 gains.
- The gate admits every salt that has no target.
- One salt in 64 that the gate rules out still goes to the processor, so the reach follows the models of the chain.
- The gate compares the raw energies that the `Result` reports. For a family with an instance offset, the final energy leaves the offset out, so the gain includes it.

## The record

The miner writes one record for each salt that the QPU sampled.
The directory is `[postprocess] records`, or `qpu-reads` beside the usage database.
The records of one UTC day share the directory `<records>/<YYYYMMDD>/`.
A lease salt has the file `<job id hex>.<salt index>.npz` there, and an explicit family job has the file `<job id hex>.npz`.
The file `trials.jsonl` in the same directory gains one line for each record.

When the records pass `records_max_gb`, the miner deletes the oldest day.
The miner never deletes the current day. When the current day alone passes the limit, the miner writes no more records that day, and it logs the count.
A record that the miner cannot write never stops a result.

One record of 48 reads on 4,560 qubits is about 36 kB and takes about 40 ms to write.
At 2 salts each second, one day of records is about 6 GB.

| Field | Content |
|---|---|
| `raw_spins`, `raw_energies` | Every QPU read of the salt, and its exact energy in milli. |
| `final_spins`, `final_energies` | The states that the processor returned, rescored on the host. Empty unless the outcome is `ok`. |
| `summary` | JSON: job id, salt index, family, nonce, time, target, policy, processor, seed, outcome, read count, best raw and final energy, host seconds, processor seconds, and error. |

The nonce rebuilds the model of the salt, together with the family parameters and the topology of the chain.
The processor is `quip-dwave-qpp sha256:<digest>` of the binary that ran.
The seed is the first 8 bytes of the salt nonce, read as a little-endian integer.
The host seconds are the wall time of the processor run, process start included.
The processor seconds are the time that the processor reports.

For a salt that goes to the processor, the miner writes the file before the processor starts, with the raw reads and the outcome `incomplete`.
The miner writes the file again when the processor ends.
If the miner stops between the two writes, the file keeps the raw reads and the outcome `incomplete`.

| Outcome | Meaning | `Result` |
|---|---|---|
| `ok` | The processor finished, and the host rescore of its state matches its report. | The final state, then every raw read. |
| `failed` | The processor exited with an error, wrote no report, or reported an energy that the host rescore does not match. The instance can also be outside the model. | Every raw read. |
| `incomplete` | A stop came before the processor finished. | No `Result`, because the stop closed the lease. |
| `raw` | No processor ran on the salt. The gate ruled it out, the queue was full, or the family has no postprocess. | Every raw read. |

## Billing

The session bills a salt's QPU access when the sampler returns. Billing comes before any abandonment check, and before the postprocess starts.
The quota ledger never holds processor startup or processor wall time.
A salt that a `Cancel` closes during its postprocess is still billed for its QPU call.

## Concurrency and stops

A lease salt frees its QPU slot as soon as its sampler call returns.
The postprocess then runs on its own executor, `threads` at a time.
The job pool stays free for QPU submissions.
An explicit job keeps its credit until its postprocess ends, because its one `Result` carries the final state.

The miner checks for a stop every 20 ms while the processor runs, and ends the processor when a stop comes.
A stop is one of these events:

- A `Cancel` covers the lease generation.
- The `deadline_ms` of the lease passes.
- The lease close deadline after a `Shutdown` passes.
- The stream closes.

The lease sends its `LeaseDone` after the postprocess of every salt finishes, or at once on a stop.

## Tests

`quip_solver_dwave/tests/test_family_postprocess.py` covers the configuration, the model built from the metadata, each trial outcome, the records and their disk limit, the reach gate, the stops, billing, and the QPU slot.
A fake processor runs on every host.
The test of the real race runs when `QUIP_DWAVE_QPP` names a built binary, as in the CI `conformance` job.
