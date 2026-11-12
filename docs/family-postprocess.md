# Family postprocess

The miner can postprocess the QPU reads of a family salt before it reports them.
The postprocess runs on the host, after the salt's QPU call, and adds its final states to the salt's `Result`.
Every raw read stays in that `Result` and in a trial record on disk.

The same path serves family leases and explicit family jobs (DW-1, QUI-2042).
An explicit job reports the energy of the plain model, without the offset of the instance (FAM-64).
Its trial record holds the same energies.

## Configuration

The postprocess runs only when `Configure.backend_toml` has a `[postprocess]` table:

```toml
[postprocess]
records = "/var/lib/quip/postprocess"   # required
threads = 1                             # processors that run at once, default 1
"quip/twisted-ice-dimer" = "qpp-race768/1"
```

Each key other than `records` and `threads` is a family id, and its value is a policy name.
A family that the table does not name gets no postprocess.

The session sends `Fatal` with exit code 64 at `Configure` when any of these conditions is true:

- `records` is missing.
- `threads` is not a positive integer.
- A policy name is unknown, or the policy is for another family.
- The table names no family.
- The processor binary is not found.

The miner never replaces a configured postprocess with raw reads alone.

## Policies

| Policy | Family | Processor | Options |
|---|---|---|---|
| `qpp-race768/1` | `quip/twisted-ice-dimer` | `quip-dwave-qpp` in `crates/qpp` | `race`, stages `4:64,1:512`, `b0=6`, `b1=20`, `single_every=4`, every raw read in the population |

`qpp-race768/1` is the read race of the study in `~/quip-data/regimes/qpu-aware-pp-20261111/`.
The race polishes every raw read with a greedy descent and keeps the 4 lowest.
It runs each of the 4 for 64 cluster sweeps, then runs the lowest for 512 sweeps, 768 sweeps in all.
Run B of that study used 30 models of `ice-twisted-w62-dimer1200`.
There the race ended 0.5 to 2.1 units below an Otsuka run that started from the best raw read.

The processor model holds couplings only.
An instance with a field is outside the model, and its trial fails.
The tiles and the pairs come from metadata sections 1 and 3 of the instance.

A new policy gets a new name.
The trial record names the policy, so results of two versions never mix.

The other families use the postprocess binaries of quip-solver-cpu through a session that the miner hosts.
That work is bead `quip-miner-dwave-a43.4`.
The miner generates no other family yet.

## The trial record

Each postprocessed lease salt has a file `<records>/<job id hex>.<salt index>.npz`, and each explicit job has a file `<records>/<job id hex>.npz`.
Each trial also adds a line to `<records>/trials.jsonl`.

| Field | Content |
|---|---|
| `raw_spins`, `raw_energies` | Every QPU read of the salt, and its exact energy in milli. |
| `final_spins`, `final_energies` | The states that the processor returned, rescored on the host. Empty unless the outcome is `ok`. |
| `summary` | JSON: job id, salt index, family, policy, processor, seed, outcome, read count, best raw and final energy, host seconds, processor seconds, and error. |

The processor is `quip-dwave-qpp sha256:<digest>` of the binary that ran.
The seed is the first 8 bytes of the salt nonce, read as a little-endian integer.
The host seconds are the wall time of the processor run, process start included.
The processor seconds are the time that the processor reports.

The miner writes the file before the processor starts, with the raw reads and the outcome `incomplete`.
The miner writes the file again when the processor ends.
If the miner stops between the two writes, the file keeps the raw reads and the outcome `incomplete`.

| Outcome | Meaning | `Result` |
|---|---|---|
| `ok` | The processor finished, and the host rescore of its state matches its report. | The final state, then every raw read. |
| `failed` | The processor exited with an error, wrote no report, or reported an energy that the host rescore does not match. The instance can also be outside the model. | Every raw read. |
| `incomplete` | A stop came before the processor finished. | No `Result`, because the stop closed the lease. |

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

`quip_solver_dwave/tests/test_family_postprocess.py` covers the configuration, the model built from the metadata, each trial outcome, the stops, billing, and the QPU slot.
A fake processor runs on every host.
The test of the real race runs when `QUIP_DWAVE_QPP` names a built binary, as in the CI `conformance` job.
