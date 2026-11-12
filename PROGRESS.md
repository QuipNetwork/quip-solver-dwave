# Regime search progress

Status as of October 15, 2026, 02:00 UTC.

Round 4 (exact coefficients on the live topology, structure ladders, cluster-aware CPU kernels) is complete and recorded in `/home/carback1/quip-data/regimes/round4/REPORT.md` and the [Round 4 Notion page](https://app.notion.com/p/3f8db8c6a08881838818c9e8df7dadf7). Round 4 charged about 352 s of QPU access. On energy, every structured cell that beat the single-flip GPU (strong pairs, ferromagnetic clusters, frustrated loops, pairs over a power-law background) is closed at 8,192 sweeps by a CPU kernel that reads the structure (pair flips, cluster flips, exact 6-spin blocks, contraction). On time, those kernels need 1.9 to 21 s of one core per model, which is 40 to 450 times the production QPU call (48 reads at 80 µs, 0.048 s of access, about 0.1 s of wall). The suite discovery run (below) starts next. The stock miner container on GPU 1 is running; the cascade miner is stopped.

The ledger is `.superpowers/sdd/2026-10-05-qpu-seeded-cascade/progress.md`. Beads is the task tracker.

## Round 4 status

| Cell | Energy verdict | One-core seconds to match at 8,192 sweeps |
| --- | --- | --- |
| `native-125`, `native-milli`, RAN-r, POWER-r | Closed by the single-flip GPU by 32,768 sweeps (0.83 card-s). | n/a |
| `native-dimer`, `dimer-r4` to `dimer-r32`, `cluster-k3`, `cluster-k4` | Closed by pair or cluster flips and contraction (24 of 24). | contraction 1.7 to 1.8, pair flips 5.3 to 5.8 |
| `dimer-power` | Closed by weighted pair flips and contraction (24 of 24). | contraction 1.9, pair flips 5.3 |
| `frust-k4`, `frust-k6` | Closed by weighted cluster flips and exact blocks (24 of 24). | cluster flips 6.9 to 7.1 |
| `cubic-dimer-pm1` | Closed by contraction (96 to 98 of 100 reach the target in 0.7 s). | contraction 0.7 |

## Suite discovery run

A codex goal run (gpt-6-astra, max effort, up to 48 hours) searches for structured families on which the production QPU call beats the strongest classical move. Workspace `/home/carback1/quip-data/regimes/qpu-suite/`: `brief.md` (criteria and protocol), `candidates.md` (19 families in three tiers, first the Zephyr pyrochlore ice glass), `research/` (four literature surveys), `capture_worker.py` (the only process that touches the QPU, hard cap 600 s), `run-goal.sh` (the driver). Deliverable: `REPORT.md` and `STATE.md` in that directory.

## Round 3 regime status

| Regime | Status | Evidence |
| --- | --- | --- |
| `cubic-dimer-pm1` | Closed (QUI-1553). The QPU wins alone and as a seed in both unit and f64. Its 1,024-read best stays ahead of CUDA MSA at 1M, 2M, and 4M sweeps (better on 43, 43, and 46 of 100), so more sweeps do not close the gap. 64 reads are about even with two A4000s. | Round 3, throughput test, deeper CUDA sweeps |
| `native-125` | Closed as a quality candidate. Open as a speed and seeding candidate. The f64 kernel needs about 100 times the QPU's access time to draw level. Seeding wins at equal sweeps (100, 71, and 62 of 100), and the QPU call is small next to 15 s to 248 s of f64 sweeps. | Round 3 report, first highlight |
| `native-pm1`, `diamond-pm1` | Closed. The QPU's reads lose to a 4 s classical run. Seeding gains a median 0.01% at deep sweeps only. | Seeding at f = 0.25 |
| Portfolio families A and C, N=28 | Closed (QUI-1554, real-data test). Settled for tuned SA. Tuned neal and scalar `cpu-sa` in f64 are never beaten by the QPU. | Portfolio sections |
| Portfolio family B, N=32 to 56 | Settled for tuned SA. The QPU loses the time-matched race once its wall time counts. | October 4 follow-up |

## Now

- `c55`: rerun the 1,024-read throughput arm at 16 calls in flight. The 4-call arm reached 37% of its access-time limit. It needs about 60 s of QPU access, a manifest, and approval.

## Next

- `cubic-dimer-pm1`: rerun the CUDA comparison after the Linear QUI-1596 fix. Merged MSA lanes understate the GPU at depth.
- Linear QUI-1596 (assigned to Rick): MSA reads collapse on dense models (`cpu-msa-f64`) and merge at depth (CUDA `msa.cu`). The fix belongs in quip-miner-cpu and quip-miner-cuda.
- Linear QUI-1598 (assigned to Rick): the quip-miner-cuda stream-mode hang with 48 concurrent sessions, and the 10-minute per-job cap.
- Repository work that waits on the user:
  - Push `e949009` (the `--prefactor` capture option) and `dc2c7a5` (the throughput capture) to MR !47 on `feat/regime-search-round2`.
  - Review and merge MR !47 (quip-miner-dwave), MR !33 (quip-miner-cpu), and taskd PR #6.
  - After PR #6 merges, repin the taskd submodule in carback-claude-config (MR !82).

## Later

- Every further paid QPU capture needs a separate estimate and approval.

## Round 3

Round 3 ran on September 30 and October 1. It used 420.24 s of QPU access time. Every capture finished with 0 failed jobs. All data is under `/home/carback1/quip-data/regimes/round3/`. The CPU runs shared a loaded host, so they count for quality only.

### Fresh QPU captures

The exploratory captures on 12 models chose 50% of each model's coupling ceiling in every regime. They chose 2000 µs for the native and diamond regimes and 400 µs for cubic-dimer. The evaluation captures then ran all 100 models per regime.

QPU results use the expected best of a random 64-read subset, because SAPI returns reads sorted by energy. Each comparison gives QPU better / CPU better against the frozen campaign.

| Regime | Setting | Reaches target, 64 reads | vs cold 32,768 sweeps | vs cold 131,072 sweeps |
| --- | --- | ---: | --- | --- |
| `cubic-dimer-pm1` | 400 µs, 1,024 reads | 68.6 expected | unit 90/10 | unit 55/45 |
| `native-pm1` | 2000 µs, 256 reads | 98.7 expected | unit 5/95 | unit 0/100 |
| `native-125` | 2000 µs, 256 reads | 91.5 expected | f64 24/76 | f64 2/98 |
| `diamond-pm1` | 2000 µs, 256 reads | 85.6 expected | unit 3/97 | unit 0/100 |

On `cubic-dimer-pm1`, 64 reads cost about 0.07 s of QPU access. `cpu-msa-unit` needs 7.5 s on one core at 131,072 sweeps to match them. Analysis: `round3/analysis/eval-summary.json`.

### QPU-seeded CPU

Each run starts `cpu-msa` from the best 64 distinct QPU reads, at the midpoint of the beta ladder (f = 0.5). The pipeline result is the better of the seed set and the refined reads. Each cell gives seeded better / tie / cold better over 100 models.

| Regime, kernel | Seeded sweeps | vs cold, same sweeps | vs cold, 131,072 sweeps |
| --- | ---: | --- | --- |
| `cubic-dimer-pm1`, unit | 32,768 | 96/3/1 | 78/17/5 |
| `native-pm1`, unit | 32,768 | 81/4/15 | 29/12/59 |
| `native-pm1`, unit | 131,072 | 60/11/29 | 60/11/29 |
| `native-125`, f64 | 32,768 | 86/1/13 | 21/0/79 |
| `native-125`, f64 | 131,072 | 23/0/77 | 23/0/77 |
| `diamond-pm1`, unit | 32,768 | 63/19/18 | 19/14/67 |
| `diamond-pm1`, unit | 131,072 | 56/20/24 | 56/20/24 |

Seeding wins at equal sweeps in every regime except `native-125` at 131,072 sweeps. It replaces 4x sweeps only on `cubic-dimer-pm1`. Summary: `round3/seeded/eval100/summary.json`.

### Portfolio families

Full report: `round3/portfolio-REPORT.md`.

- N=28, 40 draws per variant: the QPU never loses to production neal. Tuned neal 30x is never worse than the QPU, except on one `rotate` model.
- Family B is new. It draws synthetic assets from 3 statistical factors fitted to the real history, so N can exceed the real asset count. At K near N/3, the tuned neal arms disagree on all 24 models.
- Family B, N=44, K=14: the QPU beats tuned neal 30x on 5 to 7 of 12 models and loses on 2 to 4. The October 4 follow-up, with 60 models and proven optima, does not confirm this lead.
- Family B, N=56, K=18: the QPU beats production neal on 9 of 12 and loses to tuned neal 30x on 7 of 12. Chain breaks reach 58% of reads on one job.

### Other Round 3 results

- qpu-1 bench, 10 interleaved reps: quip-miner-cpu `a0786ae` cuts the 1-sweep call from 6.12 ms to 5.64 ms, faster in 10 of 10 reps. At 128 sweeps and more, the change is within noise. Data: `round3/qpu1-bench/bench-results-10rep.txt`.
- `t1e`: no copy of the original market inputs survives. Production never stored them, and the old server no longer exists.

### Follow-up, October 4

The follow-up used 57 s of QPU access time. Beads `3bl` and `0xq` are closed.

- **Exact optima** (`round3/famb-exact/`). SCIP proves the optimum on all 24 family B models and on 83 of 84 B2 models. The one exception, B2 draw005, is bounded at 7.7e-7. For a fixed support, the MIQP objective equals the production finish within 1e-9.
- **Family B2** (`round3/portfolio-family-b2/`). B2 adds 84 models: 48 at 44:14 and 12 each at 32:11, 36:12, and 40:13. The table gives models at the optimum for each arm, at the QPU's best anneal time.

| Size | Models | QPU | Production neal | Tuned neal 30x | Deep neal | Best CPU arm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32:11 | 12 | 6 | 2 | 8 | 7 | 9 |
| 36:12 | 12 | 8 | 1 | 7 | 9 | 10 |
| 40:13 | 12 | 6 | 0 | 5 | 7 | 9 |
| 44:14, pooled | 60 | 20 | 5 | 14 | 18 | 23 |
| 56:18 | 12 | 0 | 0 | 0 | 0 | 0 |

- **Time-matched race** (`round3/famb-race/`). Tuned neal 30x ran a grid of reads and sweeps on one pinned core, and each QPU job met the best grid point inside the QPU's cost. With access time only, the QPU wins most cells. The finish of about 500 QPU reads costs 0.2 s to 0.7 s, 2 to 6 times the access time. With that cost added, N=44 has 2 to 5 QPU wins per anneal time and N=56 favors the CPU. Tuned neal 30x reaches QPU quality in a median of 0.41 times that cost.
- **Chain retry at N=56** (`round3/portfolio-family-b-chains/`). Chain-strength prefactors 4.5 and 6.0 cut the median chain-break share to 0.3% or less. The median gap to the optimum at 80 µs grows from 2.3% at prefactor 3.0 to 3.6% and 4.1% at the higher prefactors. The chip pins the chain coupler at −2.0, so a higher prefactor only shrinks the problem couplings. No shorter-chain embedding exists. The `--prefactor` option in `scripts/portfolio_qpu_capture.py` is commit `e949009`.
- **`native-125` start fraction** (`round3/seeded/n125-early/`). Seeded `cpu-msa-f64` at 131,072 sweeps beats cold on 57 of 100 models at f = 0.1. It beats cold on 62 of 100 at f = 0.25. At f = 0.5 it won 23. The midpoint start caused the loss at depth.

### Follow-up, October 5

This follow-up used no QPU time.

- **Top-k finish, `2jk`, closed** (`round3/famb-topk/`). Finishing the best 64 or 128 reads wins at equal QPU access time. It loses once the measured 1.5 s QPU wall time per call counts (N=44: 3/14/19, N=56: 5/1/30).
- **Seeding at f = 0.25** (`round3/seeded/f025/`). On `cubic-dimer-pm1`, seeded runs at 8,192 sweeps beat or tie cold runs at 131,072 sweeps on 94 of 100 models, in unit and in f64. On `native-pm1`, `diamond-pm1`, and `native-125`, seeded runs lose to cold runs at 4x the sweeps on 55 to 88 of 100 models.
- **MSA read collapse** (`round3/msa-read-diversity/`, Linear QUI-1596). `cpu-msa-f64` reads collapse to one configuration on dense models. On the lattice regimes they stay as diverse as scalar `cpu-sa` reads, so the lattice baselines stand.
- **Families A and C in f64** (`round3/portfolio-msa-f64-tuned/`). Scalar `cpu-sa` with the tuned 30x range is never beaten by the QPU in any variant (QPU better on 0 of 40). `cpu-msa-f64` with the same range loses to the QPU, because its reads collapse.
- **`cubic-dimer-pm1` on the A4000** (`round3/cubic-cuda/`). The stock CUDA MSA kernel ran all 100 models at 128 reads. Each cell is QPU better / tie / CUDA better, against the QPU's best of 1,024 reads and its expected best of 64.

| Sweeps | vs QPU, 1,024 reads | vs QPU, 64 reads | Models/s per GPU |
| --- | --- | --- | --- |
| 131,072 | 71/20/9 | 55/0/45 | 6.2 |
| 1,048,576 | 43/34/23 | 23/0/77 | 0.86 |
| 2,097,152 | 43/35/22 | 21/0/79 | 0.43 (scaled) |
| 4,194,304 | 46/33/21 | 25/0/75 | 0.21 (scaled) |

- The 2M and 4M rows ran on October 9 (`analysis-deep.json`) with a copy of quip-miner-cuda whose `bench_one` waits 30 minutes instead of 2. Because they shared the GPUs with the miner, their rates are the 1M rate scaled by sweeps. The 4M run beat the 1M run on 38 models and lost on 35 (27 equal). Distinct reads per run fell to a median of 26 of 128 at 4M.
- At matched throughput, the QPU gives the better result. 64 QPU reads (about 14 models/s) beat two GPUs at 131,072 sweeps (about 12 models/s) on 66 of 100 models. 1,024 QPU reads (about 1.7 models/s) beat two GPUs at 1,048,576 sweeps (about 1.7 models/s) on 43 and lose on 23. These rates exclude the measured 1.5 s QPU wall time per call.
- At the same sweeps, CUDA is slightly weaker than `cpu-msa-unit`: 28 better, 24 tied, and 48 worse.
- At 1,048,576 sweeps, 53 of 100 CUDA runs return fewer than 64 distinct reads of 128, because lanes that meet stay merged. This is recorded on QUI-1596. The stream-mode hang and the 10-minute per-job cap are QUI-1598. The 64-read runs used the first 64 reads of each 128-read run, because the 64-read stream mode hung.

### Throughput test, October 5

`scripts/qpu_throughput.py` (commit `dc2c7a5`) ran 4 arms over the 100 `cubic-dimer-pm1` models at 400 µs, with a set number of QPU calls in flight. It charged 81.47 s against a 100 s cap. Data: `round3/qpu-throughput/`.

| Arm | Models/s, end to end | Median wall per call | Access-time limit |
| --- | --- | --- | --- |
| 64 reads, 1 in flight | 0.97 | 1.00 s | 14.2 models/s |
| 64 reads, 4 in flight | 3.67 | 1.04 s | 14.2 models/s |
| 64 reads, 16 in flight | 5.28 | 3.09 s | 14.2 models/s |
| 1,024 reads, 4 in flight | 0.62 | 5.88 s | 1.66 models/s |

Each cell compares the fresh QPU best with CUDA MSA at 128 reads: QPU better / tie / CUDA better.

| QPU arm | CUDA 131,072 sweeps | CUDA 1,048,576 sweeps |
| --- | --- | --- |
| 64 reads | 56/13/31 | 25/22/53 |
| 1,024 reads | 80/15/5 | 51/34/15 |

- Two A4000s run 131,072 sweeps at about 12.4 models/s and 1,048,576 sweeps at about 1.7 models/s.
- At 64 reads the QPU (5.3 models/s) sits between those depths, so it is about even with the GPUs at equal throughput. This is an interpolation.
- At 1,024 reads the QPU (0.62 models/s) still beats GPUs that run 2.8 times faster. The exact equal-throughput depth, about 2.6M sweeps, is untested (`77b`).
- 64-read throughput levels off near 5.3 models/s. Each problem costs about 0.19 s of service time against 0.07 s of access time.

## Round 2 record

Round 2 is complete. All CPU evidence passes independent validation, the clean runtime repeat is complete, and the [Round 2 report](https://app.notion.com/p/3eadb8c6a088810ebe26caa18b20d015) is on Notion. Commits `1f9e452`, `efd7152`, and `ae12ef9` on `feat/regime-search-round2` hold this work, and they are pushed.

### Plan and scope

- [Canonical Round 2 plan and design appendix](https://app.notion.com/p/3e3db8c6a08881dbba0bda31d12fbd52)
- [Published results and historical portfolio analysis](https://app.notion.com/p/3e2db8c6a08881529e38ce122f9eb54a)
- [Portfolio families on the QPU: one-to-one clique results](https://app.notion.com/p/3e8db8c6a08881d9a214dde488d922d1)
- [Regime search round 2: report](https://app.notion.com/p/3eadb8c6a088810ebe26caa18b20d015), with its six charts.
- Local plan: `docs/superpowers/plans/2026-09-22-regime-search-round2.md`
- Local Notion snapshot: `.superpowers/sdd/2026-09-22-regime-search-round2/canonical-notion-2026-09-23.md`

Later Notion updates and user decisions supersede the local snapshot.

| Plan tasks | State |
| --- | --- |
| 1–3 — sources, frozen models, portfolio scoring | Complete. Exact historical replay is impossible: the original market inputs do not survive (`t1e`, closed). |
| 4–5 — floating-point MSA and strict solver selection | Implemented and installed in the pinned experiment environments. |
| 6–7 — physical range audit and capture controls | Implemented and tested. |
| 8 — CPU comparisons and portfolio deadline arm | Campaign, seeded sweeps, portfolio deadline arms, and the clean timing repeat validated. |
| 9 — approved QPU captures | Physical pilot complete and reconciled. Portfolio family capture complete: 144 jobs, 25.07 s. |
| 10 — tables, charts, and regime decisions | Report and its six charts published to Notion and verified cell by cell. |
| 11 — integrated verification and handoff | Complete. |

### Validation and report

- `cpu-campaign/validation.json`: all 7,500 records match manifest `b126aa6d…`. Independent dimod rescoring verified all 6,500 completed results. Every unsupported record follows the `sa_int.rs` unit rule.
- `combined/cpu-tree-validation.json`: the quip-miner-cpu tree passes. It contributes 50 seeded-sweep records and the cross-runner agreement check.
- `portfolio-deadline/`: 24 records. All five working arms reach the same objective on each synthetic fixture. `cpu-msa-unit` refuses all four fixtures.
- `scripts/round2_report.py` builds the report and its six SVG charts. It refuses to run unless all three validations pass, and it refuses a portfolio QPU capture that differs from its manifest.

### Portfolio families

Portfolio families replace the `clique-portfolio` regime. Its campaign data stays on disk, but the report no longer shows it. Every family model is a complete graph on N selection variables and uses production's fixed Zephyr clique. Family A resamples real hourly returns from the assets API. Family C removes one kind of structure: `iid`, `shuffle`, or `rotate`.

- Small set: 400 models. The exact oracle proves the optimum of 94 models per variant.
- Large set: 160 models with at least 25 assets each. N = 28 for 148 of them.
- QPU capture: 144 jobs on `Advantage2_system1`, 25.07 s charged, 0 failed.
- Result: the QPU beats production neal on three variants and ties on `rotate`. Tuned neal 30x beats every QPU anneal time on every variant, at about 0.33 s on one core.

Do not pin a parent process with `taskset` when it runs `round2_runner.py`. The runner reads the parent's affinity as the host core count, so the contamination threshold drops and records get false contamination flags. Pin the child processes only.

### Cubic-dimer depth extension

On September 30, `cpu-msa-unit` reran the 47 `cubic-dimer-pm1` models that miss the frozen target at 131,072 sweeps. They ran in parallel, so they are quality-only.

| Sweeps | Models run | Newly reach the target | Cumulative, of 100 | Median sampling s |
| --- | ---: | ---: | ---: | ---: |
| 131,072 (campaign) | 100 | 53 | 53 | 7.5 (clean) |
| 524,288 | 47 | 28 | 81 | 35 (loaded) |
| 2,097,152 | 19 | 10 | 91 | 145 (loaded) |

The 9 remaining models end 2 to 6 energy units short of the target, a gap of 0.05% to 0.16%. `cubic-dimer-depth-extension/` holds the data and the driver.

### Quality campaign

The campaign covers 100 models per regime, five sweep depths (512 to 131,072), and three CPU kernels: `cpu-sa` (scalar), `cpu-msa-unit` (unit MSA), and `cpu-msa-f64` (floating-point MSA). It holds 7,500 records: 6,500 completed and 1,000 unsupported solver and model combinations. The parallel run used 28 workers, so its timing fields are ineligible for runtime comparisons.

### Physical range pilot

144 captures and 9,216 reads across diamond and synthetic clique, at 50%, 75%, and 100% of each model's legal ceiling. Actual QPU access was 8.637886 s, within the approved 11 s cap.

### Historical portfolio evidence

The approved read-only export recovered 4,746 production race records. The stored QPU selection rate was 79.94%. That figure measures provider selection. It does not establish an 80% quality-win rate. Original expected returns and covariance inputs remain missing.

## Evidence and working files

- Round 2 data root: `/home/carback1/quip-data/regimes/round2/`. Key paths: `cpu-campaign/`, `frozen-comparison-policy.json`, `qpu-physical-pilot/`, `historical-analysis/`, `portfolio-families*/`, `portfolio-qpu/`.
- Round 3 data root: `/home/carback1/quip-data/regimes/round3/`. Each follow-up names its directory in the Round 3 sections.
- Repository branch: `feat/regime-search-round2`, MR !47. Local commits `e949009` and `dc2c7a5` are not pushed.
- Untracked and not for commit: `PROGRESS.md`, `QPU-WINS-ANALYSIS.md`, `.beads.gate.lock`. Leave the pre-existing `QPU-WINS-ANALYSIS.md` untouched.

## Decision rules to preserve

- Keep all regimes open for research until the evidence closes them. State evidence and the next control rather than rejecting a regime from a pilot.
- Assess both quality and runtime. Equivalent quality with faster QPU runtime counts as a runtime win.
- Count QPU wall time, not only access time, before claiming a runtime win.
- Distinguish traditional `dwave-neal` SA from scalar `cpu-sa` and both MSA kernels. Check MSA read diversity before trusting a best-of-N result.
- "Full J range" means physical QPU range. Keep the logical model unchanged and compare energies in original units.
- Treat loaded-host and parallel runs as quality-only. Separate device access, provider-call time, cached-job time, and application time.
- Keep calibration and evaluation evidence separate. Historical QPU target captures cannot also provide independent target-based evaluation wins.
- Publish analysis to Notion. This file is the explicitly requested local progress record.

## Tracking and monitoring

- Beads: `c55` and `77b` are open. `t1e` closed on October 5 as unrecoverable, by user decision.
- Linear: QUI-1596 and QUI-1598, both assigned to Rick.
- No experiment job is running. Read taskd status and logs before acting on delayed events. Do not restart completed campaigns from an old notification.

## Round 4 (preliminary, October 13)

Round 4 reran two Round 3 comparisons on the exact-coefficient CUDA kernels of `quip-solver-cuda` 0.3.4-rc6 (QUI-1641), with the Round 3 QPU captures and no new QPU access. Report: [Round 4 preliminary report](https://app.notion.com/p/3f8db8c6a08881838818c9e8df7dadf7). Data: `/home/carback1/quip-data/regimes/round4/`.

| Regime | Change from Round 3 | Evidence |
| --- | --- | --- |
| `native-125` | The QPU no longer leads. The weighted I32 kernel at 32,768 sweeps and 64 reads beats the QPU's best of 256 reads on 63 of 100 models for 0.83 s of card time per model, about the QPU's access time. At 131,072 sweeps it is better on 94 of 100. | `round4/analysis-n125.json` |
| `native-milli` (new, 24 models) | Full magnitude (1,000 levels on the live topology) gives the same shape as `native-125`: the GPU is better on 21 of 24 at 32,768 sweeps and 24 of 24 at 131,072. As a quick screener, the QPU's 48 reads at 80 µs rank like a 2,048-sweep GPU pass at equal cost. | `round4/analysis-milli.json`, `round4/screener.py` |
| `ran-2` to `ran-128` (new, 24 models each) | dimod RAN-r on the live topology. The QPU's standing rises with r (the GPU matches its 256-read ceiling at 8k sweeps for r ≤ 4, at 32k for r ≥ 32) but the GPU passes it by 32k sweeps at every r. The 48-read screener ranks like a 2k-sweep GPU pass at every r. | `round4/analysis-ran*-2000.json`, `round4/screener.py` |
| `power-2` to `power-128` (new, 24 models each) | dimod POWER-r (P ∝ 1/|J|) on the live topology. A notch more favorable to the QPU than RAN-r, and the GPU still passes it by 131k sweeps at every r. | `round4/analysis-power*-2000.json` |
| `native-dimer` (new, 24 models) | The cubic-dimer structure on the mining topology: a matching at −1000 milli, other couplers ±125. The QPU's 256 reads beat the GPU on 24 of 24 up to 524k sweeps (11 card-s) and on 15 of 24 at 2M (47 card-s); its 46 ms screener call beats the GPU's 131k run on 24 of 24. A 256-read call is worth about 40 to 50 card-s here. | `round4/analysis-dimer-*.json` |
| `dimer-r1` to `dimer-r32`, `cluster-k3`, `cluster-k4` (new, 24 models each) | Ladder A (pair bond ÷ other bond, ρ): the GPU passes the QPU's 256 reads at 8k sweeps for ρ ≤ 2, 32k for ρ = 4, 2M for ρ = 8, and not through 524k for ρ = 16 and 32. Ladder C (cluster size at ρ = 8): parity at 524k for k = 3, GPU ahead at 524k for k = 4; the QPU's edge is pair-specific. | `round4/analysis-dimer-r*-2000.json`, `analysis-cluster-k*-2000.json` |
| `dimer-power`, `frust-k4`, `frust-k6` (new, 24 models each) | Dimers with power-law externals: the QPU's 256 reads beat the GPU on 24 of 24 at every rung, by 64 units at 524k (`native-dimer`: 4.75), and its 46 ms call beats the GPU's 11 card-s run on every model; the two levers multiply. Frustrated loops of random strong signs: 4-loops hold the lead to 524k (21/0/3), 6-loops to 131k only. | `round4/analysis-dimer-power-*.json`, `analysis-frust-k*-*.json` |
| Cluster-aware kernels on the ladders (codex) | Pair or cluster flips beat the QPU's 256 reads on 24 of 24 at 8,192 sweeps on every ladder cell (ρ = 4 to 32, k = 3 and 4), at 32,768 on ρ = 1 and 2; contraction takes about 1.8 s per model on one core. Not yet attacked: `dimer-power`, the frustrated loops. | `round4/dimer-moves/report-ladders.md` |
| Approaches review (codex) | Twelve ranked approaches, a devil's-advocate list, and a gated two-day plan. Net: no structure survives a kernel that reads it; the defensible December claim is an incremental hybrid contribution under stated budgets; a 20 percent share needs about 90 cards of QPU-equivalent work against the current field. | `round4/approaches/report.md` |
| Dimer-aware moves on `native-dimer` (codex) | Pair flips and contraction beat both QPU captures on 24 of 24 from 8,192 sweeps; contraction takes 1.57 s per model on one core (2.7x the QPU's access, 0.7x its wall time). The baseline kernel never reaches the QPU at 524k. | `round4/dimer-moves/report-native-dimer.md` |
| Dimer-aware moves in MSA (codex) | Pair flips reach the cubic-dimer target on 98 of 100 at 131k sweeps (baseline 51, QPU 97). Contracting the dimers to 1,054 spins reaches 96 to 98 in 0.7 s on one core, under the QPU's 0.6 s of access, and beats the QPU's 1,024-read best on 57, ties 38, loses 5 at 131k. The cubic-dimer lead is a kernel feature. Follow-up on `native-dimer` running. | `round4/dimer-moves/report.md` |
| `cubic-dimer-pm1` | Unchanged. With exact couplings the kernel scores 68/25/7 against the QPU's 1,024 reads at 131,072 sweeps, against 71/20/9 in Round 3. Reads still merge at depth. | `round4/analysis-cubic.json` |

Still open on the GPU side: the F32 lane and the deep rungs on all 100 models. The portfolio families on `quip-cuda-sa` and seeding on the GPU are also open.
