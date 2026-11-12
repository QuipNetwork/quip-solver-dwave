# QPU wins analysis

Date: 2026-09-12. The chain is the Aglais testnet (`AGLS`), read through `wss://bootnode-1.aglais.quip.network:20049/rpc` from the `QuantumPow` pallet. QPU allotment: 540 minutes (9 hours, 32,400 s) of access time per month, reset on the ninth of the month. D-Wave dashboard at the time of writing: 89,086 problems submitted this period, 1h08m24s used, 7h51m36s available, 0.046 s average per problem.

## 1. Network trends

Block time is 6.02 s, measured over the chain's first 151,799 blocks. Block 1 was produced on 2026-09-02 04:16 UTC. The first qblock was won at block 16,227, about 27 hours later, so the chain has 9.45 days of mining history.

A qblock ends only when a proof clears the difficulty threshold. No timeout exists: the threshold eases each epoch until someone clears it.

### 1.1 Qblock rate

| Window | Qblocks | Rate | Mean length |
| --- | --- | --- | --- |
| Last 24 h | 161 | 161 per day | 8.9 min |
| Last 7 days | 1,083 | 155 per day | 9.3 min |
| Since mining began | 1,442 | 153 per day | 9.4 min |

Qblock length distribution over the chain's life: tenth percentile 2.1 min, median 10.1 min, ninetieth percentile 20.2 min, longest 194 min.

### 1.2 Expected blocks per month at current trends

|  | 30-day month | 31-day month |
| --- | --- | --- |
| Substrate blocks | 430,600 | 445,000 |
| Qblocks | 4,650 | 4,800 |

We measured a 7-day rate of about 155 qblocks per day. This should increase slightly as more nodes join the network. The chain's own full days show the expected increase in competition. Days are grouped by the share of qblocks the QPU won below:

| Full UTC days | Days | Qblocks per day, total | Qblocks per day, field only | QPU share |
| --- | --- | --- | --- | --- |
| QPU won 10% or more (09-06, 09-07) | 2 | 148 | 128 | 13.5% |
| QPU won under 5% (09-05, 09-09 to 09-11) | 4 | 156 | 152 | 2.4% |
| All full days (09-04 to 09-11) | 8 | 152 | 142 | 6.2% |

We expect qblock wins to stabilize based on physical hardware limitations. We do not have enough data to know the floor yet under the current network regime.

### 1.3 The qblocks the QPU joined, by UTC hour

The QPU participated in more qblocks than planned during this measurement, and only for part of the time the qblock mining was active. This was due to an active limitation we detected in the queuing system that has since been fixed. The rounds come from the miner's own history on qpu-1, where each row spans the first to the last attempt of one generation.

Please Note:

1. Everything in this section is normalized against the qblocks the QPU participated in.
2. A qblock counts as joined when the miner submitted at least one job in the round that qblock ended.
3. Each qblock's time is the on-chain timestamp of its block, and a qblock is matched to the round that was open when it was sealed.
4. Wins are the chain's record for the QPU account. The miner's local win flag under-counts them as it was not always able to write to the chain before competitive nodes.

We sampled a period from 2026-09-09 00:00 UTC to 2026-09-12. Of the 600 qblocks in it, 576 ended a round the QPU had joined. Another 14 fell in rounds it sat out on budget, 9 landed before its next round had started, and 1 predates the record.

So we analyze from the active share of round. That is the seconds from the QPU's first to its last attempt, over the round's wall-clock from its start to the qblock. The QPU was submitting for a quarter of each round on average. That is the old per-job pacing check at work. At 540 min per month the allowance line grew by 63 min over these 3.6 days, and the QPU spent 68. So it mined a burst at the start of each round and then parked. Our update replaces the check with whole rounds so performance will change.

| Hour (UTC) | Qblocks joined | Active share of round | Jobs/s while active | QPU wins | Wins per 100 joined | Field clears in joined qblocks | Field clears per joined hour |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 26 | 0.16 | 1.27 | 0 | 0.0 | 26 | 6.5 |
| 01 | 25 | 0.22 | 1.25 | 0 | 0.0 | 25 | 7.1 |
| 02 | 25 | 0.19 | 1.27 | 0 | 0.0 | 25 | 6.1 |
| 03 | 24 | 0.20 | 1.27 | 0 | 0.0 | 24 | 6.9 |
| 04 | 27 | 0.21 | 1.30 | 0 | 0.0 | 27 | 7.9 |
| 05 | 31 | 0.18 | 1.29 | 0 | 0.0 | 31 | 7.2 |
| 06 | 22 | 0.18 | 1.10 | 1 | 4.5 | 21 | 5.8 |
| 07 | 28 | 0.27 | 1.10 | 0 | 0.0 | 28 | 7.0 |
| 08 | 23 | 0.21 | 1.21 | 1 | 4.3 | 22 | 5.9 |
| 09 | 24 | 0.18 | 1.31 | 1 | 4.2 | 23 | 5.9 |
| 10 | 28 | 0.22 | 1.30 | 1 | 3.6 | 27 | 7.7 |
| 11 | 26 | 0.17 | 1.30 | 0 | 0.0 | 26 | 6.5 |
| 12 | 27 | 0.40 | 1.33 | 1 | 3.7 | 26 | 6.4 |
| 13 | 26 | 0.35 | 1.27 | 1 | 3.8 | 25 | 6.3 |
| 14 | 24 | 0.25 | 1.22 | 1 | 4.2 | 23 | 7.2 |
| 15 | 26 | 0.40 | 0.75 | 0 | 0.0 | 26 | 6.8 |
| 16 | 17 | 0.34 | 1.11 | 0 | 0.0 | 17 | 7.6 |
| 17 | 24 | 0.39 | 0.84 | 1 | 4.2 | 23 | 6.3 |
| 18 | 22 | 0.32 | 1.61 | 2 | 9.1 | 20 | 7.3 |
| 19 | 19 | 0.25 | 1.06 | 0 | 0.0 | 19 | 6.5 |
| 20 | 23 | 0.28 | 1.00 | 0 | 0.0 | 23 | 7.3 |
| 21 | 18 | 0.26 | 1.08 | 0 | 0.0 | 18 | 6.5 |
| 22 | 20 | 0.25 | 1.11 | 1 | 5.0 | 19 | 6.6 |
| 23 | 21 | 0.23 | 1.26 | 0 | 0.0 | 21 | 7.1 |
| All | 576 | 0.25 | 1.17 | 11 | 1.9 | 565 | 6.7 |

The field runs 5.8 to 7.9 clears per joined hour, mean 6.7, which matches the 1.76e-3 per second (6.3 per hour). Wins per 100 joined qblocks is 1.9 across all hours. The 11 wins fall in 10 different hours, so no hour stands out on either measure. The QPU's rate while active was 0.75 to 1.61 jobs/s.

For scale, the full record since 2026-09-03 has 1,319 joined qblocks and 180 wins, 13.6 per 100. Its active share is 0.68, with 1.77 jobs/s while active and 5.8 field clears per joined hour. The higher win rate there is the easy difficulty of the first days, not the hour of day.

## 2. The QPU's record on chain

| Measure | Value |
| --- | --- |
| Wins since mining began | 181 |
| Wins in the last 7 days | 62 |
| Wins in the last 24 h | 1 |
| Wins this period (since 2026-09-09) | 11 of 596 qblocks, 1.8% |
| Jobs this period | 89,086 |
| Jobs per win this period | 8,100 (about 390,000 anneals at 48 reads) |
| Jobs per win, longer run (specification sample) | 6,450 (131 wins in 845,000 jobs) |
| Round age when the QPU won | tenth percentile 0.3 min, median 4.4 min, ninetieth percentile 20.2 min |

Round age at the QPU's 11 wins this period, in seconds: 48, 54, 66, 150, 174, 180, 240, 606, 606, 612 and 618. Difficulty eases every 100 blocks counted from the round's own start, about 602 s. So a threshold drop opens the round and a second one comes at 602 s. Six of the eleven wins sit inside a 60 s window after a drop, and those windows hold 14% of round time. The field put 44% of its 589 wins in the same windows, since miners resubmit stashed near misses the moment the threshold eases. It put only 1% in the first minute of a round, where a GPU has no result yet. Section 4.1 has the full-record split by round age.

## 3. Wins per month by throughput

Assumptions: 9 hours of QPU access time per month at 48 ms per job is 675,000 jobs. Rounds are 10 minutes. One win per 8,100 jobs against this week's field. Theoretical maximum throughput at 48 reads per job is 1000 / 48, about 20 jobs per second.

### 3.1 Expected wins at 9 h of QPU access time per month

Each cell with three values is low / mid / high. Mid is this period's observed rate. Low and high are what the same period's data allow at 95% confidence, as explained below the table.

| jobs/s | jobs per round | P(win a joined round), low / mid / high | rounds the budget covers | wall clock per month | wins per month (low / mid / high) |
| --- | --- | --- | --- | --- | --- |
| 1 | 600 | 3.6% / 7.1% / 12.5% | 1,170 of 4,650 | 188 h | 42 / 83 / 149 |
| 2 | 1,200 | 7.1% / 13.8% / 23.4% | 607 | 94 h | 42 / 83 / 149 |
| 3 | 1,800 | 10.5% / 19.9% / 33.0% | 419 | 63 h | 42 / 83 / 149 |
| 5 | 3,000 | 16.9% / 31.0% / 48.7% | 270 | 38 h | 42 / 83 / 149 |
| 10 | 6,000 | 31.0% / 52.3% / 73.6% | 159 | 19 h | 42 / 83 / 149 |
| 12 | 7,200 | 35.9% / 58.9% / 79.8% | 141 | 16 h | 42 / 83 / 149 |
| 14 | 8,400 | 40.5% / 64.5% / 84.5% | 129 | 13 h | 42 / 83 / 149 |
| 16 | 9,600 | 44.7% / 69.4% / 88.2% | 120 | 12 h | 42 / 83 / 149 |
| 18 | 10,800 | 48.7% / 73.6% / 90.9% | 113 | 10 h | 42 / 83 / 149 |
| 20 (theoretical max) | 12,000 | 52.3% / 77.3% / 93.1% | 108 | 9 h | 42 / 83 / 149 |

When we say that about 80% of rounds should be won by the QPU, we are referring to this last row above. Some notes to be aware of:

1. The 95% interval on 11 Poisson events runs from 5.5 to 19.7. So the same jobs could have produced anywhere from one win per 16,200 to one per 4,500.
2. Low and high apply those two rates to every cell. On top of that, jobs until a win is geometric, so a month's total carries about 20% Poisson noise of its own at 83 wins. That widens the monthly range to about 30 to 170.
3. The field and difficulty also drift from week to week by more than the sampling noise, as the daily counts in section 2 show. So the range describes this period's field and is not a guarantee.
4. On the longer-run sample we analyzed from there were 131 wins in 845,000 jobs, the mid value is 104 and the range 70 to 145.
5. The last column is flat due to the allotment is access time, so every job spends the same 48 ms of it at any rate.

There is a lot of variance by design and current technical implementation. A fast win lands early in a round at a harder threshold, which costs a little per job. The coordinator stashes near misses and resubmits them once the threshold eases, which helps slow rounds a little. Placing the 9 wall-clock hours in the field's weakest hours adds about 8% per our measurements.

### 3.2 What may happen as nodes join the network

We know the floor set by the minimum time to complete a round in the network. The worst case is still worth exploring:

Note that the network today is fielding approximately 2 PFLOPS of compute, so we this scenario is contemplating upwards of 8-10 PFLOPS. Expected wins per month are qblocks per month times the QPU's share of clearing power, where clearing power is samples times the chance each clears. The QPU's jobs are fixed by the allotment, so its share moves only with the field.

| Field's clearing power | QPU share | Wins per month |
| --- | --- | --- |
| Half of today | 3.8% | 174 |
| Today | 1.9% | 87 |
| Double | 0.9% | 44 |
| Four times | 0.5% | 22 |

This table uses 46 ms per job (704,000 jobs a month). Another QPU like this one at 1 job/s would add about 7% to the field and cost about 7% of these wins. We have no hard data with a full-queue depth implementation right now.

## 4. GPU estimates: RTX 5090, RTX 4090, RTX A4000

A GPU miner runs the whole month, so its expected wins are its share of the month's qblocks:

```
wins per month = 4,650 × share
share          = c / (H + c)
c              = the card's clearing rate, clears per second
H              = the field's clearing rate today, 1.76e-3 clears per second
```

`H` is the 7-day qblock rate, 155 per day, less the QPU's part. Adding a card adds `c` to the field, so the share is `c / (H + c)` and not `c / H`. Difficulty hardens to keep the qblock rate near 155 per day, so the month's total stays 4,650 and the newcomer takes its share from everyone.

### 4.1 Official specifications

| Card | CUDA cores | Boost clock | FP32 TFLOPS | Board power | GFLOPS per watt |
| --- | --- | --- | --- | --- | --- |
| RTX 5090 | 21,760 | 2.41 GHz | 104.8 | 575 W | 182 |
| RTX 4090 | 16,384 | 2.52 GHz | 82.6 | 450 W | 184 |
| RTX A4000 | 6,144 | 1.56 GHz | 19.2 | 140 W | 137 |
| QPU, 9 h per month | no FP32 pipeline | 80 µs anneal, 48 reads, 1 job/s | 34 to 139, equivalent while active | 12,000 W, system | 2.9 to 11.6, equivalent |

These numbers are based on the FP32 TFLOPS is cores × 2 × boost clock, the figure NVIDIA publishes on each product page. SA in general is limited by memory bandwidth contention so we are not likely to use the entire range of the chip.

GFLOPS per watt is the same figure over board power. On paper the A4000 is the least efficient of the three, yet per our observations (section 4.4) it wins the most per watt. That is because this workload does not scale with peak arithmetic.

The QPU row is an equivalence estimate and must be read against the GPU's speed of analysis. A single GPU needs about 60 s of sampling before it has a result to check. The QPU starts returning a stream of answers in about a second. So in the minute a GPU spends on its first model, the QPU has evaluated 60 jobs at 1 job/s <span underline="true">**in our observed tests**</span>. It would have 1,200 at 20 jobs/s.

The TFLOPS-equivalent is the GPU arithmetic that would clear the threshold at the QPU's rate with that latency included. So it overstates the QPU's arithmetic and understates its edge in the first minute after every threshold drop. The QPU's measured share is 6.5% of the about 2 PFLOPS active on the network during this test. Over a month the QPU's 9 hours cost 108 kWh. A 4090 running all 720 hours costs 324 kWh, a 5090 414 kWh and an A4000 101 kWh.

The latency edge shows in the round age at which wins land, over the chain's full record since 2026-09-03:

| Round age at the win | Qblocks | QPU wins | QPU share |
| --- | --- | --- | --- |
| under 30 s | 33 | 32 | 97% |
| 30 s to 1 min | 26 | 17 | 65% |
| 1 to 2 min | 67 | 21 | 31% |
| 2 to 5 min | 302 | 22 | 7% |
| 5 to 10 min | 257 | 18 | 7% |
| over 10 min | 760 | 70 | 9% |

The field won 10 of its 1,265 qblocks inside the first minute of a round, under 1%. That matches a 60 s time to first result. The QPU won 49 of its 180 there, 27%. Rounds that end inside a minute are 1% of all rounds at today's difficulty. This head start alone does not carry the QPU, as the same edge applies at every threshold drop. It grows with jobs per second.

### 4.2 Measured throughput

We generate these from the dashboard and previous experiments. The dashboard's hardware table carries the same 5090 and 4090 values. In the April fleet test the A100, at 19.5 TFLOPS, produced half the 4090's blocks. The July measurements put the 5090 at 1.9 times the A4000 across a 5.5 times TFLOPS gap. Clock and per-model latency matter more than peak arithmetic.

| Card | Measurement | Source |
| --- | --- | --- |
| RTX 4090 | 1.199 threshold-clearing blocks per minute, 60.1 s per block. Heat-bath Gibbs at the -14,900 test difficulty, 5 cards, April 2026. Led the H200 (1.118), H100 (0.972) and A100 (0.601). | Notion, "[90m Mining Test QPU v CPU v GPU](https://app.notion.com/p/342db8c6a088811e96fecec58edbb184?pvs=25)" |
| RTX 5090 | 3.2 attempts per second at peak on `quip-miner-cuda` v0.2, July 2026, 170 SMs, 2,880 MHz. Fell to 1.9 to 2.2 over hours from a starvation bug fixed in August. | GitHub quip-miner issues 19 and 21 |
| RTX A4000 | About 1.7 models per second at 140 W on the v0.3 Rust miner, July 2026. 85 jobs per minute on the 5,612-node Pegasus fixture with the `cuda` backend at batch 512, September 2026. | Technical sync notes from July. |
| QPU, reference | 113 blocks in 90 minutes at 2.5 jobs/s and 60 reads, November 2025. 1 job/s and 48 reads today. | Notion, "[90m Mining Test QPU v CPU v GPU](https://app.notion.com/p/342db8c6a088811e96fecec58edbb184?pvs=25)" |

### 4.3 Expected wins per month, full-time mining, today's field

| Card | FP32 TFLOPS | Share of the field, FLOPS share / test ratio | Wins per month, FLOPS share / test ratio | Power, estimated | kWh per month | Wins per kWh, FLOPS share / test ratio |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5090 | 104.8 | 5.0% / 17% | 232 / 810 | 575 W | 414 | 0.56 / 2.0 |
| RTX 4090 | 82.6 | 4.0% / 14% | 184 / 660 | 450 W | 324 | 0.57 / 2.0 |
| RTX A4000 | 19.2 | 1.0% / 7% | 44 / 320 | 140 W | 101 | 0.44 / 3.2 |
| QPU, 9 h per month | 34 to 139, equivalent | 1.9% observed, 6.5% while active | 83 observed, range 42 to 149 | 12,000 W, system | 108 | 0.77, range 0.39 to 1.38 |

The table implies a 2,780 TFLOPS equivalent for the QPU against the RTX 5090's 105 for FP32. This is not realizable in SA but still represents a 26x improvement in the most favorable terms.

We think that energy is the fairer comparison with the QPU, whose 12 kW draws only for its hours of use. On the FLOPS share the QPU leads every card per kWh. On the test ratio every card leads it. The QPU's share rises with jobs per second, as section 3 shows, and that is where its energy case improves.

Every GPU row assumes the card mines every round all month (full utilization). The QPU row is its 9 hours. Per watt the A4000 leads among the GPUs on the test ratio. Its 140 W board draws a quarter of a 4090's power for about half the wins. On the FLOPS share it trails, because that route scales with peak arithmetic and this workload does not. This confirms past observations from different experiments that it appears to be the best in class design for this workload.

## 5. Reads and anneal time: Can we tune the QPU?

Short answer: no, not at production difficulty. 48 reads at 80 µs is the measured optimum for wins per unit of QPU time. We note, however, that 32 reads ties it within noise, and every setting below that loses. The evidence is the paired study ["Anneal time and read count under a QPU budget (chain-h0)"](https://app.notion.com/p/3d3db8c6a0888148b8ebe6fc50cb495c?pvs=25). Its win criterion is energy at or below the target with no count or diversity gate, matching current settings.

Fewer reads make each job cheaper, so the month holds more jobs, but each job also wins less often. The question is which effect wins, and only a paired measurement on identical instances can answer it.

The fitted cost of one job on Advantage2_system1:

```
device_us = 33,769 + num_reads × (anneal_us + 211)
```

Programming time is 33.8 ms and fixed. Each read adds the anneal plus 211 µs of readout and delay. At 80 µs the first read costs 116 times what each later read costs. So a setting that adds jobs is expensive and a setting that works each job harder is cheap.

| Reads at 80 µs | Device time per job | Jobs in 9 h |
| --- | --- | --- |
| 1 | 34.1 ms | 951,000 |
| 8 | 36.1 ms | 898,000 |
| 16 | 38.4 ms | 843,000 |
| 32 | 43.1 ms | 752,000 |
| 48 | 47.7 ms (46.1 measured) | 680,000 |
| 64 | 52.4 ms | 618,000 |
| 96 | 61.7 ms | 525,000 |
| 112 | 66.4 ms | 488,000 |

It is not clear if D-Wave can optimize any part of this process, for example programming time, so we assume this is the floor. We note that, if they could take ms off any of these numbers, it would change our analysis.

### 5.2 Tuning Reads

Experiment 1 of the study: 250 `chain-h0` instances per cell, ranked hardest first, targets from -14,589 to -14,497. Win rate per job rises with reads and wins per QPU-minute falls, so across all 250 instances the cheapest cell wins. Split by target hardness the ranking reverses: on the hardest quarter 48 reads leads.

| Reads at 80 µs | Win rate per job | Wins per QPU-min | Relative, all 250 | Relative, hardest 25% | Wins per month at production, projected |
| --- | --- | --- | --- | --- | --- |
| 16 | 70.8% | 1,105 | 1.08 | 0.91 | 76 |
| 32 | 73.6% | 1,045 | 1.02 | 0.97 (at 120 µs: 0.87) | 80 |
| 48 | 78.4% | 1,021 | 1.00 | 1.00 | 83 |
| 64 | 80.0% | 962 | 0.94 | 0.91 (at 120 µs) | 76 |

The projection column applies the hardest-quarter ratio to this period's 83 wins, because production is harder than the corpus. The corpus holds instances from blocks that were won, with win rates of 37% to 96% per job. Production measures one win per 8,100 jobs. The study's own reading is that the ordering carries to production and the absolute rates do not. The only read step that separates from noise is 16 to 48, at p = 0.001. 48 to 64 is flat in every stratum, p = 0.523.

### 5.3 Anneal time at 48 reads

Experiment 2 ran 20, 40 and 80 µs at 48 reads on the same 250 instances. The 80 µs cell was repeated as a drift control, p = 0.839, one block apart.

| Anneal at 48 reads | Win rate per job | Mean energy | Device time | Wins per QPU-min, all | Wins per QPU-min, hardest 25% | Relative, hardest | Wins per month, projected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 µs, extrapolated | about 68% |  | 43.5 ms | about 940 |  | about 0.70 | about 58 |
| 10 µs, extrapolated | about 69% |  | 43.7 ms | about 950 |  | about 0.72 | about 60 |
| 20 µs | 70.8% | -14,562.8 | 43.2 ms | 984 | 515 | 0.77 | 64 |
| 40 µs | 74.0% | -14,565.8 | 44.1 ms | 1,006 | 570 | 0.85 | 70 |
| 80 µs | 79.2% | -14,569.4 | 46.1 ms | 1,032 | 672 | 1.00 | 83 |
| 120 µs | 80.4% | -14,571.0 | 48.0 ms | 1,005 | 666 | 0.99 | 82 |

20 µs saves 6.7% of device time per job and still loses, 5% across all 250 instances and 23% on the hardest quarter. It gives up 6.6 energy units to do it. 20 to 80 µs separates at p = 0.000. 120 µs is no better than 80 and costs 4.2% more, p = 0.359. Anneal time is single-peaked at 80 µs once priced.

The 5 and 10 µs rows are not measured. We took these from the cost model, which prices anneal time at only 48 µs per job for every 1 µs of anneal. We combined it with the measured trend. That trend lost 5.2 points of win rate from 80 to 40 µs and 3.2 more from 40 to 20.

Below 20 µs there is almost no device time left to save, 43.2 ms against a 33.8 ms floor. Every step down has cost win rate. The November 2025 fleet test ran the QPU at 16.13 µs and 60 reads, but against a different difficulty with different qubit topology. So it does not place a point on this curve.

### 5.4 Anneal time at 32 reads

32 reads is the other end of the measured basin, 4% ahead of 48 on the hardest third and not separable from it. Nothing below 80 µs was measured at 32 reads, so this table applies the anneal ratios measured at 48 reads to the 32-read baseline.

| Anneal at 32 reads | Device time per job | Relative, hardest, from the 48-read ladder | Wins per month, projected |
| --- | --- | --- | --- |
| 5 µs, extrapolated | 40.7 ms | about 0.70 | about 60 |
| 10 µs, extrapolated | 40.8 ms | about 0.72 | about 62 |
| 20 µs, extrapolated | 41.2 ms | 0.77 | 66 |
| 40 µs, extrapolated | 41.8 ms | 0.85 | 73 |
| 80 µs, measured | 43.1 ms (42.2 measured) | 1.00 | 86 |
| 120 µs, extrapolated | 44.4 ms | 0.99 | 85 |

The shape is the same as at 48 reads. Fewer reads make each job cheaper, so the anneal saving is a slightly larger fraction of the job. The win-rate loss below 80 µs is unchanged and still dominates.

### 5.5 Read on the hardest third of models

Experiment 3 priced each setting in both device time and wall clock, on the 120 hardest instances under live contention. A 250-instance paired confirmation covered 1, 32 and 48 reads.

| Reads at 80 µs | Win rate per job | Device time | Wins per device-second | Relative to 48 | Wins per month, projected |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.5% | 34.9 ms | 2.2 | 0.18 | 15 |
| 8 | 35.0% | 36.5 ms | 9.6 | 0.81 | 67 |
| 16 | 35.0% | 38.4 ms | 9.1 | 0.76 | 63 |
| **32** | **52.5%** | **42.2 ms** | **12.4** | **1.04** | **86** |
| **48** | **55.0%** | **46.1 ms** | **11.9** | **1.00** | **83** |
| 64 | 52.5% | 49.9 ms | 10.5 | 0.88 | 73 |
| 96 | 65.0% | 57.5 ms | 11.3 | 0.95 | 79 |

32 and 48 reads are 2% apart and not separable (p = 0.146). On the 83-instance hardest third of the confirmation run 48 led on both metrics. The single-read setting has the most jobs per second and the fewest wins per second. In the paired test it can win 0 of 117 discordant instances against 32 reads.

### 5.6 QPU Tuning Summary

| Setting | Wins per month, projected | Against 48 reads at 80 µs |
| --- | --- | --- |
| 1 read, 80 µs | 15 | -82% |
| 16 reads, 80 µs | 63 to 76 | -9% to -24% |
| 32 reads, 80 µs | 80 to 86 | -3% to +4%, not separable |
| 48 reads, 80 µs, production | 83 |  |
| 64 reads, 80 µs | 73 to 76 | -9% to -12% |
| 96 reads, 80 µs | 79 | -5% |
| 48 reads, 5 to 10 µs, extrapolated | about 58 to 60 | about -28% to -30% |
| 48 reads, 20 µs | 64 | -23% |
| 48 reads, 40 µs | 70 | -15% |
| 48 reads, 120 µs | 82 | -1% |

Lower reads do not buy wins. They buy jobs, and the jobs are worth less than the reads they gave up. Adaptive reads could be possible, as 16 reads is optimal on the easiest quarter of targets and 48 on the hardest. But the chain by design targets the hardest quartile, so this is likely not helpful. A busier QPU also may raise the race-optimal read count, since queue wait is a per-submission tax. Both reasons point us to 48 reads or above for production, never below.

## 6. Method notes

- Qblock records come from `QuantumPow.QBlocks`, read with `state_getKeysPaged` and `state_queryStorageAt`. Each record's first 32 bytes are the winner's account and its last 8 bytes are the reported device access time in microseconds.
- `QuantumPow.QBlockCount` is the number of accepted qblocks, which is also the latest qblock id. `QBlockBlockById` maps each id to its block.
- Block time and the daily counts use block number times 6.02 s from block 1's timestamp, read over the EVM JSON-RPC endpoint. That estimate wanders by minutes over the chain's life. So section 1.3 reads each qblock block's own `Timestamp.Now` by block hash. It matches that to the miner's round rows on qpu-1.
- Sources for section 4: Notion pages "90m Mining Test QPU v CPU v GPU" (2026-06-01), "Alternate Classical Implementation Comparisons" (2026-09-08), and the 2026-07-28 technical sync; GitHub `QuipNetwork/quip-miner` issues 19 and 21; NVIDIA product specifications for the three cards.
- Sources for section 5: Notion pages "Anneal time and read count under a QPU budget (chain-h0)" (2026-09-06), "QPU Throughput Tuning" (2026-05-28), "GPU vs QPU time-to-solution at h=0" (2026-06-15) and "Multi-Solver QPU Parameter Grid Test" (2026-03-31).
- The runtime metadata is version 16, which `py-substrate-interface` cannot decode. Storage keys were built by hand from `twox128` of the pallet and item names, plus `blake2_128concat` of the map key.
