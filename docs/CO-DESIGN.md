# Co-design experiment: cache-aware admission ordering

This document records the Phase-0/Phase-1 attempt to test one falsifiable question:

> On one Mac, with the same MLX executor and the same prefix-cache budget, can ordering
> requests by **predicted uncached work + predicted new KV memory** beat FIFO without
> sacrificing long requests, completion time or throughput?

Short answer: **cache-hit-aware ordering shows a reproducible benefit on 16-session
mixed traffic, but the executor still ignores the scheduler's prefill budget; this is a
narrow admission-order result, not a co-optimized engine.** The measurements, the
control-boundary evidence and the negative results are below.

---

## 1. What was missing before this phase

| layer | current implementation | gap |
|---|---|---|
| scheduler | `Scheduler` decides admission, chunking and preemption | runs on logical blocks; its prefill budget is not enforced by the executor |
| cache | `PagedPrefixCache` stores shared-prefix snapshots | success depends on copying snapshots into `BatchGenerator`, not on physical page sharing |
| executor | `MLXRunner` wraps `mlx_lm.generate.BatchGenerator` | owns its own prefill/decode batching; `max_tokens` is the output limit, not a step budget |
| cost | request cost was effectively "prompt length" | no predicted cache-hit, copy, or memory increment entered scheduling |

The review that prompted this work was right: the design borrowed DeepInfer's scheduling
semantics but did not connect cache cost, scheduler decisions and execution.

## 2. Phase 0: measure the control boundary first

Added per-step instrumentation in `Engine` (`execution_control` in `/stats`):

- what the scheduler scheduled: request ids, prefill tokens, decode slots;
- what the executor actually did: prompt tokens processed, decode tokens, step time;
- cache operations: hits, stores, reuse-vs-recompute skips.

16 sessions, 2k shared prompt, combined arm (cache + cost-aware), 1 turn:

| metric | value |
|---|---:|
| scheduled prefill tokens | 16,401 |
| actual prompt tokens processed | 41,217 |
| actual / scheduled prompt ratio | **2.51×** |
| scheduled prefill request chunks | 33 |
| requests inserted into executor | 17 |
| prefill steps with a token mismatch | 11 / 11 |
| scheduled decode tokens | 1,032 |
| actual decode tokens | 1,032 (exact) |

In the 16-session mixed run the ratio was 1.00 for the no-cache FIFO arm but 1.43–1.46
for both cache arms (and 1.12 for no-cache cost-aware): **the executor processes more
prompt tokens per step than the scheduler budgets, and the gap grows when cache hits are
involved.** `insert_segments` does not provide dynamic chunk control.

What *is* controllable: **which requests are admitted and in what order**. With a 50 ms
admission settle window (`TERRAZZO_ADMIT_SETTLE_MS=50`), all 16 agent requests reach the
scheduler as one wave; FIFO inserted a mixed first wave, while cost-aware inserted the
cached requests first (`cached=2544, uncached=31` before `cached=0, uncached=2574`).

Therefore Phase 1 tests the mechanism that can actually be controlled: **admission
order**, not prefill chunk budgeting.

## 3. Predicted cost, not "real cost"

`terrazzo/cost.py` now exposes a prediction with a confidence interval:

- prefill/copy rates can be represented as a few `(tokens, ms)` calibration points;
  otherwise a default rate is used;
- predictions outside the calibrated range are marked `confident=False`;
- auto reuse chooses copy only when even the optimistic copy range beats the pessimistic
  recompute range for the cached span; otherwise it falls back to recompute;
- the scheduler key is `predicted_uncached_tokens + memory_weight × predicted_new_kv_tokens`
  (both are estimates, not measurements).

On this workload the auto policy always chose reuse for the long shared prefix; disabling
reuse is catastrophic (`D-never`: 7.3 tok/s, 116 s p50 TTFT), while `auto`, `always` and
frozen estimates are within noise of each other. **The copy-vs-recompute crossover did not
show up at this prefix length**, so dynamic copy/recompute is not claimed as a win.

## 4. Experiment protocol

All runs: Apple M1 Max 64 GB, macOS 26.6.2, terrazzo `0.3.0`, MLX 0.32.4 / mlx-lm 0.31.3,
Qwen3-4B-4bit, one server at a time, temperature 0, server-reported token counts.

| arm | cache | admission | reuse |
|---|---|---|---|
| A | off | FIFO | — |
| B | on | FIFO | always |
| C | off | cost-aware | — |
| D | on | cost-aware | auto |

Additional variants: D-fixed (frozen cost estimates), D-always, D-never.

Workloads:

- **shared**: 16 sessions, same 2k system prompt, 1 turn × 64 tokens (2 turns with `--turns 2`).
- **mixed**: 8 sessions share the 2k system prompt; 8 have unique 2k system prompts. This is
  the workload where cache-hit information can change admission order.
- **long**: 4 sessions, 8k shared system prompt, 1 turn.

Metrics: p50/p95 TTFT, p95 total completion time, slowest session completion, throughput,
peak MLX memory, prewarm time. The slowest-session column is a fairness guard: ordering
must not make a few requests wait much longer.

## 5. Results

### 5.1 Mixed, 16 sessions, 1 turn (two independent runs with settle window)

| run | arm | tok/s | TTFT p50 | TTFT p95 | p95 total | slowest | peak MLX |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | B cache+FIFO | 11.7 | 70.87 s | 73.76 s | 87.32 s | 87.46 s | 34.5 GB |
| 1 | D cache+cost | **15.6** | **44.47 s** | **56.96 s** | **65.74 s** | **65.74 s** | **17.7 GB** |
| 2 | B cache+FIFO | 11.1 | 56.66 s | 81.48 s | 92.28 s | 92.44 s | 34.5 GB |
| 2 | D cache+cost | **12.5** | **47.07 s** | **69.84 s** | **81.75 s** | **81.87 s** | **19.9 GB** |

D vs B: throughput +13–33%, p95 TTFT −14–23%, p95 completion −11–25%, peak MLX −42–49%,
slowest session −6–25% (no fairness regression in these runs).

### 5.2 Shared prefix, 16 sessions, 1 turn (no cache-hit asymmetry)

| arm | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX |
|---|---:|---:|---:|---:|---:|
| B | 89.5 | 2.54 s | 2.95 s | 11.44 s | 9.83 GB |
| D | 87.5 | 2.73 s | 3.16 s | 11.70 s | 9.83 GB |

No gain — expected: all requests have the same predicted cost, so cost ordering is FIFO.

### 5.3 Mixed, 8 sessions, 2 turns

| arm | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX |
|---|---:|---:|---:|---:|---:|
| B | 17.4 | 11.31 s | 40.27 s | 49.88 s | 19.3 GB |
| D | 17.3 | 10.64 s | 39.78 s | 49.28 s | 18.5 GB |

No meaningful difference: with two turns and fewer sessions the admission order stops
mattering (later turns arrive asynchronously).

### 5.4 Long prompt, 4 sessions, 1 turn

| arm | tok/s | TTFT p50 | p95 total | peak MLX | prewarm |
|---|---:|---:|---:|---:|---:|
| A no cache FIFO | 1.5 | 160.0 s | 167.2 s | 18.8 GB | — |
| B cache FIFO | **40.0** | **1.09 s** | **6.39 s** | 10.5 GB | 31.9 s |
| C no cache cost | 2.0 | 119.4 s | 124.9 s | 9.0 GB | — |
| D cache cost | 37.4 | 1.10 s | 6.84 s | 10.5 GB | 32.6 s |

Prefix caching is the entire story here; cost ordering does not add anything over B.
Prewarming an 8k prefix costs ~32 s and must be amortized over requests.

### 5.5 Reuse-policy variants (mixed, 16 sessions, 1 turn)

| variant | tok/s | TTFT p50 | TTFT p95 | peak MLX |
|---|---:|---:|---:|---:|
| D auto | 12.5 | 47.57 s | 70.26 s | 19.9 GB |
| D-fixed | 12.1 | 43.65 s | 74.57 s | 25.3 GB |
| D-always | 12.3 | 42.19 s | 73.81 s | 19.5 GB |
| D-never | 7.3 | 116.32 s | 130.11 s | 21.1 GB |

Dynamic auto-reuse does not beat always/fixed at this prefix length; never reusing is a
clear loss.

## 6. Interpretation

1. **The active ingredient is cache-hit information, not ordering generically.** C (no
   cache) is effectively A; D changes decisions only because it can see cached spans.
2. **The mechanism works in a narrow window:** 16 concurrent sessions, mixed cache-hit
   asymmetry, one turn. It is neutral on shared-only traffic, neutral on 8×2 mixed, and
   irrelevant on long prompts where caching dominates.
3. **We cannot call this co-optimization.** The scheduler controls admission order, not
   prefill execution; the executor still processes a different amount of prompt work than
   the scheduler budgets (2.5× on 16 shared sessions). A controlled executor is a
   prerequisite for any stronger claim.
4. **A 16-session × 2-turn shared run hung once** (health endpoint stopped responding while
   the process stayed alive). It was not reproduced in the mixed 8×2 run and is not
   included above; it must be understood before production claims.
5. **No custom kernel is justified by this data.** The cache-copy cost was not the
   bottleneck in the tested regime; scheduling/ordering was.

## 7. Decision and next step

- Phase 1 **finds a real but narrow admission-order effect**, so the direction is alive;
  the “co-optimized engine” claim is not supported yet.
- The next experiment, if continued, is a **minimal controlled executor**: scheduler
  selects the step's requests and chunk budget, and the executor must report actual
  executed prompt tokens matching it. Only then compare D with/without chunk control.
- Do **not** start a kernel/physical-sharing rewrite from this evidence.
- Keep the documentation honest: README calls this a cache-aware scheduling experiment,
  not an algorithm-system co-design breakthrough.

## 8. Reproduce

```bash
MODEL=.../Qwen3-4B-4bit
python bench/co_design_ablation.py --mlx-model "$MODEL" \
    --arms a,b,c,d --workloads shared,mixed,long --sessions 16 --turns 1
# variants: --arms d,d_fixed,d_always,d_never --workloads mixed
```

Raw JSON is written to `bench/co_design_results.json` by default (override with `--out`).
The harness sets `TERRAZZO_ADMIT_SETTLE_MS=50` for Terrazzo arms so the admission wave
contains all sessions before the scheduler sorts it.

---

## 9. Phase 2: concurrency-cap baseline and continuous arrivals

The Phase-1 result had an obvious alternative explanation: cost ordering may simply reduce
the effective number of simultaneously active large-prefill requests. I therefore added a
**cache+FIFO + concurrency cap** baseline (`--max-num-seqs` 4/8/12) and repeated the mixed
16-session comparison, then ran a continuous-arrival version (500 ms stagger between
session starts).

### 9.1 Mixed 16×1 with FIFO concurrency caps

| arm | tok/s | TTFT p50 | TTFT p95 | p95 total | slowest | peak MLX | peak KV tokens | peak running |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B-cap4 (FIFO, 4 slots) | **14.5** | **32.45 s** | **67.44 s** | **70.74 s** | **70.74 s** | **18.5 GB** | 25,904 | 4 |
| B-cap8 (FIFO, 8 slots) | 11.9 | 54.74 s | 76.02 s | 85.59 s | 85.72 s | 24.0 GB | 36,080 | 8 |
| B-cap12 (FIFO, 12 slots) | 12.2 | 47.35 s | 79.33 s | 83.62 s | 83.62 s | 34.5 GB | 34,464 | 12 |
| B cache+FIFO (16 slots) | 11.6 | 40.89 s | 77.51 s | 88.10 s | 88.20 s | 30.3 GB | 44,784 | 16 |
| D cache+cost (16 slots) | 12.6 | 40.98 s | 71.76 s | 81.11 s | 81.11 s | 19.5 GB | 44,784 | 16 |

**A simple FIFO cap at 4 beats cost ordering on aggregate metrics here**: throughput 14.5
vs 12.6 tok/s, p95 TTFT 67.4 vs 71.8 s, p95 completion 70.7 vs 81.1 s, peak memory 18.5
vs 19.5 GB. D still beats uncapped FIFO, but the cap baseline explains that improvement
without a cost model.

This does not mean ordering has no value: in continuous arrivals cost lowered the shared
p50 to 14.7 s vs 35.0 s for cap4, i.e. it changes **service priority**. That trade-off is
only useful if serving cache-hit requests first is an explicit product requirement; the
current goal does not require it.

B and D report the same *logical* KV token peak (44,784), while cap4 peaks at 25,904.
Logical KV accounting therefore cannot explain D's lower MLX memory. Physical residency,
padding, copy buffers and allocator retention were not measured; "transient prefill/copy
buffers" is a hypothesis, not a demonstrated cause.

### 9.2 Continuous arrivals (mixed, 16 sessions, 500 ms stagger)

| arm | tok/s | TTFT p95 | p95 total | slowest | peak MLX | peak KV | shared p50 | unique p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B cache+FIFO | 11.2 | 73.25 s | 87.90 s | 88.02 s | 40.9 GB | 44,544 | 40.33 s | 70.08 s |
| D cache+cost | 12.8 | 63.99 s | 78.13 s | 78.16 s | 21.1 GB | 44,576 | **14.71 s** | 58.78 s |
| B-cap4 | **13.3** | **60.39 s** | **69.39 s** | **70.03 s** | **18.2 GB** | 28,368 | 34.95 s | **45.02 s** |

D improves both shared and unique groups over uncapped FIFO, but **cap4 is again better on
throughput, p95 and peak memory**. The D-vs-cap4 gap is not explained by the 50 ms settle
window (both Terrazzo arms share it) or by different prompts (same workload generator).

### 9.3 Causal chain, as far as the evidence goes

1. Cache-hit information changes admission order: with the cost arm the first prefill wave
   contained cached requests (`cached=2544, uncached=31`), while FIFO admitted a mix.
2. That changes the mix of active work (small-suffix requests first), and MLX peak drops
   even though the *logical* peak KV block count is identical to B.
3. **But a simpler FIFO cap at 4 achieves better aggregate throughput, p95 and memory.**
   The data therefore does not support "cost ordering has aggregate value over tuned
   concurrency"; it only supports "ordering changes service priority" (see shared p50).
   The memory mechanism remains unverified beyond logical accounting.

### 9.4 Hang investigation

The earlier 16×2 shared-prefix hang (process alive, `/health` unresponsive) was **not
reproduced** in four dedicated runs: each completed in 21.7–22.4 s at ~94 tok/s, peak MLX
9.83 GB. The failure occurred only inside a multi-arm sequence; no stack was captured at
the time, so the layer (HTTP loop, lock, worker, MLX call, cache release) remains unknown.
This is now documented as an open reliability question, not as a solved issue. Adding a
monitor that samples thread stacks (`sample <pid>`) on health failure is part of the
harness/repro notes; a full multi-arm 2-turn retry is still pending.

### 9.5 Revised decision

- The right description is **"a cache/admission-scheduling joint experiment"**, not a
  co-optimized engine.
- Tuned FIFO concurrency is the current default; cost ordering remains an experimental
  option. What was disproved is "cost ordering beats simple concurrency control on
  aggregate metrics", not the entire value of ordering.
- cap4 is the winner **in this dataset only**. Tuning and validation workloads must be
  separated; compare a pre-chosen fixed cap against a per-workload oracle cap.
- Do not pursue physical sharing or a custom executor from this evidence.

---

## 10. Next-step guardrails (after review)

- **Product path:** simple, explainable concurrency limits; tuned FIFO is the default.
  Cost ordering stays behind an experimental flag.
- **Reliability path:** capture thread stacks and request stages on timeout/hang, then
  rerun the multi-arm 16×2 sequence that produced the original hang. Four isolated
  successes do not replace that reproduction condition.
- **Research path:** first measure whether the best cap changes enough across workloads
  that one fixed cap cannot serve them. Only then evaluate automatic cap selection. The
  final answer is allowed to remain "FIFO".
- **Comparison rule:** report both a pre-selected fixed cap and a per-workload oracle cap;
  do not use the same data to pick and validate the winner.

---

## 11. Cap sweep: one fixed cap is not optimal for all workloads

External load was ~6.2 during this sweep, so only relative comparisons within the table are
meaningful. All arms are cache+FIFO, 16 sessions, 1 turn, 50 ms settle, true usage tokens.

### 11.1 Mixed traffic (8 shared + 8 unique prefixes)

| cap | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX |
|---:|---:|---:|---:|---:|---:|
| 2 | 11.2 | 63.19 s | 89.02 s | 91.40 s | 9.6 GB |
| **4** | **11.7** | **40.24 s** | **75.19 s** | **87.49 s** | 11.6 GB |
| 6 | 9.8 | 52.63 s | 98.96 s | 104.58 s | 17.3 GB |
| 8 | 8.1 | 71.79 s | 115.48 s | 125.85 s | 29.3 GB |
| 12 | 7.4 | 89.42 s | 126.90 s | 137.91 s | 29.4 GB |
| 16 | 7.3 | 72.13 s | 124.16 s | 140.63 s | 25.7 GB |

### 11.2 Shared prefix (16 sessions, same 2k prefix)

| cap | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX |
|---:|---:|---:|---:|---:|---:|
| 2 | 34.7 | 15.16 s | 26.10 s | 29.51 s | 3.9 GB |
| 4 | 38.7 | 14.18 s | 20.71 s | 26.44 s | 4.9 GB |
| 6 | 38.7 | 11.01 s | 20.47 s | 26.43 s | 5.7 GB |
| 8 | 38.0 | 15.11 s | 15.61 s | 26.93 s | 6.6 GB |
| 12 | 43.1 | 3.29 s | 17.70 s | 23.74 s | 8.2 GB |
| **16** | **52.4** | 4.00 s | **4.79 s** | **19.53 s** | 9.9 GB |

### 11.3 Fixed cap vs per-workload oracle

- Mixed: oracle cap=4; a pre-chosen cap4 matches it.
- Shared: oracle cap=16; fixed cap4 loses **26% throughput** (38.7 vs 52.4) and has
  much worse p95 (20.7 vs 4.79 s).
- Continuous mixed (500 ms stagger, earlier lower-load run): cap4 13.3 tok/s vs cap16
  11.2, so cap4 also wins there.
- Long 4×1: cap4 and cap16 are identical (both ~22 tok/s), because there are only 4
  sessions.

**Conclusion:** the best cap varies enough (4 vs 16) that one fixed value cannot serve all
workloads. This makes automatic cap selection a legitimate research question, but the
current product answer is two explicit presets: `--profile agent` (16) and
`--profile mixed` (4). Do not market a single magic cap.

## 12. Reliability status (stack dump + retry)

Added:

- `TERRAZZO_STACK_DUMP=1` registers SIGUSR1 in the server; `kill -USR1 <pid>` dumps every
  thread stack to stderr/log.
- `request_stages` in `/stats`: added / first_token / finished / aborted / failed counts.
- `scripts/watch_health.py`: polls `/health`, on failure sends SIGUSR1 and prints stats/log.

Retry status: four isolated 16×2 shared runs completed in 21.7–22.4 s (~94 tok/s). The
attempted multi-arm retry ran under load ~6.2; the no-cache arm A remained healthy
(`/health` 200) and made progress (17 first tokens, 704/2048 decode tokens) but was far too
slow to finish in a reasonable window, so it was stopped before B/C/D. No hang was
reproduced and no stack was captured. The original one-off hang remains an **open**
reliability question; the next retry should run when external load is low and keep the
full A/B/C/D 2-turn sequence.

---

## 13. Load-aware concurrency: cap4 ↔ cap16 in one process (corrected run)

Question: with one model and one long-lived server, can a simple rule change the admission
cap between 4 and 16 across `shared → mixed → shared` traffic and approach the per-phase
oracle? Rule inputs: window of 16 recent requests (admitted + pending), cache-hit ratio,
average uncached prompt tokens, queue depth, memory usage. Switches when
`avg_uncached ≥ 512` (mixed → cap4) or `avg_uncached ≤ 256` and `hit_ratio ≥ 0.5`
(shared → cap16); 6 s minimum hold; lowering affects only new admissions.

Implementation: `terrazzo/concurrency.py`, `Engine.set_max_num_seqs`,
`TERRAZZO_AUTO_CAP=1`, `POST /v1/concurrency` (`TERRAZZO_ENABLE_CONTROL=1`),
`bench/concurrency_switch.py`. All phases: 16 sessions, 1 turn × 64 tokens, 800 ms
stagger, prewarmed 2k shared prefix, 50 ms admission settle.

**Review correction:** the first version of this experiment confounded the cap with the
admission profile (`fixed4` used `default` → ADMIT_BATCH=4/WAIT=12 and a 4-wide runner
batch; the other arms used `agent` → 16/4 and a 16-wide runner batch). The table below is
the corrected run: every arm uses ADMIT_BATCH=16, ADMIT_WAIT=4, settle 50 ms and
`completion_batch_size=16`; only the admission cap differs. The harness now records the
effective `admission` and `completion_batch_size` per phase, and clears inherited
`TERRAZZO_AUTO_CAP`/`TERRAZZO_ENABLE_CONTROL`.

### 13.1 Corrected result (shared → mixed → shared)

| arm | overall tok/s | wall | shared-1 p50 / p95 | mixed p50 / p95 | shared-2 p50 / p95 | peak MLX |
|---|---:|---:|---:|---:|---:|---:|
| fixed4 | **26.4** | 116.6 s | 0.595 / 2.10 s | 33.48 / 61.00 s | 1.60 / 3.63 s | 19.1 GB |
| fixed16 | 22.8 | 134.9 s | 0.453 / 0.68 s | 54.20 / 73.95 s | 2.68 / 7.03 s | 40.5 GB |
| oracle 16/4/16 | 26.0 | 117.9 s | 0.448 / 0.61 s | 34.03 / 61.45 s | 0.48 / 0.69 s | 19.0 GB |
| auto | 23.9 | 128.4 s | **0.427 / 0.67 s** | 37.38 / 68.24 s | 2.09 / 3.76 s | 19.1 GB |

Auto decisions: cap16 shared-1; one switch to 4 in mixed; one switch back to 16 in
shared-2; no flapping. It tracks the oracle on shared-1 and stays close on mixed, but the
switch back to 16 lags in shared-2 (p50 2.09 vs 0.48 s).

**Corrected conclusion:** with the confound removed, **fixed cap4 is the best static
policy** (26.4 tok/s, 116.6 s wall) and is close to the per-phase oracle **on aggregate
throughput and total wall time only** (26.0, 117.9 s). It is not close on every phase tail:
shared-2 p95 is 3.63 s vs 0.69 s for oracle and 0.61 s for fixed4 on shared-1 vs 0.61 s
oracle. The trade-off is real: oracle keeps 16 slots on shared phases; fixed4 keeps 4.
Auto (23.9, 128.4 s) beats fixed16 on mixed tail and shared memory, but does **not** beat
fixed4 overall. The previous "auto tracks oracle / beats fixed caps" claim was confounded
and is retracted for this workload. `--profile mixed` now pins exactly the tested fixed4
configuration: admission cap 4, completion batch 16, ADMIT_BATCH 16, ADMIT_WAIT 4.

### 13.2 Validation sequence (mixed → shared → mixed)

| phase | cap after phase | tok/s | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|---:|
| mixed | 4 | 8.7 | 53.51 s | 92.33 s |
| shared | 16 | 41.9 | 2.84 s | 5.41 s |
| mixed | 4 | 8.6 | 51.96 s | 93.21 s |

The rule still switches 4→16→4 in an unseen order without within-phase flapping, but
this run used the older workload and does not establish a performance advantage over a
fixed cap.

### 13.3 Decision

- Auto cap remains an **experimental, opt-in** feature; it is not a product default.
- Tuned FIFO presets remain the default. For the tested workloads, fixed cap4 is the
  strongest simple policy; fixed16 is only useful for shared-prefix latency and costs
  memory/mixed tail.
- Further work should use the corrected single-variable harness and multiple unseen
  sequences before any automatic policy is considered for default.
- Reliability fixes from the code review (watchdog lock, request release, failure counter,
  paged/pool stats) are separate and documented in BENCHMARKS §53.
