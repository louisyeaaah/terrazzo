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
