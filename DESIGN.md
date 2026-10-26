# Design notes: what it means to "port DeepInfer to a Mac"

DeepInfer's open-source tree is ~46k lines of Python plus ~6k lines of C++, written for
one model (DeepSeek-V4.1-Flash), one class of accelerators (SM100 / Ascend 950) and one
deployment shape (Wide-EP, 64–128 GPUs, PD-disaggregated). A Mac is none of those.
So "porting" has to be split into layers, and each layer gets a different answer.

## Layer-by-layer verdict

| Layer | DeepInfer | Portable to M1? | What this project does |
|---|---|---|---|
| Entrypoint (`entrypoints/llm.py`) | `LLM.generate` → `step()` loop | ✅ pure Python | `engine.py:generate` |
| Engine orchestration (`engine/llm_engine.py`) | scheduler + workers + post-process | ✅ shape, ⚠️ details | `engine.py:step` |
| Scheduler (`csrc/engine/scheduler.hpp`, 1329 lines) | chunked prefill, LCFS, DP voting | ✅ algorithm, ❌ voting | `scheduler.py` (~200 lines) |
| Block manager (`csrc/engine/block_manager.hpp`, 242 lines) | paged KV, ref-count | ✅ algorithm | `blocks.py` (~180 lines) |
| Worker / model runner (`worker/*.py`, 2805 lines) | CUDA streams/graphs, pin memory, GDR | ❌ | replaced by `runner.py` + mlx-lm |
| Model (`model_executor/v4_modules/*`, 5197 lines) | MLA/CSA2/Engram/mHC, V4.1-only | ❌ | unused (any mlx-lm model works) |
| Kernels (`tilelang_ops/`, 43 `*_tl` per backend) | CUDA / Ascend only | ⚠️ TileLang Metal exists but has open correctness bugs | ❌ |
| KV cache store / prefix hit | closed source | — | `paged_kv.py` `PagedPrefixCache` (added) |

Rough arithmetic from the audit in `../deepinfer-study/notes/05-mac-port-blockers.md`: the pure-CPU
engine shell is ~7–9k lines and *could* compile on macOS; the compute path and model
layer are 100% CUDA/V4.1-specific and would be ~3k lines to rewrite. This project takes
the cheap side of that trade: **re-implement the shell (~900 lines of Python), borrow the
executor from mlx-lm, and delete the rest.**

## Why the scheduler is the interesting part

DeepInfer's production value is mostly in the cluster (Wide-EP, EPLB, 3FS). What remains
portable *and* instructive is the single-rank scheduler:

1. **Chunked prefill** — a long prompt is admitted in `prefill_chunk_size` token chunks
   spread over several steps, so decode traffic can interleave instead of stalling behind
   a 32k-token prefill. In DeepInfer this is `_schedule_prefill`; here it is the
   `while waiting and budget > 0` loop in `scheduler.py:schedule`.
2. **Decode append-slot** — each decoding sequence needs one new block every
   `block_size` tokens. DeepInfer's `_append_slot` reserves it before the step; without
   it you discover the shortage mid-step. `_ensure_blocks` mirrors this.
3. **LCFS preemption** — when the pool is exhausted the victim is the sequence with the
   *shortest output*, i.e. the one whose recomputation costs least. DeepInfer's
   `_schedule_decode` sorts by `output_len` and preempts from that end; same here.
4. **Push-front requeue** — a preempted request goes back to the *front* of `waiting`, so
   it resumes before newer arrivals. `scheduler.py:_preempt`.
5. **Ref-counted blocks** — a block is returned to the free list only at `ref_count == 0`,
   which is what makes prefix-cache sharing (and forked sequences in DeepInfer) safe.

## Deliberate deviations

| Deviation | Why |
|---|---|
| No DP all-reduce step-type voting | DeepInfer needs every rank in an EP group to issue the same step type or the MoE all-to-all deadlocks. One Mac has one rank, so a step may freely mix prefill and decode — better TTFT. |
| No zero-overhead scheduling (ZOS) | ZOS exists to hide the CPU-side `schedule → post-process` gap behind GPU work started one step earlier. mlx-lm already issues `mx.async_eval` inside its step, so the GPU is fed; adding ZOS would only make output latency worse. |
| Logical paged KV, not physical | A real paged-attention kernel needs a custom attention that gathers K/V from a block pool. mlx-lm's caches are contiguous per sequence. The block manager therefore enforces capacity/admission/preemption and does the prefix-hash bookkeeping; the KV itself stays in MLX objects. This is the biggest fidelity gap and is documented in README limitations. |
| Prefix cache is exact-match only | A partial-prefix cache needs the block pool to be the source of truth for KV, i.e. the physical paging above. Exact match still captures the common agent pattern: a fixed system prompt. |
| `stop_preempted_request` honoured | DeepInfer drops preempted requests instead of recomputing when configured; that flag exists here too. |

## v0.2: what changed and why

v0.1 already had a `BlockManager` and a prefix cache, but the KV itself lived in
per-request MLX caches, so sharing was limited to *identical prompts*. v0.2 adds three
objects:

| Object | Role | DeepInfer analogue |
|---|---|---|
| `SharedPrefix` | frozen block-aligned per-layer KV of a prefix | a cached KV prefix in the (closed) 3FS store |
| `PagedPrefixCache` | ref-counted prefix blocks + LRU/byte budget | KVCache store metadata + eviction |
| `PagedKVCache` | drop-in `KVCache` returning `concat(prefix, suffix)` | the per-sequence view over paged blocks |

Why `concat` and not a gather kernel: MLX exposes no paged-attention primitive, so the
cache protocol (`update_and_fetch` returns dense K/V) forces a materialised view. The
copy is measurable but small: 0.5% decode throughput at a 592-token prefix, because
attention already streams the same bytes. Two real paging benefits survive anyway:
**the prefix KV is stored once** (87 MB instead of 87 MB × N) and **prefill skips it**
(TTFT 14.3× lower in the benchmark).

Correctness argument: for token-identical prefixes the KV is bit-identical, so attaching
a shared prefix is mathematically the same as recomputing it. The e2e test asserts greedy
token equality against a no-prefix session.

What is still missing for "real" paging:
1. a block-scatter KV write (write K/V slices straight into the pool instead of a
   per-sequence buffer) — needs custom attention;
2. partial-prefix sharing (share the first N blocks even if later tokens differ) — the
   hash chain in `blocks.py` already computes what is needed, but the executor has to
   accept a prefix shorter than the cached one;
3. ~~unifying the paged path with the continuous-batching path~~ — done in v0.3 at the
   cache level (seeded dense caches); a true single path still needs block-scatter KV
   writes inside attention.

## v0.3: one prefix cache, two execution paths

v0.2's shared prefix only worked on a hand-rolled sequential loop. v0.3 reuses the same
`PagedPrefixCache` from both paths:

* **batch path** — `Engine._insert_new` looks the prefix up with the block hash chain,
  builds per-layer `mlx_lm` `KVCache` objects whose `keys/values/offset` are the shared
  arrays, and hands them to `BatchGenerator.insert_segments(caches=..., all_tokens=...)`.
  The batcher then prefills only the suffix, and the request batches normally with other
  requests. When the sequence grows, `KVCache.update_and_fetch` allocates a new buffer and
  copies — this is the known memory cost of unifying with a dense-cache executor.
* **paged path** — `PagedKVCache` keeps returning `concat(prefix, suffix)`, so the prefix
  is never copied; the trade-off is one request at a time.

Block-granularity partial prefixes: `PagedPrefixCache.lookup` walks the hash chain block
by block and stops at the first block that is not cached, returning `hit_len` separately
from the stored entry. A prompt that shares 3 of 8 cached blocks still reuses those 3
blocks, and `PagedKVCache` slices the stored arrays to `hit_len`.

### A finding worth writing down: chunk-boundary numerics

Sharing is exact aliasing, but *recomputing* a prefix is not bit-identical to *reusing*
it, because the two computations have different attention shapes. On this 4-bit model:

| comparison | result |
|---|---|
| paged cache vs. recompute with the same chunk split | bit-identical logits (test asserts this) via `concat` |
| paged cache vs. one-shot prefill of the whole prompt | max logit diff ≈ 0.1–0.5 |
| prefill 160 tokens vs. prefill 320 tokens, compare first 160 positions' KV | differs (layer 20: up to 4.3) |
| prefill 160 vs. 160+160 chunked | identical |

So the network amplifies bf16/kernel-shape differences across 40 layers, and at the first
decode step the top-2 margin is frequently exactly 0 — greedy tokens can legitimately
flip. Equivalence tests therefore compare logits with a tolerance, not tokens; the CLI
benchmarks use temperature 0 only for timing, not for equality claims.

## v0.4: block pool, and the cost of a gather-based read

v0.3's prefix is a frozen dense tensor. It is shared, but:
* the executor copies it per sequence (copy-on-grow),
* it cannot be evicted at block granularity,
* it is not a real KV cache — just a read-only tensor.

v0.4 adds `PagedKVPool`: one preallocated `[num_blocks, H, block_size, D]` tensor per
layer, a per-sequence block table, and ref-counted blocks. Writing is a scatter
(`pool[block, :, offset:offset+n, :] = k`); reading is a gather of the sequence's blocks.

Two correctness traps worth recording:

1. **Per-layer length, not table length.** The model calls `update_and_fetch` once per
   layer for the same tokens. If the table advanced its length on the first layer, layer
   1 would RoPE at the wrong offset and read half-empty blocks. Fix: the table keeps
   `layer_length[layer]` and `PoolKVCache.offset` returns *its own* length. This bug cost
   a 3.9-logit divergence before it was found; the numeric test caught it.
2. **Ownership of prefix blocks.** Temporary caches used to prefill a prefix must not
   free its blocks on `__del__` (`auto_release=False`); the prefix entry owns them.

Cost of the gather: measured 73.0 → 53.9 tok/s (−26%) on decode, because each step
materialises `[1, H, L, D]` per layer instead of reading a pre-concatenated buffer.

### The Metal kernel: two attempts

Attempt 1 — one 128-thread threadgroup per head with a shared-memory tree reduction:
**correct but 2× slower than the gather baseline.** Every token paid ~9
`threadgroup_barrier`s.

Attempt 2 — one *simdgroup* (32 lanes) per head, `D/32` dims per lane, `simd_sum` for the
q·k reduction, online softmax in registers: no barriers inside the token loop. That was
≈ break-even, because 32 simdgroups × 450 serial token steps underutilises the GPU.

Attempt 3 — **split-K**: 4 simdgroups per head, each covering every 4th token, combined
once at the end through threadgroup memory. Now 32 heads × 4 simdgroups = 128 simdgroups
in flight. Measured relative to the gather baseline (decode):

| context | gather | split-K kernel |
|---|---:|---:|
| 400 tokens | 53.2 tok/s | **67.3 tok/s** |
| 1024 tokens (per-layer microbench) | 0.98 ms | **0.54 ms** |
| 4096 tokens (per-layer microbench) | 2.05 ms | **0.73 ms** |

Remaining ~9% gap to the dense-prefix path (73–75 tok/s) is 36 kernel launches per step
plus a hand-written kernel versus MLX's tuned SDPA; vectorised `half4` loads and
fusing the q reshape are the obvious next optimisations.

## What a "next version" would need

1. **Physical paged attention on MLX** — a `PagedKVCache` implementing
   `update_and_fetch` + a gather kernel over a block pool. This is the single change that
   would make prefix sharing between different prompts and real block-level eviction
   possible.
2. **Batch-level token budget** — mlx-lm's `BatchGenerator` owns the per-step batching
   decision; to make `max_num_batched_tokens` authoritative (and to charge prefill and
   decode against the same budget, as DeepInfer does) the generation loop has to be
   inlined, not delegated.
3. **A fair scheduler → executor interface** — DeepInfer passes `SchedulerOutputs` (seq
   metadata + block tables) into the runner. Inlining mlx-lm's `PromptProcessingBatch` /
   `GenerationBatch` would let the scheduler hand over exactly the sequences it chose.

## Number check

Measured on M1 Max 64 GB, Qwen3-4B 4-bit, greedy, 128 output tokens:

| Case | terrazzo | mlx-lm `BatchGenerator` | Ollama |
|---|---:|---:|---:|
| single request | 95.3 tok/s | 77–98 | 67.1 |
| 4 concurrent aggregate | 124.1 tok/s | 112.5 | 100.4 |
| 8 concurrent aggregate | 108.4 tok/s | 103.3 | 105.0 |
| prefix cache, warm TTFT | 0.099 s vs 0.955 s (9.5×) | prompt cache | — |

The engine is within noise of the executor it wraps, which is the right sanity check:
the ported scheduler adds correctness and control, not throughput. The 1.35× gap to
Ollama on a single request is kernel quality (MLX 4-bit vs llama.cpp Q4_K_M), not
scheduler design — and that gap closes to ~1.0× at 8-way concurrency, exactly as
predicted in `../deepinfer-study/DeepInfer-vs-Ollama-实测.md`.
