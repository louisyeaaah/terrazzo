# terrazzo vs Ollama and vLLM (English)

> Data date: 2026-10-27 (current code v0.9 + prefix-cache fix).
> Machine: Apple M1 Max (10-core CPU / 24-core GPU / 64 GB / 400 GB/s).
> Model: `mlx-community/Qwen3-4B-4bit` (MLX 4-bit, 2.28 GB);
> Ollama serves `qwen3:4b` (GGUF Q4_K_M, 2.5 GB); vLLM runs through the **vllm-metal** plugin
> (MLX backend, same MLX weights).
> Test status: **171 tests OK (1 skipped — needs an extra OlMoE model)**.

---

## 0. Bottom line

- **vs Ollama**: decode **2.08x**, first-request TTFT with a long shared prefix **5x+**, default
  4-way concurrency **+30%**. But **Ollama wins cold prefill and warm same-prompt prefix reuse**,
  and it wins model management/ecosystem by an order of magnitude. **There is no qualitative leap.**
- **vs vLLM (vllm-metal on Apple Silicon)**: we are faster on single-request and 4-way concurrency
  at 4B size (v0.3 measurement: 86.8 vs 67.8 tok/s; 89.2 vs 63.3 aggregate); but in the **warm
  agent long-prefix workload vLLM is faster** (v0.7 measurement: 67.9 vs 54.6 tok/s), and vLLM is
  far more complete as an engine (paged prefill, multi-GPU, production features). We are a Mac
  reference implementation, not a vLLM replacement.
- Our honest positioning: **agent inference-gateway semantics on a Mac** (zero-prefill first request,
  prefix sharing, backpressure/cancellation/observability, structured output) — not "a faster Ollama",
  and not "vLLM on a Mac".

---

## 1. Measurement protocol and fairness

- **One service at a time**; `ollama stop` before MLX runs so the two stacks never share GPU/memory.
- **Random front marker** in every system prompt, so cold starts cannot reuse another target's
  (or another round's) prefix.
- Model warmed with a throwaway prompt that does not share the measured prefix; temperature 0;
  identical `max_tokens`; medians (decode: 3 runs).
- Scripts: `bench/head_to_head.sh`, `bench/h2h_long_prompts.py`, `bench/compare_ollama.py`,
  `bench/api_bench.py`; full regression: `TERRAZZO_MAC_MODEL=<snapshot> python -m unittest discover -s tests`.
- **Known interference**: during the 2026-10-26 long-prompt Ollama runs the machine carried
  unrelated external load (Ghidra headless etc., load average 6–29). **Ollama absolute numbers are
  contaminated**; structural conclusions (who hits cache, order-of-magnitude gaps) still hold.
- **Quantization differs**: MLX 4-bit (2.28 GB) vs GGUF Q4_K_M (2.5 GB) — not bit-identical.
- **vLLM numbers are historical** (v0.3 / v0.7 phases) and were **not re-run** after the current v0.9
  changes; use them for capability/order-of-magnitude context only.

---

## 2. Versus Ollama

### 2.1 Decode (each engine's own counters, 128 tokens, median of 3, short context)

| Engine | decode tok/s | ratio |
|---|---:|---:|
| Ollama `qwen3:4b` (num_ctx 8192) | 48.4 | 1.00x |
| **terrazzo (MLX 4-bit)** | **100.7** | **2.08x** |

Source: `bench/compare_ollama.py` (2026-10-26, quiet window). This is a **short-context** result;
at 4k context we measure ~60–72 tok/s and a like-for-like long-context ratio has not been run yet.

### 2.2 ~440-token system prompt: TTFT / prefix reuse / 4-way concurrency

| Target | cold TTFT | warm same | warm diff | 4-way aggregate |
|---|---:|---:|---:|---:|
| terrazzo default | 1.040 s | 0.076 s | 0.236 s | **54.1 tok/s** |
| terrazzo pre-registered (`--system-file`) | **0.169 s** | 0.186 s | 0.306 s | **58.1 tok/s** |
| Ollama | 0.811 s | **0.033 s** | **0.111 s** | 41.8 tok/s |

- **Cold start**: Ollama's llama.cpp prefill is faster (0.811 vs 1.040 s).
- **Warm prefix reuse**: Ollama's slot cache is fastest (0.033 / 0.111 s); we are 0.076–0.306 s.
- **First request**: pre-registration gives us a hit on the very first request (0.169 s vs 0.811 s,
  **4.8x**) — Ollama must complete one request before its cache exists.
- **Concurrency**: our default is ~30% ahead; Ollama defaults to `num_parallel=1`, and with 4 it
  historically reached ~100 tok/s (a configuration issue, not a capability gap).

### 2.3 Long system prompts 2k / 4k / 8k (after the 2026-10-27 fix)

| Target | length | cold TTFT | warm same TTFT | warm diff TTFT |
|---|---:|---:|---:|---:|
| terrazzo default | 2k / 4k / 8k | 4.77 / 12.22 / 23.71 s | 0.145 / 0.170 / 0.254 s | 0.217 / 0.304 / 0.443 s |
| terrazzo pre-registered | 2k / 4k / 8k | **0.202 / 0.297 / 0.406 s** | 0.155 / 0.247 / 0.245 s | 0.158 / 0.245 / 0.319 s |
| Ollama (minimal ctx) | 2k / 4k / 8k | 3.37 / 15.68 / 40.39 s* | 0.034 / 0.113 / 0.118 s* | 0.123 / 0.243 / 0.396 s* |

\* contaminated by external load — do not quote the absolutes. Structural facts: pre-registration
still wins at 8k (cold 0.406 s), and after the fix the default configuration also caches 8k prompts
(warm same 0.254 s, meeting the ≤0.5 s target).

### 2.4 Where we beat Ollama / where Ollama beats us

**Ours**: 2x decode (MLX 4-bit stack), first-request long prefix (pre-registration), block-granular
partial prefix sharing, serving semantics (503 + Retry-After backpressure, abort by request_id,
hard memory cap, `/stats`, 1-hour soak 2161/2161 with 0 errors), JSON-Schema structured output.

**Ollama's**: cold prefill (llama.cpp), warm prefix reuse (slot cache), model management/downloads/
ecosystem, maturity.

---

## 3. Versus vLLM (vllm-metal on Apple Silicon)

### 3.1 How vLLM runs on a Mac

Upstream vLLM's macOS support is an **experimental source build** (CPU backend). What actually runs
is the **vllm-metal plugin** (`vllm-project/vllm-metal`, Apache-2.0): upstream vLLM engine /
scheduler / **paged block manager** + mlx-lm model layer + its own **paged varlen Metal attention
kernels** (including speculative decoding, GQA, M5 NAX prefill).

### 3.2 Single request / 4-way concurrency (v0.3 measurement, 2026-10-24)

| Engine (same machine/model) | single decode | 4-way aggregate | cold TTFT | warm TTFT |
|---|---:|---:|---:|---:|
| **terrazzo v0.3 batch** | **86.8** | **89.2** | 0.155 s | 0.126 s |
| vLLM + vllm-metal | 67.8 (cold) / 68.3 (warm) | 63.3 | 1.358 s | **0.107 s** |

- At 4B we are ~25% faster single-request and ~40% higher aggregate; vLLM has the faster warm
  TTFT (0.107 vs 0.126 s).
- vLLM's value at this size is not speed; it is the complete scheduler/API/paged kernels and
  larger-model support.

### 3.3 Agent long-prefix multi-session (v0.7 measurement, historical)

Workload: 2544-token shared system prompt, 4 concurrent sessions, 2 turns each, 64 tokens per turn
(`bench/agent_workload.py`).

| Engine | cold first round | warm aggregate | warm TTFT p50 | memory |
|---|---:|---:|---:|---:|
| terrazzo (pre-registered prefix) | **59.8 tok/s / 0.66 s** | 54.6 tok/s | 0.82 s | MLX peak **4.8 GB** |
| vLLM-Metal (`--gpu-memory-utilization 0.2`) | 36.5 tok/s / 3.75 s | **67.9 tok/s** | **0.48 s** | KV pool 7.32 GiB + weights |
| vLLM-Metal (default 0.92) | — | 37.2 tok/s | 3.65 s | KV pool **44.65 GiB** |

**Conclusion (we corrected an earlier apples-to-oranges comparison)**: once warm, vLLM is 24% faster
(67.9 vs 54.6 tok/s) with 40% lower TTFT; our only real advantages are the **cold first round**
(0.66 s vs 3.75 s, 5.5x) and **memory** (4.8 GB vs 7.3–44.7 GB). Root cause: our batch engine copies
the shared prefix into each sequence's dense KV cache (4 × 375 MB), while vLLM's paged blocks truly
share. This is a known limitation; a paged runtime exists but currently runs sequentially.

### 3.4 Capability matrix

| Capability | vLLM (NVIDIA / vllm-metal) | terrazzo |
|---|---|---|
| Paged KV (block table / ref count / CoW) | ✅ PagedAttention | ✅ block pool (v0.4, sequential path) |
| Paged **prefill** kernel | ✅ | ❌ prefill uses gather + SDPA |
| Paged **decode** kernel | ✅ | ✅ Metal split-K (q_len=1, on by default for long context) |
| Continuous batching | ✅ | ✅ (via mlx-lm BatchGenerator) |
| Block-level prefix sharing | ✅ automatic prefix caching | ✅ registered + block-granular partial hits |
| Multi-GPU / TP / PP | ✅ | ❌ single machine |
| Structured output / tools / metrics | ✅ | ✅ JSON Schema + tools/function calling; metrics ✅ |
| Production maturity | production-grade | research prototype (~4k LoC, 171 tests) |

---

## 3.5 Combining the strengths: B succeeded, A is a negative result (2026-10-27)

**B: vLLM-Metal + our startup pre-registration (`scripts/vllm_prewarm_proxy.py`)**
The proxy sends one warmup request with the shared system prompt to fill vLLM's prefix
cache, then forwards traffic. Agent workload (4 sessions x 2 turns x 64 tokens, 3257-token
shared prompt):

| Target | aggregate tok/s | TTFT p50 |
|---|---:|---:|
| vLLM-Metal raw (cold cache) | 16.8 | 9.11 s |
| **vLLM-Metal + prewarm proxy** | **38.2** | **1.73 s** |
| terrazzo (pre-registered prefix) | 34.4 | **1.09 s** |

Prewarming cuts vLLM's first-round TTFT by **5.3x**; the combined picture is still
"vLLM +11% throughput, we have better TTFT", and single-stream decode drops from ~100 to
~47-68 tok/s. **The mechanism works; it is not a qualitative leap.**

**Follow-up (2026-10-27 evening)**: after the scheduler round-robin chunked prefill +
`--profile agent` (16 slots) + `POST /v1/prewarm`, the same agent workload measures
**TTFT p50 2.26 s / p95 2.87 s, 95.8 tok/s aggregate** (baseline was 8.0-11.1 s /
36-44.8 tok/s). OpenAI `tools`/function calling is now supported (Qwen3 template +
`<tool_call>` parsing, streaming and non-streaming); `/v1/embeddings` is still missing.

**A: moving vLLM-style paged prefix sharing into the batch path (negative, not integrated)**
Packed shared-prefix attention microbenchmark (32 q heads / 8 kv heads, prefix 3400, B=4,
bf16, 36 layers):

| Variant | per layer | vs baseline | correctness |
|---|---:|---:|---|
| baseline: 4x SDPA | 1.77 ms | 1.00x | exact |
| packed plain (no mask) | 0.93-1.77 ms | 1.33-1.86x | wrong (sees other rows' suffixes) |
| packed mask (exact) | 1.08 ms | 2.17x* | exact |
| **packed + per-step concat (real integration)** | 1.60 ms | **1.11x** | exact |
| MMA v2 + lse merge | 23.4 ms | 0.05x | exact |

\* that 2.17x baseline includes concat inside the loop; adding the concat the packed path
needs every step leaves a **1.11x** attention gain (~+5% per step), far from the +20% target.
The MMA kernel v2 is numerically correct but ~10x slower than SDPA for this shape (the
earlier "MMA 3-4x faster" was a causal early-break measurement artifact). The memory target
would need the full paged-sharing rewrite (weeks), and our paged decode path already
measured slower. **Conclusion: A is not integrated; the practical combine ceiling is +5-11%.**

## 4. Decision matrix

| Your situation | Recommendation |
|---|---|
| Just want models on a Mac, GUI/model management/one-click pulls | **Ollama** |
| Mac agent service: long shared system prompts, low first-request latency, backpressure/abort/observability/structured output | **terrazzo** (pre-registered prefix + serving semantics) |
| Production, paged prefill, multi-GPU, tools/metrics, large models | **vLLM (Linux/NVIDIA)**; if it must be a Mac, try vllm-metal |
| Research on Mac inference systems, hacking scheduler/kernels | terrazzo (small, readable, tested, negative results documented) |

---

## 5. Limitations (read before quoting)

1. Only one 4B model was tested; 27B-class, MoE and tool-use workloads are untested.
2. vLLM numbers are historical (v0.3/v0.7) and neither the current v0.9 code nor newer vllm-metal
   was re-run.
3. Ollama long-prompt absolutes were affected by external load on 2026-10-26; warm reuse and cold
   prefill need a quiet re-run.
4. Quantizations differ (MLX 4-bit vs GGUF Q4_K_M); part of the decode gap comes from the quant
   format and runtime, not from scheduling innovation.
5. Serving semantics (503/abort/memory cap/soak) are implemented and tested but have no formal SLO
   comparison against vLLM/Ollama yet.
6. Concurrency conclusions use 4 sessions; at 16 sessions our TTFT degrades noticeably (cold prefill
   serialization in the scheduler, see BENCHMARKS §38).

---

## 6. Reproduction

```bash
# full test suite (171 tests OK, 1 skipped)
TERRAZZO_MAC_MODEL=<Qwen3-4B-4bit snapshot> python -m unittest discover -s tests

# vs Ollama: decode (authoritative counters)
python bench/compare_ollama.py --mlx-model <snapshot> --ollama-model qwen3:4b --num-ctx 8192

# vs Ollama: TTFT / prefix / concurrency (start terrazzo server + ollama first)
bash bench/head_to_head.sh

# vs Ollama: 2k/4k/8k long system prompts
python bench/h2h_long_prompts.py --mlx-model <snapshot> --lengths 2000,4000,8000
```
