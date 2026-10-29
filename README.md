# Terrazzo

[![CI](https://github.com/louisyeaaah/terrazzo/actions/workflows/ci.yml/badge.svg)](https://github.com/louisyeaaah/terrazzo/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/website-louisyeaaah.github.io%2Fterrazzo-e2725b)](https://louisyeaaah.github.io/terrazzo/)

**Local LLM inference for Apple Silicon.** · [Project page](https://louisyeaaah.github.io/terrazzo/) · [简体中文](README.zh.md)

Terrazzo is an inference runtime and server for macOS / Apple Silicon, built on
[MLX](https://github.com/ml-explore/mlx) and `mlx-lm`. It ports the scheduler and
block-manager semantics of **DeepInfer** (a cluster-scale C++/CUDA engine) to a single
Mac, then wraps them in OpenAI- and Ollama-compatible APIs with first-class support for
long shared system prompts — the shape of real agent traffic.

```text
           DeepInfer (research engine)                    Terrazzo (this project)
  ┌──────────────────────────────────────┐     ┌────────────────────────────────────────┐
  │ LLMEngine.step()                     │ ──▶ │ engine.py      Engine.step()            │
  │ C++ Scheduler (chunked prefill, LCFS)│ ──▶ │ scheduler.py   Scheduler                │
  │ C++ BlockManager (paged KV, refcount)│ ──▶ │ blocks.py      BlockManager             │
  │ KV store / prefix hit (closed source)│ ──▶ │ paged_kv.py    PagedPrefixCache         │
  │ model runner (CUDA graphs)           │ ──▶ │ runner.py      MLX runner (mlx-lm)      │
  │ FlashMLA / DeepGEMM / DeepEP         │ ──▶ │ ——— MLX kernels, no equivalents ———     │
  │ internal server                      │ ──▶ │ server.py      OpenAI + Ollama API      │
  └──────────────────────────────────────┘     └────────────────────────────────────────┘
```

## Highlights

- **Engine** — paged KV block manager with ref-counting, continuous batching,
  round-robin chunked prefill, LCFS preemption, speculative n-gram decoding.
- **Prefix caching** — exact shared-prefix snapshots stored once and reused across
  requests; automatic registration on the first batch; `POST /v1/prewarm` /
  `--system-file` to register a system prompt before traffic arrives.
- **Serving** — OpenAI `/v1/chat/completions` (streaming, JSON Schema constrained
  decoding, `tools`/function calling) and Ollama `/api/chat|generate|tags|ps|show|create|copy|delete`,
  plus `/v1/prewarm`, `/stats`, `/v1/abort`.
- **CLI** — Ollama-style model management: `run`, `pull`, `list`, `show`, `rm`, `cp`,
  `create -f Modelfile`, `ps`, `prewarm`, `serve`; Modelfile supports `FROM` / `SYSTEM` /
  `PARAMETER`.
- **Operations** — bearer auth (constant-time compare), optional TLS, global rate
  limit, memory guard (503 + `Retry-After`), bounded concurrency, per-request
  deadlines, a worker hang watchdog (`/stats.worker`), request cancellation,
  structured `/stats`.

## Performance (measured 2026-10-27)

Environment: Apple M1 Max (24-core GPU, 64 GB), macOS 26.6.2, load average 2–4,
`terrazzo` 0.3.0 (commit `8d00b0c`), MLX 0.32.4 / mlx-lm 0.31.3. Ollama 0.34.4 with
`qwen3:4b` (GGUF Q4_K_M); vLLM 0.32.0 + vllm-metal, `--max-model-len 4096
--gpu-memory-utilization 0.2`. One service at a time, warmed model, temperature 0.
Token counts are server-reported (`usage` for OpenAI-style APIs, `eval_count` for the
native Ollama API), never SSE chunk counts.

### Decode, short prompt (128 tokens)

| engine | decode tok/s | vs Ollama |
|---|---:|---:|
| terrazzo (MLX 4-bit) | **101.1** | **1.51×** |
| Ollama qwen3:4b (num_ctx 8192) | 67.0 | 1.00× |

### Agent workload: 2.6k shared system prompt, 2 turns × 64 tokens

| backend / config | sessions | aggregate tok/s | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|---:|
| terrazzo (`--profile agent` + `/v1/prewarm`) | 1 | **59.4** | **0.29 s** | 0.37 s |
| terrazzo | 4 | **74.1** | **0.60 s** | 1.44 s |
| terrazzo | 16 | 88.7 | 2.56 s | 3.49 s |
| Ollama 0.34.4 (native `/api/chat`) | 1 | 19.6 | 2.21 s | 4.21 s |
| Ollama | 4 | 32.6 | 5.16 s | 7.80 s |
| Ollama | 16 | 40.6 | 22.39 s | 25.04 s |
| vLLM-Metal | 1 | 43.5 | 0.19 s | 0.24 s |
| vLLM-Metal | 4 | 43.8 | 3.25 s | 5.78 s |
| vLLM-Metal | 16 | **122.1** | 1.57 s | 2.00 s |

vLLM's single-stream number is the second (steady-state) run: the first run after
startup is 15.1 tok/s / 2.95 s because of kernel warm-up. Its 4/16-session rows are
steady state as well.

### Model support and caveats

- **Prefix-cache path (supported):** Qwen3 family (tested with Qwen3-4B) with
  `--profile agent` / `--profile mixed`. This is the production path.
- **Experimental native snapshot:** Qwen3.5 / Gemma 4 can now prewarm and restore native
  full-prefix cache state (ArraysCache / RotatingKVCache), but strict cached-vs-no-cache
  output equality still fails in the initial tests. They are not production presets yet.
- No "which model is fastest overall" claim. Generation-behavior diagnostics, benchmark
  caveats and invalidated results live in
  [BENCHMARKS §54–62](BENCHMARKS.md); performance comparisons are no longer a homepage claim.
- **Cache correctness gate is not fully green yet:** Qwen3-4B single-request, concurrent,
  snapshot-reuse and abort-then-reuse match the no-cache path; one multi-turn branch differs
  by one token at turn 2. Qwen3.5 native snapshots also differ from no-cache on a single
  request. See [BENCHMARKS §58–62](BENCHMARKS.md). Cache gains stay unclaimed until resolved.
- Decode-only numbers exclude queueing, prefill and first-token wait; do not read them as
  end-to-end speed.

### Long system prompts: pre-registration is the differentiator

| prompt | terrazzo default cold TTFT | terrazzo prewarmed cold TTFT | Ollama cold TTFT | terrazzo warm-same | Ollama warm-same |
|---|---:|---:|---:|---:|---:|
| 2k | 4.37 s | **0.154 s** | 3.20 s | 0.153 s | **0.033 s** |
| 4k | 8.91 s | **0.242 s** | 6.95 s | 0.125 s | **0.044 s** |
| 8k | 19.87 s | **0.343 s** | 17.61 s | 0.081 s | **0.060 s** |

Decode rate on the same prompts (prewarmed): **82.8 / 74.6 / 57.0 tok/s** at 2k/4k/8k
vs Ollama 45.2 / 43.6 / 38.0 tok/s.

### Long-context decode (terrazzo)

| context | decode tok/s | TTFT |
|---:|---:|---:|
| 1k | 87.2 | 1.95 s |
| 2k | 76.3 | 3.93 s |
| 4k | 70.7 | 8.36 s |
| 8k | 57.2 | 19.21 s |

### Memory

Peak MLX memory from `/stats`: **3.9 GB** with 1 session, **4.9 GB** with 4,
**9.85 GB** with 16 concurrent sessions; process RSS ~2.47 GB; a 2544-token prefix
snapshot is ~375 MB. vLLM's KV cache lives in its engine process, so its 1.1 GB RSS
is not a comparable number.

### Honest positioning after these numbers

- **vs Ollama**: decode 1.5×, agent throughput 1.8–2.2× at 4/16 sessions, and
  prewarmed first-token latency 10–50× better on 2k–8k shared prompts. Ollama still
  wins cold prefill *without* pre-registration and pure warm-prefix reuse.
- **vs vLLM-Metal**: terrazzo is faster single-stream (59.4 vs 43.5 tok/s) and at
  4 sessions (74.1 vs 43.8), but vLLM-Metal wins the 16-session batch (122.1 vs 88.7)
  and its TTFT (1.57 vs 2.56 s). There is no throughput crown at high concurrency.
- **Summary**: terrazzo's edge is agent-style traffic with a pre-registered long
  system prompt at low/medium concurrency on one local model. It is not a drop-in
  throughput replacement for vLLM at scale, and not faster than Ollama for cold
  prefill or pure prefix reuse.

Reproduce with `bench/compare_ollama.py`, `bench/h2h_long_prompts.py --only all`, and
`bench/agent_workload.py` (exact commands, raw rows and caveats: [BENCHMARKS.md](BENCHMARKS.md) §48).
Earlier table rows (e.g. 2.08× decode) came from loaded-machine runs and are superseded.

### Co-design status (2026-10-27)

I tested one mechanism that connects the cache and the scheduler without writing a kernel:
**cache-hit-aware admission ordering** (predicted uncached work + predicted new KV bytes,
instead of FIFO). It is a limited cache/admission-scheduling experiment, not full-stack
co-optimization. The honest summary:

- Mixed 16-session traffic, single turn: cost ordering beat uncapped FIFO (p95 TTFT
  57–70 s vs 74–81 s; peak memory 18–20 GB vs 30–41 GB).
- **A plain FIFO concurrency cap at 4 slots did better on aggregate metrics**: 14.5 vs
  12.6 tok/s, p95 TTFT 67.4 vs 71.8 s, peak 18.5 vs 19.5 GB. Continuous arrivals showed
  the same overall ordering (cap4 best, cost second, uncapped FIFO last), but cost did
  change service priority: shared-prefix p50 14.7 s vs 35.0 s for cap4. That trade-off
  only matters if the product goal is "serve cache-hit requests first"; our current goal
  does not require it.
- FIFO-16 and cost-ordering had the same *logical* KV token count (44,784). That only
  shows logical KV accounting cannot explain the memory difference; physical residency,
  padding, copies and allocator retention were not measured, so the "transient
  prefill/copy buffers" explanation is still a hypothesis.
- cap4 is the winner **in this dataset**, not a universal default. Tuning and validation
  workloads must be separated; future comparisons should report both a pre-chosen fixed
  cap and the per-workload oracle cap.
- Shared-only traffic, 8 sessions × 2 turns and 8k prompts showed no ordering benefit;
  prefix caching itself remains the dominant lever.
- Auto-cap experiment (shared → mixed → shared, one process): after fixing the
  admission-profile confound in the first run (fixed4 used different ADMIT_BATCH/WAIT and
  runner batch), **fixed cap4 is the best static policy**: 26.4 tok/s / 116.6 s wall vs
  auto 23.9 / 128.4 s and fixed16 22.8 / 134.9 s; oracle 16/4/16 is 26.0 / 117.9 s. That
  "close to oracle" applies to aggregate throughput/total wall only: fixed4's shared-phase
  p95 is 2.10/3.63 s vs oracle 0.61/0.69 s, a real latency trade-off. Auto switches
  16→4→16 without flapping and beats fixed16 on mixed tail, but it does not beat fixed4
  overall. The earlier "auto tracks oracle / beats both fixed caps" claim was confounded
  and is retracted. `--profile mixed` now matches the tested fixed4 config (cap4,
  completion batch 16, ADMIT_BATCH 16, ADMIT_WAIT 4). Auto remains opt-in; manual presets
  stay the default.

The scheduler still cannot control the executor's prefill budget (16.4k scheduled vs 41.2k
actual prompt tokens, 2.5×), so the only controllable mechanism is admission/order.
**Conclusion: the right label is "cache/admission-scheduling joint experiment", not
"co-optimized engine". Tuned FIFO concurrency is the current default; cost ordering stays
as an experimental option. What was disproved is "cost ordering beats simple concurrency
control on aggregate metrics", not the entire value of ordering.**
Full protocol, cap scan, continuous-arrival numbers and the unresolved 16×2 hang:
[docs/CO-DESIGN.md](docs/CO-DESIGN.md).

## Quickstart

```bash
git clone https://github.com/louisyeaaah/terrazzo.git && cd terrazzo
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

terrazzo pull mlx-community/Qwen3-4B-4bit     # Hugging Face snapshot (not an Ollama tag)
terrazzo serve --model Qwen3-4B-4bit --port 8080
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "terrazzo",
  "messages": [{"role":"user","content":"Explain KV cache in one sentence."}],
  "max_tokens": 128, "temperature": 0
}'
```

### Agent traffic (shared long system prompt)

```bash
# register the system prompt once, at startup or against a running server
terrazzo serve --model Qwen3-4B-4bit --profile agent --system-file system.txt
terrazzo prewarm --base http://127.0.0.1:8080 --system-file system.txt
```

16 concurrent sessions × 2 turns × 64 tokens with a 2.6k-token shared prompt:
**TTFT p50 2.56 s / p95 3.49 s, 88.7 tok/s aggregate** (prewarm + `--profile agent`),
versus 8–11 s p50 on the default 8-slot configuration. These are server-usage-counted
tokens with real multi-turn answers; see [BENCHMARKS.md §48](BENCHMARKS.md).

## CLI (Ollama mapping)

| Ollama | Terrazzo |
|---|---|
| `ollama pull qwen3:4b` | `terrazzo pull mlx-community/Qwen3-4B-4bit` |
| `ollama run` | `terrazzo run [MODEL] [PROMPT]` |
| `ollama create -f Modelfile` | `terrazzo create NAME -f Modelfile` |
| `ollama list / show / rm / cp / ps` | `terrazzo list / show / rm / cp / ps` |
| `ollama serve` | `terrazzo serve` |
| `ollama push` | not supported (no Terrazzo registry) |
| — | `terrazzo prewarm --system-file ...` |

Model store: `~/.terrazzo/models` (`TERRAZZO_MODELS_DIR` overrides). Modelfile:
`FROM`, `SYSTEM`, `PARAMETER` (`temperature/top_p/top_k/num_predict/stop/num_ctx/seed`);
`TEMPLATE` is rejected on purpose — MLX models carry their own chat template.

## HTTP API

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions` | ✅ streaming, JSON Schema `response_format`, `tools`/`tool_calls` |
| `POST /v1/completions`, `/v1/abort`, `/v1/prewarm` | ✅ |
| `POST /api/chat`, `POST /api/generate` | ✅ NDJSON streaming, `options`, `format`, `think`, tools |
| `GET /api/version`, `/api/tags`, `/api/ps` | ✅ |
| `POST/DELETE /api/show`, `/api/create`, `/api/copy`, `/api/delete` | ✅ |
| `/api/embed`, `/api/embeddings`, `/api/pull`, `/api/push` | explicit 501 |

Tool calling uses the model's chat template (Qwen3 is tool-aware) and parses
`<tool_call>` blocks into proper `tool_calls`; tool requests disable `<think>` by default.

## Architecture map

| DeepInfer mechanism | DeepInfer source | Terrazzo |
|---|---|---|
| `step()` loop + streaming outputs | `engine/llm_engine.py` | `engine.py` |
| chunked prefill under a token budget | `csrc/engine/scheduler.hpp` | `scheduler.py` |
| FIFO waiting queue, LCFS preemption | `csrc/engine/scheduler.hpp` | `scheduler.py` |
| paged KV block manager, ref-count + free list | `csrc/engine/block_manager.hpp` | `blocks.py` |
| shared-prefix KV store (closed source upstream) | — | `paged_kv.py` |
| model execution | CUDA graphs + custom kernels | `runner.py` (mlx-lm) |
| CUDA kernels (FlashMLA/DeepGEMM/DeepEP), DP voting, EP/PD | cluster-only | not ported |

Design rationale and fidelity notes: [DESIGN.md](DESIGN.md). Operations:
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Repository layout

```text
terrazzo/          engine, scheduler, blocks, paged KV, server, CLI, tools
bench/             reproducible benchmarks (head-to-head, agent workload, prefill, ...)
experiments/       negative-result kernels kept for reproducibility (not installed)
tests/             default test suite
docs/              quickstart, comparison, operations
BENCHMARKS.md      every measurement, including the ones that lost
```

## Tests and benchmarks

```bash
python -m unittest discover -s tests                  # core suite (255 tests, 1 skipped)
python -m unittest discover -s experiments/tests -v   # experimental kernels (31 tests)
python scripts/ci_core_tests.py                       # pure-Python subset with MLX blocked
bash bench/head_to_head.sh                            # Ollama head-to-head
python bench/agent_workload.py --base ... --sessions 16   # --api ollama for native eval_count
```

CI runs two jobs: a Linux job that executes the pure-Python suite with MLX blocked
(proving the scheduler/HTTP layer does not need Apple Silicon) and a macOS job that
installs the pinned MLX dependencies and runs the same suite with MLX available.
Benchmark numbers, dependency pins and the tested matrix are documented in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).

## Docs

- [docs/QUICKSTART.en.md](docs/QUICKSTART.en.md) · [简体中文](docs/QUICKSTART.zh.md) — install, API, tool calling, limitations
- [docs/COMPARISON.en.md](docs/COMPARISON.en.md) · [简体中文](docs/COMPARISON.zh.md) — vs Ollama / vLLM, decision matrix
- [BENCHMARKS.md](BENCHMARKS.md) — all benchmark data and negative results
- [DESIGN.md](DESIGN.md) — what is ported and why
- [Project page](https://louisyeaaah.github.io/terrazzo/) — measured results, honest positioning and quick start (also [中文](https://louisyeaaah.github.io/terrazzo/zh.html))
- [docs/CO-DESIGN.md](docs/CO-DESIGN.md) — the cache/scheduler co-design experiment: control boundary, A/B/C/D ablation, negative results
- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) — pinned/tested MLX versions and the runtime-patch contract
- [MAC-TURBO.md](MAC-TURBO.md) — MacTurbo: predicting reachable tok/s before serving

## Limitations (nothing faked)

- **No Ollama registry / GGUF tags.** `pull` downloads Hugging Face MLX snapshots; use
  Ollama if you need GGUF.
- **No `/v1/embeddings`** — generation models only.
- Single machine, single loaded model; no multi-GPU/distributed.
- p50 agent TTFT is 2.28 s under load (target was ≤2 s): the remaining cost is mlx-lm's
  per-sequence copy of the shared prefix; the paged-sharing fix was evaluated and lost.
- mlx-lm is pinned to `>=0.31.3,<0.32` because runtime patches touch its cache internals;
  the engine refuses to start outside that range unless `TERRAZZO_SKIP_MLX_LM_CHECK=1`.
- Forcing tool calls (`tool_choice` specific/`required`) is best-effort; chat templates
  have no grammar hook.
- Experimental Metal kernels in `experiments/` are **slower** than MLX/SDPA; they are
  kept as documented negative results, not product paths.
- A hung GPU call cannot be preempted: a heartbeat watchdog marks the runtime unhealthy
  (`503`, `/stats.worker.hung`), and `TERRAZZO_HANG_RESTART=1` exits for a supervisor
  to restart. See [docs/OPERATIONS.md](docs/OPERATIONS.md#known-gaps-residual-risks).

## License

Apache-2.0.
