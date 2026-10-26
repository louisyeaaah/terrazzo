# Terrazzo

**Local LLM inference for Apple Silicon.** · [简体中文](README.zh.md)

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
- **Operations** — bearer auth, global rate limit, memory guard (503 + `Retry-After`),
  request cancellation, structured `/stats`, soak-tested.

## Why Terrazzo? (honest positioning)

All numbers below are measured on the same Apple M1 Max (24-core GPU, 64 GB) against
**Ollama 0.34.4** running `qwen3:4b` (GGUF Q4_K_M), with `terrazzo` serving
`mlx-community/Qwen3-4B-4bit`. Full methodology and negative results: [BENCHMARKS.md](BENCHMARKS.md),
[COMPARISON.en.md](docs/COMPARISON.en.md).

| Dimension | Terrazzo | Ollama | Note |
|---|---:|---:|---|
| Decode, short context | **100.7 tok/s** | 48.4 tok/s | 2.08×; MLX 4-bit vs GGUF Q4_K_M |
| First request, pre-registered prefix | **0.169 s** | 0.811 s | Ollama only caches after the first request |
| 8k pre-registered prefix, cold | **0.406 s** | 23.7 s (default) / 40.4 s | pre-registration is the point |
| Warm prefix reuse (same prompt) | 0.076 s | **0.033 s** | Ollama is faster here |
| Cold prefill (no prefix) | 1.040 s | **0.811 s** | Ollama is faster here |
| 4-way aggregate, default config | **54–58 tok/s** | 41.8 tok/s | Ollama gains at `num_parallel=4` |

Terrazzo is **not** a qualitative leap over Ollama: decode is ~2× and first-request
prefix handling is much better, but cold prefill and pure warm reuse still favor Ollama.
The realistic product slot is "agent inference gateway on a Mac": pre-registered long
system prompts, block-granular sharing, structured output, backpressure and observability.

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

16 concurrent sessions × 2 turns × 64 tokens with a 2.5k-token shared prompt:
**TTFT p50 2.26 s / p95 2.87 s, 95.8 tok/s aggregate** (prewarm + `--profile agent`),
versus 8–11 s p50 on the default 8-slot configuration. See [BENCHMARKS.md §44](BENCHMARKS.md).

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
python -m unittest discover -s tests                  # core suite (200 tests, 1 skipped)
python -m unittest discover -s experiments/tests -v   # experimental kernels (31 tests)
bash bench/head_to_head.sh                            # Ollama head-to-head
python bench/agent_workload.py --base ... --sessions 16
```

## Docs

- [docs/QUICKSTART.en.md](docs/QUICKSTART.en.md) · [简体中文](docs/QUICKSTART.zh.md) — install, API, tool calling, limitations
- [docs/COMPARISON.en.md](docs/COMPARISON.en.md) · [简体中文](docs/COMPARISON.zh.md) — vs Ollama / vLLM, decision matrix
- [BENCHMARKS.md](BENCHMARKS.md) — all benchmark data and negative results
- [DESIGN.md](DESIGN.md) — what is ported and why
- [MAC-TURBO.md](MAC-TURBO.md) — MacTurbo: predicting reachable tok/s before serving

## Limitations (nothing faked)

- **No Ollama registry / GGUF tags.** `pull` downloads Hugging Face MLX snapshots; use
  Ollama if you need GGUF.
- **No `/v1/embeddings`** — generation models only.
- Single machine, single loaded model; no multi-GPU/distributed.
- p50 agent TTFT is 2.26 s under load (target was ≤2 s): the remaining cost is mlx-lm's
  per-sequence copy of the shared prefix; the paged-sharing fix was evaluated and lost.
- Forcing tool calls (`tool_choice` specific/`required`) is best-effort; chat templates
  have no grammar hook.
- Experimental Metal kernels in `experiments/` are **slower** than MLX/SDPA; they are
  kept as documented negative results, not product paths.

## License

Apache-2.0.
