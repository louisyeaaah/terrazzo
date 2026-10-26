# terrazzo quick start (Ollama-style)

> Goal: pull models, serve an OpenAI-compatible API and call it like Ollama on a Mac,
> with first-class support for long shared system prompts (agent traffic).

## Install

```bash
git clone https://github.com/louisyeaaah/terrazzo.git && cd terrazzo
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
# or: bash scripts/install.sh
```

Requirements: Apple Silicon, macOS 14+, Python 3.10+, MLX (`pip install -e .` pulls mlx / mlx-lm).

## Model management (ollama pull / list / show / rm)

```bash
# pull an MLX snapshot from Hugging Face (Ollama registry tags are NOT supported)
terrazzo pull mlx-community/Qwen3-4B-4bit

terrazzo list                 # NAME / SIZE / MODIFIED / ARCH
terrazzo list --json
terrazzo show Qwen3-4B-4bit   # config summary + file list
terrazzo rm Qwen3-4B-4bit

# existing local directory: symlink into the store instead of copying
terrazzo import /path/to/mlx-model --name qwen3-local

# store location: $TERRAZZO_MODELS_DIR (default ~/.terrazzo/models)
```

## Serve

```bash
terrazzo serve --model Qwen3-4B-4bit --port 8080
# OpenAI-compatible (/v1/chat/completions), continuous batching, prefix cache
```

### Agent workload (long shared system prompt + many sessions)

```bash
# 1) pre-register the system prompt at startup (no prefill on the first request)
terrazzo serve --model Qwen3-4B-4bit --profile agent \
    --system-file system_prompt.txt
# note: --system/--system-file also acts as the default system prompt for requests
# that do not carry a system message

# or prewarm a running server:
terrazzo prewarm --base http://127.0.0.1:8080 --system-file system_prompt.txt
# → {"status":"prewarmed","registered":true,"tokens":2544,"ms":5992}
```

Measured (M1 Max, 16 sessions x 2 turns x 64 tokens, 2.5k shared prefix): TTFT p50
**2.14 s** / p95 3.07 s, aggregate **96 tok/s**; without prewarm and with the default
8 slots it is 8-11 s p50. See BENCHMARKS §44.

### Runtime status

```bash
terrazzo ps --base http://127.0.0.1:8080        # /api/ps
curl -s localhost:8080/api/tags                       # model list
curl -s localhost:8080/stats                          # steps / prefix hits / memory
```

## OpenAI API

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "terrazzo",
  "messages": [{"role":"system","content":"You are a terse assistant"},
               {"role":"user","content":"Explain KV cache in one sentence"}],
  "max_tokens": 128, "temperature": 0
}'
```

Supported: `stream`, `response_format={"type":"json_schema", ...}` (constrained
decoding), `/v1/models`, `/v1/abort`, `/stats`, `/health`. Auth/limits: `--api-key`,
`--rate-limit`; memory cap: `--memory-cap-mb` (503 + Retry-After when exceeded).

## Tool calling (OpenAI `tools`)

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "terrazzo",
  "messages": [{"role":"user","content":"Weather in Beijing?"}],
  "tools": [{"type":"function","function":{
      "name":"get_weather","description":"Get weather",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                    "required":["city"]}}}],
  "max_tokens": 128
}'
```

- The response has `finish_reason: "tool_calls"` and
  `message.tool_calls[*].function.arguments` as a JSON string; the model may also answer
  directly (`finish_reason: "stop"`).
- **Multi-turn**: send `assistant.tool_calls` back unchanged and append
  `{"role":"tool","tool_call_id":...,"content":...}`; the server normalizes OpenAI wire
  details such as `content: null`.
- **`tool_choice`**: `auto` (default) / `none` (no tools injected, no parsing) / a specific
  function (only that tool + "must call" instruction) / `required` (must-call instruction).
  Forcing is best-effort (MLX chat templates have no grammar hook).
- Requests with tools disable `<think>` by default (`enable_thinking=false`); pass
  `"thinking": true` to opt back in.
- Duplicate identical calls are de-duplicated; streaming works but tool calls arrive in the
  final SSE chunk (coarse-grained, protocol compatible). Verified with Qwen3-4B: single-turn,
  multi-turn tool results, two tools, and forced function.

## Ollama compatibility: CLI and HTTP API

**CLI (covers the common Ollama commands)**

```bash
terrazzo run base "Explain KV cache in one sentence"  # one-shot with a prompt; REPL without
terrazzo run terse                                    # interactive; Modelfile SYSTEM/PARAMETER applied

terrazzo create terse -f Modelfile         # create from a Modelfile (FROM/SYSTEM/PARAMETER)
terrazzo show terse --modelfile            # print the stored Modelfile
terrazzo cp terse terse2                   # copy (symlink, weights are not duplicated)
terrazzo stop terse                        # single resident model: nothing to unload
terrazzo push terse                        # explicitly unsupported (no registry)
terrazzo version
```

Modelfile: `FROM <store-name-or-path>`, `SYSTEM` (incl. triple-quoted multi-line),
`PARAMETER` (`temperature/top_p/top_k/num_predict/stop/num_ctx/seed`). `TEMPLATE`/`ADAPTER`/`MESSAGE`
are rejected with a clear error (MLX models carry their own chat template).

**HTTP API (Ollama clients can call these directly)**

| Endpoint | Status |
|---|---|
| `POST /api/chat`, `POST /api/generate` | ✅ NDJSON streaming + non-streaming; `options` (temperature/top_p/top_k/num_predict/stop), `format` (json/schema), `think` (default false; reasoning in `message.thinking` when true), tools |
| `GET /api/version`, `GET/POST /api/tags`, `GET /api/ps` | ✅ |
| `POST/DELETE /api/show`, `/api/create`, `/api/copy`, `/api/delete` | ✅ |
| `/api/embed`, `/api/embeddings` | ❌ 501 (generation models only) |
| `/api/pull`, `/api/push` | ❌ 501 (no registry; use CLI `pull <hf-repo>`) |

```bash
curl localhost:8080/api/chat -d '{"model":"terse","stream":false,
  "messages":[{"role":"user","content":"hello"}]}'
curl localhost:8080/api/generate -d '{"model":"terse","prompt":"1+1=?","stream":false}'
```

## Ollama command mapping

| Ollama | terrazzo | Notes |
|---|---|---|
| `ollama pull qwen3:4b` | `terrazzo pull mlx-community/Qwen3-4B-4bit` | HF repo id, not an Ollama tag |
| `ollama list` | `terrazzo list` | local store |
| `ollama show` | `terrazzo show` | config / file summary |
| `ollama rm` | `terrazzo rm` | symlinks only remove the link |
| `ollama run` | `terrazzo chat` / `generate` | supports `--json-schema` |
| `ollama serve` | `terrazzo serve` | OpenAI + `/api/*` (chat/generate/show/create/copy/delete/version/tags/ps) |
| `ollama create` | `terrazzo create -f Modelfile` | FROM/SYSTEM/PARAMETER; TEMPLATE unsupported |
| `ollama cp` | `terrazzo cp` | symlink copy |
| `ollama stop` | `terrazzo stop` | single resident model, no unload |
| `ollama push` | `terrazzo push` | explicitly unsupported (no registry) |
| `ollama ps` | `terrazzo ps` | via `/api/ps` |
| — | `prewarm` | register a long system prompt up front (agent traffic) |

## Limitations (nothing faked)

- **No Ollama registry / GGUF tags**: this runtime serves MLX snapshots; `pull` says so
  explicitly and points at HF repo ids. Keep Ollama for GGUF.
- Capabilities follow mlx-lm (generation models); **tools/function calling is supported**
  (requires a tool-aware chat template, e.g. Qwen3); **`/v1/embeddings` is not implemented
  yet** (roadmap).
- Single machine, single model; no multi-GPU/distributed.
- No CUDA on macOS; 4B 4-bit decode is ~60-100 tok/s. The performance ceiling and all
  negative results are documented in `BENCHMARKS.md`.
