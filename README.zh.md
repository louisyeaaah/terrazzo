# Terrazzo

**Apple Silicon 上的本地 LLM 推理服务。** · [English](README.md)

Terrazzo 是面向 macOS / Apple Silicon 的推理运行时与服务端，基于
[MLX](https://github.com/ml-explore/mlx) 和 `mlx-lm`。它把 **DeepInfer**（集群级 C++/CUDA
引擎）的调度器与块管理语义移植到单台 Mac，再封装成 OpenAI / Ollama 兼容 API，
并针对真实 agent 流量的形态——**长共享系统提示 + 多会话**——做了专门优化。

```text
           DeepInfer（研究用引擎）                        Terrazzo（本项目）
  ┌──────────────────────────────────────┐     ┌────────────────────────────────────────┐
  │ LLMEngine.step()                     │ ──▶ │ engine.py      Engine.step()            │
  │ C++ Scheduler（chunked prefill/LCFS）│ ──▶ │ scheduler.py   Scheduler                │
  │ C++ BlockManager（paged KV/引用计数）│ ──▶ │ blocks.py      BlockManager             │
  │ KV store / 前缀命中（闭源）           │ ──▶ │ paged_kv.py    PagedPrefixCache         │
  │ model runner（CUDA graphs）          │ ──▶ │ runner.py      MLX runner（mlx-lm）     │
  │ FlashMLA / DeepGEMM / DeepEP         │ ──▶ │ ——— MLX 内核，无对应物 ———              │
  │ 内部 server                          │ ──▶ │ server.py      OpenAI + Ollama API      │
  └──────────────────────────────────────┘     └────────────────────────────────────────┘
```

## 核心能力

- **引擎**：paged KV 块管理（引用计数 + free list）、连续批处理、跨请求 round-robin
  chunked prefill、LCFS 抢占、n-gram 投机解码。
- **前缀缓存**：共享前缀快照只算一次、跨请求复用；首批请求自动注册；`POST /v1/prewarm` /
  `--system-file` 可在流量到来前预注册系统提示。
- **服务端**：OpenAI `/v1/chat/completions`（流式、JSON Schema 约束解码、`tools`/function
  calling）+ Ollama `/api/chat|generate|tags|ps|show|create|copy|delete`，另有
  `/v1/prewarm`、`/stats`、`/v1/abort`。
- **CLI**：Ollama 风格模型管理——`run`、`pull`、`list`、`show`、`rm`、`cp`、`create -f
  Modelfile`、`ps`、`prewarm`、`serve`；Modelfile 支持 `FROM` / `SYSTEM` / `PARAMETER`。
- **运维**：Bearer 鉴权、全局限流、内存守卫（503 + `Retry-After`）、请求取消、结构化
  `/stats`、可选 `--profile agent`。

## 为什么选 Terrazzo（诚实定位）

以下数字均在同一台 Apple M1 Max（24 核 GPU / 64GB）上测量，对比 **Ollama 0.34.4** 的
`qwen3:4b`（GGUF Q4_K_M）与 `terrazzo` 服务的 `mlx-community/Qwen3-4B-4bit`。完整口径与
负结果见 [BENCHMARKS.md](BENCHMARKS.md) 和 [COMPARISON.zh.md](docs/COMPARISON.zh.md)。

| 维度 | Terrazzo | Ollama | 说明 |
|---|---:|---:|---|
| 短上下文 decode | **100.7 tok/s** | 48.4 tok/s | 2.08×；MLX 4-bit vs GGUF Q4_K_M |
| 预注册前缀的首请求 | **0.169 s** | 0.811 s | Ollama 要等首请求结束才有缓存 |
| 8k 预注册前缀冷启动 | **0.406 s** | 23.7 s（默认）/ 40.4 s | 预注册是关键 |
| 纯前缀复用（同提示） | 0.076 s | **0.033 s** | 这一项 Ollama 更快 |
| 冷 prefill（无前缀） | 1.040 s | **0.811 s** | 这一项 Ollama 更快 |
| 4 并发默认配置聚合 | **54–58 tok/s** | 41.8 tok/s | Ollama 调到 `num_parallel=4` 会反超 |

Terrazzo **不是**对 Ollama 的"质变"：decode 约 2×、首请求前缀处理明显更好，但冷 prefill
和纯 warm 复用仍输给 Ollama。它真正的定位是"Mac 上的 agent 推理网关"：长系统提示预注册、
块粒度前缀共享、结构化输出、背压/取消/可观测。

## 快速开始

```bash
git clone https://github.com/louisyeaaah/terrazzo.git && cd terrazzo
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

terrazzo pull mlx-community/Qwen3-4B-4bit     # Hugging Face 快照（不是 Ollama tag）
terrazzo serve --model Qwen3-4B-4bit --port 8080
```

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "terrazzo",
  "messages": [{"role":"user","content":"用一句话解释 KV cache。"}],
  "max_tokens": 128, "temperature": 0
}'
```

### Agent 流量（共享长系统提示）

```bash
# 启动时或对运行中的服务预注册系统提示
terrazzo serve --model Qwen3-4B-4bit --profile agent --system-file system.txt
terrazzo prewarm --base http://127.0.0.1:8080 --system-file system.txt
```

16 会话 × 2 轮 × 64 token、2.5k 共享提示：**TTFT p50 2.26 s / p95 2.87 s，聚合 95.8 tok/s**
（prewarm + `--profile agent`）；默认 8 槽配置下 p50 为 8–11 s。见 [BENCHMARKS.md §44](BENCHMARKS.md)。

## CLI（Ollama 对照）

| Ollama | Terrazzo |
|---|---|
| `ollama pull qwen3:4b` | `terrazzo pull mlx-community/Qwen3-4B-4bit` |
| `ollama run` | `terrazzo run [MODEL] [PROMPT]` |
| `ollama create -f Modelfile` | `terrazzo create NAME -f Modelfile` |
| `ollama list / show / rm / cp / ps` | `terrazzo list / show / rm / cp / ps` |
| `ollama serve` | `terrazzo serve` |
| `ollama push` | 不支持（没有 Terrazzo registry） |
| — | `terrazzo prewarm --system-file ...` |

模型库默认 `~/.terrazzo/models`（`TERRAZZO_MODELS_DIR` 可覆盖）。Modelfile 支持
`FROM`、`SYSTEM`、`PARAMETER`（`temperature/top_p/top_k/num_predict/stop/num_ctx/seed`）；
`TEMPLATE` 会明确报错——MLX 模型自带 chat template。

## HTTP API

| 接口 | 状态 |
|---|---|
| `POST /v1/chat/completions` | ✅ 流式、JSON Schema `response_format`、`tools`/`tool_calls` |
| `POST /v1/completions`、`/v1/abort`、`/v1/prewarm` | ✅ |
| `POST /api/chat`、`POST /api/generate` | ✅ NDJSON 流式、`options`、`format`、`think`、tools |
| `GET /api/version`、`/api/tags`、`/api/ps` | ✅ |
| `POST/DELETE /api/show`、`/api/create`、`/api/copy`、`/api/delete` | ✅ |
| `/api/embed`、`/api/embeddings`、`/api/pull`、`/api/push` | 明确返回 501 |

工具调用使用模型自带 chat template（Qwen3 支持 tools），把输出里的 `<tool_call>` 解析成
标准 `tool_calls`；带 tools 的请求默认关闭 `<think>`。

## 架构映射

| DeepInfer 机制 | DeepInfer 源码 | Terrazzo |
|---|---|---|
| `step()` 循环 + 流式输出 | `engine/llm_engine.py` | `engine.py` |
| 带 token 预算的 chunked prefill | `csrc/engine/scheduler.hpp` | `scheduler.py` |
| FIFO 等待队列、LCFS 抢占 | `csrc/engine/scheduler.hpp` | `scheduler.py` |
| paged KV 块管理、引用计数 + free list | `csrc/engine/block_manager.hpp` | `blocks.py` |
| 共享前缀 KV store（上游闭源） | — | `paged_kv.py` |
| 模型执行 | CUDA graphs + 自定义内核 | `runner.py`（mlx-lm） |
| CUDA 内核（FlashMLA/DeepGEMM/DeepEP）、DP 投票、EP/PD | 集群专用 | 未移植 |

设计取舍与移植说明：[DESIGN.md](DESIGN.md)；运维：[docs/OPERATIONS.md](docs/OPERATIONS.md)。

## 仓库结构

```text
terrazzo/          引擎、调度器、块管理、paged KV、服务端、CLI、tools
bench/             可复跑的基准（head-to-head、agent workload、prefill……）
experiments/       为负结果保留的实验内核（不参与安装）
tests/             默认测试套件
docs/              快速开始、对比、运维
BENCHMARKS.md      所有测量数据，包括输掉的那些
```

## 测试与基准

```bash
python -m unittest discover -s tests                  # 核心套件（200 个测试，1 skipped）
python -m unittest discover -s experiments/tests -v   # 实验内核（31 个测试）
bash bench/head_to_head.sh                            # 与 Ollama 的 head-to-head
python bench/agent_workload.py --base ... --sessions 16
```

## 文档

- [docs/QUICKSTART.zh.md](docs/QUICKSTART.zh.md) · [English](docs/QUICKSTART.en.md) — 安装、API、工具调用、限制
- [docs/COMPARISON.zh.md](docs/COMPARISON.zh.md) · [English](docs/COMPARISON.en.md) — vs Ollama / vLLM、决策矩阵
- [BENCHMARKS.md](BENCHMARKS.md) — 全部基准数据与负结果
- [DESIGN.md](DESIGN.md) — 移植了什么、为什么
- [MAC-TURBO.md](MAC-TURBO.md) — MacTurbo：服务启动前预测可达 tok/s

## 限制（不假装支持）

- **不支持 Ollama registry / GGUF tag**：`pull` 下载 Hugging Face 的 MLX 快照；需要 GGUF 请继续用 Ollama。
- **不支持 `/v1/embeddings`**：只有生成模型。
- 单机、单模型常驻；不支持多卡/分布式。
- 负载下 agent TTFT p50 为 2.26 s（目标 ≤2 s）：剩余成本来自 mlx-lm 把共享前缀复制进每个序列；
  paged 共享方案评估过并失败（见 BENCHMARKS）。
- 强制工具调用（指定函数/`required`）是 best-effort：chat template 没有语法约束钩子。
- `experiments/` 里的 Metal 内核**比 MLX/SDPA 慢**，只作为负结果保留，不是产品路径。

## License

Apache-2.0。
