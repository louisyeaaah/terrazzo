# Terrazzo

[![CI](https://github.com/louisyeaaah/terrazzo/actions/workflows/ci.yml/badge.svg)](https://github.com/louisyeaaah/terrazzo/actions/workflows/ci.yml)

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
- **运维**：Bearer 鉴权（常量时间比较）、可选 TLS、全局限流、内存守卫（503 + `Retry-After`）、
  并发/排队上限、请求 deadline、worker 卡死 watchdog（`/stats.worker`）、请求取消、结构化
  `/stats`、可选 `--profile agent`。

## 性能实测（2026-10-27）

环境：Apple M1 Max（24 核 GPU / 64GB）、macOS 26.6.2、load average 2–4、`terrazzo` 0.3.0
（commit `8d00b0c`）、MLX 0.32.4 / mlx-lm 0.31.3。Ollama 0.34.4 + `qwen3:4b`
（GGUF Q4_K_M）；vLLM 0.32.0 + vllm-metal（`--max-model-len 4096
--gpu-memory-utilization 0.2`）。同一时间只跑一个服务、模型已热身、temperature 0。
token 数全部来自服务端计数（OpenAI 风格用 `usage`，Ollama 原生接口用 `eval_count`），
不用 SSE 块计数。

### 短提示 decode（128 token）

| 引擎 | decode tok/s | 相对 Ollama |
|---|---:|---:|
| terrazzo（MLX 4-bit） | **101.1** | **1.51×** |
| Ollama qwen3:4b（num_ctx 8192） | 67.0 | 1.00× |

### Agent 负载：2.6k 共享系统提示、2 轮 × 64 token

| 后端 / 配置 | 会话数 | 聚合 tok/s | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|---:|
| terrazzo（`--profile agent` + `/v1/prewarm`） | 1 | **59.4** | **0.29 s** | 0.37 s |
| terrazzo | 4 | **74.1** | **0.60 s** | 1.44 s |
| terrazzo | 16 | 88.7 | 2.56 s | 3.49 s |
| Ollama 0.34.4（原生 `/api/chat`） | 1 | 19.6 | 2.21 s | 4.21 s |
| Ollama | 4 | 32.6 | 5.16 s | 7.80 s |
| Ollama | 16 | 40.6 | 22.39 s | 25.04 s |
| vLLM-Metal | 1 | 43.5 | 0.19 s | 0.24 s |
| vLLM-Metal | 4 | 43.8 | 3.25 s | 5.78 s |
| vLLM-Metal | 16 | **122.1** | 1.57 s | 2.00 s |

vLLM 单会话取第二次（稳态）运行；首次运行包含 kernel 热身（15.1 tok/s / 2.95 s）。
4/16 会话均为稳态。

### 长系统提示：预注册是差异点

| 提示长度 | terrazzo 默认 cold TTFT | terrazzo 预注册 cold TTFT | Ollama cold TTFT | terrazzo warm-same | Ollama warm-same |
|---|---:|---:|---:|---:|---:|
| 2k | 4.37 s | **0.154 s** | 3.20 s | 0.153 s | **0.033 s** |
| 4k | 8.91 s | **0.242 s** | 6.95 s | 0.125 s | **0.044 s** |
| 8k | 19.87 s | **0.343 s** | 17.61 s | 0.081 s | **0.060 s** |

同样提示下的 decode（预注册）：**82.8 / 74.6 / 57.0 tok/s**（2k/4k/8k），
Ollama 为 45.2 / 43.6 / 38.0 tok/s。

### 长上下文 decode（terrazzo）

| 上下文 | decode tok/s | TTFT |
|---:|---:|---:|
| 1k | 87.2 | 1.95 s |
| 2k | 76.3 | 3.93 s |
| 4k | 70.7 | 8.36 s |
| 8k | 57.2 | 19.21 s |

### 内存

`/stats` 的 MLX 峰值：1 会话 **3.9 GB**、4 会话 **4.9 GB**、16 会话 **9.85 GB**；
进程 RSS 约 2.47 GB；2544-token 前缀快照约 375 MB。vLLM 的 KV 在其 engine 进程里，
1.1 GB RSS 不可直接比较。

### 实测后的诚实定位

- **vs Ollama**：decode 1.5×；4/16 会话 agent 吞吐 1.8–2.2×；2k–8k 共享提示的
  预注册首 token 快 10–50×。Ollama 仍在"不预注册的冷 prefill"和"纯 warm 前缀复用"上更快。
- **vs vLLM-Metal**：terrazzo 单流（59.4 vs 43.5 tok/s）和 4 会话（74.1 vs 43.8）更强；
  但 vLLM-Metal 的 16 会话批处理（122.1 vs 88.7）和 TTFT（1.57 vs 2.56 s）更好。
  高并发下我们没有吞吐优势。
- **总结**：terrazzo 的甜点是"单机、低/中并发、预注册长系统提示的 agent 流量"；
  不是 vLLM 的规模化吞吐替代品，也不是 Ollama 冷 prefill/纯前缀复用的对手。

复跑：`bench/compare_ollama.py`、`bench/h2h_long_prompts.py --only all`、
`bench/agent_workload.py`（完整命令、原始数据与 caveat 见 [BENCHMARKS.md](BENCHMARKS.md) §48）。
早先表格里的数字（如 2.08× decode）来自有负载的机器，已被本轮取代。

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

16 会话 × 2 轮 × 64 token、2.6k 共享提示：**TTFT p50 2.56 s / p95 3.49 s，聚合 88.7 tok/s**
（prewarm + `--profile agent`）；默认 8 槽配置下 p50 为 8–11 s。数字用服务端 usage 计 token、
多轮喂真实回答，见 [BENCHMARKS.md §48](BENCHMARKS.md)。

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
python -m unittest discover -s tests                  # 核心套件（226 个测试，1 skipped）
python -m unittest discover -s experiments/tests -v   # 实验内核（31 个测试）
python scripts/ci_core_tests.py                       # MLX 被 block 的纯 Python 子集
bash bench/head_to_head.sh                            # 与 Ollama 的 head-to-head
python bench/agent_workload.py --base ... --sessions 16   # --api ollama 用原生 eval_count
```

CI 两个 job：Linux 上 block 掉 MLX 跑纯 Python 套件（证明调度器/HTTP 层不依赖 Apple Silicon），
macOS 上安装固定版本 MLX 依赖跑同一套 + MLX 相关测试。benchmark 数字、依赖固定和测试矩阵见
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)。

## 文档

- [docs/QUICKSTART.zh.md](docs/QUICKSTART.zh.md) · [English](docs/QUICKSTART.en.md) — 安装、API、工具调用、限制
- [docs/COMPARISON.zh.md](docs/COMPARISON.zh.md) · [English](docs/COMPARISON.en.md) — vs Ollama / vLLM、决策矩阵
- [BENCHMARKS.md](BENCHMARKS.md) — 全部基准数据与负结果
- [DESIGN.md](DESIGN.md) — 移植了什么、为什么
- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) — 固定/验证过的 MLX 版本与 runtime patch 兼容契约
- [MAC-TURBO.md](MAC-TURBO.md) — MacTurbo：服务启动前预测可达 tok/s

## 限制（不假装支持）

- **不支持 Ollama registry / GGUF tag**：`pull` 下载 Hugging Face 的 MLX 快照；需要 GGUF 请继续用 Ollama。
- **不支持 `/v1/embeddings`**：只有生成模型。
- 单机、单模型常驻；不支持多卡/分布式。
- 负载下 agent TTFT p50 为 2.28 s（目标 ≤2 s）：剩余成本来自 mlx-lm 把共享前缀复制进每个序列；
  paged 共享方案评估过并失败（见 BENCHMARKS）。
- mlx-lm 固定在 `>=0.31.3,<0.32`（runtime patch 依赖其 cache 内部实现）；超出范围时引擎拒绝启动，
  除非显式设置 `TERRAZZO_SKIP_MLX_LM_CHECK=1`。
- 强制工具调用（指定函数/`required`）是 best-effort：chat template 没有语法约束钩子。
- `experiments/` 里的 Metal 内核**比 MLX/SDPA 慢**，只作为负结果保留，不是产品路径。
- GPU 调用无法抢占：心跳 watchdog 会把 runtime 标记为 unhealthy（`503`、`/stats.worker.hung`），
  `TERRAZZO_HANG_RESTART=1` 时退出交给 supervisor 重启；详见
  [docs/OPERATIONS.md](docs/OPERATIONS.md#known-gaps-residual-risks)。

## License

Apache-2.0。
