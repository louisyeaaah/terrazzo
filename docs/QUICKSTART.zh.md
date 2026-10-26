# terrazzo 快速开始（Ollama 风格）

> 目标：在 Mac 上像用 Ollama 一样拉模型、起服务、调 OpenAI API；面向 agent 的长系统提示场景。

## 安装

```bash
git clone https://github.com/louisyeaaah/terrazzo.git && cd terrazzo
python3 -m venv .venv && . .venv/bin/activate
pip install -e .
# 或：bash scripts/install.sh
```

要求：Apple Silicon、macOS 14+、Python 3.10+、MLX（`pip install -e .` 会带上 mlx / mlx-lm）。

## 模型管理（对应 ollama pull / list / show / rm）

```bash
# 从 Hugging Face 下载 MLX 快照（Ollama registry 的 tag 不支持，见"限制"）
terrazzo pull mlx-community/Qwen3-4B-4bit

terrazzo list                 # NAME / SIZE / MODIFIED / ARCH
terrazzo list --json
terrazzo show Qwen3-4B-4bit   # config 摘要 + 文件列表
terrazzo rm Qwen3-4B-4bit

# 已有本地目录：软链进模型库，不复制数据
terrazzo import /path/to/mlx-model --name qwen3-local

# 模型库位置：$TERRAZZO_MODELS_DIR，默认 ~/.terrazzo/models
```

## 起服务

```bash
terrazzo serve --model Qwen3-4B-4bit --port 8080
# 默认：OpenAI 兼容（/v1/chat/completions）、连续批处理、前缀缓存
```

### agent 场景（长共享系统提示 + 多会话）

```bash
# 1) 预注册系统提示：启动时就把前缀算好（首请求不吃 prefill）
terrazzo serve --model Qwen3-4B-4bit --profile agent \
    --system-file system_prompt.txt
# 注意：--system/--system-file 同时作为「默认系统提示」——请求没有 system 消息时自动注入

# 或对运行中的服务预热：
terrazzo prewarm --base http://127.0.0.1:8080 --system-file system_prompt.txt
# → {"status":"prewarmed","registered":true,"tokens":2544,"ms":5992}
```

实测（M1 Max，16 会话 × 2 轮 × 64 token，2.5k 共享前缀）：TTFT p50 **2.14 s** / p95 3.07 s、
聚合 **96 tok/s**；不预热/默认 8 槽时 p50 8–11 s。见 BENCHMARKS §44。

### 查看运行状态

```bash
terrazzo ps --base http://127.0.0.1:8080        # /api/ps
curl -s localhost:8080/api/tags                       # 模型列表
curl -s localhost:8080/stats                          # 步骤/前缀命中/内存
```

## OpenAI API

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "terrazzo",
  "messages": [{"role":"system","content":"你是严谨的助理"},
               {"role":"user","content":"用一句话解释 KV cache"}],
  "max_tokens": 128, "temperature": 0
}'
```

兼容：`stream`、`response_format={"type":"json_schema", ...}`（结构化输出）、
`/v1/models`、`/v1/abort`、`/stats`、`/health`。
鉴权/限流：`--api-key`、`--rate-limit`；内存上限：`--memory-cap-mb`（超限 503+Retry-After）。

## Tool calling（OpenAI tools）

```bash
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "terrazzo",
  "messages": [{"role":"user","content":"北京今天天气？"}],
  "tools": [{"type":"function","function":{
      "name":"get_weather","description":"查询天气",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                    "required":["city"]}}}],
  "max_tokens": 128
}'
```

- 返回 `finish_reason: "tool_calls"`，`message.tool_calls[*].function.arguments` 是 JSON 字符串；
  模型也可能直接回答（`finish_reason: "stop"`）。
- **多轮工具调用**：把 `assistant.tool_calls` 原样回传，并追加 `{"role":"tool","tool_call_id":...,"content":...}`
  即可；服务端会规范化 `content: null` 等 OpenAI 报文。
- **`tool_choice`**：`auto`（默认）/ `none`（不注入工具、不解析工具调用）/ 指定函数（只注入该工具并提示必须调用）
  / `required`（提示必须调用）；指定/required 是 best-effort（MLX chat template 无语法约束钩子）。
- 带 tools 的请求默认关闭 `<think>` 推理（`enable_thinking=false`），可用 `"thinking": true` 显式打开。
- 模型偶发重复输出同一调用会被去重；流式模式支持，但工具调用在最后一个 SSE chunk 一次性给出
  （粗粒度，协议兼容）。实测 Qwen3-4B 单轮/多轮/双工具/强制函数均正确。

## Ollama 兼容：CLI 与 HTTP API

**CLI（覆盖 Ollama 常用命令）**

```bash
terrazzo run base "用一句话解释 KV cache"   # 有 prompt 一次性输出；无 prompt 进入 chat
terrazzo run terse                         # 交互式 REPL（自动读取 Modelfile 的 SYSTEM/PARAMETER）

terrazzo create terse -f Modelfile         # 从 Modelfile 创建（FROM/SYSTEM/PARAMETER）
terrazzo show terse --modelfile            # 打印 Modelfile
terrazzo cp terse terse2                   # 复制模型（软链，不复制权重）
terrazzo stop terse                        # 单模型常驻，无需卸载（说明性提示）
terrazzo push terse                        # 明确不支持：没有 terrazzo registry
terrazzo version
```

Modelfile 支持：`FROM <store 名或路径>`、`SYSTEM`（含三引号多行）、`PARAMETER`
（`temperature/top_p/top_k/num_predict/stop/num_ctx/seed`）。**不支持** `TEMPLATE`/`ADAPTER`/`MESSAGE`
（会明确报错：MLX 模型自带 chat template）。

**HTTP API（Ollama 客户端可直接调用）**

| 接口 | 状态 |
|---|---|
| `POST /api/chat`、`POST /api/generate` | ✅ NDJSON 流式 + 非流式；`options`（temperature/top_p/top_k/num_predict/stop）、`format`（json/schema）、`think`（默认 false，true 时推理放 `message.thinking`）、tools |
| `GET /api/version`、`GET/POST /api/tags`、`GET /api/ps` | ✅ |
| `POST/DELETE /api/show`、`/api/create`、`/api/copy`、`/api/delete` | ✅ |
| `/api/embed`、`/api/embeddings` | ❌ 501（只支持生成模型） |
| `/api/pull`、`/api/push` | ❌ 501（无 registry，用 CLI `pull <hf-repo>`） |

```bash
curl localhost:8080/api/chat -d '{"model":"terse","stream":false,
  "messages":[{"role":"user","content":"你好"}]}'
curl localhost:8080/api/generate -d '{"model":"terse","prompt":"1+1=?","stream":false}'
```

## 与 Ollama 命令对照

| Ollama | terrazzo | 说明 |
|---|---|---|
| `ollama pull qwen3:4b` | `terrazzo pull mlx-community/Qwen3-4B-4bit` | HF 仓库 id，不是 Ollama tag |
| `ollama list` | `terrazzo list` | 本地模型库 |
| `ollama show` | `terrazzo show` | config/文件摘要 |
| `ollama rm` | `terrazzo rm` | 软链只删链接 |
| `ollama run` | `terrazzo chat` / `generate` | 支持 `--json-schema` |
| `ollama serve` | `terrazzo serve` | OpenAI 兼容 + `/api/tags`、`/api/ps` |
| `ollama ps` | `terrazzo ps` | 走 `/api/ps` |
| 无 | `prewarm` | 启动/运行中预注册长系统提示（agent 场景） |

## 限制（不假装支持）

- **不支持 Ollama registry / GGUF tag**：本运行时服务 MLX 快照（`models.py` 会明确报错并指向 HF id）。
  GGUF 模型请继续用 Ollama。
- 模型能力跟随 mlx-lm（生成类模型）；**tools/function calling 已支持**（依赖模型 chat template
  支持 tools，如 Qwen3），**/v1/embeddings 尚未支持**（路线图）。
- 单机单模型；无多卡/分布式。
- macOS 无 CUDA；Metal 上 decode 约 60–100 tok/s（4B 4-bit），性能上限见 `BENCHMARKS.md` 的负结果记录。
