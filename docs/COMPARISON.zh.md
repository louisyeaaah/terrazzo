# terrazzo 效果对比：Ollama 与 vLLM（中文）

> 数据日期：2026-10-27（当前代码 v0.9 + 前缀缓存修复）。
> 机器：Apple M1 Max（10 核 CPU / 24 核 GPU / 64 GB / 400 GB/s）。
> 模型：`mlx-community/Qwen3-4B-4bit`（MLX 4-bit，2.28 GB）；
> Ollama 用 `qwen3:4b`（GGUF Q4_K_M，2.5 GB）；vLLM 通过 **vllm-metal** 插件（MLX 后端，同一份 MLX 权重）。
> 测试状态：**171 tests OK（1 skipped，需额外 OlMoE 模型）**。

---

## 0. 一句话结论

- **vs Ollama**：decode **2.08×**、首请求长前缀 TTFT **5× 以上**、默认 4 并发 **+30%**；
  但**冷 prefill 和热同问前缀复用 Ollama 更快**，模型管理/生态 Ollama 完胜。**没有质的飞跃。**
- **vs vLLM（Mac 上的 vllm-metal）**：4B 尺寸的单请求与 4 并发我们更快（历史 v0.3 测量：86.8 vs 67.8；
  89.2 vs 63.3）；但 **warm 的 agent 长前缀场景 vLLM 更快**（v0.7 测量：67.9 vs 54.6 tok/s），
  且 vLLM 的功能/架构完整得多（paged prefill、多卡、生产特性）。我们是 Mac 上的参考实现，不是它的替代品。
- 我们的真实定位：**Mac 上的 agent 推理网关语义**（首请求零 prefill、前缀共享、背压/取消/可观测、结构化输出），
  不是"更快的 Ollama"，也不是"Mac 版 vLLM"。

---

## 1. 测试条件与公平性

- **一次只跑一个服务**；跑 Ollama 前先 `ollama stop` 释放显存，避免两个栈争 GPU/内存。
- **随机前置标记**：每个目标使用带随机 marker 的系统提示，保证冷启动没有跨目标/跨轮的前缀复用。
- 先热身模型（不与待测提示共享前缀）；温度 0；相同 `max_tokens`；报告值取中位数（decode 3 次）。
- 复跑脚本：`bench/head_to_head.sh`、`bench/h2h_long_prompts.py`、`bench/compare_ollama.py`、
  `bench/api_bench.py`；完整回归：`TERRAZZO_MAC_MODEL=<snapshot> python -m unittest discover -s tests`。
- **已知干扰**：2026-10-26 的 Ollama 长提示测量期间，本机有无关的 Ghidra headless 分析等外部负载
  （load average 6–29 波动），**Ollama 的绝对值被污染**；结构性结论（谁命中缓存、数量级差异）不受影响。
- **量化不同**：我们是 MLX 4-bit（2.28 GB），Ollama 是 GGUF Q4_K_M（2.5 GB），不是逐 bit 相同。
- **vLLM 数字是历史测量**（v0.3 / v0.7 阶段），当前 v0.9 代码之后**没有重跑**，仅作能力与量级参考。

---

## 2. 对 Ollama

### 2.1 decode（各自计数器，128 token，3 次中位数，短上下文）

| 服务 | decode tok/s | 比值 |
|---|---:|---:|
| Ollama `qwen3:4b`（num_ctx 8192） | 48.4 | 1.00× |
| **terrazzo（MLX 4-bit）** | **100.7** | **2.08×** |

来源：`bench/compare_ollama.py`（2026-10-26，空闲窗口）。注意这是**短上下文**结论；
4k 上下文我们约 60–72 tok/s，长上下文比例尚未做同条件对比。

### 2.2 ~440 token 系统提示：TTFT / 前缀复用 / 4 并发

| 目标 | 冷 TTFT | 热同问 | 热换问题 | 4 并发聚合 |
|---|---:|---:|---:|---:|
| terrazzo 默认 | 1.040 s | 0.076 s | 0.236 s | **54.1 tok/s** |
| terrazzo 预注册前缀（`--system-file`） | **0.169 s** | 0.186 s | 0.306 s | **58.1 tok/s** |
| Ollama | 0.811 s | **0.033 s** | **0.111 s** | 41.8 tok/s |

- **冷启动**：Ollama 的 llama.cpp prefill 更快（0.811 vs 1.040 s）。
- **热前缀复用**：Ollama 的 slot 缓存最快（0.033 / 0.111 s），我们 0.076–0.306 s。
- **首请求**：预注册前缀让我们第一次请求就命中（0.169 s，Ollama 0.811 s，**4.8×**）——Ollama 必须先跑完一次请求才有缓存。
- **并发**：我们默认配置领先约 30%；Ollama 默认 `num_parallel=1`，调到 4 时历史测得约 100 tok/s（配置问题，非能力问题）。

### 2.3 长系统提示 2k / 4k / 8k（2026-10-27 修复后）

| 目标 | 长度 | 冷 TTFT | 热同问 TTFT | 热换问题 TTFT |
|---|---:|---:|---:|---:|
| terrazzo 默认 | 2k / 4k / 8k | 4.77 / 12.22 / 23.71 s | 0.145 / 0.170 / 0.254 s | 0.217 / 0.304 / 0.443 s |
| terrazzo 预注册 | 2k / 4k / 8k | **0.202 / 0.297 / 0.406 s** | 0.155 / 0.247 / 0.245 s | 0.158 / 0.245 / 0.319 s |
| Ollama（最小 ctx） | 2k / 4k / 8k | 3.37 / 15.68 / 40.39 s* | 0.034 / 0.113 / 0.118 s* | 0.123 / 0.243 / 0.396 s* |

\* 外部负载污染，绝对值不可直接引用；结构性事实：**预注册前缀在 8k 仍成立（冷 0.406 s）**，
默认配置在本轮修复后也能在 8k 热请求上命中（0.254 s ≤ 0.5 s 目标）。

### 2.4 我们比 Ollama 强 / 弱在哪

**强**：decode 2×（MLX 4-bit 执行栈）、首请求长前缀（预注册）、块粒度前缀共享（部分命中）、
服务语义（503+Retry-After 背压、按 request_id 取消、内存硬上限、`/stats`、1 小时 soak 2161/2161 无错）、
JSON Schema 结构化输出。

**弱**：冷 prefill（llama.cpp 更快）、热前缀复用（slot 缓存更快）、模型管理/下载/生态、成熟度。

---

## 3. 对 vLLM（Mac 上的 vllm-metal）

### 3.1 vLLM 在 Mac 上怎么跑

官方 vLLM 的 macOS 支持是**实验性源码构建**（CPU 后端）；实际可用的是 **vllm-metal 插件**
（`vllm-project/vllm-metal`，Apache-2.0）：沿用上游 vLLM 的 engine / scheduler / **paged block manager**，
模型层用 mlx-lm，自带 **paged varlen Metal attention kernel**（含投机解码、GQA、M5 NAX prefill）。

### 3.2 单请求 / 4 并发（v0.3 阶段实测，2026-10-24）

| 引擎（同机同模型） | 单请求 decode | 4 并发聚合 | 冷 TTFT | 热 TTFT |
|---|---:|---:|---:|---:|
| **terrazzo v0.3 batch** | **86.8** | **89.2** | 0.155 s | 0.126 s |
| vLLM + vllm-metal | 67.8（冷）/ 68.3（热） | 63.3 | 1.358 s | **0.107 s** |

- 4B 尺寸我们单请求快约 25%，并发高约 40%；vLLM 热 TTFT 更快（0.107 vs 0.126 s）。
- vLLM 的价值不在这个尺寸的速度，而在完整调度器/API/分页内核/更大模型支持。

### 3.3 agent 长前缀多会话（v0.7 阶段实测，历史数据）

工作负载：2544-token 共享系统提示，4 会话并发、每会话 2 轮、每轮 64 token（`bench/agent_workload.py`）。

| 引擎 | 冷启动首轮 | warm 聚合吞吐 | warm TTFT p50 | 内存 |
|---|---:|---:|---:|---:|
| terrazzo（预注册前缀） | **59.8 tok/s / 0.66 s** | 54.6 tok/s | 0.82 s | MLX peak **4.8 GB** |
| vLLM-Metal（`--gpu-memory-utilization 0.2`） | 36.5 tok/s / 3.75 s | **67.9 tok/s** | **0.48 s** | KV pool 7.32 GiB + 权重 |
| vLLM-Metal（默认 0.92） | — | 37.2 tok/s | 3.65 s | KV pool **44.65 GiB** |

**结论（已修正过一版的错误比法）**：warm 之后 vLLM 比我们快 24%（67.9 vs 54.6）、TTFT 低 40%；
我们唯一的真实优势是**冷启动首轮**（0.66 s vs 3.75 s，5.5×）和**内存**（4.8 GB vs 7.3–44.7 GB）。
原因：我们的 batch 引擎会把共享前缀**复制进每个序列**的稠密 KVCache（4 × 375 MB），
vLLM 的 paged block 是真正的共享——这是已知限制，也是后续工作（paged runtime 已有，但当前是顺序执行）。

### 3.4 能力对照

| 能力 | vLLM（NVIDIA / vllm-metal） | terrazzo |
|---|---|---|
| 分页 KV（block table / ref count / CoW） | ✅ PagedAttention | ✅ block pool（v0.4，顺序路径） |
| paged **prefill** kernel | ✅ | ❌ prefill 走 gather + SDPA |
| paged **decode** kernel | ✅ | ✅ Metal split-K（q_len=1，长上下文默认启用） |
| 连续批处理 | ✅ | ✅（借 mlx-lm BatchGenerator） |
| 块级前缀共享 | ✅ 自动前缀缓存 | ✅ 注册式 + block 粒度 partial |
| 多卡 / TP / PP | ✅ | ❌ 单机 |
| 结构化输出 / tools / 指标 | ✅ | ✅ JSON Schema + tools/function calling；指标 ✅ |
| 生产成熟度 | 生产级 | 研究原型（~4k 行、171 测试） |

---

## 3.5 尝试过"集各家之长"：B 成功、A 负结果（2026-10-27）

**B：vLLM-Metal + 我们的启动预注册前缀（已实现，`scripts/vllm_prewarm_proxy.py`）**
启动时用同一段系统提示给 vLLM 的 prefix cache 发一次 warmup，再转发请求。agent 场景
（4 会话 × 2 轮 × 64 token、3257-token 共享提示）：

| 目标 | 聚合 tok/s | TTFT p50 |
|---|---:|---:|
| vLLM-Metal 原始（冷缓存） | 16.8 | 9.11 s |
| **vLLM-Metal + 预热代理** | **38.2** | **1.73 s** |
| terrazzo（预注册前缀） | 34.4 | **1.09 s** |

预热把 vLLM 首轮 TTFT 压了 **5.3×**；但组合后的真实格局是"吞吐 vLLM 高 11%、TTFT 我们更好"，
且单流 decode 从 ~100 掉到 ~47–68 tok/s。**机制成立，但不产生质变。**

**后续进展（2026-10-27 晚）**：调度器 round-robin chunked prefill + `--profile agent`（16 槽）
+ `POST /v1/prewarm` 预注册之后，同一 agent 场景 **TTFT p50 2.26 s / p95 2.87 s、聚合 95.8 tok/s**
（原基线 8.0–11.1 s / 36–44.8 tok/s）。另外已支持 OpenAI `tools`/function calling
（Qwen3 模板 + `<tool_call>` 解析，非流式与流式均可），`/v1/embeddings` 仍未支持。

**A：把 vLLM 式 paged 前缀共享搬进 batch（负结果，未集成）**
共享前缀打包 attention 的微基准（32 q heads / 8 kv heads、前缀 3400、B=4、bf16、36 层）：

| 方案 | 每层 | 相对 baseline | 正确性 |
|---|---:|---:|---|
| baseline：4×SDPA | 1.77 ms | 1.00× | exact |
| packed plain（无 mask） | 0.93–1.77 ms | 1.33–1.86× | ❌ 看到别人的 suffix |
| packed mask（exact） | 1.08 ms | 2.17×* | exact |
| **packed + 每步 concat（真实集成）** | 1.60 ms | **1.11×** | exact |
| MMA v2 + lse 合并 | 23.4 ms | 0.05× | exact |

\* 该 2.17× 的 baseline 含 concat；加上 packed 每步必需的 concat 后，真实收益只有 **1.11×**（整步约 +5%），
达不到 +20% 目标。mma_flash_attn v2 数值正确但该形状比 SDPA 慢 10×（此前"MMA 快 3–4×"是 causal
提前 break 的测量假象）。内存目标需要完整 paged 共享重写（数周），且我们的 paged decode 路径此前已
实测更慢。**结论：A 不做集成，现有上限是 +5–11% 的组合收益。**

## 4. 决策矩阵

| 你的场景 | 建议 |
|---|---|
| 只想在 Mac 上跑模型、要 GUI/模型管理/一键下载 | **Ollama** |
| Mac 上的 agent 服务：长共享系统提示、要首请求低延迟、要背压/取消/可观测/结构化输出 | **terrazzo**（预注册前缀 + 服务语义） |
| 生产、要 paged prefill、多卡、tools/指标、大模型 | **vLLM（Linux/NVIDIA）**；坚持 Mac 则试 vllm-metal |
| 研究 Mac 推理系统、改调度/内核 | terrazzo（小、可读、有完整测试与负结果记录） |

---

## 5. 局限（重要）

1. 只测了一个 4B 模型；27B 级、MoE、tools 工作负载未测。
2. vLLM 数字来自 v0.3/v0.7 历史测量，当前 v0.9 代码与 vllm-metal 新版本都未重跑。
3. Ollama 长提示的绝对数字受 2026-10-26 外部负载影响；热前缀复用与冷 prefill 需要空闲重测。
4. 量化不同（MLX 4-bit vs GGUF Q4_K_M），decode 差异部分来自量化格式与执行栈，而不是调度创新。
5. 服务语义（503/abort/内存上限/soak）有实现与测试，但未做与 vLLM/Ollama 的正式 SLO 对比。
6. 并发结论基于 4 会话；16 会话下我们的 TTFT 会明显变差（调度器冷 prefill 串行化，见 BENCHMARKS §38）。

---

## 6. 复现

```bash
# 完整测试（171 tests OK, 1 skipped）
TERRAZZO_MAC_MODEL=<Qwen3-4B-4bit snapshot> python -m unittest discover -s tests

# vs Ollama：短 decode（权威口径）
python bench/compare_ollama.py --mlx-model <snapshot> --ollama-model qwen3:4b --num-ctx 8192

# vs Ollama：TTFT/前缀/并发（需要先起 terrazzo server 与 ollama）
bash bench/head_to_head.sh

# vs Ollama：2k/4k/8k 长系统提示
python bench/h2h_long_prompts.py --mlx-model <snapshot> --lengths 2000,4000,8000
```
