# terrazzo v0.3 vs 现有 Mac 推理方案：实测对比

> 测量时间 2026-10-24，本机 Apple M1 Max（10 核 CPU / 24 核 GPU / 64 GB / 400 GB/s）。
> 模型：`mlx-community/Qwen3-4B-4bit`（MLX 4-bit，2.28 GB）；Ollama 用 `qwen3:4b`（GGUF Q4_K_M，2.5 GB）。
> 方法：**一次只跑一个服务**（避免 GPU 争用），每个服务先热身模型，再用**随机前缀**保证冷启动公平；温度 0，输出 64 token。
> 复现：`bench/run_clean_bench.sh` + `bench/api_bench.py`。

---

## 1. 一句话结论

| 对手 | 谁快 | 差多少 |
|---|---|---|
| **Ollama** | terrazzo | 单请求 decode **+38~48%**（90–96 vs 47–65 tok/s），4 并发 **2.1×**（96 vs 45 tok/s） |
| **mlx-lm server** | terrazzo | 端到端 **+20~25%**（decode 90 vs 75，短 prompt 96 vs 80），并发 **+13%**（96 vs 85） |
| **前缀复用** | 各有千秋 | 冷启动预注册：**0.21 s vs 1.03 s（5×）**；热请求复用：Ollama 0.10 s 最快、我们 0.14–0.20 s、mlx-lm 0.25 s |
| **冷 prefill** | Ollama | llama.cpp prefill 更快：**0.72 s vs 1.03–1.09 s**（同样的 ~440 token 前缀） |
| **内存共享前缀** | terrazzo（paged 模式独有） | 前缀 KV 只存一份；批处理模式会 copy-on-grow |
| **成熟度/生态** | Ollama / LM Studio / mlx-lm | 我们只有 3k 行、45 个测试、3 天历史，差距是数量级的 |

**一句话**：这一版在"单机长系统提示 + 中低并发"这个窄场景里打得过所有 Mac 方案，但它不是产品——是套在 mlx-lm 上的调度层/实验台。

---

## 2. 实测数据（同一台机器、同一模型、同一批 prompt）

### 2.1 长系统提示场景（~440 token 系统提示 + 用户问题，输出 64 token）

| 服务 | 冷启动 TTFT | 重复同问 TTFT | 换问题同系统 TTFT | decode（冷/热） | 短 prompt decode |
|---|---:|---:|---:|---:|---:|
| **terrazzo batch（默认）** | 1.034 s | **0.041 s** | 0.143 s | **90 / 90 tok/s** | **96 tok/s** |
| **terrazzo batch（启动预注册前缀）** | **0.209 s** | 0.196 s | 0.196 s | 90 | 96 tok/s |
| **terrazzo paged（默认）** | 1.092 s | 0.961 s | 0.963 s | 76 | 80 tok/s |
| **terrazzo paged（预注册前缀）** | 0.283 s | 0.154 s | **0.155 s** | 72 | 80 tok/s |
| mlx-lm server（官方，带 LRU prompt cache） | 1.087 s | 0.159 s | 0.254 s | 75 / 76 | 80 tok/s |
| Ollama（llama.cpp, Q4_K_M） | 0.715 s | **0.025 s** | **0.100 s** | 65 / 65 | 47 tok/s |

读法：
- **冷启动**：Ollama 的 prefill（llama.cpp）比 MLX 快约 30%；三家冷 TTFT 都在 0.7–1.1 s。
- **前缀复用**：三家的 prompt cache 都有效。Ollama 命中后最快（0.025 s），它是纯 KV 复用；我们 0.041 s 已经很接近。
- **换问题但同系统提示**（真实 agent 场景）：Ollama 0.100 s、我们 0.143 s、mlx-lm 0.254 s —— 我们的 block 粒度 partial 命中比 mlx-lm 的 nearest-cache 更细。
- **启动预注册前缀**：这是 v0.3 的独有效果——**第一次请求就不用重算前缀**（1.03 s → 0.21 s，5×）。Ollama/mlx-lm 只能等第一次请求自己跑完。
- **paged 模式的默认配置没有自动缓存**：不预注册时它每个请求都重算前缀（0.96 s），这是设计取舍（按引用共享需要显式注册）。

### 2.2 并发（4 个请求同时打，每个输出 64 token，共用长系统提示）

| 服务 | 总耗时 | 聚合吞吐 | 单请求吞吐 |
|---|---:|---:|---:|
| **terrazzo batch** | **2.70 s** | **94.9 tok/s** | 23.7 tok/s |
| terrazzo batch（预注册） | 2.69 s | 95.2 tok/s | 23.8 tok/s |
| mlx-lm server | 3.03 s | 84.5 tok/s | 21.1 tok/s |
| terrazzo paged（预注册） | 4.29 s | 59.7 tok/s | 14.9 tok/s |
| Ollama（默认 `num_parallel=1`） | 5.82 s | 44.0 tok/s | 11.0 tok/s |

- 批处理路径 ≈ mlx-lm +13%，是同一执行层上更薄的 server 层带来的。
- Ollama 默认串行（`num_parallel=1`），所以并发吞吐只有 44；把它调到 4 能到 ~100（上一轮测过），**但那是配置问题，不是能力问题**。
- paged 模式是顺序执行：并发下只有 60 tok/s，符合预期（用内存换并发）。

### 2.3 单请求原始吞吐（无系统提示，短 prompt）

| 服务 | decode |
|---|---:|
| terrazzo（batch） | **96 tok/s** |
| mlx-lm server | 80 tok/s |
| Ollama | 47 tok/s |

三家跑的是**同一架构的 4B 模型**，差异来自量化格式（MLX 4-bit 2.28 GB vs GGUF Q4_K_M 2.5 GB）与执行栈（MLX vs llama.cpp）。

---

## 3. 能力对比（不只是速度）

| 能力 | terrazzo v0.3 | mlx-lm（库/server） | Ollama | LM Studio（mlx-engine） | vLLM / SGLang |
|---|---|---|---|---|---|
| macOS / Apple Silicon | ✅ | ✅ | ✅ | ✅ | ❌ |
| 模型覆盖 | 跟随 mlx-lm（137 架构） | 137 | GGUF 生态 | MLX 生态 | 数百（NVIDIA） |
| 连续批处理 | ✅（BatchGenerator） | ✅ | ✅（需调 `num_parallel`） | ✅ | ✅ |
| 前缀缓存 | ✅ 注册式 + 自动；**block 粒度 partial**；批处理内共享 | ✅ LRU prompt cache（exact/nearest） | ✅ 同前缀复用（slot） | ✅ | ✅ |
| 分页 KV 记账 / LCFS 抢占 | ✅（逻辑层） | ❌ | ❌ | ❌ | ✅ |
| 共享前缀"只存一份 KV" | ✅ 仅 paged 模式（顺序） | ❌ | ❌ | ❌ | ✅（vLLM prefix caching） |
| OpenAI 兼容 API + 流式 | ✅ 子集 | ✅ | ✅ | ✅ | ✅ |
| 结构化输出 / tools | ❌ | ✅（outlines 类、logits processors） | ✅ | ✅ | ✅ |
| 模型管理 / 下载 UI | ❌ | 基本 | ✅✅（`ollama pull`） | ✅✅ | ❌ |
| 量化 | 跟随 mlx-lm | 4/6/8-bit、KV 量化 | GGUF 各种 | MLX | FP8/INT4 |
| 多机 / EP / PD 分离 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 生产成熟度 | 研究原型（3k 行、45 测试） | 生产级（7.4k★） | 极成熟 | 商业产品 | 生产级 |

---

## 4. 为什么数字是这样（机制层面的解释）

1. **decode 吞吐 ≈ 同一水平**：v0.3 的 batch 路径就是 mlx-lm 的 `BatchGenerator`，模型、kernel、量化完全相同。raw decode 差异主要来自 server 层：我们的 per-token SSE + 直接 `tokenizer.decode([id])` 比 mlx-lm server 的 response-generator 管线薄，所以端到端快 20% 左右。
2. **Ollama 更快的是 prefill、更慢的是 decode**：llama.cpp 的 prefill kernel 在这颗 GPU 上更成熟；而它的 decode 受 GGUF 量化/内存布局影响，比 MLX 4-bit 慢约 30%。
3. **前缀缓存三家都有**，差别在"命中颗粒度"与"什么时候开始命中"：
   - Ollama/llama.cpp：按 KV slot 的公共前缀命中，命中后极快（0.025 s）。
   - mlx-lm：`LRUPromptCache.fetch_nearest_cache` 按 token 前缀找最近的缓存，但需要至少一次完整请求把缓存"喂"出来。
   - v0.3：block 哈希链 → **可以只共享前 N 个块**；`--system-file` 让服务启动时就注册，所以**第一次请求**就命中。
4. **批处理内共享前缀是 v0.3 特有的组合**：mlx-lm 的批处理器只认稠密 per-layer cache，所以我们的做法是把共享前缀数组"预置"进每个序列的 cache（copy-on-grow）。省钱在 prefill/TTFT，不省显存——要省显存只能走 paged（顺序）。
5. **"效果"层面没有神奇之处**：同一模型、greedy 解码，输出分布相同；共享前缀只是不改 KV 的别名。数值差异（logit 0.1–0.5）来自 chunk 边界 kernel shape，top-2 打平时会翻 token——这一点已写进 DESIGN.md。

---

## 5. 该用哪个？（决策表）

| 你的场景 | 建议 | 原因 |
|---|---|---|
| 只想在 Mac 上跑模型、要 GUI/模型管理 | **Ollama / LM Studio** | 生态、稳定性、量化选项；冷 prefill 还更快 |
| 要在 Mac 上做推理系统研究、改调度 | **mlx-lm + 本项目的调度层** | 137 架构、成熟执行层；本项目给 DeepInfer 式调度与可观测性 |
| 长系统提示（agent/工具 schema） + 中低并发 | **terrazzo batch（预注册前缀）** | 首请求 TTFT 5×，4 并发吞吐 2.1× Ollama |
| 显存紧张、前缀很长、并发不高 | **terrazzo paged** | 前缀 KV 只存一份，TTFT 0.15 s |
| 要生产、要 tools/结构化输出/多模型 | Ollama/LM Studio，或直接调 API | 我们功能子集，别拿去当生产 |
| 要极致吞吐/多卡 | **vLLM/SGLang on NVIDIA** | Mac 上没有可比对象；DeepInfer 本体也只在这类硬件上 |

---

## 6. 测不到 / 不确定的部分（诚实交代）

1. **LM Studio（mlx-engine）没测**：未安装；它基于 mlx-lm，预期与 mlx-lm server 接近，但没有数据。
2. **vLLM/SGLang 无法在 Mac 上跑**，与它们比较只能在 NVIDIA 上做——本报告不涉及。
3. **DeepInfer 本体仍然跑不了**（510 GB 权重 + SM100），所以这里是"DeepInfer 风格引擎" vs 市面 Mac 方案，不是 DeepInfer vs Ollama。
4. **单次测量**：每个 case 只跑一轮（除短 prompt 取 2 次中位数），并发数字可能有 ±10% 波动；没有做置信区间。
5. **量化不完全对齐**：Ollama 是 GGUF Q4_K_M（2.5 GB），MLX 是 4-bit group-64（2.28 GB），体积差 9%，会自然让 MLX 侧快一点。
6. **paged 模式没有自动缓存**：不预注册前缀时它不共享，这是设计上的取舍，不是 bug。

---

## 7. 下一步最值得做的两件事

1. **paged attention kernel（MLX gather）**：把"前缀只存一份"和"连续批处理"合到一条路径上——那样 batch 模式就不必 copy-on-grow，paged 模式也能并发。这是唯一能同时赢下内存和吞吐的改动。
2. **server 层补齐**：结构化输出（logits processor）、tools/function-call 协议、embeddings；不然永远只是实验台。

---

## 8. v0.4 分页 KV（block pool + Metal kernel）与前三版的对比

同一台 M1 Max、同一模型（Qwen3-4B-4bit）、同一随机前缀（~440 token 系统提示）、
一次只起一个服务、4 并发、64 token 输出：

| 运行时 | 冷启动 TTFT | 同问复热 | 换问同系统 | 单请求 decode | 4 并发聚合 |
|---|---:|---:|---:|---:|---:|
| **v0.3 batch**（连续批处理 + 稠密前缀） | 0.155 s | 0.126 s | 0.198 s | **86.8 tok/s** | **89.2 tok/s** |
| v0.3 paged（稠密前缀，顺序） | **0.114 s** | 0.093 s | **0.155 s** | 78.5 tok/s | 66.3 tok/s |
| v0.4 pool（真 block pool + gather） | 0.176 s | **0.080 s** | 0.247 s | 52.1 tok/s | 42.6 tok/s |
| **v0.4 pool + Metal split-K** | 0.168 s | **0.079 s** | 0.247 s | **69.7 tok/s** | 55.8 tok/s |

内存（400-token 前缀 = 25 个 16-token 块）：

| | 前缀 KV | 3 个请求之后 |
|---|---:|---|
| v0.3 稠密前缀（batch） | 59 MB（一个连续数组） | 每个活跃序列 copy-on-grow 一份 |
| v0.4 block pool | 59 MB（25 块，引用计数） | **仍是 25 块**（`used_blocks=[25,25,25]`） |

Metal kernel 的两次提速：
- 单层/36 层隔离测试（L=450）：gather 5.47 ms/step → Metal **1.78 ms/step（3.1×）**；
- 服务端端到端：pool 52.1 → **69.7 tok/s（+34%）**。

结论：
1. **v0.4 的收益是「内存共享是真的」**，不是速度：真共享 + 块级引用计数 + CoW 已经可证明，
   但吞吐仍落后 v0.3 batch 约 20%（batch 用的是 mlx-lm 调优过的稠密 cache + 连续批处理）。
2. **Metal kernel 把 gather 的代价从 −38% 拉回到 −20%**，隔离测试里 3× 领先 gather；
   端到端没完全兑现，是因为每步 36 次 kernel 启动 + 手写 kernel vs MLX 的 SDPA。
3. **TTFT 上 pool 最好（0.079–0.080 s）**：decode 不再为前缀做 gather，热请求只付后缀 prefill。
4. 并发上 pool 模式仍是顺序执行（42.6/55.8），要同时拿到「共享前缀 + 批处理」还需要
   把 pool 接进连续批处理（v0.5 的工作）。

---

## 9. 和 vLLM、Ollama 的关系：一个能实测，一个只能讲能力

### 9.1 vLLM：**能跑在 Apple Silicon 上**（2026-10-24 修正并实测）

**修正**：本页早期版本称「vLLM 在 Mac 上不存在」。这个结论只基于 PyPI 轮子，是错的。

实际情况（2026-10-24 在本机核实）：

1. **官方 vLLM 支持 Apple Silicon（实验性）**：`docs/getting_started/installation/cpu.apple.inc.md`
   写明「vLLM has experimental support for macOS with Apple Silicon」，需从源码构建
   （`VLLM_TARGET_DEVICE` 自动设为 `cpu`），PyPI 确实没有 macOS 轮子。
2. **vllm-metal 插件**（`vllm-project/vllm-metal`，1.9k★，Apache-2.0，2026-10 仍活跃）
   提供 macOS arm64 预编译轮子：上游 vLLM 的 engine/scheduler/**paged block manager**
   + mlx_lm 模型层 + 自己的 **paged varlen Metal attention kernel**（还有投机解码、GQA、M5 NAX prefill）。

本机实测安装运行（M1 Max，Qwen3-4B-4bit，同一个 MLX 权重）：

```bash
# Python 3.12 + uv，装完约 1.6 GB
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
vllm serve <Qwen3-4B-4bit 目录> --port 8401 --max-model-len 4096
# → Application startup complete，/health 200
```

同一套 benchmark 脚本（`bench/api_bench.py`，同提示词、64 token、4 并发）：

| 引擎（同机同模型） | 单请求 decode | 4 并发聚合 | 冷 TTFT | 热 TTFT |
|---|---:|---:|---:|---:|
| **terrazzo v0.3 batch** | **86.8** | **89.2** | 0.155 s | 0.126 s |
| mlx-lm server | 75–80 | 84.5 | 1.09 s | 0.159 s |
| **vLLM + vllm-metal** | 67.8（冷）/ 68.3（热） | **63.3** | 1.358 s | **0.107 s** |
| Ollama（qwen3:4b） | 47–65 | 44（默认 `num_parallel=1`） | 0.715 s | 0.025 s |

读法：
- **vLLM-Metal 能跑，但在这台机器的 4B 模型上比我们的 batch 引擎慢 ~25%**（67 vs 87），
  并发聚合也低（63 vs 89）；热 TTFT 反而是它最快（0.107 s）。
- 它的价值不在这个尺寸的速度，而在**完整的 vLLM 调度器/API/分页内核/投机解码**，
  以及 27B 级模型支持（Qwen3.8-27B 等）——这些是我们要追的能力，不是速度。
- 结论修正：不存在「vLLM 上不了 Mac」这回事；存在的是「vLLM 在 Mac 上性能一般、但功能完整」。

能力对照（同为「分页 KV + 连续批处理 + 前缀缓存」这一族）：

| 能力 | vLLM（NVIDIA/Linux） | terrazzo v0.4 |
|---|---|---|
| 分页 KV（block table） | ✅ PagedAttention | ✅ block pool + ref count + CoW（v0.4） |
| paged **prefill** kernel | ✅ | ❌ prefill 走 gather + SDPA |
| paged **decode** kernel | ✅ | ✅ Metal split-K（q_len=1） |
| 连续批处理 | ✅ | ✅（借 mlx-lm 的 BatchGenerator） |
| 块级前缀共享 | ✅ 自动前缀缓存 | ✅ 需注册前缀，按 block 引用 |
| 多卡 / TP / PP | ✅ | ❌ 单机 |
| 结构化输出 / tools / 指标 | ✅ | ❌（仅 OpenAI 兼容基础子集） |
| 成熟度 | 生产级 | 研究原型（~4k 行 + 75 测试） |

**结论**：vLLM 是这类引擎的成熟形态，我们是它在 Mac 上的一次可运行的「缩小重实现」；
它的核心机制（分页 + 批处理 + 前缀缓存）我们已经有了骨架，缺的是 paged prefill、
生产特性与多卡。

### 9.2 Ollama：可以同机实测，且我们更快

同机、同模型族、一次只起一个服务、4 并发、64 token（Ollama 是 GGUF Q4_K_M 2.5GB，
我们/MLX 是 4-bit 2.28GB，量化不完全一致）：

| 指标 | terrazzo v0.3 batch | terrazzo v0.4 pool+Metal | mlx-lm server | Ollama |
|---|---:|---:|---:|---:|
| 单请求 decode | **86.8–96 tok/s** | 69.7 | 75–80 | 47–65 |
| 4 并发聚合 | **89–95 tok/s** | 55.8 | 84.5 | 44（默认 `num_parallel=1`；调到 4 约 100） |
| 冷 TTFT（~440 token 系统提示） | 1.03 s | **0.17 s（预注册前缀）** | 1.09 s | **0.72 s**（prefill 更快） |
| 热 TTFT | 0.04–0.13 s | 0.079 s | 0.159 s | **0.025 s** |
| 换问同系统前缀 | 0.143–0.198 s | 0.155–0.247 s | 0.254 s | **0.100 s** |
| 前缀 KV 复用粒度 | block 级（共享 + CoW） | block 级（引用计数，真一份） | token 前缀 LRU | KV slot 复用 |

读法：
- **decode 吞吐我们赢**（比 Ollama 快 30–50%），因为 MLX 4-bit kernel 在这颗 GPU 上比 llama.cpp 的 GGUF 快；
- **prefill / 热命中 Ollama 赢**（llama.cpp 的 prefill 更快、KV slot 复用更直接）；
- **并发 Ollama 默认配置吃亏**（`num_parallel=1`），调到 4 能到约 100 tok/s，但那是拿单请求延迟换的；
- Ollama 强在模型管理、GGUF 生态、稳定性——这些是工程产品力，不是本项目的目标。

---

## 10. v0.6：MoE 校准、KV 量化、投机解码与 200 tok/s 的真实条件

详见 [MAC-TURBO.md](MAC-TURBO.md)。三方实测（M1 Max，Qwen3-4B / OLMoE-1B-7B）：

### 10.1 MoE 校准（OLMoE-1B-7B：激活 1.18B / 总 6.9B）

| 指标 | 数值 |
|---|---:|
| 两项式预测 | 158.8 tok/s |
| 实测单流（mlx-lm 路径） | **164.9–167.4 tok/s** |
| terrazzo batch 引擎单流 | 132.9–138.3 tok/s |
| **4 并发聚合** | **284.4–285.1 tok/s** |

结论：MoE 的「激活参数」确实决定速度；**小激活 MoE 聚合吞吐可超 200**，单流接近 200。

### 10.2 KV 量化（Qwen3-4B，48 token 输出；空闲窗口测得，复跑 `experiments/bench/kv_quant.py`）

| KV 精度 | decode tok/s | KV 缓存 | 与 fp16 输出 |
|---|---:|---:|---|
| fp16 | 78.6 | 37.7 MB | — |
| 8-bit | 64.6（−18%） | 20.1 MB | 首 token 一致 |
| 4-bit | 63.6（−19%） | 10.6 MB | **首 token 即不同** |

### 10.3 投机解码（4B + 0.6B draft）

基线 76.1 tok/s；投机 k=3 → 36.2、k=5 → 27.4（**负收益**，输出与基线一致）。
原因：验证边际成本 7.3 ms/token，draft 0.6B 要 5.08 ms/token，不够便宜。

### 10.4 200 tok/s 的真实条件（校准后的两项式模型）

`t_step ≈ bytes_per_token × 1000 / (0.664 × 峰值带宽) + 3.8 ms`

| 芯片 | 单流 200 tok/s 的激活参数预算（4-bit） |
|---|---:|
| M1 Max / M3 Max | 0.57B |
| M4 Max | 0.77B |
| M3 / M4 Ultra | 1.13B |

**dense 7B–27B 单流 200 tok/s 不可达；MoE 激活 3B 单流也不可达（Ultra 143）；只有小激活模型或批处理聚合可达。**

---

## 11. v0.7 首测：agent 长前缀多会话（MacInfer 靶场）

工作负载：**一个 2544-token 的共享系统提示，4 个会话并发，每会话 2 轮，每轮 64 token**
（`bench/agent_workload.py`，OpenAI streaming，同机同模型 Qwen3-4B-4bit）。

| 引擎（同机同模型） | 聚合吞吐 | TTFT p50 | TTFT p95 | 内存 |
|---|---:|---:|---:|---:|
| **terrazzo（预注册前缀 + 连续批处理）** | **59.8 tok/s** | **0.66 s** | **1.01 s** | MLX active **2.6 GB** / peak 4.8 GB；前缀 KV 375 MB 共享一份 |
| vLLM-Metal（默认 `gpu-memory-utilization=0.92`） | 37.2 tok/s | 3.65 s | 6.74 s | KV pool **44.65 GiB**（日志实测） |
| vLLM-Metal（`--gpu-memory-utilization 0.2`） | 38.0 tok/s | 3.66 s | 6.76 s | KV pool **7.32 GiB** + 权重 |

**含 warmup 的重复测量（更公平，每栈 3 次，取后两次的中位数）**：

| 引擎 | 冷启动首轮 | **warm 聚合吞吐** | **warm TTFT p50** | 内存 |
|---|---:|---:|---:|---:|
| terrazzo（预注册前缀） | **59.8 tok/s / 0.66 s** | 54.6 tok/s | 0.82 s | MLX peak 4.8 GB |
| vLLM-Metal（0.2） | 36.5 tok/s / 3.75 s | **67.9 tok/s** | **0.48 s** | KV pool 7.32 GiB + 权重 |
| mlx-lm server | 15.0 tok/s / 13.3 s | — | — | RSS 2.6 GB |
| Ollama（默认 num_parallel=1） | 24.1 tok/s / 6.3 s | — | — | — |

**修正结论（重要）**：上一版把「冷 vLLM」和「热我们」放在一起比，是错的。
warm 之后 **vLLM-Metal 比我们快 24%（68 vs 55 tok/s）、TTFT 低 40%（0.48 vs 0.82 s）**；
我们唯一的真实优势是**冷启动第一轮**：启动时就把前缀注册好，首个用户请求 TTFT 0.66 s，
而 vLLM 首轮要 3.75 s。内存上我们仍然领先（4.8 GB vs 7.3–44.7 GB），因为 vLLM 预分配 KV pool。

**为什么 warm 后我们更慢（已定位）**：batch 引擎的 `_seed_caches` 在首个 growth step 会把
共享前缀**复制进每个序列**的稠密 KVCache（4 会话 × 375 MB），既慢又吃内存；
vLLM 的 paged block 是真共享。下一轮的核心工作就是让 batch 路径也用 v0.4 的 block pool（真共享），
而不是复制。

原始单次测量（保留）：
1. **吞吐 1.57×、TTFT 5.5× 优于 vLLM-Metal**：我们把系统提示在启动时注册成共享前缀，
   请求只 prefill 后缀；vLLM 虽然有 prefix cache（日志 86.7% 命中率），
   但首轮 4 个 2544-token prefill 在 `max_num_batched_tokens=2048` 下被串行化。
2. **内存**：我们按需分配（前缀 375 MB + 活跃块），vLLM 预分配 KV pool（默认 44.65 GiB，
   降到 0.2 也有 7.32 GiB）。这是长前缀多会话场景的结构性差异。
3. 公平性说明：vLLM 的默认内存配置不是为小模型调的；但即使调到 0.2，其 KV pool 仍是我们的 2.8×。
   后续要跑 mlx-lm / Ollama 两条基线，并重复 3 次取中位数。

复跑：

```bash
# 我们的
terrazzo serve --model <dir> --port 8501 --num-gpu-blocks 4096 --system-file /tmp/agent_sys.txt
python bench/agent_workload.py --base http://127.0.0.1:8501 --stats http://127.0.0.1:8501/stats     --model terrazzo --label ours --sessions 4 --turns 2 --system-file /tmp/agent_sys.txt

# vLLM-Metal
vllm serve <dir> --port 8502 --max-model-len 8192 --gpu-memory-utilization 0.2
python bench/agent_workload.py --base http://127.0.0.1:8502 --model qwen3-4b-mlx --label vllm \
    --sessions 4 --turns 2 --system-file /tmp/agent_sys.txt
```

### 11.1 Round 2：稳定复测与瓶颈定位

每栈 3 次、取中位数（4 会话 × 2 轮 × 64 token，2544-token 共享前缀）：

| 引擎 | 聚合吞吐（中位数） | TTFT p50 | 内存 |
|---|---:|---:|---:|
| terrazzo（预注册前缀） | 62.4 tok/s | 0.70 s | MLX peak 4.8 GB |
| vLLM-Metal（`gpu-memory-utilization=0.2`） | 61.0 tok/s | **0.49 s** | KV pool 7.32 GiB + 权重（≈9.6 GB） |

**结论**：warm 之后两者吞吐打平（62 vs 61），vLLM TTFT 更好（0.49 vs 0.70）；
内存我们仍是 0.5×。Round 1 里「54.6 vs 67.9」的差距是测量污染（当时机器上还有别的服务），已修正。

**瓶颈定位（engine 级测量，非 HTTP）**：
- 4 序列、2500-token 上下文时，engine 稳态 step **59.6 ms**（p90 79.3 ms）；
- 单序列 20.4 ms/step，2 序列 22.4 ms，4 序列 39–60 ms —— 权重读取已摊薄，
  增量几乎全部来自**长上下文 decode attention**（dense SDPA，q_len=1）；
- 粗算：每步每序列读 KV 375 MB ≈ 0.94 ms，但实测 ~8 ms —— 稠密 SDPA 在 q_len=1、
  L≈2600 时效率很低（没有 flash-decode 路径）。

**下一个杠杆**：把 v0.4 的 Metal paged decode kernel 从单序列扩展到**批内多序列**
（每序列一张 block table，一个 kernel launch 处理整个 batch）。微基准里该 kernel 在
L=1024 时对 gather 是 1.8×、L=4096 时 2.8×；若同比例作用于长上下文 decode，
4 序列 step 可从 ~60 ms 降到 ~40 ms，对应聚合吞吐 ~90 tok/s（≈1.4× vLLM），
才有机会过 1.2× 门限。

### 11.2 Round 3：批量 Metal decode kernel（已实现，但只到 parity）

实现 `dense_metal.py` + `batch_attn_patch.py`：把 mlx-lm 在 **q_len=1 的 batch decode**
时调用的 `mx.fast.scaled_dot_product_attention` 替换成自写的 split-K simdgroup kernel
（支持左填充、GQA、B≤8，L≤2600 实测正确，maxdiff ≤ 1.6e-3，7 个测试全绿）。

**孤立微基准（B=4，Hq=32，Hk=8，D=128，fp16）**：

| L | SDPA | Metal kernel | 加速 |
|---:|---:|---:|---:|
| 256 | 0.45 ms | 0.34 ms | 1.35× |
| 1024 | 0.70 ms | 0.46 ms | 1.53× |
| 2600 | 0.91 ms | 0.51 ms | **1.79×** |

**但端到端没有兑现**：engine 里 4 序列 200-token 生成的稳态 step：

| 路径 | 中位数 | 最小 |
|---|---:|---:|
| SDPA（原路径） | 57.6 ms | 39.0 ms |
| Metal kernel | 64.8 ms | 39.6 ms |

**原因（新测到的硬地板）**：直接跑模型本体（`BatchKVCache`，B=4，L=2600）的 decode step
**中位 64.9 ms / 最小 60.6 ms**——我们的 engine（39–60 ms）已经贴着 MLX 执行层的下限。
注意力只占 0.9 ms × 36 层 ≈ 33 ms 的「名义」时间，但 MLX 每层 ~7 次 dispatch（36 层 ≈ 250 次
kernel launch）的固定开销占了大头，换注意力算子救不回来。

**结论**：1.2× 门限靠「换注意力 kernel」达不到。下一个真正的杠杆是**减少 dispatch 数量**：
把 KV 预分配到固定 `max_len`（形状静态化）+ `mx.compile` 整段 decode，把 250 次 launch 合并；
这也是 vLLM-Metal 与 mlx-lm 都没做的事（两者都直接用 mlx_lm 的逐层 Python 路径）。
风险：静态形状需要在会话层面预留 KV（内存换速度），但我们在内存上仍有 2× 余量（4.8 GB vs 9.6 GB）。

### 11.3 Round 4：8 会话扩展、注意力消融、mx.compile 探针

**8 会话（2 轮 × 64 token，2544-token 共享前缀）**：

| 引擎 | 聚合吞吐 | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|
| terrazzo | **52.5 tok/s** | **1.16 s** | **2.00 s** |
| vLLM-Metal（0.25） | 48.6 tok/s | 3.96 s | 6.83 s |

8 会话时我们领先 8%，TTFT 领先 3.4×；4 会话时是持平（62 vs 61）。并发越高我们相对越好。

**step 时间分解（模型本体，B=4，L=2600）**：

| 配置 | step 中位数 |
|---|---:|
| 原样（SDPA） | 66.5 ms |
| 注意力替换为 zeros（消融） | 39.2 ms |

→ 注意力真实成本 ≈ **27 ms/步**（其余 39 ms = 权重读取 + 4 token 的 matmul + MLP）。

**mx.compile 探针（否决「dispatch 开销」假设）**：固定形状 [4,8] 的整模型前向，
eager 81.2 ms vs `mx.compile` 80.4 ms —— **没有收益**。MLX 的异步求值已经把逐层调用
组织成图，真正的成本是计算/访存本身，不是 launch 次数。上一轮提出的「静态形状 + compile」
路线被这组数据否决，不再投入。

**结论**：要过 1.2× 门限，只剩「把 27ms 的注意力打到 ~12ms」这一条路：需要让自写 kernel
真正快过 MLX SDPA（当前只是持平）。Round 5 的方向：`half4` 向量化加载、寄存器内半精度累加、
每迭代处理 2 个 token 提升 ILP。

### 11.4 Round 5：并发扩展曲线与门限结论

同一负载（2544-token 共享前缀，2 轮 × 64 token），不同并发下的聚合吞吐：

| 并发 | terrazzo | vLLM-Metal | 比值 |
|---:|---:|---:|---:|
| 4 | 62.4 tok/s | 61.0 tok/s | 1.02× |
| 8 | **52.5 tok/s** | 48.6 tok/s | **1.08×** |
| 16 | 50.6–51.5 tok/s | **79.1 tok/s** | **0.64×** |

- 我们最优区间在 8 会话（1.08×），16 会话被反超：vLLM 的 paged attention 在长上下文、大 batch 下
  伸缩性明显更好；我们的 BatchKVCache 是稠密左填充，batch 越大注意力成本越高。
- 把 `max-num-seqs` 提到 32、blocks 提到 16384 后，16 会话从 42.5 → 51.5（TTFT 13.4s → 2.26s），
  但仍低于 vLLM。
- 向量化 `half4` 的 batch kernel 在 B=16、L=2600 的实模型里仍是持平（min 206 vs 184ms），
  说明 MLX SDPA（GQA 广播、无 mask 物化）已经是高效实现；孤立微基准里的 2–3× 主要来自
  我构造的 repeat+mask 基线，不是公平对比。

**门限结论（4 个 round 的一致结果）**：在同机同模型同负载下，聚合吞吐无法达到最佳方案的 1.2×
（最好 1.08× @ 8 并发），内存 0.5× 达标。原因是执行层地板：step 时间由 MLX 逐层执行 +
稠密 BatchKVCache 决定，自写注意力 kernel 打不过 MLX SDPA。要继续提升需要自写 4-bit 量化
matmul 与 paged attention（相当于重写 MLX 执行层），已超出本 goal 的范围。

### 11.5 Round 6（回归主线）：前缀缓存内存上限 + 自动前缀的诚实结论

在 v0.3/v0.4 线上做的两处修复：

1. **前缀快照上限**：`EngineConfig.prefix_cache_max_tokens = 2048`。此前每个长 prompt 快照
   按全 prompt 长度存（2544-token ≈ 375 MB），两个条目就撞到 512 MB 预算并触发淘汰；
   现在单个快照固定 ≤ 2048 token（实测 132 MB），内存可控。
2. **admission 移到 worker 线程**：请求不再从 HTTP 线程直接 `add_request`（MLX 流是线程局部的），
   改为 worker 线程统一准入，避免线程亲和性隐患。

**自动前缀（experimental，默认关）**：即使做了上述修复，自动注册的前缀在实测中
**只命中约 896 / 2048 token**（block-hash 链在登记条目里只覆盖了一部分块），
4 会话 2 轮的 agent 负载 TTFT 仍是 17–21 s、吞吐 6.9–8.5 tok/s，**比不启用更慢**。
因此默认关闭，用环境变量 `TERRAZZO_AUTO_PREFIX=1` 显式开启。

**推荐部署姿势（实测最好）**：

```bash
terrazzo serve --model <dir> --system-file system.txt \
    --num-gpu-blocks 8192 --max-num-seqs 16
```

即启动时预注册系统提示（TTFT 0.66–0.70 s），而不是依赖自动发现。

下一轮要修的是 `PagedPrefixCache.store/lookup` 的块覆盖问题（为什么 2048 只命中 896），
修好之后自动前缀才可能成为默认路径。

### 11.6 自动前缀：真 bug 找到并修了（engine 已好，server 并发仍待修）

**根因**：`Engine._maybe_store_prefix` 把「完整 prompt 的 token 列表」和「截断到 `prefix_cache_max_tokens` 的 KV 快照」一起传给 `PagedPrefixCache.store`。
`store` 自己按 token 列表算块哈希（2544 token = 159 块），而快照只有 2048 token——条目登记了 159 块，
但只有 128 块的 KV；后续 `_seed_caches` 按 2544 去切片，拿到 2048 的数组，
`BatchKVCache.merge` 直接抛 `broadcast_shapes (1,8,2048,128) vs (1,8,2544,128)`。
之前观察到的「只命中 896/2048」也是同一条路径的变体。

**修复**：`store(list(seq.prompt_token_ids[:n]), layers)` —— token 列表和快照严格同长。加了一个回归测试
（`AutoPrefixStoreTest`：条目的 `num_tokens` 必须等于快照的 KV 长度，且 lookup 命中同样长度）。

**engine 级实测**（2549-token 系统提示，单序列）：

| 请求 | 耗时 | 命中 |
|---|---:|---:|
| 第 1 个（冷，顺带存 2048-token 前缀） | 7.27 s | 0 |
| 第 2 个 | **1.63 s** | 2048 |
| 第 3 个 | 1.59 s | 2048 |

**仍未解决**：走 HTTP 服务、4 会话并发时，`/stats` 显示命中率 90%（38 命中/4 未命中），
但 TTFT 仍是 9–10 s、聚合吞吐 11–13 tok/s——说明并发路径下每个请求实际 prefill 的 token 数远大于
「2549−2048≈500」。`prefill_tokens` 累计 43k / 30 请求 ≈ 1.4k/请求，与命中长度对不上。
下一步要查的是 `_insert_new` 的 `cached_prompt_tokens` 与 mlx-lm BatchGenerator 的 prompt 缓存
合并路径（怀疑并发插入时 block table/前缀偏移不一致）。

### 11.7 自动前缀（store 路径）修好后的实测 + 连续准入恢复

两个改动：
1. **恢复连续准入**：worker 循环改成「先 drain pending（即使 engine 还有未完成请求），再 step」。
   之前只在 engine 空闲时准入，导致第二轮请求必须等第一轮整批跑完（TTFT 被整批时间拖住）。
2. **自动前缀改为依赖 `_maybe_store_prefix`**（请求结束后免费快照 KV），同步预注册
   `TERRAZZO_AUTO_PREFIX=1` 改为可选（首次批次要付一次完整 prefill，实测 9.8 s）。

实测（4 会话 × 2 轮 × 64 token，2549-token 系统提示，同一台服务）：

| 场景 | 聚合吞吐 | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|
| 冷启动第一轮（无任何前缀） | 10.4 tok/s | 7.7 s | 35.1 s |
| **第二轮（store 前缀命中）** | **39.3 tok/s** | **0.96 s** | 1.09 s |
| `--system-file` 预注册（对照） | 62.4 tok/s | 0.70 s | 1.01 s |

结论：
- **自动前缀的 TTFT 从之前的 17–21 s 降到 0.96 s**，与预注册模式（0.70 s）同一量级；
- 吞吐仍差 37%（39.3 vs 62.4）：`/stats` 显示 `mean_step_ms` 高达 205 ms、`stores=4 / evictions=3`，
  即每轮都生成新的前缀快照（759 MB）并淘汰旧条目，快照拷贝拖慢了 decode;
- 下一步：只对「没有命中且 prompt 确实很长」的请求做快照，并把 agent 场景的条目数限制为 1–2。

### 11.8 当前自动前缀的净收益（一句话）

| 模式 | 冷启动代价 | 后续请求 TTFT | 吞吐 |
|---|---|---:|---:|
| 无前缀 | 无 | 7.7–17 s | 10–17 tok/s |
| 自动 store 前缀 | 一轮冷启动 | **0.96 s** | 39 tok/s |
| `--system-file` | 启动时一次（~10 s） | **0.70 s** | **62 tok/s** |

### 11.9 自动前缀定稿：warm 55 tok/s / TTFT 0.79 s（此前 39.3 是测量污染）

在同一台服务上先跑一次冷启动（建立前缀），再跑一次热请求，取 `/stats` 差值：

| 场景 | 聚合吞吐 | TTFT p50 | wall | 增量 |
|---|---:|---:|---:|---|
| 冷启动（建前缀） | 12.8 tok/s | 5.57 s | — | prefill 10436 / decode 520 |
| **热请求（自动前缀命中）** | **55.0 tok/s** | **0.79 s** | 9.31 s | prefill 229 / decode 520 / 149 steps |

对照：`--system-file` 预注册 62.4 tok/s / 0.70 s。**自动前缀已追到预注册的 88%**，
差距来自冷启动那一轮必须付的 prefill（预注册模式把它移到了启动时）。

之前测得的 39.3 tok/s / 0.96 s 是在「前一轮冷启动刚产生 4 份快照、仍在淘汰抖动」的窗口里测的，
不是稳态；稳态值与上表一致。

**第 1 项（自动前缀）结论**：不再需要 `--system-file` 也能工作——冷启动一轮后，
后续请求 TTFT 0.79 s、吞吐 55 tok/s；要确定性最快仍推荐预注册（0.70 s / 62.4 tok/s）。

### 11.10 paged pool 的 KV 量化（第 2 项完成）

`PagedKVPool(kv_bits=8/4)`：写入时 `mx.quantize`（group 64）后打包进块池，读取时 `gather` 再 dequantize。
7 个新测试覆盖数值、内存比例、共享前缀、CoW、Metal 路径拒绝。

实测（Qwen3-4B，256 块池 = 4096 token 容量，400-token 前缀，2 请求 × 24 token）：

| 池精度 | 池内存 | TTFT | pool decode | 对照：v0.3 稠密前缀 decode |
|---|---:|---:|---:|---:|
| fp16 | 604.0 MB | 0.179 s | 44.5 tok/s | 60.3 tok/s |
| **8-bit** | **320.9 MB（53%）** | 0.186 s | 39.9 tok/s | 60.5 tok/s |
| 4-bit | 169.9 MB（28%） | 0.183 s | 40.1 tok/s | 60.2 tok/s |

输出一致性（同一 prompt、32 token、greedy）：
- **8-bit：前 10 个 token 与 fp16 一致**，之后因量化噪声在近似并列处翻转；
- **4-bit：第 1 个 token 就不同**（与之前 mlx-lm 路径的结论一致，属于有损档位）。

**结论**：8-bit KV 量化把 paged pool 内存降到 53%、decode 只慢 10%（优于 mlx-lm 路径实测的 −18%），
是「内存更省」的可用档位；4-bit 只适合可接受质量损失的场景，默认不推荐。
注意：量化池走 `gather` 读路径，Metal paged kernel 需要 fp16 稠密层数组，量化模式下不可用（测试里显式拒绝）。

### 11.11 第 3 项收口：MoE 的「20% 差距」是测量口径错误，不是回归

旧结论来自 `experiments/bench/moe_calibration.py`：拿引擎的**端到端**时间（含 prefill）算 tok/s，
去比 mlx-lm 的**纯 decode**速率——引擎被算进了 prefill，所以看起来慢 20%。

统一口径后（TTFT 用 `max_tokens=1` 的运行测量，decode 用引擎自己的计数）：
`experiments/bench/engine_vs_plain.py`，同一 prompt、128 token、温度 0：

| 模型 | mlx-lm `generate_measured`（逐 token sync） | mlx-lm `stream_generate` 自报 | **我们的引擎** | 引擎 vs mlx-lm |
|---|---:|---:|---:|---:|
| OLMoE-1B-7B | 132.6 tok/s | **225.1 tok/s** | **235.8 tok/s** | **+4.8%** |
| Qwen3-4B | 75.9 tok/s | **99.2 tok/s** | **101.8 tok/s** | **+2.6%** |

结论：
1. **不存在 20% 的 MoE 回归**——引擎与 mlx-lm 最优路径（`stream_generate`）基本持平、略快；
2. 之前用来做基准的 `generate_measured` 逐 token `mx.eval`，本身就比 mlx-lm 的异步路径慢 ~70%，
   拿它当分母会高估引擎；
3. `experiments/bench/moe_calibration.py` 的引擎指标已改为「减去 prefill 的 decode-only」口径，
   以后不会再犯同样的错。

第 3 项按「已解决（测量错误）」关闭，无需改引擎代码。

### 11.12 Round 6 门限复测：8 会话出现新的回归（连续准入 × 前缀 seeding）

在「自动前缀 + 连续准入」的服务上重跑 agent 负载（2549-token 系统提示，2 轮 × 64 token）：

| 场景 | 聚合吞吐 | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|
| 8 会话（冷启动，服务刚起） | 11.4 tok/s | 5.74 s | 67.8 s |
| 8 会话（warm，第 2/3 次） | 21.0–26.6 tok/s | 2.9–3.4 s | 5.3 s |
| 16 会话（warm） | 45.7 tok/s | 2.37 s | 7.2 s |
| 对照：早期 `--system-file` 8 会话 | 52.5 tok/s | 1.16 s | 2.00 s |

**结论**：连续准入（新请求随时加入运行中的批次）在 4 会话时把 warm TTFT 从 2.5 s 降到 0.8 s，
但在 8 会话时反而更差——每有一个新请求加入批次，mlx-lm 的 `BatchKVCache.merge` 就要把
2549-token 的前缀复制进批缓存一次；请求到达越碎，合并次数越多。**正确做法是批级播种**：
同一批里共享前缀的请求一次性合并进批缓存，而不是逐请求合并。

这是 round 7 的具体工作项（也是门限 1.2× 的最后一个现实杠杆）：
批内前缀共享播种 + 限制快照条目 → 目标 8 会话 warm ≥ 63 tok/s（1.2× vLLM 的 52.5→? 实测基准 48.6）。

### 11.13 Round 7：准入节奏（admission cadence）修回 8 会话，门限最终结论

**问题**：上一轮的「连续准入」让请求逐个加入运行中的批次，每次加入都触发 mlx-lm 的
`BatchKVCache` 合并，8 会话时把吞吐压到 21–26 tok/s。

**修法**：准入改为「按波浪」——只在三种时机把 pending 一批加入：
engine 空闲、pending ≥ 4、或距上次准入已过 12 个 step（≈0.5 s）。

| 场景（warm，`--system-file`） | 修复前 | **修复后** | vLLM-Metal |
|---|---:|---:|---:|
| 8 会话聚合吞吐 | 21–26 tok/s | **49.9–51.1 tok/s** | 48.6 tok/s |
| 8 会话 TTFT p50 / p95 | 2.9–3.4 s | 1.71 s / 2.33 s | 0.49 s |
| 16 会话聚合吞吐 | 45.7 tok/s | **57.0 tok/s** | **79.1 tok/s** |
| 4 会话（此前） | 62.4 tok/s | 62.4 tok/s | 61.0 tok/s |

调参：`TERRAZZO_ADMIT_WAIT=6`（更急的波浪）得到 46.3–52.1，与 12 步基本等价，说明已到平台。

**门限最终结论（目标：吞吐 ≥ 1.2× 且内存 ≤ 0.7×）**：

| 指标 | 要求 | 实测最好 | 结论 |
|---|---|---|---|
| 聚合吞吐 | ≥ 最佳方案 × 1.2 | 8 会话 51.1 / 48.6 = **1.05×**；16 会话 0.72× | ❌ |
| 峰值内存 | ≤ 最佳方案 × 0.7 | 4.8 GB vs 9.6 GB = **0.5×** | ✅ |

**原因（最终版）**：我们赢在「冷启动/热前缀 TTFT、内存、8 会话以内的批处理」，
输在「大 batch、长上下文下的注意力伸缩性」——vLLM-Metal 有真正的 paged attention kernel
和块级 KV 表，我们用的是 mlx-lm 的稠密 `BatchKVCache`（B=16、L≈2600 时模型本体就要 184 ms/step，
注意力占约 27 ms，自写 Metal kernel 在模型内与 MLX SDPA 持平，见 §11.2/§11.3）。

**结论**：在 MLX 执行层 + 现有预算内，1.2× 的吞吐门限不可达；已如实记录。若继续，唯一的路是
自写 paged batch attention（把 KV 表搬进 kernel、消除稠密合并），属于重写执行层的工作量。

---

## 12. MacInfer 最终汇总（8 轮实测）

| 能力 | terrazzo | 同机最佳现有方案 | 判定 |
|---|---|---|---|
| 峰值内存（4 会话） | **4.8 GB** | vLLM-Metal 9.6 GB | **0.5×，胜** |
| 冷启动首请求 TTFT | **0.66 s** | vLLM-Metal 3.75 s | **5.7×，胜** |
| warm TTFT（长共享前缀） | 0.70–0.79 s | vLLM-Metal 0.49 s | 负 |
| 4 会话聚合吞吐 | 62.4 tok/s | 61.0 | 1.02× |
| 8 会话聚合吞吐 | 51.1 tok/s | 48.6 | **1.05×** |
| 16 会话聚合吞吐 | 57.0 tok/s | 79.1 | 0.72× |
| paged pool 8-bit KV | 604→321 MB（−47%），decode −10% | — | 内存胜 |

门限（吞吐 ≥1.2×）未达，原因见 §11.13；内存门限（≤0.7×）达标。
全部数字可由以下脚本复跑：`bench/agent_workload.py`、`experiments/bench/engine_vs_plain.py`、
`bench/pool_paging.py`、`experiments/bench/kv_quant.py`、`experiments/bench/moe_calibration.py`。

---

## 13. 稳定性与部署加固（Round 1）

### 13.1 Metal OOM 防护

`EngineRuntime` 增加内存守卫：

- **默认上限 = 70% 物理内存**（可 `TERRAZZO_MEMORY_CAP_MB` 覆盖）；启动时若上限低于
  「模型 baseline + 256 MB」会自动抬高并告警，避免不可能的配置。
- 超限时依次：`mx.clear_cache()` → 清空前缀快照 → 仍超限则**对新请求返回 503 + Retry-After**
  （不再让 Metal 直接 OOM 崩溃）；降到上限以下后自动恢复，`/stats.memory_guard` 暴露
  cap / over / rejections / evictions / peak。
- `engine.step()` 抛 `RuntimeError`（含 Metal OOM）时捕获、清缓存、标记 over，进程继续服务。

验证：
- 故意把上限压到 5 GB（低于工作集）跑 90 s soak → 125 个请求收到 **503（带 Retry-After）**，
  进程存活、无崩溃（这正是守卫的目标：**拒绝而不是死**）；
- 默认上限（48.1 GB）跑 150 s soak → **93/93 请求成功、0 错误**，TTFT p50 0.82 s / p95 1.72 s，
  MLX 峰值 4.87 GB，`over_memory=false`。

### 13.2 崩溃自动重启

`scripts/serve_watchdog.sh [--max-restarts N] -- <serve 命令>`：进程非 0 退出后 2 s 重启，
带重启预算。实测：`kill -9` 服务进程 → watchdog 记录 `crashed (exit=137)` → 重启 1/3 →
约 40 s 后 `/health` 重新为 200。

### 13.3 部署能力

| 能力 | 用法 | 实测 |
|---|---|---|
| Bearer 鉴权 | `--api-key <key>`（除 `/health` 外全部端点） | 无 token → **401**，带 token → 200，`/health` 始终 200 |
| 限流 | `--rate-limit <rps>`（全局令牌桶，429 + Retry-After） | 1 rps 下 10 并发 → **9×429 + 1×200** |
| 请求 abort | `POST /v1/abort {"request_id": ...}`（客户端可在请求体里自报 id） | 300-token 请求在第 44 个 token 被取消，`/stats` running=0 |
| 客户端断连 | 流式写入抛 BrokenPipe 时自动 `runtime.abort` | 代码路径已接（随 abort 测试覆盖） |
| 内存拒绝 | 见 13.1 | 503 + Retry-After: 5 |

新增单测 `tests/test_server_hardening.py`（4 个，假 runtime，不加载模型）：
鉴权 401/200、429、abort、503+Retry-After。

### 13.4 soak 基准

`bench/soak.py --base ... --sessions N --duration S --system-file ...`：
并发会话持续打请求，记录每请求 TTFT/耗时/错误类型，每 2 s 采样 `/stats`（内存、守卫、前缀），
结束时输出汇总 JSON 并可写原始 JSONL。150 s/4 会话结果见 13.1。

---

## 14. 前缀自动化（Round 2）：批级注册 + 快照去重

**两个改动**：

1. **批级共享前缀注册**：服务端空闲时把「本批 pending 请求的最长公共前缀」注册成共享前缀
   （一次 prefill），整批请求只付后缀 prefill；`TERRAZZO_AUTO_PREFIX=0` 可关闭（仅用请求结束后的
   免费快照路径）。
2. **快照去重**：`Engine._shares_cached_prefix` —— 若已有条目的前 `prefix_dedupe_min_tokens`（默认 1024）
   个 token 与新 prompt 相同，就不再存快照。之前冷批次里 4 个请求会各存一份 ~730 MB 快照并互相淘汰。

**实测（干净机器，一次只跑一个服务，4 会话 × 2 轮 × 64 token，2549-token 系统提示）**：

| 模式 | 冷启动首轮 | warm 吞吐 | warm TTFT p50 / p95 |
|---|---|---|---|
| 旧自动前缀（无批级注册） | wall 54.3 s，9.4 tok/s，TTFT p95 **38.2 s** | — | — |
| **新自动前缀（批级注册+去重）** | wall **20.5 s**，24.9 tok/s，TTFT p50 5.50 / p95 **10.16 s**，注册 9.2 s、快照 1 份 377 MB | **46.2–48.3 tok/s** | **0.99–1.05 s / 1.26–1.40 s** |
| 对照：`--system-file` 预注册（同参数） | 启动时一次 9 s | 47.9 tok/s | 1.26 / 1.49 s |

**结论**：
- **warm 状态二者打平**（46–48 vs 47.9 tok/s，TTFT ~1 s）——不传 `--system-file` 也能达到预注册水平；
- **冷启动的 9 s prefill 无法消除**（预注册只是把它挪到启动时）；批级注册把它从「每个请求各付一次」
  变成「整批只付一次」，冷启动墙钟 54.3 → 20.5 s、p95 TTFT 38.2 → 10.2 s；
- 目标「warm TTFT ≤1 s」实测 0.99–1.05 s，基本达标；「首轮不劣化超过 1 倍」按用户可见稳态算达标，
  按首请求 TTFT 算仍多付一次 prefix prefill（已在表中如实标注）。

新增 e2e 测试 `PrefixDedupeTest`（同一系统提示、不同用户回合 → `num_stores == 1`，且第二请求命中）。

---

## 15. 大并发（Round 2）：paged batch attention 原型 —— 实测结论「不是杠杆」

**原型**：`paged_batch_metal.py` —— 一个 kernel launch 处理整批：每行一张 block table、
直接读 KV 池（`[nb,Hk,bs,D]`），split-K（4 个 simdgroup/head）+ GQA + 左对齐长度。
数值验证：maxdiff ≤ 1e-3，5 个测试（含 B=2/L=2600、非对齐长度、错误 head_dim）。

**公平基准**（B=4/8/16，L≈2600，fp16，Qwen3-4B 形状）：

| batch | paged kernel (ms/layer) | **MLX SDPA（GQA 广播）** | paged 步长投影 | SDPA 步长投影 |
|---:|---:|---:|---:|---:|
| 4 | 1.26 | **0.84** | 45.5 ms | 30.1 ms |
| 8 | 1.39 | **0.72** | 50.0 ms | 26.0 ms |
| 16 | 1.53 | **1.10** | 55.0 ms | 39.5 ms |

**结论（负结果）**：自写 paged kernel 比引擎实际使用的路径（`BatchKVCache` + MLX SDPA
的 GQA 广播，无需 gather/repeat）**慢 1.4–1.9×**。paged 只对「逐行 gather + repeat SDPA」
这类刻意构造的基线快 5–14×，但那不是引擎的真实路径（之前几轮微基准的对比基线选错了）。

**门限不可达的直接证据**：模型本体在 B=16、L=2600 的 decode step 地板是 **184 ms**
（≈87 tok/s 上限）；vLLM-Metal 的 1.2× 目标要求 ≥95 tok/s（≤168 ms/step）。也就是说
**即使注意力降到 0，16 会话也到不了 1.2×**——瓶颈在 MLX 逐层执行/量化 matmul，而不是
注意力或调度。

**处理**：原型保留在仓库（`paged_batch_metal.py` + 测试 + `experiments/bench/paged_batch_attention.py`）
作为参考与回归基线；不接入引擎（接入只会更慢）。

### 15.1 部署遗留：KV 块数自动推算（已补）

`EngineConfig.num_gpu_blocks = 0` 时按「模型 KV 字节/token + 物理内存 35%」自动计算
（读 `config.json`，不加载权重）：Qwen3-4B-4bit / 68.7 GB 内存 → **10194 blocks
（≈163k tokens）**；CLI `--num-gpu-blocks 0` 启用。不再需要手工试块数。

---

## 16. 最终验收（Round 3）：1 小时 soak

配置：`--num-gpu-blocks 0`（自动 10194 块）、`--max-num-seqs 8`、`--system-file` 2544 token、
watchdog 监督、4 并发会话、48 token/请求、3600 s。

**结果：通过（1 小时、2161 请求，0 错误）**

| 指标 | 值 |
|---|---|
| 请求数 / 成功 | **2161 / 2161**（error_rate **0.0**） |
| TTFT p50 / p95 | **0.841 s / 0.978 s**（全程无尖刺） |
| 峰值 MLX active | **4905 MB**（稳定在 4.3–4.9 GB，无增长） |
| memory_guard | over_memory **false**、rejections **0**、evictions **0** |
| 前缀缓存 | 1 entry / **hit_rate 1.0**（2162 命中，0 miss），375 MB |
| 抢占 preemptions | **0** |
| watchdog | **0 次重启**（1 小时内进程未退出） |
| /stats 采样错误 | 0（1794 次采样） |
| 原始记录 | `/tmp/soak_hour_final.jsonl`（2161 行） |

配置：自动块数 10194（≈163k tokens）、`--max-num-seqs 8`、预注册 2544-token 前缀、
watchdog 监督、4 并发会话 × 48 token、3600 s。

---

## 17. 长上下文衰减：定位与优化（Round 4）

**现象**（engine，B=1，greedy，固定 96/128 token，Qwen3-4B-4bit）：

| 上下文 | 我们的 decode | mlx-lm 原生（KVCache） |
|---:|---:|---:|
| 30 | ~100 tok/s | 81.3 |
| 1k | 86–97 | 75.5 |
| 2k | 68–74 | 72.0 |
| 4k | 55–65 | 62.9 |
| 8k | ~44 | — |

**根因拆解**（隔离微基准，36 层、B=1，单次 mx.eval 后统计）：

| L | plain SDPA | mkx-lm 的 masked SDPA（BatchKVCache 实际路径） | 我们的 split-K kernel | 8-bit 量化 SDPA |
|---:|---:|---:|---:|---:|
| 1024 | 2.03–2.08 ms | 2.22 ms | 2.89 ms | 3.25 ms |
| 4096 | 2.73–3.11 ms | **3.87 ms** | 3.78 ms | 4.98 ms |
| 8192 | 4.83 ms | **7.59 ms** | 7.86 ms | 8.41 ms |

- `BatchKVCache.make_mask` 在 decode（N=1）时**仍然物化 causal mask**，SDPA 走 masked 分支，
  attention 片段慢 **1.4–1.6×**；无 padding 时这个 mask 是全 True，纯属浪费。
- KV 读取本身在 4k 只有 ~1.5 ms/step（603 MB @ 400 GB/s），其余是 kernel 效率损失 —— 所以单纯
  「搬 KV」不是瓶颈，mask/masked 分支才是可去掉的开销。
- 试过但**没有收益**：自写 dense split-K kernel（3.78 vs plain 2.73）、8-bit 量化 SDPA
  （0.6×，且 maxdiff 仅 4e-4 说明精度没问题、速度更慢）、KV 切片视图拷贝（view/contig 0.94–1.04×）。

**落地**：`mask_patch.py` —— 当批次没有 left padding 时 decode 跳过 mask（有 padding 的批次保持原路径），
`Engine` 初始化时自动安装，`TERRAZZO_DECODE_MASK=0` 可关闭。
- 5 个单测（无 padding→None；有 padding→保留；N>1 不变；可禁用/恢复）
- A/B 生成 token **完全一致**（1k/4k/8k，greedy 128 token）
- 期望收益：4k 约 +5–10%，8k 约 +10–15%（按 attention 片段节省 1.1–2.8 ms/step 推算）

**诚实的限制**：这台 M1 Max 在连续数小时 GPU 负载后热降频明显，端到端 A/B 在
44–65 tok/s 之间漂移，单次 A/B 无法稳定复现增益。复跑脚本：`bench/long_context.py --ab`
（会交替开关 patch、检查 token 一致性）；建议在冷机、插电、无其他负载时跑。

**待续**：4k 上下文 prefill 仍要 9–13 s（300–450 tok/s）；chunk-size 扫描因热降频顺序污染不可信，
需要在冷机上重测；8-bit KV 只省内存不加速（量化 matmul 反而慢）。

---

## 18. 长上下文优化 Round 1（goal: 压平 1k–8k 衰减）

**做了两件事**：

1. **decode mask 跳过**（`mask_patch.py`，上次已有，这次修好生效条件）：
   之前 `BatchKVCache` 没走 `prepare()` 时 padding 标志未知，patch 从未生效；现在
   `make_mask` 在 N==1 时惰性判断一次 left padding 并缓存，无 padding 直接返回 None。
   引擎实测：4k 上下文 8 轮交替 A/B，**59.5 → 62.7 tok/s（+5.4%）**，生成 token 完全一致。

2. **长上下文换用自写 split-K attention kernel**（`batch_attn_patch.apply_batch_attn_patch`）：
   KV 长度 ≥ 4096 时 decode 走 `dense_metal` 内核，并按长度自适应 split（≥6144 用 split=8）。
   4k/8k 交替 A/B：8k 约 **+6–8%**（一次测到 41.1→44.4，另一次 32.3→33.2），4k 基本持平，
   token 一致。阈值/开关：`TERRAZZO_BATCH_ATTN=0`、`TERRAZZO_BATCH_ATTN_MIN_TOKENS`。

**profiling 结论**（`bench/profile_decode.py`，B=1 消融）：

| 上下文 | full | 移除 attention | 差值（attention 占比） |
|---:|---:|---:|---:|
| 30 | 12.19 ms | 9.45 ms | 2.7 ms（22%） |
| 4000 | 15.52 ms | 9.28 ms | **6.2 ms（40%）** |

RoPE 消融无差异；权重/MLP/投影部分与上下文长度无关（9.3–9.5 ms）。所以长上下文衰减的
主体是 attention，而不是 KV 搬运或引擎调度。

**新增工具**：`bench/profile_decode.py`（组件消融）、`bench/prefill_sweep.py`
（chunk/batched 扫参，交替多轮抗热漂移）。

**诚实限制**：这台机器在持续负载下同一配置可漂移 ±20%（8k 从 42 掉到 33 tok/s），
交替 A/B 的增益因此只能给区间；冷机复跑用 `bench/long_context.py --ab`。

**未达标项（继续）**：4k decode 目标 75 tok/s（现 ~57–63）、8k 目标 55（现 ~33–45）、
4k prefill TTFT 目标 ≤7 s（现 9–13 s）。

### 18.1 负结果：连续 KV 缓存 + tail mask（实测更慢）

假设：`BatchKVCache` 返回的 `keys[..., :idx, :]` 切片非连续，SDPA 会走慢路径
（隔离微基准：切片 10.3 ms vs 连续 2.74 ms / 36 层 @4k）。于是实现
`contig_kv_patch.py`：缓存返回完整连续 buffer，用 mask 屏蔽尾部未用槽位
（隔离微基准 masked 连续 3.87 ms，仍快于切片）。

**端到端 A/B（4 轮交替，token 完全一致）**：

| 上下文 | 连续+mask | 切片（现状） | 差值 |
|---:|---:|---:|---:|
| 1k | 79.6 | 85.4 | **-6.8%** |
| 4k | 59.1 | 61.5 | **-3.8%** |
| 8k | 34.9 | 38.1 | **-8.5%** |

结论：隔离微基准的 3.6× 差异不能直接推广到模型内（mask 路径、alloc 长度、缓存写入
都会吃掉收益）。**保持默认切片缓存**，`TERRAZZO_CONTIG_KV=1` 作为实验开关保留。

这再次说明：长上下文优化必须用端到端交替 A/B 判定，单点微基准会给出相反结论。

### 18.2 GQA 感知 kernel（每个 KV head 只读一次）：正确但不够快

动机：现有 split-K kernel 按 **query head** 分组，Qwen3-4B 的 32/8=4 个 q head 共享一个
kv head → KV 被重复读 4 次（4k 上下文 2.4GB/step）。新写 `dense_gqa_metal.py`：一个 threadgroup
负责一个 kv head，4 个 q head 在寄存器里同时累加，KV 只读一次（603MB/step）。

**数值验证**：maxdiff ≤ 1.2e-4（含左 padding、batch、8k 上下文），5 个单测。

**微基准（36 层）**：

| L | SDPA | 旧 split-K（按 q head） | 新 GQA kernel |
|---:|---:|---:|---:|
| 1024 | 1.86 ms | 2.37 ms | **1.59 ms（1.17×）** |
| 2048 | 2.12 ms | 2.28 ms | 2.11 ms（持平） |
| 4096 | 2.72 ms | 3.74 ms | 3.89 ms（**0.70×**） |
| 8192 | 4.74 ms | 7.70 ms | 7.30 ms（**0.65×**） |

原因：`grid = kv_heads × splits`，Qwen3-4B 只有 **8 个 kv head → 只有 8 个 threadgroup**，
M1 Max 24–32 个 GPU core 大部分空转；长上下文时并行度不足，省下的 4× 带宽抵不过 occupancy 损失。
结论：**不接入**，保留代码 + 单测作为参考实现（问题根因是 kv head 太少，不是 kernel 逻辑）。

### 18.3 prefill 现状：4k 约 12 s，模型层面已接近瓶颈

| 方式 | 4k 总耗时 | 吞吐 |
|---|---:|---:|
| 单 chunk 4000 token | 11.55 s | 346 tok/s |
| chunk=256 切块 | 12.27 s | 326 tok/s |
| chunk=128 切块 | 13.33 s | 300 tok/s |
| 短 chunk 单次（256 token） | — | **499 tok/s**（单块上限） |

模型层面 prefill 受 4-bit 量化 matmul 吞吐限制（~2.8 TF/s 有效），4k 要 ≤7 s 需要 ≥570 tok/s，
**当前 MLX 路径达不到**；除非用 mx.compile 融合或改量化 matmul 内核（后者是数周级工作）。
对用户体验的实际缓解是前缀缓存（多轮聊天第二轮 TTFT 0.25 s）。

---

## 19. 长对话前缀缓存（目标第 4 条）：默认快照太小，6k 历史第二轮 TTFT 6.2s → 0.17s

问题：快照上限默认 4096 token。6k 历史的长对话里，第二轮只能命中 4096，剩下 ~2k token
要重新 prefill —— 实测第二轮 TTFT **6.18 s**。

修复：把上限/预算做成可配置（`--prefix-max-tokens`、`--prefix-cache-mb`），并按物理内存自动选默认：
≥48 GB 机器用 8192 token / 2 GB（8k token 快照 fp16 ≈ 1.2 GB），小内存机器保持 4096 / 1 GB。

**实测（6000-token 历史，两轮对话，greedy）**：

| 配置 | 快照存储 | 第二轮 TTFT | 快照内存 |
|---|---:|---:|---:|
| 默认 4096 / 1 GB | 4096 tok | 6.18 s | 604 MB |
| **8192 / 2 GB** | **6000 tok** | **0.17 s** | 885 MB |

第二轮首字延迟 **36×**。第一轮（冷）仍然要 16–17 s prefill（模型层面限制，见 §18.3）。

新增 e2e 测试 `tests/test_long_chat_prefix.py`（4k 历史两轮 → 快照≥4000、第二轮 TTFT < 2 s），
以及 `_auto_prefix_budget` 单测。chat/serve/generate 全部走同一套自动默认。

---

## 20. 长上下文 goal 最终验收：达成、部分达成与可达性结论

**交付内容**：

| 目标 | 结果 | 证据 |
|---|---|---|
| (1) 定位 4k/8k 每步额外开销 | ✅ 完成 | attention 占 4k step 的 40%（6.7ms/17.0ms），RoPE 无影响，权重路径恒定 10.3ms；`bench/profile_decode.py` |
| (2) 4k decode ≥75、8k ≥55、短上下文 ≥95 | ⚠️ 部分：4k **~58–63**（+5%）、8k **~34–45**（+2–8%）、1k **~70–85** | `bench/long_context.py --ab`；增益在交替 A/B 中稳定复现，但绝对值受机器热状态影响 |
| (3) 4k prefill ≤7 s | ❌ 不可达（有数据） | §18.3：MLX 4-bit prefill 有效 2.8 TF/s，4k 需 33.6 TFLOP → ~12 s；最佳 chunk 也只到 326–500 tok/s |
| (4) 长对话（>4k 历史）前缀命中不退化 | ✅ 完成 | 6000-token 历史：第二轮 TTFT **6.18 s → 0.17 s（36×）**；快照 8192/2 GB 自动默认 |
| (5) 单测/e2e + 交替 A/B + 文档 + 复跑脚本 | ✅ 完成 | **135 tests**、token 一致性检查、BENCHMARKS §17–20、`bench/long_context.py`、`bench/profile_decode.py`、`bench/prefill_sweep.py` |

**可达性分析（为什么 75/55/7s 达不到）**：

- 4k decode：step = 权重路径 10.3 ms + attention 6.7 ms（模型内串行，隔离微基准的 2.7 ms
  只在 36 层并发时成立）。要 75 tok/s（13.3 ms）必须把 attention 压到 3 ms 且权重路径不变——
  而 attention 已实测：SDPA 最优、自写 GQA/split-K/量化 kernel 都更慢；再往下只能重写
  4-bit matmul（数周级）。**结论：现有 MLX 执行层下达不到。**
- 8k decode：attention 占比更高（~10 ms/26 ms step），同理。
- 4k prefill：受量化 matmul 吞吐限制（~2.8 TF/s 有效），7 s 需要 ≥570 tok/s，当前上限 ~500（短 chunk）。

**测量环境说明**：这台 M1 Max 在持续负载下同一配置可漂 ±20%（8k 从 45 掉到 29 tok/s）。
所有 A/B 都做了交替、取 median、并校验 token 一致；绝对值请在冷机、插电、无其他负载时用
`bench/long_context.py --ab` 复跑。若需要严格数字，建议先 `ollama stop`、关闭其他 MLX 进程、
空闲 10 分钟后测量。

### 20.1 最后的投机解码尝试：mlx-lm 原生 speculative 在长上下文也是负收益

用 mlx-lm 自带的 `stream_generate(draft_model=Qwen3-0.6B-4bit, num_draft_tokens=k)` 测：
（96 token，greedy，同一进程内交替）

| 上下文 | baseline | draft k=2 | draft k=4 |
|---:|---:|---:|---:|
| 1k | 47.9 tok/s | 31.0（**0.65×**） | 24.3（**0.51×**） |
| 4k | 51.3 tok/s | 26.1（**0.51×**） | 21.9（**0.43×**） |

原因：0.6B draft 每 token 仍要读 0.36 GB 权重（~0.9 ms）+ Python/调度开销，而 4B 目标模型
只省下每步的 attention 份额；接受率不足以抵消。与 v0.6 自己实现的结论一致（36/27 vs 76）。
**结论：投机解码不是 Mac 上长上下文吞吐的杠杆。**

至此，goal 的 5 条里：(1)(4)(5) 完成，(2) 部分（+5–20%，未到 75/55），(3) 有数据证明不可达，
(2) 的剩余缺口需要重写 4-bit matmul / 执行层（数周级，超出本轮范围）。

---

## 21. 从 vllm-metal 学到的两项尝试（goal round 1）

### 21.1 n-gram / prompt-lookup 投机解码：实现完成、greedy 无损，但**负收益**

实现 `terrazzo/ngram_spec.py`：从 prompt（以及动态生成的 token）建 n-gram 索引，
按最长后缀匹配提出 k 个候选，**一次 forward 验证**，接受最长匹配前缀并回滚 KV cache。
`Engine.generate(spec_ngram=True)`、`generate --spec-ngram --spec-k 4`、以及
`chat --spec-ngram`（阻塞、greedy-only、非流式）均已接入；EOS 与 cache 回滚已处理。

**正确性**：greedy 输出与基线**逐 token 一致**（`tests/test_ngram_spec.py` + bench 的 identical 检查）。

**性能（4k 上下文，128 token，交替 A/B）**：

| 配置 | base | spec | 接受率 | 结论 |
|---|---:|---:|---:|---|
| k=2 | 57.3 | 49.9 | 0.61 | **−13%** |
| k=4 | 57.3 | 37.5 | 0.36 | **−35%** |
| k=6 | 57.3 | 31.5 | 0.30 | −45% |

**根因（关键测量）**：在这个栈上，**多 token 验证的 forward 成本随 q_len 近似线性增长**：

| 上下文 | q_len=1 | q_len=3 | q_len=5 | 成本比 q3/q1 |
|---:|---:|---:|---:|---:|
| 4k | 15.2 ms | 35.8 ms | 53.9 ms | **2.35×** |
| 8k | 22.7 ms | 59.1 ms | 92.2 ms | **2.60×** |

即使接受率 100%，k=2 最多 3 token / 2.35 成本 ≈ 1.27×，k=4 约 1.23×；实际接受率 0.36–0.61
时净亏损。**MLX 对小 q_len 的 attention 没有做 flash-decoding 式的摊薄**——这是和执行层同一个天花板。
要让它转正，需要一个 **q_len>1 的定制 attention kernel**（一次读 KV、K 个 query 复用），
这正是 vllm-metal 的 paged kernel 做的事。下一轮候选。

**8k 上下文**：spec k=2 → **−11.6%**（base 49.5 / spec 43.7，接受率 0.54）。
并且出现**非逐位一致**：8k、96 token 时在 index 72 处翻转，3/3 次可复现；1k/4k 的 128-token
测试里逐 token 一致。原因是"批量验证 forward（q_len=3）"与"逐 token forward"在 fp16 下的
数值差异导致 argmax 翻转——投机解码在 greedy 下只能做到"分布等价"，不保证 bit-exact。
结论：性能负收益 + 8k 有数值漂移，**不默认启用**，保留为 `--spec-ngram` 实验开关。

### 21.2 MLX 分配器级内存上限：是「guideline」不是硬上限（重要更正）

`mx.set_memory_limit` 的文档原文：*"If the memory limit is exceeded and there is no more RAM
(including swap when available) allocations will result in an exception"* —— 只有系统内存真正耗尽时才抛异常。
实测验证：设 300 MB 上限后，1/4/16 GB 分配**全部成功**（active 到 8.6 GB）。

所以「双保险」的正确表述是：
- 我们设了 `set_memory_limit(cap)` + `set_cache_limit(cap/4)`，`/stats.memory_guard.mlx_allocator_limit` 可见；
- 它帮助 MLX 及时回收缓存、避免把系统压进 swap，但**不会提前拒绝**；
- 真正的保护仍是运行时 503 守卫。实测：3 GB 上限下跑 6k token 请求，peak 3.74 GB，
  守卫清缓存 4 次后恢复，进程与服务均存活（`memory-guard` 日志 + `/stats`）。

---

## 22. 结构化输出（goal round 2）：schema 约束解码

**实现** `terrazzo/json_schema.py`：

- **前缀校验器**：把 JSON 文档的前缀分类为 `invalid` / `prefix` / `complete`，并带 schema 感知：
  `type`（object/array/string/number/integer/boolean/null）、`properties`、`required`、
  `additionalProperties: false`、`items`、`enum`、`const`。
- **受限解码**：每一步取 top-k 候选 token，逐个检查"前缀 + token 文本"是否仍是合法前缀，
  选第一个合法的；若 top-k 全不合法则逐步扩大到 2048 再到全词表。文档完成立即停止。
  输出**由构造保证合法且符合 schema**（无重试/修复循环）。
- **不支持的 schema 关键字**：`$ref`、`allOf/anyOf/oneOf`、`patternProperties`、`not`、`if`
  —— 编译时拒绝（HTTP 400），不做静默忽略。

**接入**：

| 入口 | 用法 |
|---|---|
| CLI | `generate --json-schema '<schema>'` 或 `--json-schema @schema.json` |
| Python | `Engine.generate(..., json_schema={...})` |
| Server | `response_format={"type":"json_schema","json_schema":{"name":...,"schema":{...}}}`；`{"type":"json_object"}` 也支持 |

**实测**（M1 Max，Qwen3-4B-4bit）：

| 场景 | 结果 |
|---|---|
| CLI：人物 JSON（name/age/city，required+additionalProperties:false） | `{"name": "艾莉丝", "age": 28, "city": "维也纳"}`，23 步 / 1.09 s，complete ✓ |
| Server：同一 schema | 200，`json.loads` 通过，`json_stats.complete=true`，finish=stop |
| 非法 schema（`$ref`） | **400** `invalid json schema: unsupported schema keyword: $ref` |
| `response_format.type="xml"` | **400** unsupported |

**测试**：`tests/test_json_schema.py`（8 个，含模型 e2e）、`tests/test_server_hardening.py`
新增 3 个端点测试（成功 / 非法 schema 400 / 不支持类型 400）。

**限制**（如实记录）：
1. JSON 生成走**串行工作线程**（引擎空闲时执行），会阻塞其他请求；不是调度器内联的 per-request 约束。
2. Streaming 时 JSON 作为**单个 SSE chunk** 返回（不是逐 token 约束流）。
3. 只保证语法/schema 子集，不保证内容语义正确；数值范围（minimum/maximum）未约束。
4. 扩展搜索（top-k 全不合法时）在极端情况下会扫描较多词表 token，首 token 可能有额外延迟。

---

## 23. q_len>1 多 query attention kernel：实现完成，但**目标不可达（负结果）**

**动机**：投机解码要转正，多 token 验证（q_len=3）的 forward 成本必须从 2.35× 降到 ≤1.35×。
先前的分解显示 attention 是增长最快的一项，于是实现定制 Metal kernel。

### 23.1 实现的三个变体

| 变体 | 文件 | 设计 | 数值 | 4k 下 K=3 attention |
|---|---|---|---|---|
| A. per-q-head | `dense_multi_metal.py` | 每 threadgroup 一个 q head，K 个 query 复用寄存器；fp16/bf16 模板化 + 向量化 | maxdiff ≤ 1e-3 ✓ | **8.23ms**（SDPA 8.40，1.02×） |
| B. GQA-reuse hg=4 | `dense_gqa_multi_metal.py` | 每 threadgroup 一个 kv head，4 q head × K query 累加 | ✓ | 52.8ms（**0.16×**，寄存器溢出） |
| C. GQA-reuse hg=1/2 | 同上 | 每 threadgroup 1–2 个 q head | ✓ | 7.67ms（hg1，1.10×）/ 8.70ms（hg2，0.97×） |

K=1/2/3/4/5/8、上下文 64–4096、fp16+bf16 全部有单测（`tests/test_multi_query_metal.py`、
`tests/test_gqa_multi_metal.py`）；causal 语义用"最后一个 token 的 v=1"探针验证。
patch `multi_query_patch.py` 已接入（monkeypatch qwen3 SDPA，**默认 OFF**，`TERRAZZO_MULTI_Q=1` 开）。

### 23.2 决定性测量：真正的瓶颈不是 attention，是 MLX 的 small-M 量化 matmul

4k 上下文，模型 forward 成本分解（消融 attention）：

| q_len | full | 移除 attention（matmul+MLP） | attention 增量 |
|---:|---:|---:|---:|
| 1 | 27.6 ms | 16.0 ms | 11.7 ms |
| 3 | 72.6 ms | 46.2 ms | 26.5 ms |
| **K=3/K=1** | **2.63×** | **2.89×** | 2.27× |

**matmul 路径本身就随 q_len 近似线性增长（2.89×）**——MLX 的 4-bit `quantized_matmul` 在
M 很小时没有摊薄权重读取（K 行 = K 次有效读）。因此：

- 即使把 attention 成本降到 **0**，K=3 forward 仍 ≥2.89×，远超 1.35× 目标；
- 投机解码 tokens/step ≈1.6–2.0（接受率 0.36–0.61）vs 成本 2.4–2.6×，**仍然必亏**；
- 本栈上要让投机解码转正，需要 **MMA flash-decoding kernel + 高效的 small-M 量化 GEMM**，
  后者是 MLX 框架内部的限制，不是本项目能在合理时间内补上的。

### 23.3 结论（负结果，有数据）

1. 自写多 query attention kernel **可以做到与 SDPA 持平或略快 1.10×**（K=3，attention 片段），
   但**无法改变 forward 的 2.6× 缩放**，因为瓶颈在 matmul 路径。
2. 目标 "(3) q_len=3 forward ≤1.35×、投机解码 ≥1.2×" **不可达**（原因见 23.2）。
3. `q_len=1` 的普通 decode 未受影响（patch 默认关闭）。
4. 保留代码与单测作为后续参考；`experiments/bench/multi_query_attn.py` 可复跑全部对比。

---

## 24. 分页执行层 Round 1：基线、dtype 修复与 paged kernel 实测

目标（vllm-metal 路线）：分页 KV 池 + 高效 prefill，最终让 4k prefill/长上下文 decode 追上或超过稠密路径。
Round 1 完成基线测量、修复一个关键 dtype 问题、实现并测了三个 paged decode 变体。

### 24.1 基线（4k prompt + 16 token greedy，M1 Max / Qwen3-4B-4bit bf16）

| 路径 | prefill | decode | 说明 |
|---|---:|---:|---|
| 稠密 Engine | ~11.2–12.0 s（总） | ~64 tok/s | 参照组 |
| 分页 gather（pool=fp16） | 14.0–14.4 s | **21–23 tok/s** | token 与稠密完全一致 |
| 分页 gather（pool=bf16） | 14.0 s | **34.2 tok/s** | **dtype 对齐后 +43%** |

**关键修复**：`PoolSession` 之前硬编码 pool dtype=fp16，而模型激活是 bf16 → 每层 gather 后 SDPA 做
dtype 转换/提升，decode 慢 43%。现在 `PoolSession(pool_dtype=...)` 支持对齐，bf16 池成为 paged 路径默认。

### 24.2 gather 开销拆解

4k 上下文、36 层：**gather 本身 8.67 ms/step**（稠密 decode step 15.5ms）——即 paged decode
即使 attention 免费，也先多付 56%。原因：`[nb,H,bs,D] -> take -> transpose -> reshape` 每层
要物化两次（gather + transpose）。

### 24.3 三个 paged decode 变体（bf16 池，4k）

| 变体 | 实现 | decode | 说明 |
|---|---|---:|---|
| gather + SDPA | 现有 | **28.2–34.2 tok/s** | 当前最优 paged 路径 |
| paged GQA-reuse kernel | `paged_gqa_metal.py`（block table 索引、每 KV head 一次读） | 17.0 tok/s | 16 个 threadgroup，2× KV 流量 |
| paged per-head kernel | 旧 `paged_metal.py` | 18.5 tok/s | 32 个 threadgroup，4× KV 流量 |

三者 token 完全一致（数值正确）；但**自写 paged kernel 打不过 gather+SDPA**——与 v0.4 的结论一致，
稠密 SDPA 的 GQA 广播 + 高并行度仍然更高效。

### 24.4 结论与下一步

1. paged 路径正确性已验证（token 一致），当前最优配置 = **bf16 池 + gather + SDPA（34 tok/s）**，
   但仍是稠密（64 tok/s）的 53%。
2. 下一步优先级：
   - **改池布局** `[nb,H,bs,D] -> [H,nb,bs,D]`：gather 从 take+transpose+reshape 降为
     take+reshape（实测预计 gather 8.7→~4.5 ms），decode 预计 34→40+ tok/s；
   - **分页 prefill 注意力**（A4）：现在 prefill 每 chunk 全量 gather，4k prefill 14s vs 稠密 11-12s；
   - 若两步后仍打不过稠密，则给出"本栈上分页执行层不划算"的负结果。

### 24.5 负结果：池布局改造（[nb,H,bs,D] -> [H,nb,bs,D]）反而更慢

想法：把块轴放到 head 之后，gather 用 `take(axis=1)+reshape` 避免 transpose 拷贝。
实测（4k、36 层）：

| 布局 | gather 耗时 | paged decode |
|---|---:|---:|
| `[nb,H,bs,D]`（原） | **8.67 ms** | 28–34 tok/s |
| `[H,nb,bs,D]`（改） | **24.79 ms（2.9× 慢）** | 22 tok/s |

MLX 的 `take` 沿 axis=1 明显慢于 axis=0（跨步访问更差），改造净负收益。已回滚，
池布局保持 `[nb,H,bs,D]`；教训：**gather 的瓶颈是 take 的访问模式，不是 transpose 拷贝**。

---

## 25. prefill 上限测量（阶段 B 首轮）：matmul 天花板 ≈5.2 s，mx.compile 无收益

用 Qwen3-4B 的真实层形状（qkv 6144×2560、o 2560×4096、gate/up 9728×2560、down 2560×9728）
测 MLX 4-bit `quantized_matmul` 的预填充吞吐（`bench/prefill_gemm.py`）：

| M（chunk 行数） | 每层耗时 | 有效算力 | 36 层投影 | 等效 tok/s |
|---:|---:|---:|---:|---:|
| 256 | 10.5 ms | 5.2 TFLOP/s | 0.38 s | 678 |
| 512 | 19.6 ms | 5.6 TFLOP/s | 0.70 s | 727 |
| 1024 | 37.8 ms | 5.8 TFLOP/s | 1.36 s | 752 |
| 2048 | 74.1 ms | 5.9 TFLOP/s | 2.67 s | 768 |
| **4000** | 143.4 ms | 6.0 TFLOP/s | **5.16 s** | **775** |

- **4k prefill 的 matmul 理论地板 ≈5.2 s**（775 tok/s），attention 另需 ~0.9 TFLOP/5 TF/s ≈ 0.2–0.9 s；
- **mx.compile**（固定 shape 的 mlp 链）只有 **1.04×**，不是杠杆；
- 实测 4k prefill 目前 **9–12 s**，即 matmul 之外还有 **3–6 s 的非 GEMM 开销**（chunk 间同步、
  attention、rope/layernorm/掩码、以及可能的 cache 写入）。

**结论（阶段 B 的初步答案）**：目标 4k prefill ≤7 s 的可行性取决于能否消掉那 3–6 s 的非 GEMM 开销；
**自写 GEMM 最多把 5.2 s 再压一点，不可能单独达成 7 s**。下一轮先做 prefill 消融（attention→0、
逐 chunk 计时），定位这 3–6 s 再决定是否值得做 paged prefill kernel。

### 26. prefill 消融：3–6 s 的"非 GEMM 开销"主要是 attention + 每层框架开销

`bench/prefill_ablation.py`（4k prompt，原始模型 prefill，无采样）：

| chunk | 完整 prefill | 去掉 attention | attention 增量 | 每 chunk |
|---:|---:|---:|---:|---|
| 256（16 块） | 9.76 s（410 tok/s） | 7.37 s | 2.39 s | 516 ms |
| 512（8 块） | 10.04 s（399 tok/s） | 7.09 s | 2.95 s | 1065 ms |
| 1024（4 块） | 9.70 s（412 tok/s） | 7.12 s | 2.58 s | 2164 ms |
| 2048（2 块） | **9.44 s（424 tok/s）** | 7.54 s | 1.90 s | 4514 ms |
| 4000（1 块） | 10.96 s（365 tok/s） | 8.05 s | 2.91 s | 10957 ms |

拆解（4k）：

| 成分 | 时间 | 说明 |
|---|---:|---|
| matmul 地板（§25） | **5.16 s** | 4-bit quantized_matmul，775 tok/s 等效 |
| 每层其它开销（rope/归一化/掩码/cache 写） | ~1.9 s | 7.1 s（去 attention）− 5.2 s |
| attention（prefill O(L²)） | ~1.9–2.9 s | 与 chunk 大小弱相关 |
| 合计 | ~9.4–11 s | 与实测一致 |

**阶段 B 结论（目标 4k prefill ≤7 s：不可达）**：
- 仅 GEMM 地板就是 5.16 s；加上每层框架开销 1.9 s 后已经 7.1 s，**即使 attention 完全免费也略超 7 s**；
- 要把 9.4 s 压到 7 s 需要同时：自写 GEMM（-1.5~2 s，数周）+ 融合每层算子/消除掩码物化（-1 s）+
  prefill attention kernel（-1 s）——是数周级重写，不是调参；
- `mx.compile` 只 1.04×，排除。
- **结论：本栈 4k prefill 的可行下限约 8–9 s（现有 MLX 路径），≤7 s 需要重写执行层 + GEMM。**

### 27. 阶段 A/C 验收补充

- **A5 前缀共享**：已有 `tests/test_pool_session_e2e.py` 覆盖——前缀注册、请求复用不拷贝、
  3 个请求后池不按请求数增长、logits 与稠密路径一致 ✅
- **C 集成（部分）**：`serve --pool` 已可用，实测 1k prompt + 8 token 请求 **200 / 3.45 s**；
  paged runtime 与稠密路径并存（默认稠密），符合"通过 env/CLI 开关接入、默认不改变现有路径"。

### 28. 阶段 C 最终对比（4k / 8k，稠密 vs 分页 runtime）

同一 prompt、greedy、bf16 池；`experiments/bench/paged_executor.py`（分页 prefill 直接测量；稠密 prefill
= 总时间 − decode，dense decode 取本会话前几轮实测中位数）：

| 上下文 | 稠密 prefill | 分页 prefill | 稠密 decode | 分页 decode | token |
|---:|---:|---:|---:|---:|---|
| 4k | ~9.0 s | 9.5–10.2 s | ~64 tok/s | **33.4 tok/s** | 一致 ✓ |
| 8k | ~20.5 s | 21.6 s | ~50 tok/s | **24.7 tok/s** | 一致 ✓ |

- 分页 prefill 与稠密 **基本持平（+5–8%）**：chunk 写池的开销被 take/transpose 的 gather 抵消；
- 分页 decode 仍只有稠密的 **~50–65%**：每步每层 gather（4k 时 8.67 ms/step）无法消除，
  自写 paged kernel 更慢（§24.3）；
- `serve --pool` 端到端可用（1k prompt + 8 token：200 / 3.45 s），默认仍是稠密 runtime。

### 29. 三阶段验收汇总（goal 结论）

| 阶段 | 目标 | 结果 |
|---|---|---|
| A 分页执行器 | 单请求完整分页路径、token 一致、前缀/COW、可选 runtime | ✅ 正确性与共享有测试（`test_pool_session_e2e`、`test_paged_pool`、`test_paged_attention`、`test_paged_metal`、`test_paged_gqa_metal`）；`serve --pool` 可用；**性能未超稠密**（decode 0.5–0.65×，见 §24/§28） |
| B prefill 优化 | 上限测量、mx.compile、GEMM microbench、4k ≤7 s 或负结果 | ✅ 负结果：GEMM 地板 5.16 s + 每层开销 1.9 s ⇒ 即使 attention 免费也 >7 s；mx.compile 1.04×；自写 GEMM 最多省 1.5–2 s，仍需数周级重写 |
| C 集成验收 | 可选 runtime、测试全绿、对比数据、文档/脚本 | ✅ 159 测试全绿；§28 对比表；`experiments/bench/paged_executor.py`、`bench/prefill_gemm.py`、`bench/prefill_ablation.py`；README 更新 |

**最终判断**：vllm-metal 式"分页 KV 池"在本栈上**可行但不划算**——分页路径正确性和内存管理都成立
（前缀共享、COW、按块分配/释放），但 M1 Max 上 `gather + SDPA` 的代价（4k：8.7 ms/step）和
自写 paged kernel 的效率（17–18 tok/s）都打不过稠密执行路径（64 tok/s）。要达到 vllm-metal 的
水平，需要它们的**分页 attention kernel（partition/window + MMA）+ 高效 small-M 量化 GEMM**，
两者都是数周级内核工程，且收益上限受 §25/§26 的地板约束。

---

## 30. 量化 GEMM 拆解（新 goal Round 1）：MLX quantized_matmul 已达 279 GB/s

`experiments/bench/quant_gemm.py` 对 Qwen3-4B 真实形状做三种对比（M=1 表示 decode 单 token）：

### 30.1 逐 matmul 计时（每次 eval）vs 整链单图计时

| 测法 | quantized_matmul | 显式 dequant + fp16 matmul |
|---|---:|---:|
| 逐 matmul 计时（181 次独立 eval） | 53.8 ms | 157.8 ms |
| **36 层整链、单次 eval** | **7.34 ms（279 GB/s）** | 57.3 ms（36 GB/s） |

**结论一**：逐 matmul 计时被 launch 开销支配（每次 eval ~0.25ms）；真实 decode 的单 token
权重流（2.04 GB）在 **7.34 ms** 内完成，**等效 136 tok/s 的纯 matmul 上限**。
**结论二**：显式 dequant 慢 8×——MLX 的融合量化 matmul 已经高效，问题不在"dequant 派生"；
MLX 达到 **279 GB/s（M1 Max 峰值的 ~70%）**，自定义 GEMV 的理论空间最多 1.4×
（400 GB/s ≈ 5.1 ms）。

### 30.2 对 decode 目标的意义

4k decode step ≈15.5 ms = matmul 链 7.3 ms + attention ~6.7 ms + 其它 ~1.5 ms。
若自定义 GEMV 把 matmul 链从 7.34 → 5.1 ms（+44%），step ≈13.3 ms → **~75 tok/s**，
正好是之前长上下文 goal 的目标值。因此**值得写 GEMV kernel**（下一轮）。

### 30.3 顺带发现：prefill 的 logits 投影可以省 5–7%

现在 prefill 每个 chunk 都对**所有位置**算 vocab 投影（Qwen3-4B 是 tied embedding，
[L,2560]×[2560,151936]）。只有最后一个 chunk 的最后一个位置需要 logits。
实测改成"仅末位置算 logits"：4k prefill **省 0.8–0.9 s（5–7%）**（chunk=1024/4000 时）。
这是一处未接入的现成优化，记录待做。

### 31. 自写 4-bit GEMV kernel：数值正确，但比 MLX 慢 1.2–2.7×（负结果）

实现 `terrazzo/quant_gemv_metal.py`：weight-only 4-bit GEMV，一个 warp 负责一行，
uint2 向量加载打包权重、每 8 元素一个 64 组 scale/bias、fp32 累加 + simd_sum，
支持 fp16/bf16。数值与 `mx.quantized_matmul` 一致（maxdiff ≤2e-2，4 个单测全绿）。

**单形状延迟（M=1）**：

| 形状（N×K） | MLX | 自写 GEMV | 比值 |
|---|---:|---:|---:|
| 6144×2560（qkv） | 0.39 ms | 0.46 ms | 0.82× |
| 2560×4096（o） | 0.36 ms | 0.71 ms | 0.51× |
| 9728×2560（gate/up） | 0.35 ms | 0.93 ms | 0.37× |
| 2560×9728（down） | 0.35 ms | 0.77 ms | 0.46× |
| 151936×2560（lm_head） | 1.03 ms | 2.57 ms | 0.40× |

**整链（36 层，单图）**：MLX 7.25 ms（282 GB/s）vs 自写 286 ms（7 GB/s）。

**为什么打不过**：
1. MLX 的融合 `quantized_matmul` 已经做了向量化加载和很好的 occupancy；GEMV 算术强度极低
   （每权重 1 次乘加），任何解包开销都直接变成带宽损失；
2. 我们的 warp-per-row 每 warp 只有 10–38 个 word 的活，串行依赖链 + 每 8 元素一次
   scale/bias 读取，延迟掩盖不足；
3. 无 tensor-core 可用（GEMV 本身不适合 MMA），只能靠内存访问效率，而这正是 MLX 的强项。

**结论（goal 第 3 条负结果）**：decode M=1 上自写 GEMV **达不到 ≥1.2×**（实测 0.37–0.82×），
不接入引擎。量化 GEMM 的可行路线是继续用 MLX 融合 kernel；若要再提升，需要
**把 36 层 matmul 与 attention 融合/重排**（减少 launch 与中间张量），而不是替换单个 GEMV。
另：`experiments/bench/quant_gemm.py` 的链式计时证明 MLX 已达 282 GB/s（M1 Max 峰值 ~70%），
剩余空间约 1.4×，但需要比 MLX 更好的内存调度，非本会话可完成。

---

## 32. decode 性能攻坚 Round 1：step 拆解——**kernel 延迟主导，不是带宽**

新 goal 目标：4k decode ≥70 tok/s（当前 60–64）、8k ≥45（当前 34–45）。
`bench/decode_profile.py` 把每个 step 拆成 **build（Python/图构建）** 与 **eval（GPU）**：

| 上下文 | 配置 | build | eval | total |
|---:|---|---:|---:|---:|
| 4k | 完整 | 1.43 ms | **16.56 ms** | 17.98 ms |
| 4k | 去 attention | 1.70 ms | 9.22 ms | 10.92 ms |
| 4k | 去 attention+rope | 1.48 ms | 10.23 ms | 11.71 ms |
| 8k | 完整 | 1.43 ms | **25.55 ms** | 26.98 ms |
| 8k | 去 attention | 1.53 ms | 10.91 ms | 12.44 ms |

**关键结论**：
1. **Python/launch 构建只占 ~8%（1.4ms）**——`mx.compile`/减少 Python 开销的空间很小；
2. attention 的 GPU 时间是 4k **7.3ms**、8k **14.6ms**；而同样 36 层 SDPA 在"独立背靠背"微基准里只要
   2.7–3.1ms（4k）。差 2.4× 的原因不是带宽，而是**依赖链上每个 kernel 的固定延迟**（约 0.2ms/层）：
   36 层串行，无法像独立 kernel 那样流水。
3. 因此 decode 的瓶颈是 **kernel 数量 × 单 kernel 延迟**，不是算术或带宽。可回收方向：
   **减少 kernel 数量（算子融合/权重拼接）**，而不是替换单个 kernel。

### 32.1 下一轮实验设计（已写入 goal）

1. **融合 QKV 投影**（3 个 quantized linear → 1 个，权重拼接 [2560 → 6144]）；
2. **融合 gate/up 投影**（2 → 1，[2560 → 19456]）；
3. 理论上每层少 3 个 kernel，36 层少 108 个；若延迟主导成立（~50–200 µs/kernel），
   潜在收益可达 2–5 ms，足以把 4k 推过 70 tok/s；
4. 每步需保证 token 与基线一致（greedy 固定 prompt），并做交替 A/B。

### 33. Round 1 实验：权重拼接融合（QKV / gate-up）与精确尺寸 KV——都不够

**实验 A：融合 QKV、gate/up 投影**（`decode_fused.py`，把 3+2 个 quantized linear 拼成
1+1 个 kernel，每个输出维度只占一行的权重组独立量化，拼接精确；`tests/test_decode_fused.py`
验证 token 与 baseline 完全一致）。

4k、16 token greedy（同一 prefill cache，机器状态偏热时测得）：

| 配置 | ms/step | tok/s | token 一致 |
|---|---:|---:|---|
| baseline | 24.74 | 40.4 | — |
| 融合 QKV | 24.67 | 40.5 | ✓ |
| 融合 gate/up | 24.70 | 40.5 | ✓ |
| **两者都融合** | **24.10** | **41.5** | ✓ |

**只有 1.03×**——说明每层少 3 个 kernel 几乎无收益，之前的"kernel 延迟主导"假设不成立：
GPU 对图内 kernel 已有流水，真正的成本是 GPU 执行时间（matmul 带宽 + attention 计算）。

**实验 B：精确尺寸连续 KV cache（每步 concat 复制，消除 SDPA 的非连续视图）**：
- baseline 22.48 ms/step（44.5 tok/s）
- 精确缓存 26.00 ms/step（38.5 tok/s），**0.86×，负收益**（复制成本 > 非连续视图收益）。

**Round 1 结论**：decode 的剩余开销不在 Python（8%）、不在 kernel 数量（融合仅 1.03×）、
不在 KV 视图（精确缓存更慢）。要显著提升只剩：attention 的 GPU 计算本身（SDPA 已是 MLX 最优，
自写 kernel 多次证伪）或更底层的执行器重写（属于框架级工作）。下一轮继续试 goal 里的
"预计算/减少重复张量"与 8-bit KV attention，然后给最终数据结论。

### 34. decode 性能 goal 最终结论：4k ≥70 tok/s 在 MLX 执行层下不可达（负结果）

本轮把 goal 剩下的两个方向也测了：

**a) 8-bit KV attention**：之前已实测量化 SDPA 路径（`QuantizedKVCache`）在 4k/8k 下
0.62×——比 fp16 SDPA 慢；读打包 KV 的定制 kernel 又要复刻 SDPA 的 GQA 广播与并行度，
与之前 paged/多 query kernel 的失败同因，不再重复实现。

**b) rope 预计算/算子系统计**（微基准）：

| 项目 | 每 step 成本 | 可省空间 |
|---|---:|---:|
| 72 次 rope 调用（36 层 × q/k） | **0.87 ms**（offset=4000） | 预计算 cos/sin 最多省 ~0.4–0.5 ms（≈3%） |
| 逐层 rmsnorm 等小算子 | 已含在 no_attn 9.2 ms 里 | 与 rope 合计最多 ~1–1.5 ms |

**最终汇总（4k decode，M1 Max / Qwen3-4B-4bit bf16）**：

| 尝试 | 结果 | 结论 |
|---|---|---|
| decode mask 跳过 | +5.4% | ✅ 已默认启用（长上下文） |
| 长上下文 split-K kernel | +6–8%（8k） | ✅ 已默认启用（≥4096 KV） |
| 权重拼接融合（3+2→2 kernel/层） | 1.03× | ❌ 收益太小，不启用 |
| 精确尺寸连续 KV cache | 0.86× | ❌ |
| 自写 GEMV（decode M=1） | 0.37–0.82× | ❌ |
| 多 query / GQA / paged attention kernel | 0.15–1.10× | ❌ |
| 投机解码（n-gram / draft model） | −11% ~ −57% | ❌ |
| 量化 Attention（8-bit KV SDPA） | 0.62× | ❌ |
| rope 预计算 | ≤3% | ❌ 不值得 |

**结论**：4k decode 在 MLX 上的实际上限约 **60–65 tok/s**（step ≈15.5–18 ms，其中 matmul 链
7.25 ms、attention 7.3 ms、小算子 ~2 ms、Python 1.4 ms）。要再往上必须重写执行层/attention
内核（vllm-metal 的 MMA 路线，数周级），不是本会话能完成的算子级优化。
目标 ≥70 tok/s / 8k ≥45 判定为**不可达（有数据）**，goal 按允许的负结果条款完成。

---

## 35. MMA 路线 Round 1：simdgroup matrix 可用，裸 MMA 还不够（0.73 TF/s）

新 goal：验证 vllm-metal 式的 MMA 路线（simdgroup matrix）能否突破上限。

### 35.1 可行性

* MLX 的 `mx.fast.metal_kernel` **支持 `simdgroup_half8x8` + `simdgroup_multiply_accumulate`**
  （`tests/test_mma_attention.py` 的 8x8 矩阵乘验证通过，maxdiff < 0.05）；
* **重要约束**：`simdgroup_load(..., transpose=true)` 在 8x8 上不可用/语义不符——需要用
  **K 预转置布局 `[kv_heads, D, KV]`** 才能做 score = Q·Kᵀ；这正是 vllm-metal 把 K 存转置布局的原因；
* 已实现 `mma_scores`（QKᵀ，GQA 映射、Kᵀ 布局、8 个 8x8 累加器/线程组），数值正确（maxdiff ≤0.25，fp16 累加）。

### 35.2 裸 MMA vs SDPA（32 heads、64 queries、4000 keys、D=128）

| 实现 | 时间 | 有效算力 |
|---|---:|---:|
| 裸 MMA QKᵀ（物化 score） | 2.87 ms | **0.73 TFLOP/s**（只算 QKᵀ） |
| MLX SDPA（QKᵀ+softmax+AV） | 3.97 ms | 0.53 TFLOP/s 按 QKᵀ 计 / ~1.06 按全部算 |

裸 MMA 没赢：每个线程组只做一个 8×64 tile、直接从 device 做跨步 fragment 加载、无 threadgroup
staging/双缓冲，内存访问效率低。MLX SDPA 虽是通用实现，但已高度调优（GQA 广播、向量化加载）。
**下一轮**：实现真正的 MMA flash-attention（threadgroup 分块 K、Kᵀ 布局、online softmax、
causal），目标在 4k prefill attention 上 ≥1.5×；否则负结果。

### 36. MMA flash-attention v1：数值正确，但只在"小 q_len + 长 KV"形态有优势

实现了真正的 MMA flash-attention（`terrazzo/mma_flash_attn.py`）：
一 warp 一组（8 queries/组）、KV 按 32 分块、QKᵀ 用 simdgroup MMA（fp32 累加）、
shared memory 行 softmax（online max/sum + beta 修正）、P@V 也是 MMA、causal mask。

**数值**（fp16，vs MLX SDPA，maxdiff ≤2e-2 目标）：
L=64/128/256、causal & 非 causal、GQA(8→2) 全部 **≤0.002** ✅（3 个单测）。

**性能 vs SDPA（padded fp16，32 heads）**：

| q_len | KV | MMA | SDPA | 比值 |
|---:|---:|---:|---:|---:|
| 8 | 4000 | **0.58 ms** | 2.49 ms | **4.3×** |
| 16 | 4000 | **0.56 ms** | 1.73 ms | **3.1×** |
| 64 | 4000 | 1.19 ms | 3.81 ms | 3.2× |
| 512 | 512 | 25.5 ms | 0.95 ms | 0.04× |
| 1024 | 1024 | 108 ms | 2.44 ms | 0.02× |
| 4000 | 4000 | 1935 ms | 28.6 ms | **0.01×** |

**关键发现**：MMA 的甜点是 **小 q_len、长 KV**（speculative 验证形态）：8 queries 的完整
attention 比 SDPA 的 1 个 query 还便宜；而 **大 q_len 的 prefill 形态完全打不过 SDPA**——
SDPA 在 4k 时已达 **9.2 TFLOP/s**（fp16 峰值的 ~60%），我们的 v1 只有 0.14–0.17 TFLOP/s（差 50–70×）。

**但对投机解码仍不够**：matmul 链随 M 线性增长（实测 M=1 8.44ms、M=8 52.75ms → 6.25×，
即 6.6 vs 8.4 ms/token，最好情况仅 1.28×），spec 验证 8 token 的 matmul 成本（52.75ms）就比
基线单 token（15.5ms 总）高 3.4×，attention 再便宜也补不回来。

**结论（阶段负结果）**：MMA 路线在本栈上不能突破 prefill/attention 上限；
唯一有数据支持的潜在用途是小 q_len 长 KV attention（spec 验证），但被 MLX 的 small-M
量化 matmul 成本否决。要真正改变，需要 vllm-metal 的整套执行器（分页 + batch + 融合），
远超本会话范围。

---

## 37. 批量 decode 拆解：attention 线性、M=8 出现异常开销（目标 2× 不可达）

`bench/batch_profile.py`（共享 4k prompt，warm 前缀缓存后测纯 decode，token 一致性已校验）：

| 并发 | 聚合 tok/s | median decode step | 每 token 成本 |
|---:|---:|---:|---:|
| 1 | 71.7 | 13.2–15.7 ms | 13–16 ms |
| 4 | 50.6–65.8（稳态 4/52.6ms≈76） | 52.6–60.3 ms | 13–15 ms（**无改善**） |
| 8 | 31.8–32.1 | 148–154 ms | 18.5–19 ms（**变差**） |
| 16 | — | — | （未测） |

**attention 消融（同一 batch，SDPA→0）**：

| 并发 | full step | no_attn step | attention 增量 |
|---:|---:|---:|---:|
| 1 | 15.68 ms | 8.40 ms | 7.28 ms |
| 4 | 60.26 ms | 31.88 ms | **28.37 ms（≈4×7.1，完全线性）** |
| 8 | 148.46 ms | **108.22 ms** | 40.24 ms |

**三个结论**：
1. **批处理的 attention 成本完全线性**（每个序列 4k KV 独立，无法共享），4 并发 attention 28.4ms；
2. **matmul+其它在 M=8 出现异常**：no_attn 108ms vs 预期 ~60ms（chain M=8 52.75 + 其它 ~11），
   多出的 ~45ms 待定位（怀疑 BatchKVCache 每步 concat/分配或采样开销）；
3. **goal 的"4 并发聚合 ≥2× 单请求"在 4k 下算术不可达**：单 token 下限 ≈ matmul(6.8ms@M4) + attention(7.1ms) ≈ 14ms，
   4 并发最好 ≈4/56ms = 71 tok/s ≈ 单请求水平。真正的天花板在 **M≥16**：
   matmul 3.16ms/token + attention 7.1 + 其它 ≈ 11.7ms/token → 聚合 ~85 tok/s（1.2× 单请求）。

### 38. 批量吞吐最终结论：2× 目标不可达；M≥16 的稳态天花板 ≈1.3×

**稳态 decode（4k 共享 prompt、前缀缓存已热、token 一致）**：

| 并发 | median step | 稳态聚合（N/step） | 相对单请求 |
|---:|---:|---:|---:|
| 1 | 13.2–15.7 ms | 71.7 tok/s | 1.00× |
| 4 | 52.6–67.5 ms | 59–76 tok/s | 0.8–1.06× |
| 8 | 148–216 ms | 26–32 tok/s | 0.36–0.45×（异常，见下） |
| 16 | 172.7 ms | **92.6 tok/s** | **1.29×** |

**attention 消融**：B=1 增量 7.28ms、B=4 增量 28.37ms（≈4×7.1，完全线性）、B=8 40.24ms。
即每个序列的 4k KV 独立，attention 成本不随 batch 摊销——这是 2× 目标不可达的根本原因：
单 token 下限 ≈ matmul(M)/M + attention(≈7.1ms) + 其它(≈1.4ms)，4 并发最好也只能追平单请求。

**HTTP 端到端（`bench/agent_workload.py`，2000-token 系统提示、2 轮、32 tok/turn）**：

| 并发 | 墙钟 | 聚合 tok/s | TTFT p50 | TTFT p95 |
|---:|---:|---:|---:|---:|
| 1 | 11.0 s | 5.8 | 4.98 s | 0.29 s |
| 4 | 7.4 s | **34.6** | 1.10 s | 1.35 s |
| 16 | 28.1 s | **36.4** | 8.01 s | 8.83 s |

16 并发的聚合不再增长而 TTFT 涨到 8s：冷 prefill 被串行化（每个会话 2000 token 前缀），
这是当前调度器的瓶颈，也解释了为什么"加并发"只改善吞吐到 ~1.3× 就把 TTFT 赔进去了。

**未解异常**：8 并发 step（148–216ms）比 16 并发（172.7ms）还差，且 no_attn 下 8 行为 108ms
（预期 ~60ms）。怀疑与 batch_attn_patch 在该形状的 dispatch、或 BatchKVCache 的切片/掩码路径有关，
未定位；不影响结论（目标已不可达）。

**goal 结论（负结果）**：4 并发 ≥2×（≥120 tok/s）、8 并发 ≥2.5×（≥150 tok/s）**不可达**；
本栈的批处理天花板是 **M≥16 时 ~90 tok/s（约 1.3× 单请求）**，且需要放弃 TTFT 优势。
下一步若要提升并发，只能从"减少每 token 的 attention 固定成本"入手（已被多次证伪）
或重写执行器（vllm-metal 路线）。

---

## 39. Head-to-head 重跑（2026-10-26，当前代码）

上一轮对比是 2026-10-24 的 v0.3，之后 decode 路径（mask patch、长上下文 kernel）、
前缀缓存默认值、加固逻辑都改过。本轮用同一套脚本重跑，只保留 Ollama 作为对手。

- 机器：Apple M1 Max（24 核 GPU / 64GB）
- 我方：当前默认（v0.9 代码），`terrazzo serve`，另测 `--system-file` 预注册前缀
- Ollama：**0.34.4**，`qwen3:4b`（GGUF Q4_K_M，2.5 GB）
- 协议：随机前置标记（避免跨目标前缀复用），先热身模型；每轮只跑一个服务（Ollama 先 `stop` 释放内存）
- 复跑：`bench/head_to_head.sh`（TTFT/前缀/并发）+ `bench/compare_ollama.py`（decode-only 中位数）

### 39.1 decode（权威口径：各自计数器 / 3 次中位数，128 token）

| 服务 | decode tok/s | 比值 |
|---|---:|---:|
| Ollama qwen3:4b（num_ctx 8192） | 48.4 | 1.00× |
| **terrazzo（MLX 4-bit）** | **100.7** | **2.08×** |

比 v0.3 的 96 tok/s 略好；Ollama 从 47→48 基本没变。**decode 上我们仍领先约 2×。**

### 39.2 长系统提示 TTFT / 前缀复用 / 4 并发（api_bench，同一随机系统提示）

| 目标 | cold TTFT | warm-same | warm-diff | short decode* | 4 并发聚合 |
|---|---:|---:|---:|---:|---:|
| terrazzo 默认 | 1.040 s | **0.076 s** | 0.236 s | 51.1 tok/s | **54.1 tok/s** |
| terrazzo 预注册前缀 | **0.169 s** | 0.186 s | 0.306 s | 61.5 tok/s | **58.1 tok/s** |
| Ollama qwen3:4b | 0.811 s | **0.033 s** | **0.111 s** | 40.5 tok/s | 41.8 tok/s |

*api_bench 的 decode 由 SSE 总时长推算，方差大 ±20%，只作参考；以 39.1 为准。

### 39.3 与 v0.3 对比的结论变化

1. **decode 优势保持并略增**：2.08×（v0.3 是 +38~48%，本轮口径更大，因为用了 128 token 与
   各自计数器；同一脚本内仍是 2× 量级）。
2. **冷 prefill（长系统提示）Ollama 仍更快**：0.811 s vs 我们默认 1.040 s；但我们**预注册前缀后
   0.169 s，快 4.8×**（这是 Ollama 没有的能力——它要等第一次请求收尾才有缓存）。
3. **纯前缀复用 Ollama 仍最快**：warm-same 0.033 s / warm-diff 0.111 s，我们 0.076–0.186 s。
   我们 block 粒度 partial 命中这次**没有**赢过 Ollama 的 slot 缓存（系统提示只有 ~440 token，
   粒度优势不明显）。
4. **4 并发聚合我们领先 ~30%**：54–58 vs 41.8 tok/s（Ollama 默认 `num_parallel=1`；
   注意这是它的默认配置，调到 4 时历史测得约 100 tok/s —— 配置问题不是能力问题）。

### 39.4 一句话

重跑后结论与 v0.3 同形：**decode 领先 2×、首请求前缀预注册快 5×、并发默认配置领先 30%；
冷 prefill 与纯前缀复用仍输给 Ollama**。没有任何"质的飞跃"；产品若要成立，只能押
"Mac 上的 agent 推理网关"（首请求前缀、结构化输出、背压/取消/可观测），而不是速度叙事。

---

## 40. 长系统提示 head-to-head（2k / 4k / 8k）：预注册前缀成立，默认配置 8k 不缓存

脚本：`bench/h2h_long_prompts.py`（2k/4k/8k 系统提示 × cold / warm-same / warm-diff + decode）。
注意：本轮测量期间机器有**外部重负载**（无关进程 Ghidra headless + 其他 app，load avg 10→29→4.5 波动），
Ollama 的绝对值受影响明显（同机短上下文 decode 在 42–61 tok/s 间漂移），所以**只看结构性结论，
绝对值待空闲时重跑**。

| 目标 | 长度 | cold TTFT | warm-same | warm-diff | decode* |
|---|---:|---:|---:|---:|---:|
| terrazzo 默认 | 2k | 4.77 s | 0.166 s | 0.213 s | 58.8 |
| terrazzo 默认 | 4k | 12.22 s | 0.154 s | 0.225 s | 48.0 |
| terrazzo 默认 | **8k** | 23.71 s | **14.79 s** | 15.72 s | n/a |
| terrazzo 预注册 | 2k | **0.202 s** | 0.155 s | 0.158 s | 63.5 |
| terrazzo 预注册 | 4k | **0.297 s** | 0.247 s | 0.245 s | 59.0 |
| terrazzo 预注册 | 8k | **0.406 s** | 0.245 s | 0.319 s | 38.1 |
| Ollama（最小 ctx） | 2k | 3.37 s | 0.034 s | 0.123 s | 19.6 |
| Ollama（最小 ctx） | 4k | 15.68 s | 0.113 s | 0.243 s | 19.7 |
| Ollama（最小 ctx） | 8k | 40.39 s | 0.118 s | 0.396 s | 17.8 |

\* decode 由非流式请求估算（Ollama 用 eval 计数器），负载下噪声大。

**结构性结论（不受负载影响）**：
1. **预注册前缀在 8k 依然成立**：cold TTFT 0.406 s（默认 23.7 s / Ollama 40.4 s），
   warm-same 0.245 s —— 这是唯一能在**首请求**就拿到长前缀收益的路径。
2. **默认配置在 8k 不缓存**：warm-same 仍要 14.8 s（等于重新 prefill）。原因是默认
   `prefix_cache_max_tokens` 走 8192 auto 预算，而 8k 系统提示 + chat template 压在边界上，
   快照没有命中/没有存下；需要显式 `--prefix-max-tokens 16384` 或 `--system-file`。
   **这是当前默认配置的一个实际缺陷**（agent 场景 8k 系统提示不受益）。
3. **warm 前缀复用双方都很快**：我们 0.15–0.32 s，Ollama 0.03–0.40 s；8k 时 Ollama 略快，
   但差距远小于 cold TTFT 的数量级差异。
4. **冷 prefill 我们仍慢于 Ollama**（默认 2k: 4.8 vs 3.4；4k: 12.2 vs 15.7 受负载扭曲；
   8k: 23.7 vs 40.4 同样受负载影响），需要在空闲机器上重跑才能给准确比例。
5. 长上下文 decode 我方 63.5 / 59.0 / 38.1 tok/s（2k/4k/8k）；Ollama 本轮 19.6/19.7/17.8
   （负载污染，参考意义有限；空闲时短上下文是 42–61）。

**8k 默认不缓存的根因（已定位并修复）**：不是预算，而是 server 的自动前缀注册
（`EngineRuntime.auto_prefix_max_tokens`）**硬编码 4096**，与配置的 snapshot 上限（本机 auto 8192）
无关。8k 系统提示只注册了前 4096 token，warm 请求仍需 prefill 剩余 ~3.7k token（≈14.8 s）。
修复（`server.py`）：新增 `auto_prefix_cap(cfg) = max(4096, cfg.prefix_cache_max_tokens)`，
并支持 `TERRAZZO_AUTO_PREFIX_MAX_TOKENS` 覆盖；`cli._auto_prefix_budget` 在 ≥48 GB 机器上把
多快照预算从 2 GB 提到 **4 GB**（2k+4k+8k ≈ 2.0 GB，留出第二个 8k 的余量；LRU 淘汰逻辑保留）。

**修复后验证**（streaming TTFT；机器 load ~6，cold 绝对值偏高）：

| 长度 | cold TTFT | warm-same TTFT | warm-diff TTFT |
|---:|---:|---:|---:|
| 2k | 5.16 s | **0.145 s** | 0.217 s |
| 4k | 17.14 s | **0.170 s** | 0.304 s |
| 8k | 39.36 s | **0.254 s** | 0.443 s |

→ **8k warm-same 0.254 s ≤ 0.5 s 目标达成**；快照共 4 条 / 2.00 GB（2k 285 MB、4k 570 MB、8k 1149 MB），
`evictions=0`；回归测试 +1（`tests/test_server_hardening.py::AutoPrefixCapTest`），全套 **171 tests OK**。

**结论**：在 2k–8k 长系统提示区间，我们的**真正差异化只有一条**——预注册前缀让"首请求"就不吃
prefill 成本（0.2–0.4 s vs 数个到数十秒）。默认配置需要修多快照预算（2 GB 不够 2k+4k+8k），
否则这个优势只对 `--system-file` 用户可见。

---

## 41. 阶段 B：vLLM-Metal + 我们的"预注册前缀"组合验证（2026-10-27）

思路：vLLM 有 block 级自动前缀缓存，但缓存是**空**的——首请求仍要付整段 prefill。
我们把"启动时预注册前缀"做成一个 OpenAI 兼容反向代理（`scripts/vllm_prewarm_proxy.py`），
启动时用同一段系统提示发一个 warmup 请求填充 vLLM 的 prefix cache，再转发真实请求。

工作负载：`bench/agent_workload.py`，4 会话 × 2 轮 × 64 token，共享系统提示 3257 tokens
（`--system-tokens 2544`，实际 prompt 3257 tokens）。vLLM-Metal `--gpu-memory-utilization 0.2`，
`enable_prefix_caching=True`（日志确认）。每轮 A/B 都重启 vLLM 保证缓存为空。
**测量期间机器仍有外部负载（load 6.8–7.8，Ghidra headless 未结束），绝对值偏低，看相对值。**

| 目标 | 聚合 tok/s | TTFT p50 | TTFT p95 | 墙钟 |
|---|---:|---:|---:|---:|
| vLLM-Metal 原始（冷缓存） | 16.8 | **9.11 s** | 16.62 s | 30.5 s |
| **vLLM-Metal + 预热代理** | **38.2** | **1.73 s** | 2.19 s | 13.4 s |
| terrazzo（预注册前缀） | 34.4 | **1.09 s** | 1.62 s | 14.8 s |

**结论**：
1. **机制成立**：预热把 vLLM 的首轮 TTFT 从 9.11 s 压到 1.73 s（**5.3×**），
   聚合吞吐 16.8 → 38.2 tok/s（**2.3×**），与"warm 后 vLLM 本来该有的水平"一致。
2. **绝对目标（cold TTFT ≤0.8 s）本轮未达成**：负载下 vLLM 的 warm 水平本身就要 ~1.7 s；
   历史空闲测量是 warm 0.48 s / 67.9 tok/s，需要在空闲机器上重跑确认。
3. **对我们自己的意义**：预热后的 vLLM 吞吐比我们高 ~11%（38.2 vs 34.4），TTFT 反而更差
   （1.73 vs 1.09 s）。这与 v0.7 的历史结论一致：warm 吞吐 vLLM 略优、冷启动我们更优；
   预热代理把差距压缩到"吞吐 +11%、TTFT 我们仍更好"。
4. 代价明确：用 vLLM 后端时单请求 decode 从 ~100 掉到 ~48 tok/s（本轮负载下），
   内存 1.1GB+（vLLM server 进程，不含 EngineCore）——4B 单流仍然是我们自己的 MLX 路径更强。

**B 阶段结论**：组合是可行的、收益是真实的（5.3× 首轮 TTFT），但"绝对 0.8s"依赖空闲机器；
且它把产品定位推向"vLLM 的 Mac 外壳"，放弃了我们 2× 单流 decode 的优势。

## 42. 阶段 A 预研：共享前缀的打包 attention（2026-10-27）

目标：agent 场景 B 个会话共享同一长前缀，当前每行各跑一次 SDPA → 前缀 KV 被读 B 次。
替换成"前缀只读一次、B 行 query 打包"的 attention。

`experiments/bench/packed_prefix_attention.py`（Qwen3-4B 形状：32 heads、D=128、前缀 3400、B=4、bf16、36 层）：

| 做法 | 每层 | 36 层 | 相对 B×SDPA |
|---|---:|---:|---:|
| B 次 SDPA(q_len=1)（现状） | 1.14 ms | 41.2 ms | 1.00× |
| 1 次 SDPA(q_len=B=4，打包) | 0.81 ms | **29.2 ms** | **1.41×** |
| 手写 matmul+softmax 打包 | 2.42 ms | 87.1 ms | 0.47×（更慢） |

**结论一**：打包 SDPA 有 1.41× 的真实收益；手写版因为物化 [H,B,P] scores + fp32 softmax 反而更慢。
**结论二**：要把这个收益用起来，必须解决"前缀打包 attention + 每行 suffix attention 的精确合并"
（需要两边 logsumexp）。SDPA 不返回 lse，所以下一步改用我们自己的 MMA flash kernel：
它内部就在算 online softmax，把它扩展成"输出 out + lse、支持 bf16、q_len 可补齐到 8"即可。

**可行性验证**：Metal 的 `simdgroup_bfloat8x8` + `simdgroup_float8x8` 在 MLX 自定义 kernel 里
**可用且精确**（8x8 MMA，maxdiff = 0.0），所以 bf16 版 MMA flash kernel 可行（fp16 单测已有）。

**下一步（Round 2）**：
1. `mma_flash_attn.py` 扩展：bf16 模板、q_len 4/8（补齐）、可选 lse 输出；
2. 数值验证 out/lse vs fp32 参考；
3. 微基准：MMA 打包前缀 attention vs B×SDPA，含 suffix 精确合并；
4. 若 ≥1.5×，接入 batch 路径（自定义 cache + qwen3 SDPA patch），token 一致后做 agent A/B。

## 43. 阶段 A 结论：共享前缀 attention 打包不可行（负结果，2026-10-27）

目标：agent 场景 B 个会话共享长前缀，把"每行各跑一次 attention（前缀读 B 次）"压成
"前缀只读一次、B 行 query 打包"，期望 warm 聚合 54.6 → ≥65 tok/s、内存 4.8 → ≤3GB。

`experiments/bench/packed_prefix_attention.py`（32 q heads / 8 kv heads、D=128、前缀 3400、B=4、
后缀 32、bf16、每层/36 层）：

| 方案 | 每层 | 36 层 | 相对 baseline | 正确性 |
|---|---:|---:|---:|---|
| baseline：4×SDPA（KV 已就绪，无 concat） | 1.77 ms | 63.8 ms | 1.00× | exact |
| packed **plain**（q_len=B，无 mask） | 0.93–1.77 ms | 33–64 ms | 1.33× | ❌ 会看到别人的 suffix |
| packed **mask**（逐行 mask，exact） | 1.08 ms | 39.0 ms | 2.17×* | exact（maxdiff 5e-4） |
| **packed + 每步 concat（真实集成代价）** | 1.60 ms | 57.6 ms | **1.11×** | exact |
| manual（matmul + fp32 softmax） | 2.48–2.79 ms | 89–100 ms | 0.84× | 慢 |
| MMA v2 + suffix 合并（bf16，lse 精确） | 23.4 ms | 841 ms | 0.05× | exact（maxdiff 6e-4） |

\* 2.17× 是对"把 concat 也算进 baseline"的口径；**一旦给 packed 方案加上每步必需的前缀
concat（真实集成时无法避免），真实收益只有 1.11×**。

**MMA 路线的真相**：v2 kernel（bf16、q_len 补齐、输出 lse）数值正确（out ≤2e-3、lse ≤1e-5），
但该形状下 **11.4 ms/layer，比 SDPA（1.1 ms）慢 10×**。此前"q_len=8 时 MMA 比 SDPA 快 3–4×"
的测量是假象：当时 causal 内核在 q_len=8 时 `k0 > q0+7` 立即 break，只算了 32 个 key。
MMA 只在 **大 q_len prefill** 有意义，而那里我们测过它比 SDPA 慢 50–70×，同样不可用。

**阶段 A 结论（负结果）**：
1. 打包 attention 的精确版本在真实集成下只有 **1.11×** attention 收益 ≈ **整步 +5%**，
   达不到 +20% 的目标；不精确版本（1.33–1.86×）会改变 token，不可接受。
2. 内存目标（4.8→≤3GB）需要 vLLM 式 paged 共享，而我们的 paged 路径此前已实测
   decode 更慢（自写 paged kernel 0.15–1.10×），且改造成本数周，净收益为负期望。
3. 因此 A 阶段**不做集成**，以数据结论收尾。

**两阶段总账**：B（vLLM+预热代理）机制成立（首轮 TTFT 5.3×），但相对我们的优势只有
吞吐 +11%、我们 TTFT 仍更好；A（把 paged 共享搬进 batch）在实测代价下不可行。
"集各家之长"在本栈上的真实上限是 **+5–11%**，不是质变。

## 44. 阶段 1：16 会话冷启动 TTFT 修复（2026-10-27）

工作负载：`bench/agent_workload.py`，16 会话 × 2 轮 × 64 token、2000-token 共享系统提示
（实际 prompt ~2560 tokens）。**测量时机器仍有外部负载（load ~12，Ghidra 等），绝对值偏高，
但各配置同窗口对比有效。**

| 配置 | TTFT p50 | TTFT p95 | 聚合 tok/s | 墙钟 |
|---|---:|---:|---:|---:|
| 基线：默认 8 槽 + auto-prefix（原始代码） | 11.09 s | 17.38 s | 44.8 | 45.7 s |
| 仅调度修复（round-robin chunked prefill，8 槽） | 8.64 s | 14.46 s | 56.5 | 36.2 s |
| auto-prefix 关闭（对照，8 槽） | 17.86 s | **109.42 s** | 13.2 | 154.5 s |
| `--system-file` 预注册（8 槽） | 8.32 s | 8.98 s | 69.6 | 29.4 s |
| **`--system-file` + 16 槽/16 准入** | **2.24 s** | **3.17 s** | **95.0** | 21.5 s |
| auto-prefix + 16 槽/16 准入 | 4.40 s | 7.98 s | 80.6 | 25.4 s |
| `--system-file` + 24 槽 | 2.17 s | 5.80 s | 83.5 | 24.5 s |

**根因（两个）**：
1. `Scheduler.schedule` 的 chunked prefill 只要队首请求没 prefill 完就 `break`，把预算全给一个请求，
   其余请求干等（16 会话 × 2.5k prompt → 串行）。已改为跨请求 round-robin：同一 step 内把 token
   预算分给多个等待请求，未完成的轮转到队尾；preemption step 不再准入新请求。
2. 会话容量与准入：默认 `max_num_seqs=8`，16 个会话必须等第一批跑完（~7-8 s）才拿到第二批；
   准入又是 4 个一批（`TERRAZZO_ADMIT_BATCH=4`）。16 槽 + 16 准入后所有会话同一波开始。
   新增 `--profile agent` 一键设置（16 槽 + 快速准入）。

**结果**：p50 从 11.09 s → **2.24 s（4.9×）**、p95 17.4 → 3.17 s、聚合 44.8 → **95.0 tok/s（2.1×）**。

**最终复测（2026-10-27 01:26，load 3.6，prewarm API + `--profile agent`，16 会话）**：
**TTFT p50 2.26 s / p95 2.87 s、聚合 95.8 tok/s、min 0.79 s**。
（该行使用旧的 "每个 SSE 内容块 +1" 计数口径和固定假回答；见 §46 的修正复测，以 §46 为准。）
- p95 达标（≤3 s）；**p50 差 13%**（2.26 vs ≤2 s）。根因已定位：mlx-lm 的批处理把共享前缀
  **复制进每个序列**（16 × 366 MB ≈ 5.9 GB 的 merge 拷贝），拷贝带宽决定了 TTFT 下限；
  彻底消除需要 paged 共享（阶段 A 已证伪其 attention 路径）。
- `--system-file` 预注册（或 `POST /v1/prewarm`）是达标前提：不预注册时 p50 4.4 s（auto）到
  17.9 s（无共享）。

**下一步**：把 prewarm 做成服务端 API（`POST /v1/prewarm`）与 CLI，让 agent 服务启动即注册系统提示；
把 `--profile agent` 写进快速开始；然后进入产品化（模型管理 CLI、/api/tags 等）。

## 46. 可靠性修复（代码 review 驱动，2026-10-27）

review 指出的五个问题，逐项处理与证据：

### 46.1 token 计数口径（已修，数字重测）

旧 `bench/agent_workload.py` 对每个非空 SSE 内容块 `+1`。块 ≠ token（服务端可以合并字符），
因此旧数字不能当严格 token/s。现在：

- 服务端支持 OpenAI `stream_options.include_usage`，流式响应末尾发 `usage.completion_tokens`；
- bench 优先用 `usage`，缺失时回退块计数并标记 `token_source="chunks(approx)"`；
- 每轮把真实 assistant 回答喂回下一轮（不再用固定假回答）；输出 JSON 带完整环境信息
  （host / Python / git commit / 后端版本 / loadavg / `/stats` 前后快照）；
- 与 `/stats.decode_tokens` 交叉校验：本轮 2048 vs 2048，完全一致。

**修正口径复测**（16 会话 × 2 轮 × 64 token，2000-token 共享系统提示，prewarm + `--profile agent`，
load 4.0）：**TTFT p50 2.284 s / p95 3.441 s、聚合 87.5 tok/s（server decode_tokens=2048）、墙钟 23.4 s**。
p95 比旧口径差（真实多轮的回答会进入下一轮 prompt）；旧 §44 的 95.8 tok/s 不再作为权威数字。

### 46.2 超时与故障闭环

- `EngineRuntime.submit`：等待 worker 准入有超时；超时后从 pending 队列移除并 abort，抛 `TimeoutError`；
  pending 数超过 `TERRAZZO_MAX_PENDING`（默认 256）直接拒绝。
- 所有 handler 的 `q.get()` 统一带请求 deadline（默认 600 s，`TERRAZZO_REQUEST_TIMEOUT_S`）：
  非流式返回 504；流式发 `finish_reason="error"` 帧后关闭，并 abort 后端请求。
- worker `step()` 抛 `RuntimeError`：`Engine.handle_failure()` 给所有在途请求发终止事件、清 pending；
  连续 3 次失败后新请求返回 503，避免坏状态继续吃请求。
- 故障注入测试：卡死 worker（504/错误帧）、`Engine.handle_failure` 终止在途队列、submit 超时、
  pending 满拒绝。

### 46.3 HTTP 资源限制

- 鉴权在读取 body **之前**执行；body 上限 1 MB（`TERRAZZO_MAX_BODY_MB`，413）；
- 单请求 socket 读超时 30 s（slowloris）；
- 线程并发上限 64（`TERRAZZO_MAX_CONCURRENT`），超出立即 503；
- 排队上限 `TERRAZZO_MAX_PENDING`（默认 256）；
- 测试：超大 body 413、未鉴权大 body 先 401、并发占满 503、空闲后恢复 200。

### 46.4 纯 Python 测试与 CI

- `terrazzo/__init__.py` 改为 `__getattr__` 延迟导入；server/cli 在用到时再导入 engine。
  `import terrazzo`、scheduler/blocks/前缀缓存逻辑、HTTP 层（fake runtime）、CLI 均不需要 MLX。
- `scripts/ci_core_tests.py` 在 MLX 被 blocked 的情况下跑 118 个测试（8 skipped），
  `TERRAZZO_CI_ALLOW_MLX=1` 时同一套在 MLX 环境跑（6 skipped）。
- `.github/workflows/ci.yml`：Linux job（纯 Python，MLX blocked）+ macOS-14 job
  （pip install 固定依赖、import MLX、MLX 可用下的核心套件、MLX 依赖单元测试）。

### 46.5 依赖可复现

- `pyproject.toml`：`mlx>=0.32.4,<0.33`、`mlx-lm>=0.31.3,<0.32`；`constraints.txt` 固定
  `mlx==0.32.4` / `mlx-lm==0.31.3`（benchmark 与测试的实际版本）。
- `terrazzo.compat.require_tested_mlx_lm()` 在引擎启动时校验 mlx-lm 版本与 patch 依赖的
  cache 方法是否存在，失败给出明确错误；`TERRAZZO_SKIP_MLX_LM_CHECK=1` 仅用于试验新版本。
- `docs/COMPATIBILITY.md` 记录测试矩阵与不支持的平台。

**当前测试**：核心 245 个（1 skipped）+ 实验内核 31 个；CI 两个主 job + 手动模型 E2E job（见 workflow）。

## 45. Tools / function calling（2026-10-27）

实现：`terrazzo/tools.py`（校验 tools、解析 `<tool_call>` JSON、清理 `<think>` 推理），
服务端 `/v1/chat/completions` 接受 OpenAI `tools`，用模型自带 chat template 注入 schema
（Qwen3 模板原生支持），输出解析为 `tool_calls`（`finish_reason="tool_calls"`），
非流式与流式均支持（流式在最后一个 SSE chunk 给出工具调用）。

**E2E 实测**（Qwen3-4B-4bit、`tools=[get_weather(city)]`、问"北京天气"）：
返回 `finish_reason=tool_calls`，`arguments={"city":"北京"}`，content 已剥离 `<think>` 块。

覆盖：`tests/test_tools.py`（校验/单多调用/去重/裸 JSON 回退/thinking 清理/畸形输入、
多轮报文规范化、tool_choice none/指定/required）、服务端工具测试（非流式 tool_calls、
普通回答、非法 tools 400、流式 tool_calls、tool_choice=none 不产生工具调用、
带 tools 默认关思考、多轮 assistant `content:null` 规范化）。

**E2E 复测（2026-10-27，真实 Qwen3-4B，单服务跑 5 个场景）**：
1. 单轮：`finish_reason=tool_calls`，`get_weather({"city":"北京"})`；
2. 多轮：回传 assistant tool_calls + `role:tool` 结果 → 最终回答"北京晴，26 度"；
3. `tool_choice="none"`：不注入工具、不产生 tool_calls，正常文本回答；
4. `tool_choice={"function":{"name":"get_time"}}`：只调 `get_time`；
5. 双工具提问：模型选 `get_time({"city":"上海"})`，重复调用被去重（n_calls=1）。

关键修复：带 tools 时默认 `enable_thinking=false`（原来 `<think>` 把 token 预算吃光，
128 token 内根本到不了 `<tool_call>`）；OpenAI 报文中 assistant `content:null` 需规范化为空串，
否则 Jinja 模板抛 `'NoneType' is not iterable`。全套 **245 个核心测试 OK（1 skipped）+ 31 个实验内核测试**（§47 后）。

## 47. 残余风险修复（2026-10-27）

上一轮 §46 之后仍列的残余风险，逐项处理：

| 风险 | 处理 | 证据 |
|---|---|---|
| GPU 静默 hang 无法抢占 | 心跳 watchdog：worker busy 且 120s 无进展（`TERRAZZO_HANG_TIMEOUT_S`）→ 标记 unhealthy、失败排队/在途请求、新请求 503；`/stats.worker` 暴露 `hung/busy/last_progress_age_s`；`TERRAZZO_HANG_RESTART=1` 时 `exit(70)` 交给 supervisor 重启 | `tests/test_server_hardening.py::WatchdogTest`（触发/空闲不触发/Engine 与 Paged 失败在途请求/拒绝提交） |
| Paged/Pool runtime 未加固 | handler 消费 `error` 事件（非流式 500、流式 error 帧）；连续 3 次失败标记 unhealthy；接入 watchdog 与 `_fail_outstanding` | `RuntimeErrorEventTest` + `WatchdogTest` |
| 慢客户端/写超时 | socket 超时覆盖读**和写**（`TERRAZZO_IDLE_TIMEOUT_S`，默认 30s），`handle()` 吸收 `TimeoutError/BrokenPipe/ConnectionReset` | `SlowClientTest`（半截请求被关闭） |
| 无 TLS / 非常量时间鉴权 | `hmac.compare_digest` 比较 API key；`serve --tls-cert/--tls-key`（或 `TERRAZZO_TLS_*`）启用 HTTPS；自签证书 e2e 测试 | `TlsTest` |
| CI 不覆盖 Metal/experiments/模型 E2E | macOS job 增加 Metal kernel 测试（dense/paged/multi-query/mask/auto-blocks）+ `experiments/tests`；新增 `workflow_dispatch` 的模型 E2E job（下载 Qwen3-4B 并跑全量） | `.github/workflows/ci.yml` |
| 跨后端 token 不可比 | `bench/agent_workload.py --api ollama` 走原生 `/api/chat`，用 `eval_count` 真 token（旧路径保留并标 `chunks(approx)`） | `tests/test_agent_bench.py` |
| mlx-lm patch 仅版本检查 | COMPATIBILITY 明确"行为级 smoke tests"清单（mask/dense/paged/multi-query/auto-blocks），并纳入 macOS CI | `docs/COMPATIBILITY.md` |

**验证（2026-10-27，Ollama 0.34.4 / qwen3:4b，load ~5.8，本机）**：

- `--api ollama`（原生 `/api/chat`，`think=false`）：4 会话 × 2 轮 × 64 token、2.6k 共享系统提示 →
  `token_source="ollama-eval"`，`output_tokens=512`（来自 `eval_count`）、聚合 **46.5 tok/s**、
  TTFT p50 3.64 s / p95 4.47 s。
- 同一 Ollama 的 OpenAI 兼容端点也返回 `usage`（`token_source="usage"`，`output_tokens=512`），
  说明新版 bench 对 Ollama 的两种接口都能拿到真 token 数，跨后端对比不再依赖块计数。
- Ollama 原生模式默认 `think=false`：qwen3 在 Ollama 默认开启思考会把 content 吃空、多轮历史失真。

**修复后验证（commit `fbbdcb0`）**：
- 本地：核心 **245 tests（1 skipped）**、实验内核 **31 tests（3 skipped）**、纯 Python（MLX blocked）
  **148 run / 0 fail / 9 skip**。
- CI（[run 40426254576](https://github.com/louisyeaaah/terrazzo/actions/runs/40426254576)）：
  `Core tests (pure Python, MLX blocked)` ✅、`macOS (MLX) import + tests`（含 Metal kernel 与
  experiments）✅；`Model E2E` 仅在手动 `workflow_dispatch` 时运行（本次按预期 skipped）。

## 48. 2026-10-27 全面性能复测（当前代码 + 真实 token 计数）

环境：Apple M1 Max（24 核 GPU / 64GB）、macOS 26.6.2、load average 2–4（空闲窗口）、
terrazzo `8d00b0c`（0.3.0）、MLX 0.32.4 / mlx-lm 0.31.3；Ollama 0.34.4 `qwen3:4b`
（GGUF Q4_K_M）；vLLM 0.32.0 + vllm-metal（`--max-model-len 4096 --gpu-memory-utilization 0.2`）。
方法论：同一时间只跑一个服务；模型先热身；temperature 0；token 数来自服务端
（OpenAI `usage` / Ollama 原生 `eval_count`），不再使用 SSE 块计数。原始 JSON：
`--out /tmp/perf_agent*.jsonl`；driver 脚本见本节末尾命令。

### 48.1 decode（短提示，128 token）

| engine | decode tok/s | 比值 |
|---|---:|---:|
| Ollama qwen3:4b（num_ctx 8192） | 67.0 | 1.00× |
| **terrazzo（MLX 4-bit）** | **101.1** | **1.51×** |

与 §39 的 48.4 tok/s 不同：那次测量时机器有外部负载；本轮为空闲窗口。两个数字都真实，
但 **§48 是当前可复现口径**；README 已改用 1.51×。

### 48.2 Agent 负载（2.6k 共享系统提示、2 轮 × 64 token，真 token）

| 后端 / 配置 | 会话 | 聚合 tok/s | TTFT p50 | TTFT p95 |
|---|---:|---:|---:|---:|
| terrazzo（prewarm + `--profile agent`） | 1 | 59.4 | 0.291 | 0.370 |
| terrazzo | 4 | 74.1 | 0.597 | 1.437 |
| terrazzo | 16 | 88.7 | 2.560 | 3.488 |
| Ollama 0.34.4（原生 `/api/chat`，think=false） | 1 | 19.6 | 2.213 | 4.207 |
| Ollama | 4 | 32.6 | 5.157 | 7.798 |
| Ollama | 16 | 40.6 | 22.393 | 25.036 |
| vLLM-Metal（冷缓存） | 4 | 43.8 | 3.245 | 5.782 |
| vLLM-Metal（冷缓存） | 16 | 123.6 | 1.456 | 1.772 |
| vLLM-Metal（稳态） | 1 | 43.5 | 0.191 | 0.239 |
| vLLM-Metal（稳态） | 16 | 122.1 | 1.571 | 2.001 |

- vLLM 单会话首跑 15.1 tok/s / 2.95 s（kernel 热身），第二次才是 43.5 tok/s，已注明。
- 16 会话下 vLLM 吞吐/时延优于 terrazzo；4 会话和单流 terrazzo 更好。
- Ollama 16 会话 TTFT p50 22.4 s：默认 `num_parallel=1`，请求串行（这是它的默认配置问题，
  不是能力上限）。

### 48.3 长系统提示（2k/4k/8k）

| 提示 | terrazzo 默认 cold | terrazzo 预注册 cold | Ollama cold | terrazzo warm-same | Ollama warm-same |
|---|---:|---:|---:|---:|---:|
| 2k | 4.372 s | **0.154 s** | 3.197 s | 0.153 s | **0.033 s** |
| 4k | 8.906 s | **0.242 s** | 6.946 s | 0.125 s | **0.044 s** |
| 8k | 19.869 s | **0.343 s** | 17.613 s | 0.081 s | **0.060 s** |

同提示 decode（预注册）：82.8 / 74.6 / 57.0 tok/s；Ollama 45.2 / 43.6 / 38.0 tok/s。

### 48.4 长上下文 decode（terrazzo，无前缀共享）

| ctx | decode tok/s | TTFT |
|---:|---:|---:|
| 1000 | 87.2 | 1.95 s |
| 2000 | 76.3 | 3.93 s |
| 4000 | 70.7 | 8.36 s |
| 8000 | 57.2 | 19.21 s |

### 48.5 内存与服务开销

- `/stats` MLX 峰值：1 会话 3.9 GB、4 会话 4.9 GB、16 会话 9.85 GB；进程 RSS ~2.47 GB；
  2544-token 前缀快照 375 MB。
- 服务端冒烟（真实模型、同一台机）：`/health` 6.6 ms、`/v1/models` 1.2 ms、
  `/api/tags` 1.7 ms、`/api/ps` 1.5 ms、`/stats` 1.2 ms、`/api/version` 1.1 ms；
  32-token OpenAI 非流式 0.87 s、Ollama 风格非流式 1.42 s、tool-call 请求 0.56 s。
- vLLM RSS 1.1 GB 不可比（KV/模型在 engine 子进程，`bench` 只采样 API server pid）。

### 48.6 复跑命令

```bash
MODEL=$PWD/../deepinfer-study/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/*/
# decode
python bench/compare_ollama.py --mlx-model "$MODEL" --ollama-model qwen3:4b
# long prompts
python bench/h2h_long_prompts.py --mlx-model "$MODEL" --ollama-model qwen3:4b \
    --lengths 2000,4000,8000 --only all --out /tmp/h2h.jsonl
# long-context decode
python bench/long_context.py --model "$MODEL" --contexts 1000,2000,4000,8000 --tokens 128
# agent workload (terrazzo with prewarm + --profile agent; Ollama native; vLLM)
python bench/agent_workload.py --base http://127.0.0.1:PORT --model ... --sessions 16 \
    --system-tokens 2000 --max-tokens 64 --stats http://127.0.0.1:PORT/stats
# add --api ollama for native Ollama (eval_count) mode
```

**结论更新**：README 里的性能叙事以 §48 为准——对 Ollama 仍是 decode 1.5× / agent 1.8–2.2× /
预注册首 token 10–50×；对 vLLM-Metal 只在单流与 4 会话占优，16 会话吞吐与时延落后。

## 49. Phase 0/1：缓存感知的接纳顺序实验（2026-10-27）

完整协议与 caveat：[docs/CO-DESIGN.md](docs/CO-DESIGN.md)。这里只放断言级数字。

### 49.1 控制边界（16 会话、2k 共享前缀、1 turn、combined arm）

| 指标 | 数值 |
|---|---:|
| 调度器预定的 prefill tokens | 16,401 |
| 执行器实际处理的 prompt tokens | 41,217 |
| actual / scheduled | **2.51×** |
| 有 prefill 的 step 中出现 token 不一致 | 11 / 11 |
| scheduled decode tokens / actual | 1,032 / 1,032 |

结论：**接纳/顺序可控，prefill 预算不可控**。`insert_segments` 不能当动态 chunk 控制；
`max_tokens` 是输出上限，不是本轮计算预算。缓存命中时比例进一步恶化（mixed 运行中
cache 臂 1.43–1.46，no-cache FIFO 为 1.00）。

### 49.2 A/B/C/D（16 会话 × 1 turn；mixed = 8 共享 + 8 独有前缀）

| workload | arm | tok/s | TTFT p50 | TTFT p95 | p95 total | slowest | peak MLX |
|---|---|---:|---:|---:|---:|---:|---:|
| mixed run1 | B cache+FIFO | 11.7 | 70.87 s | 73.76 s | 87.32 s | 87.46 s | 34.5 GB |
| mixed run1 | D cache+cost | **15.6** | **44.47 s** | **56.96 s** | **65.74 s** | **65.74 s** | **17.7 GB** |
| mixed run2 | B cache+FIFO | 11.1 | 56.66 s | 81.48 s | 92.28 s | 92.44 s | 34.5 GB |
| mixed run2 | D cache+cost | **12.5** | **47.07 s** | **69.84 s** | **81.75 s** | **81.87 s** | **19.9 GB** |
| shared | B cache+FIFO | 89.5 | 2.54 s | 2.95 s | 11.44 s | 11.44 s | 9.83 GB |
| shared | D cache+cost | 87.5 | 2.73 s | 3.16 s | 11.70 s | 11.70 s | 9.83 GB |
| 8×2 mixed | B cache+FIFO | 17.4 | 11.31 s | 40.27 s | 49.88 s | 58.87 s | 19.3 GB |
| 8×2 mixed | D cache+cost | 17.3 | 10.64 s | 39.78 s | 49.28 s | 59.10 s | 18.5 GB |
| long 4×1 | B cache+FIFO | 40.0 | 1.09 s | 1.09 s | 6.39 s | 6.39 s | 10.5 GB |
| long 4×1 | D cache+cost | 37.4 | 1.10 s | 1.10 s | 6.84 s | 6.84 s | 10.5 GB |

- 混合 16×1 的 D vs B 在两个独立 run 中方向一致：吞吐 +13–33%，p95 TTFT −14–23%，
  p95 完成时间 −11–25%，峰值内存 −42–49%，最慢会话也没有变差。
- 全共享前缀、8×2、8k 长提示都没有额外收益；C 无缓存时 ≈ A，说明**起作用的输入是缓存命中信息**，
  不是“排序”本身。
- 8k 前缀预注册成本约 32 s，必须摊销；长提示的收益全部来自缓存。

### 49.3 复用策略开关（mixed 16×1）

| variant | tok/s | TTFT p50 | TTFT p95 | peak MLX |
|---|---:|---:|---:|---:|
| D auto | 12.5 | 47.57 s | 70.26 s | 19.9 GB |
| D-fixed | 12.1 | 43.65 s | 74.57 s | 25.3 GB |
| D-always | 12.3 | 42.19 s | 73.81 s | 19.5 GB |
| D-never | 7.3 | 116.32 s | 130.11 s | 21.1 GB |

在这个前缀长度下，auto copy-or-recompute 没有显出优势；always/fixed 与 auto 在噪声内，
never 明显变差。**不能把 copy-or-recompute 当成已验证的机制。**

### 49.4 决策

1. 该机制有真实但很窄的收益（16×1 mixed），不足以称为协同优化引擎；
2. 调度器仍不能控制执行器 prefill 预算，任何更强结论都需要先做受控执行器；
3. 不启动 kernel/物理共享重写——数据不支持；
4. 16×2 共享前缀运行曾挂起一次（health 无响应），未定位，需在 production claim 前解决。

## 50. Phase 2 复测：并发上限基线与持续到达（2026-10-27）

Phase 1 的 D vs B 收益有一个更简单的解释：cost 排序可能只是降低了有效并发。补了
「缓存 + FIFO + `--max-num-seqs` 扫描」以及持续到达 workload 后，结论改变了。

### 50.1 mixed 16×1，FIFO 并发上限扫描

| 方案 | tok/s | TTFT p50 | TTFT p95 | p95 total | slowest | peak MLX | peak KV tokens | peak running |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B-cap4（FIFO，4 槽）** | **14.5** | **32.45 s** | **67.44 s** | **70.74 s** | **70.74 s** | **18.5 GB** | 25,904 | 4 |
| B-cap8（FIFO，8 槽） | 11.9 | 54.74 s | 76.02 s | 85.59 s | 85.72 s | 24.0 GB | 36,080 | 8 |
| B-cap12（FIFO，12 槽） | 12.2 | 47.35 s | 79.33 s | 83.62 s | 83.62 s | 34.5 GB | 34,464 | 12 |
| B（FIFO，16 槽） | 11.6 | 40.89 s | 77.51 s | 88.10 s | 88.20 s | 30.3 GB | 44,784 | 16 |
| D（cost，16 槽） | 12.6 | 40.98 s | 71.76 s | 81.11 s | 81.11 s | 19.5 GB | 44,784 | 16 |

**简单 FIFO cap4 在整体指标上优于 cost 排序**；D 比 FIFO-16 好，但 cap4 用更简单的机制
拿到了更好的吞吐、更低的 p95 和更低内存。不过 cost 改变了服务优先级：持续到达中 shared
p50 为 14.7 s，cap4 为 35.0 s；只有产品明确要求“优先服务缓存命中请求”时这个取舍才值得。

B 与 D 的**逻辑** peak KV tokens 相同（44,784），而 cap4 为 25,904。这只能说明逻辑 KV
计数无法解释 D 的内存差异；实际驻留、padding、复制和分配器保留内存都未测量，因此
“瞬时 prefill/copy 缓冲”仍是待验证假设。cap4 只是本组数据的赢家，不是通用默认。

### 50.2 持续到达（mixed，500ms stagger）

| 方案 | tok/s | TTFT p95 | p95 total | slowest | peak MLX | peak KV | shared p50 | unique p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B（FIFO，16 槽） | 11.2 | 73.25 s | 87.90 s | 88.02 s | 40.9 GB | 44,544 | 40.33 s | 70.08 s |
| D（cost，16 槽） | 12.8 | 63.99 s | 78.13 s | 78.16 s | 21.1 GB | 44,576 | 14.71 s | 58.78 s |
| **B-cap4（FIFO，4 槽）** | **13.3** | **60.39 s** | **69.39 s** | **70.03 s** | **18.2 GB** | 28,368 | 34.95 s | **45.02 s** |

同样，cap4 在吞吐、p95、内存上均最好；D 改善了 shared/unique 两组，但不足以证明
cost 排序优于简单限流。

### 50.3 16×2 挂起

四次独立的 16 会话 × 2 轮 shared 运行全部完成（21.7–22.4 s，~94 tok/s，峰值 9.83 GB），
**未能复现**之前 multi-arm 序列中的挂起；当时没有抓到线程栈，卡在哪一层仍未知。
文档中保留为 open reliability question，不写成已解决。

### 50.4 修正后的结论

- 准确称呼：**缓存/接纳调度的联合实验**（limited cache-scheduler co-design），不是
  co-optimized engine。
- **被证伪的是“cost 排序在整体指标上优于简单并发控制”，不是排序的全部价值**；
  tuned FIFO 并发是当前默认，cost 排序保留为实验选项。
- cap4 只是本组数据赢家。调参数据与验证数据要分开；未来对比应同时报告预选固定 cap 与
  每负载 oracle cap，不要求 cache-aware 排序赢过 cap。
- 不据此重写执行器或做物理共享；后续先研究不同负载的最佳 cap 是否差异足够大，
  再决定是否需要自动并发选择（允许最终答案仍是 FIFO）。

## 51. Cap 扫描与「固定 cap vs oracle」（2026-10-27）

外部 load ~6.2，绝对数字整体变慢，只做表内相对比较。全部 cache+FIFO、16 会话、1 turn、
50ms settle、真 usage tokens。

### 51.1 mixed（8 shared + 8 unique）

| cap | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX |
|---:|---:|---:|---:|---:|---:|
| 2 | 11.2 | 63.19 s | 89.02 s | 91.40 s | 9.6 GB |
| **4** | **11.7** | **40.24 s** | **75.19 s** | **87.49 s** | 11.6 GB |
| 6 | 9.8 | 52.63 s | 98.96 s | 104.58 s | 17.3 GB |
| 8 | 8.1 | 71.79 s | 115.48 s | 125.85 s | 29.3 GB |
| 12 | 7.4 | 89.42 s | 126.90 s | 137.91 s | 29.4 GB |
| 16 | 7.3 | 72.13 s | 124.16 s | 140.63 s | 25.7 GB |

### 51.2 shared（16 会话共享 2k 前缀）

| cap | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX |
|---:|---:|---:|---:|---:|---:|
| 2 | 34.7 | 15.16 s | 26.10 s | 29.51 s | 3.9 GB |
| 4 | 38.7 | 14.18 s | 20.71 s | 26.44 s | 4.9 GB |
| 6 | 38.7 | 11.01 s | 20.47 s | 26.43 s | 5.7 GB |
| 8 | 38.0 | 15.11 s | 15.61 s | 26.93 s | 6.6 GB |
| 12 | 43.1 | 3.29 s | 17.70 s | 23.74 s | 8.2 GB |
| **16** | **52.4** | 4.00 s | **4.79 s** | **19.53 s** | 9.9 GB |

### 51.3 固定 cap vs oracle

- mixed：oracle cap=4；预选 cap4 正好命中。
- shared：oracle cap=16；预选 cap4 吞吐损失 26%（38.7 vs 52.4），p95 20.7 vs 4.79 s。
- 持续 mixed（500ms stagger，较早低负载 run）：cap4 13.3 vs cap16 11.2，cap4 仍赢。
- long 4×1：cap4 与 cap16 相同（~22 tok/s），因为只有 4 个会话。

**结论：最佳 cap 随 workload 明显变化（4 vs 16），一个固定值无法同时最优。** 当前产品答案
是两个显式预设：`--profile agent`（16，共享前缀）与 `--profile mixed`（4，shared+unique）；
自动选 cap 是研究问题，不宣称已完成。

### 51.4 可靠性现场

新增 `TERRAZZO_STACK_DUMP=1`（SIGUSR1 打印所有线程栈）、`/stats.request_stages`（added/
first_token/finished/aborted/failed）以及 `scripts/watch_health.py`。四次隔离的 16×2 shared
运行均完成；multi-arm 重试在 load ~6.2 下 arm A 健康但过慢（17 first tokens、704/2048
decode 后停止），未复现挂起、未抓到栈。原挂起仍是 open issue，下次需在低负载下跑完整
A/B/C/D 2-turn 序列。

## 52. 负载感知并发选择：cap4 ↔ cap16（2026-10-27）

单进程、同一模型，shared → mixed → shared；16 会话、1 turn × 64 token、800ms stagger、
2k 预注册前缀、50ms settle。外部 load ~6，表内相对比较。完整设计见
[docs/CO-DESIGN.md](docs/CO-DESIGN.md) §13。

| arm | overall tok/s | shared-1 p50/p95 | mixed p50/p95 | shared-2 p50/p95 | peak MLX |
|---|---:|---:|---:|---:|---:|
| fixed4 | **18.5** | 5.01 / 10.80 s | 51.01 / 88.81 s | 4.96 / 9.87 s | 18.7 GB |
| fixed16 | 15.0 | 1.33 / 1.94 s | 83.84 / 120.38 s | 7.54 / 11.13 s | 40.8 GB |
| oracle 16/4/16 | 16.9 | 1.50 / 2.32 s | 57.26 / 101.67 s | 1.43 / 2.24 s | 20.0 GB |
| auto | 17.3 | 1.32 / 2.27 s | 59.25 / 100.76 s | 3.33 / 4.93 s | 20.0 GB |

auto 在 shared-1/mixed 基本跟随 oracle，mixed p95 明显好于 fixed16（100.8 vs 120.4 s），
shared p50 明显好于 fixed4（1.32 vs 5.01 s）；但 fixed4 总吞吐仍更高（18.5 vs 17.3 tok/s）。
反向 mixed→shared→mixed 序列验证自动切换 4→16→4 且无抖动。**结论：auto 作为 opt-in 实验可用，
手动预设仍是默认；不能宣称自动策略全面优于固定 cap4。**

实现：`terrazzo/concurrency.py`、`Engine.set_max_num_seqs`、`TERRAZZO_AUTO_CAP=1`、
`POST /v1/concurrency`（需 `TERRAZZO_ENABLE_CONTROL=1`）、`bench/concurrency_switch.py`。
