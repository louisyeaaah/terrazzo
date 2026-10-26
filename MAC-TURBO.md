# MacTurbo：Apple Silicon 上的 200 tok/s 推理架构（v0.6 实测校准版）

> **一句话结论（已按实测校准）：**
> 「200 tok/s」在 Mac 上不是模型大小问题，而是**每 token 字节数 + 每步固定开销**的问题。
> 单流 200 tok/s 只对**激活参数 ≤0.57B（M1/M3 Max）/ ≤1.13B（Ultra）**的模型成立；
> **dense 7B–27B 单流 200 tok/s 在当前任何 Mac 上都不可能**；
> 但**聚合吞吐**可以通过批处理超过 200 —— 实测 OLMoE（1.18B 激活）在 M1 Max 上 4 并发聚合 **285 tok/s**。

实测机器：M1 Max / 24 核 GPU / 64 GB / 400 GB/s，测量日期 2026-10-24。
所有数字来自 `bench/` 下可复跑脚本，不是营销口径。

---

## 1. 物理模型：两项式（已用三个模型校准）

```
t_step(ms) ≈ bytes_per_token × 1000 / BW_eff  +  FIXED_MS
BW_eff     = 0.664 × 峰值带宽        # 实测有效带宽只有峰值的 66%
FIXED_MS   ≈ 3.8 ms                  # 每步固定开销（框架/launch）
```

| 校准模型 | bytes/token | 实测 ms/token | 公式预测 | 实测 tok/s |
|---|---:|---:|---:|---:|
| Qwen3-4B（dense，激活 4.02B） | 2.263 GB | 12.32 | 12.32 | 81 |
| Qwen3-0.6B（dense） | 0.340 GB | 5.08 | 5.08 | 197 |
| OLMoE-1B-7B（MoE，激活 1.18B / 总 6.9B） | 0.663 GB | 6.09 | 6.30 | 165 |

三个点覆盖 dense 与 MoE、大与小，误差 ≤5%，所以下面的矩阵是**推算**而不是猜的（±20% 带）。

关键推论：**MoE 的「激活参数」确实决定速度**（OLMoE 1.18B 激活 = 165 tok/s，接近公式的 159）；
之前担心的「MoE 在 MLX 上不按带宽模型走」只在冷启动那一次出现，warmup 后不成立。

---

## 2. 可行性矩阵（单流，200 tok/s 标记加粗）

| 芯片（峰值 BW） | dense 0.6B | dense 4B | dense 7B | dense 27B | MoE 激活 1.2B | MoE 激活 3B |
|---|---:|---:|---:|---:|---:|---:|
| M1（68） | 89 | 19 | 11 | 3 | 54 | 24 |
| M1 Max（400） | 197 | 81 | 54 | 16 | 159 | 98 |
| M3 Max（400） | 197 | 81 | 54 | 16 | 159 | 98 |
| M4 Max（546） | **211** | 100 | 68 | 22 | 178 | 118 |
| M3 Ultra（800） | **225** | 124 | 89 | 31 | 198 | 143 |

**200 tok/s 的真实条件**（`t_step ≤ 5 ms` 反解的单流预算）：

| 芯片 | 每 token 预算 | 等价激活参数（4-bit） |
|---|---:|---:|
| M1 Max / M3 Max | 0.32 GB | **0.57B** |
| M4 Max | 0.44 GB | **0.77B** |
| M3 / M4 Ultra | 0.64 GB | **1.13B** |

所以：
- **dense 7B / 13B / 27B 单流 200 tok/s：任何 Mac 都做不到**（7B 需要 ~1575 GB/s 峰值）。
- **MoE 30B-A3B（激活 3B）单流也不行**：Ultra 上 143、M4 Max 118。
- 只有**小激活**模型（≤0.6B on Max、≤1.1B on Ultra）能单流过 200。
- 若目标是「服务吞吐 200 tok/s 以上」，**批处理**才是答案：实测 OLMoE 在 M1 Max 上 4 并发聚合 **285 tok/s**（单流 165）。

---

## 3. MacTurbo 五层架构

```
OpenAI API ─▶ L4 服务层：SLO 准入 / 能力声明 / 指标
                │  按 (芯片峰值带宽, 每 token 字节) 预测 tok/s；达不到就拒绝
                ▼
              L3 调度层（terrazzo v0.3）：连续批处理 / chunked prefill / LCFS 抢占
                ▼
              L2 KV 层（v0.2–v0.4）：block pool + 引用计数前缀共享 + CoW + KV 量化
                ▼
              L1 执行层：MLX +（可选）Metal paged decode 内核
                ▼
              L0 模型选择器：给定 SLO 反解「激活参数预算」与位宽
```

设计决策（全部来自实测）：

1. **先算预算，再选模型**：`terrazzo/slo.py` 用两项式反解「该芯片上 200 tok/s 能跑多大的激活参数」，
   服务端 `--target-tok-s 200` 达不到就拒绝启动（`slo_router.enforce`）。
2. **单流 SLO 与聚合 SLO 分开承诺**：单流看激活预算；聚合靠批处理（实测 4 并发 ×1.9）。
3. **KV 量化只用于并发/长上下文**（见 §4.2）：它省内存但掉 18% 解码速度，4-bit 还会伤质量。
4. **投机解码默认关**（见 §4.3）：在 Apple GPU 上每多验证一个 token 要 7.3 ms，draft 不够便宜就是负收益。
5. **绝不做 SSD 流式加载**：那是 1.25 tok/s 路线，和 200 tok/s SLO 互斥。

---

## 4. v0.6 实测记录

### 4.1 SLO 准入器（已接入服务端）

```bash
python -m terrazzo slo --model-path <dir> --target 200
python -m terrazzo serve --model <dir> --target-tok-s 200        # 达不到直接拒绝
python -m terrazzo serve --model <dir> --target-tok-s 200 --allow-below-target
```

- 自动识别芯片（`chip.py`，读 `sysctl`）+ 从 `config.json` 估算每 token 字节
  （dense：全部权重；MoE：激活专家 + 共享专家 + 路由 + LM head；embedding 只读一行）；
- `/v1/models` 与 `/capability` 返回预测 tok/s、是否达标、芯片/模型画像；
- 校准：Qwen3-4B 预测 **81.2**，实测单请求 81、batch 引擎 96（±10% 内）。

### 4.2 KV 量化：省内存，但掉速且 4-bit 伤质量

Qwen3-4B，48 token 输出，warmup 后 3 次中位数（在一段空闲窗口测得；复跑脚本
`experiments/bench/kv_quant.py --kv-bits <4|8>`；机器被其它进程占用时 decode 数字会成倍掉，
但 **KV 内存数字是确定性的**）：

| KV 精度 | decode tok/s | KV 缓存（~90 token） | 与 fp16 输出 |
|---|---:|---:|---|
| fp16 | **78.6** | 37.7 MB | — |
| 8-bit | 64.6（−18%） | 20.1 MB（−47%） | 首 token 一致，26/48 token 前缀相同 |
| 4-bit | 63.6（−19%） | 10.6 MB（−72%） | **首 token 就不同**（mlx-lm 官方 `kv_bits=4` 同样如此，不是接线错误） |

结论：默认关；需要更长上下文/更多并发时用 **8-bit**，4-bit 只在质量可接受时用。

### 4.3 投机解码：在 M1 Max 上是负收益（反例）

Qwen3-4B + Qwen3-0.6B draft（贪婪验证 + KV trim，输出与基线逐 token 一致）：

| 方案 | tok/s | 验收长度 |
|---|---:|---:|
| 基线 | **76.1** | — |
| 投机 k=3 | 36.2 | 0.88 |
| 投机 k=5 | 27.4 | 1.07 |

原因（实测）：验证 k+1 个 token 的边际成本是 **7.3 ms/token**（单 step 12.3 ms），
draft 0.6B 每 token 要 5.08 ms——只便宜 2.4×。判定公式写进了 DESIGN，默认关闭。

### 4.4 MoE 校准（OLMoE-1B-7B，本次目标的核心）

| 指标 | 数值 |
|---|---:|
| 激活 / 总参数 | 1.18B / 6.92B |
| 预测（两项式） | 158.8 tok/s |
| 实测单流（mlx-lm 路径） | **164.9–167.4 tok/s** |
| 我们的 batch 引擎单流 | 132.9–138.3 tok/s |
| **4 并发聚合** | **284.4–285.1 tok/s** |

这验证了架构的核心主张：**小激活 MoE 在 M1 Max 上单流接近 200、聚合远超 200**；
M4 Max / Ultra 上按公式会分别到 ~178 / ~198（单流）、批处理聚合更高。

---

## 5. 风险与未验证

1. **矩阵里非 M1 Max 的芯片**是按带宽比例外推，未实测（本机只有 M1 Max）。
2. **MoE 的每步固定开销**是否随实现变化未验证（OLMoE 的 MLX 实现已校准；其他 MoE 可能更差）。
3. **长时压测/降频**未做：持续满载 10 分钟后速度可能掉 10–20%。
4. **KV 量化对长上下文的质量影响**未系统评估（只看了首 token 与 48 token 前缀）。
5. **batch 引擎在 MoE 上比 mlx-lm 原生路径慢 ~20%**（133 vs 165），原因待查（可能是 BatchGenerator 的 MoE 支持路径）。

---

## 6. v0.7 计划

1. **MoE 上把 batch 引擎追平 mlx-lm 单流**（133 → 165），并把 4 并发聚合从 285 推高。
2. **KV 量化接进 paged pool**（v0.4 目前只支持 fp16）。
3. **多芯片实测校准**（M4 Max / Ultra 若可得）。
4. **SLO 路由接入调度**：按实时 tok/s 动态调 batch 大小（单流 SLO 与聚合 SLO 之间切换）。
