"""Microbenchmark: paged batch decode attention vs the dense path.

Compares, for batch sizes 4/8/16 and L~2600 tokens:
  * paged kernel  : pool + per-row block table, one kernel launch (proposed);
  * gather+sdpa   : per-row gather into a dense [B,Hk,L,D] then one SDPA call
                    (this is what a dense BatchKVCache path pays per layer);
  * sdpa only     : dense SDPA on pre-gathered KV (attention-only lower bound).

Per-layer numbers are multiplied by 36 (Qwen3-4B layers) to project a decode step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.paged_batch_metal import (  # noqa: E402
    gather_rows,
    paged_batch_attention_metal,
)

BS = 16
Hq, Hk, D = 32, 8, 128
LAYERS = 36


def bench(fn, n: int = 20) -> float:
    fn()
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="4,8,16")
    ap.add_argument("--tokens", type=int, default=2600)
    args = ap.parse_args()

    rows = []
    for B in [int(x) for x in args.batches.split(",")]:
        L = args.tokens
        per_row_blocks = (L + BS - 1) // BS
        nb = per_row_blocks * B
        k_pool = mx.random.normal((nb, Hk, BS, D)).astype(mx.float16)
        v_pool = mx.random.normal((nb, Hk, BS, D)).astype(mx.float16)
        lengths = mx.array([L - (i % 3) * 7 for i in range(B)], dtype=mx.int32)
        maxb = per_row_blocks
        table = []
        for b in range(B):
            table.append(list(range(b * per_row_blocks, (b + 1) * per_row_blocks)))
        block_table = mx.array(table, dtype=mx.int32)
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
        scale = D ** -0.5

        kk, vv = gather_rows(k_pool, v_pool, block_table, lengths)
        # fair in-engine baseline: MLX SDPA supports GQA directly (no repeat)
        sdpa_gqa = bench(lambda: mx.fast.scaled_dot_product_attention(q, kk, vv, scale=scale))
        mask = None  # rows are padded to the same length; timing only

        paged = bench(lambda: paged_batch_attention_metal(
            q, k_pool, v_pool, block_table, lengths, scale))
        gather = bench(lambda: gather_rows(k_pool, v_pool, block_table, lengths))
        dense = bench(lambda: mx.fast.scaled_dot_product_attention(
            q, mx.repeat(kk, Hq // Hk, axis=1), mx.repeat(vv, Hq // Hk, axis=1),
            scale=scale, mask=mask))

        rows.append({
            "batch": B, "context": L,
            "paged_ms": round(paged, 3),
            "gather_ms": round(gather, 3),
            "sdpa_ms": round(dense, 3),
            "sdpa_gqa_ms": round(sdpa_gqa, 3),
            "step_sdpa_gqa_ms": round(sdpa_gqa * LAYERS, 1),
            "paged_vs_sdpa_gqa": round(sdpa_gqa / max(1e-9, paged), 2),
            "gather_sdpa_ms": round(gather + dense, 3),
            "step_paged_ms": round(paged * LAYERS, 1),
            "step_gather_sdpa_ms": round((gather + dense) * LAYERS, 1),
            "paged_vs_gather_sdpa": round((gather + dense) / paged, 2),
        })
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
