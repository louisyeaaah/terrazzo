"""Microbench: multi-query (K-token verification) attention vs MLX SDPA.

Compares, per 36-layer cost, our multi-query Metal kernel against the SDPA path the
model actually uses (mask="causal") for K = 1..5 at 4k/8k context, then reports the
K/QK=1 cost ratio — the number that decides whether speculative decoding can pay off.

    python experiments/bench/multi_query_attn.py [--contexts 4000,8000] [--queries 1,2,3,5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

D, Hq, Hk, LAYERS = 128, 32, 8, 36


def bench(fn, n: int = 10) -> float:
    import mlx.core as mx

    fn()
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="4000,8000")
    ap.add_argument("--queries", default="1,2,3,5")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    import mlx.core as mx

    from experiments.dense_gqa_multi_metal import gqa_multi_attention_metal
    from terrazzo.dense_multi_metal import multi_query_attention_metal

    for L in [int(c) for c in args.contexts.split(",")]:
        rows = {}
        for QK in [int(q) for q in args.queries.split(",")]:
            best = {"sdpa": 1e9, "kernel": 1e9, "gqa": 1e9}
            for _ in range(args.repeats):
                ks = [mx.random.normal((1, Hk, L, D)).astype(mx.float16) for _ in range(LAYERS)]
                vs = [mx.random.normal((1, Hk, L, D)).astype(mx.float16) for _ in range(LAYERS)]
                qs = [mx.random.normal((1, Hq, QK, D)).astype(mx.float16) for _ in range(LAYERS)]
                lengths = mx.array([L], dtype=mx.int32)
                starts = mx.array([L - QK], dtype=mx.int32)
                scale = D ** -0.5

                def sdpa():
                    out = []
                    for i in range(LAYERS):
                        out.append(mx.fast.scaled_dot_product_attention(
                            qs[i], ks[i], vs[i], scale=scale, mask="causal"))
                    return out

                def kernel():
                    return [multi_query_attention_metal(qs[i], ks[i], vs[i],
                                                        lengths, starts, scale)
                            for i in range(LAYERS)]

                def gqa():
                    return [gqa_multi_attention_metal(qs[i], ks[i], vs[i],
                                                      lengths, starts, scale)
                            for i in range(LAYERS)]

                best["sdpa"] = min(best["sdpa"], bench(sdpa))
                best["kernel"] = min(best["kernel"], bench(kernel))
                best["gqa"] = min(best["gqa"], bench(gqa))
            rows[QK] = best
            print(f"L={L:>5} K={QK}: sdpa={best['sdpa']:6.2f}ms  "
                  f"per-head={best['kernel']:6.2f}ms ({best['sdpa'] / best['kernel']:.2f}x)  "
                  f"gqa-reuse={best['gqa']:6.2f}ms ({best['sdpa'] / best['gqa']:.2f}x)", flush=True)
        base_sdpa = rows[1]["sdpa"]
        base_gqa = rows[1]["gqa"]
        print(f"  K/K=1 ratios: sdpa="
              + ",".join(f"{k}:{v['sdpa'] / base_sdpa:.2f}" for k, v in rows.items())
              + "  gqa-reuse="
              + ",".join(f"{k}:{v['gqa'] / base_gqa:.2f}" for k, v in rows.items()),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
