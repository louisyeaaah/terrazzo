"""Where does 4k prefill time go? Chunk-size sweep + attention ablation.

    python bench/prefill_ablation.py --model <mlx-dir> --context 4000

Measures a raw model prefill (no sampling) with chunk sizes 256..4000 and with
attention replaced by zeros, to separate GEMM time (measured ~5.2 s floor in
bench/prefill_gemm.py) from attention / chunk-overhead time.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=4000)
    ap.add_argument("--chunks", default="256,512,1024,2048,4000")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm.models.qwen3 as qwen3
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(args.model)
    text = (FILLER * (args.context // 20 + 60))[:]
    ids = tok.encode(text)[: args.context]
    orig_sdpa = qwen3.scaled_dot_product_attention

    def fake_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        return mx.zeros_like(queries)

    def run(chunk: int, ablate_attention: bool) -> tuple[float, list[float]]:
        qwen3.scaled_dot_product_attention = fake_sdpa if ablate_attention else orig_sdpa
        cache = make_prompt_cache(model)
        per_chunk = []
        t0 = time.perf_counter()
        for start in range(0, len(ids), chunk):
            piece = ids[start:start + chunk]
            c0 = time.perf_counter()
            logits = model(mx.array([piece]), cache=cache)
            mx.eval(logits)
            per_chunk.append(time.perf_counter() - c0)
        total = time.perf_counter() - t0
        qwen3.scaled_dot_product_attention = orig_sdpa
        return total, per_chunk

    print(f"prefill {len(ids)} tokens (median of {args.repeats}):")
    for chunk in [int(c) for c in args.chunks.split(",")]:
        full = [run(chunk, False) for _ in range(args.repeats)]
        noattn = [run(chunk, True) for _ in range(args.repeats)]
        best = min(full, key=lambda r: r[0])
        best_na = min(noattn, key=lambda r: r[0])
        n_chunks = len(best[1])
        print(f"chunk={chunk:>5} chunks={n_chunks:>3}: full={best[0]:5.2f}s "
              f"({len(ids)/best[0]:5.0f} tok/s)  no_attn={best_na[0]:5.2f}s  "
              f"attn_delta={best[0]-best_na[0]:4.2f}s  "
              f"per_chunk_first={best[1][0]*1000:5.0f}ms last={best[1][-1]*1000:5.0f}ms",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
