"""Decode step breakdown: graph build (Python/launch) vs GPU eval, and per-op increments.

For a 4k/8k prompt this measures, per decode step:
  build  : wall time for model(...) returning lazy logits (Python dispatch + graph)
  eval   : wall time for mx.eval(logits) (GPU execution)
and repeats the same split with attention removed, rope removed, and both.

    python bench/decode_profile.py --model <mlx-dir> [--contexts 4000,8000]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


class _NoRope:
    def __call__(self, x, offset=0, **kwargs):
        return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="4000")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx_lm.models.qwen3 as qwen3
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load(args.model)
    orig_sdpa = qwen3.scaled_dot_product_attention
    orig_ropes = [layer.self_attn.rope for layer in model.layers]

    def fake_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        return mx.zeros_like(queries)

    def set_config(attn: bool, rope: bool):
        qwen3.scaled_dot_product_attention = orig_sdpa if attn else fake_sdpa
        for layer, orig in zip(model.layers, orig_ropes):
            layer.self_attn.rope = orig if rope else _NoRope()

    for ctx in [int(c) for c in args.contexts.split(",")]:
        text = (FILLER * (ctx // 20 + 60))[:]
        ids = tok.encode(text)[:ctx]
        rows = {}
        for name, (attn, rope) in {
            "full": (True, True),
            "no_attn": (False, True),
            "no_rope": (True, False),
            "no_attn_no_rope": (False, False),
        }.items():
            set_config(attn, rope)
            cache = make_prompt_cache(model)
            pre = ids[:-1]
            for start in range(0, len(pre), 1024):
                mx.eval(model(mx.array([pre[start:start + 1024]]), cache=cache))
            token = ids[-1]
            builds, evals = [], []
            for i in range(args.steps + 4):
                t0 = time.perf_counter()
                logits = model(mx.array([[token]]), cache=cache)
                t_build = time.perf_counter() - t0
                t1 = time.perf_counter()
                mx.eval(logits)
                t_eval = time.perf_counter() - t1
                token = int(mx.argmax(logits[0, -1]))
                if i >= 4:
                    builds.append(t_build * 1000)
                    evals.append(t_eval * 1000)
            rows[name] = (statistics.median(builds), statistics.median(evals))
        set_config(True, True)
        print(f"ctx={ctx}:")
        for name, (b, e) in rows.items():
            print(f"  {name:>16}: build={b:5.2f}ms  eval={e:6.2f}ms  total={b+e:6.2f}ms",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
