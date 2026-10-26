"""Where does a decode step go at long context? Ablation profiler (B=1).

Measures wall time for N decode steps of a single sequence at two context lengths,
with model components ablated one at a time:

  full           everything
  no_attn        qwen3's SDPA call returns zeros (keeps qkv/rope/cache/o_proj)
  no_rope        rotary embedding is the identity
  no_attn_no_rope  both

    python bench/profile_decode.py --model <mlx-dir> [--contexts 30,4000]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def steps(model, tok, ctx: int, n: int, ctx_pos: int) -> float:
    """Return ms per decode step (prompt prefilled, cache warm)."""
    text = (FILLER * (ctx // 20 + 40))[:]
    ids = tok.encode(text)[:ctx]
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids])[:, :-1], cache=cache)
    mx.eval(logits)
    last = ids[-1]
    for _ in range(3):  # warm
        logits = model(mx.array([[last]]), cache=cache)
        last = int(mx.argmax(logits[0, -1]))
        mx.eval(logits)
    t0 = time.perf_counter()
    for _ in range(n):
        logits = model(mx.array([[last]]), cache=cache)
        last = int(mx.argmax(logits[0, -1]))
        mx.eval(logits)
    return (time.perf_counter() - t0) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="30,4000")
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    model, tok = load(args.model)
    import mlx_lm.models.qwen3 as qwen3

    orig_sdpa = qwen3.scaled_dot_product_attention
    roped_layers = [layer.self_attn.rope for layer in model.layers]
    originals = list(roped_layers)

    class NoRope(nn.Module):
        def __call__(self, x, offset=0, **kwargs):
            return x

    def fake_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        return mx.zeros_like(queries)

    def set_rope(enabled: bool) -> None:
        for layer, original in zip(model.layers, originals):
            layer.self_attn.rope = original if enabled else NoRope()

    configs = {
        "full": (orig_sdpa, True),
        "no_attn": (fake_sdpa, True),
        "no_rope": (orig_sdpa, False),
        "no_attn_no_rope": (fake_sdpa, False),
    }
    for ctx in [int(c) for c in args.contexts.split(",")]:
        row = {}
        # two passes, take min, to damp thermal noise
        for _ in range(2):
            for name, (sdpa, rope_on) in configs.items():
                qwen3.scaled_dot_product_attention = sdpa
                set_rope(rope_on)
                ms = steps(model, tok, ctx, args.steps, ctx)
                row[name] = min(row.get(name, 1e9), ms)
        print(f"ctx={ctx:>5}: " + "  ".join(f"{k}={v:6.2f}ms" for k, v in row.items()), flush=True)
        qwen3.scaled_dot_product_attention = orig_sdpa
        set_rope(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
