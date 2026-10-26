"""A/B: fused decode forward (one kernel per qkv / gate-up) vs the stock model call.

    python experiments/bench/decode_fusions.py --model <mlx-dir> [--context 4000] [--tokens 16]

Both paths share a baseline-prefilled KV cache; greedy tokens must match exactly.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from experiments.decode_fused import build_fused, fused_decode_step

    model, tok = load(args.model)
    text = (FILLER * (args.context // 20 + 60))[:]
    ids = tok.encode(text)[: args.context]
    fused = build_fused(model)

    def prefill_caches():
        caches = make_prompt_cache(model)
        pre = ids[:-1]
        for start in range(0, len(pre), 1024):
            mx.eval(model(mx.array([pre[start:start + 1024]]), cache=caches))
        return caches

    configs = {
        "baseline": None,
        "fused_qkv": {"fuse_qkv": True, "fuse_gate_up": False},
        "fused_gate_up": {"fuse_qkv": False, "fuse_gate_up": True},
        "fused_both": {"fuse_qkv": True, "fuse_gate_up": True},
    }
    results = {}
    ref_tokens = None
    for name, kwargs in configs.items():
        caches = prefill_caches()
        token = ids[-1]
        tokens, times = [], []
        for i in range(args.tokens):
            if kwargs is None:
                t0 = time.perf_counter()
                logits = model(mx.array([[token]]), cache=caches)
                mx.eval(logits)
                dt = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                logits = fused_decode_step(model, fused, token, caches, **kwargs)
                mx.eval(logits)
                dt = time.perf_counter() - t0
            token = int(mx.argmax(logits[0, -1]))
            if i >= 2:
                times.append(dt * 1000)
            tokens.append(token)
        if ref_tokens is None:
            ref_tokens = tokens
        results[name] = {
            "step_ms": round(statistics.median(times), 2),
            "tok_s": round(1000 / statistics.median(times), 1),
            "same": tokens[: len(ref_tokens)] == ref_tokens,
        }
        print(f"{name:>14}: {results[name]['step_ms']:6.2f} ms/step  "
              f"{results[name]['tok_s']:5.1f} tok/s  tokens_match={results[name]['same']}",
              flush=True)
    base = results["baseline"]["step_ms"]
    for name in configs:
        if name != "baseline":
            print(f"  speedup {name}: {base / results[name]['step_ms']:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
