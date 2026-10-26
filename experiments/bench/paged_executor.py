"""Paged KV executor vs the dense engine: correctness + prefill/decode numbers.

    python experiments/bench/paged_executor.py --model <mlx-dir> [--context 4000] [--tokens 32]

Runs the same greedy prompt through the dense engine (mlx-lm BatchGenerator) and the
paged PoolSession (block pool + block table + prefix sharing), compares token ids, and
reports prefill/decode timing for both.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def dense(engine, tok, prompt_ids, n_tokens):
    from terrazzo.sequence import SamplingParams

    params = SamplingParams(max_tokens=n_tokens, temperature=0.0, ignore_eos=True)
    t0 = time.perf_counter()
    text = engine.generate(prompt_token_ids=prompt_ids, params=params)
    total = time.perf_counter() - t0
    st = engine.stats()
    return {
        "tokens": st.get("decode_tokens", 0),
        "prefill_s": round(st.get("prefill_s", 0.0), 3),
        "total_s": round(total, 3),
        "text": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--variants", default="gather,gqa,perhead",
                    help="paged decode variants to compare (gather/gqa/perhead)")
    ap.add_argument("--pool-dtype", default="bfloat16",
                    help="bfloat16 (matches model activations) or float16")
    args = ap.parse_args()

    import os

    import mlx.core as mx
    from mlx_lm import load

    from terrazzo import Engine, EngineConfig
    from experiments.paged_attn_patch import (apply_paged_attn_patch,
                                                disable_paged_attn_patch)
    from terrazzo.pool_session import PoolSession

    model, tok = load(args.model)
    text = (FILLER * (args.context // 20 + 40))[:]
    prompt = tok.encode(text)[: args.context] + tok.encode("请用三句话总结以上内容")

    engine = Engine(EngineConfig(model=args.model, num_gpu_blocks=0,
                                 enable_prefix_cache=False))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pool_dtype = mx.bfloat16 if args.pool_dtype == "bfloat16" else mx.float16
    session = PoolSession(model, tok, block_size=16, num_blocks=2048,
                          prefill_chunk_size=1024, pool_dtype=pool_dtype,
                          eos_token_ids=tuple(getattr(tok, "eos_token_ids", []) or []))
    results = {}
    gather_tokens = None
    for variant in variants:
        os.environ["TERRAZZO_PAGED_KERNEL"] = "perhead" if variant == "perhead" else "gqa"
        if variant == "gather":
            disable_paged_attn_patch()
        else:
            apply_paged_attn_patch()
        stats = []
        for _ in range(args.rounds):
            ps = session.generate(prompt, max_tokens=args.tokens, temperature=0.0,
                                  stop_on_eos=False)
            stats.append(ps)
        best = min(stats, key=lambda st: st.decode_s)
        rate = best.decode_tokens / max(best.decode_s, 1e-9)
        if gather_tokens is None:
            gather_tokens = best.tokens
        results[variant] = {
            "decode_tok_s": round(rate, 1),
            "prefill_s": round(best.prefill_s, 2),
            "decode_s": round(best.decode_s, 3),
            "same_tokens": best.tokens == gather_tokens,
        }
        print(f"paged[{variant:7s}]: prefill {results[variant]['prefill_s']}s  "
              f"decode {results[variant]['decode_s']}s -> {rate:.1f} tok/s  "
              f"same={results[variant]['same_tokens']}", flush=True)
    disable_paged_attn_patch()

    rows = []
    for _ in range(args.rounds):
        d = dense(engine, tok, prompt, args.tokens)
        t0 = time.perf_counter()
        ps = session.generate(prompt, max_tokens=args.tokens, temperature=0.0,
                              stop_on_eos=False)
        paged_total = time.perf_counter() - t0
        rows.append({
            "dense_prefill_s": d["prefill_s"],
            "dense_total_s": d["total_s"],
            "paged_prefill_s": round(ps.prefill_s, 3),
            "paged_decode_s": round(ps.decode_s, 3),
            "paged_total_s": round(paged_total, 3),
            "paged_prefill_tokens": ps.prefill_tokens,
            "paged_prefix_hit": ps.prefix_hit_tokens,
            "same_text": d["text"] == tok.decode(ps.tokens, skip_special_tokens=True)
            or (d["text"].strip() == tok.decode(ps.tokens, skip_special_tokens=True).strip()),
            "dense_tokens": tok.decode(ps.tokens, skip_special_tokens=True) == d["text"],
        })
        last = rows[-1]
        print(f"round: dense(prefill {last['dense_prefill_s']}s, total {last['dense_total_s']}s) "
              f"paged(prefill {last['paged_prefill_s']}s/{last['paged_prefill_tokens']} tok, "
              f"decode {last['paged_decode_s']}s, total {last['paged_total_s']}s, "
              f"prefix_hit={last['paged_prefix_hit']}) same={last['dense_tokens']}", flush=True)
    print("pool stats:", session.stats()["pool"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
