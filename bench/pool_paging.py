"""Benchmark: paged pool (v0.4) vs dense shared prefix (v0.3).

    python bench/pool_paging.py --model mlx-community/Qwen3-4B-4bit

Reports for the same 400-token system prefix and 3 sequential requests:
  * TTFT / decode throughput of both implementations
  * KV bytes for the prefix, and whether the pool grows per request
  * pool block accounting (used / shared) vs v0.3's dense arrays
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm import load  # noqa: E402

from terrazzo.paged_session import PagedSession  # noqa: E402
from terrazzo.pool_session import PoolSession  # noqa: E402

SYSTEM = (
    "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
    "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--kv-bits", type=int, default=None, dest="kv_bits",
                    help="quantize the paged pool (8 or 4); default fp16")
    args = ap.parse_args()

    model, tok = load(args.model)
    eos = tuple(tok.eos_token_ids)
    sys_ids = tok.encode(SYSTEM)
    user_ids = tok.encode("用户问：KV cache 有什么好处？")
    prompt = sys_ids + user_ids
    bs = 16

    # ---- v0.3: dense frozen prefix arrays
    dense = PagedSession(model, tok, eos_token_ids=eos)
    p = dense.register_prefix(sys_ids)
    dense_stats = []
    for _ in range(args.requests):
        st = dense.generate(prompt, max_tokens=args.max_tokens)
        dense_stats.append(st.as_dict())
    dense_prefix_mb = p.nbytes / 1e6 if p else 0

    # ---- v0.4: block pool + per-block refs
    pool = PoolSession(model, tok, num_blocks=256, eos_token_ids=eos,
                       kv_bits=args.kv_bits)
    entry = pool.register_prefix(sys_ids)
    pool_used_after_register = pool.stats()["pool"]["used_blocks"]
    pool_stats = []
    used_after_each = []
    for _ in range(args.requests):
        st = pool.generate(prompt, max_tokens=args.max_tokens)
        pool_stats.append(st.as_dict())
        used_after_each.append(pool.stats()["pool"]["used_blocks"])
    pool_final = pool.stats()["pool"]

    per_token_bytes = (
        pool.n_layers * 2 * pool.pool.n_kv_heads * pool.pool.head_dim
        * (2 if pool.pool.kv_bits is None else pool.pool.kv_bits / 8)
    )
    prefix_bytes = (entry.num_tokens * per_token_bytes) if entry else 0

    print(json.dumps({
        "engine": "terrazzo-v0.4",
        "pool_kv_bits": pool.pool.kv_bits,
        "case": "prefix",
        "system_tokens": len(sys_ids),
        "dense_prefix_tokens": p.num_tokens if p else 0,
        "dense_prefix_mb": round(dense_prefix_mb, 1),
        "pool_prefix_tokens": entry.num_tokens if entry else 0,
        "pool_prefix_blocks": len(entry.blocks) if entry else 0,
        "pool_prefix_mb_theory": round(prefix_bytes / 1e6, 1),
        "pool_used_after_register": pool_used_after_register,
    }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "engine": "terrazzo-v0.4",
        "pool_kv_bits": pool.pool.kv_bits,
        "case": "ttft_decode",
        "dense_ttft_s": round(statistics.median(x["ttft_s"] for x in dense_stats), 3),
        "pool_ttft_s": round(statistics.median(x["ttft_s"] for x in pool_stats), 3),
        "dense_decode_tok_s": round(statistics.median(x["decode_tok_s"] for x in dense_stats), 1),
        "pool_decode_tok_s": round(statistics.median(x["decode_tok_s"] for x in pool_stats), 1),
        "dense_prefill_tokens": dense_stats[0]["prefill_tokens"],
        "pool_prefill_tokens": pool_stats[0]["prefill_tokens"],
    }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "engine": "terrazzo-v0.4",
        "pool_kv_bits": pool.pool.kv_bits,
        "case": "memory",
        "requests": args.requests,
        "dense_equivalent_mb": round(dense_prefix_mb * args.requests, 1),
        "pool_used_blocks_after_each": used_after_each,
        "pool_shared_blocks": pool_final["shared_blocks"],
        "pool_peak_blocks": pool_final["peak_blocks"],
        "pool_reserved_mb": round(pool_final["nbytes"] / 1e6, 1),
        "note": "pool_reserved is the whole preallocated pool; the prefix itself is one copy",
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
