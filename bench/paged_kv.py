"""Benchmark: shared-prefix paged KV vs dense recomputation.

    python bench/paged_kv.py --model mlx-community/Qwen3-4B-4bit
    python bench/paged_kv.py --model <path> --system-tokens 600 --requests 4

Reports:
  * TTFT of a request that recomputes the whole prompt (dense)
  * TTFT of a request that attaches to a registered shared prefix (paged)
  * decode throughput of both paths (the paged path pays one concat per step)
  * memory: the shared prefix is stored once, regardless of how many requests use it
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm import load  # noqa: E402

from terrazzo.paged_session import PagedSession  # noqa: E402

SYSTEM = (
    "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
    "以下是背景资料："
    + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算。" * 12
)


def build_system(tok, target_tokens: int) -> list[int]:
    ids = tok.encode(SYSTEM)
    while len(ids) < target_tokens:
        ids = ids + tok.encode("补充说明：长上下文推理需要缓存复用。")
    return ids[:target_tokens]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--system-tokens", type=int, default=600)
    ap.add_argument("--requests", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    model, tok = load(args.model)
    eos = tuple(tok.eos_token_ids)
    sys_ids = build_system(tok, args.system_tokens)
    user_ids = tok.encode("用户问：KV cache 有什么好处？")

    session = PagedSession(model, tok, eos_token_ids=eos)
    plain = PagedSession(model, tok, eos_token_ids=eos)

    t0 = time.perf_counter()
    prefix = session.register_prefix(sys_ids)
    register_s = time.perf_counter() - t0
    print(json.dumps({
        "engine": "terrazzo-v0.2",
        "case": "register_prefix",
        "system_tokens": len(sys_ids),
        "prefix_tokens": prefix.num_tokens if prefix else 0,
        "register_s": round(register_s, 3),
        "prefix_kv_mb": round(prefix.nbytes / 1e6, 1) if prefix else 0,
        "blocks": len(prefix.block_ids) if prefix else 0,
    }, ensure_ascii=False), flush=True)

    prompt = sys_ids + user_ids
    dense_ttft, paged_ttft, dense_tps, paged_tps = [], [], [], []
    for i in range(args.repeat):
        st = plain.generate(prompt, max_tokens=args.max_tokens)
        dense_ttft.append(st.ttft_s)
        if st.decode_s:
            dense_tps.append(st.decode_tok_s)
        st2 = session.generate(prompt, max_tokens=args.max_tokens)
        paged_ttft.append(st2.ttft_s)
        if st2.decode_s:
            paged_tps.append(st2.decode_tok_s)
        if i == 0:
            first = st2.as_dict()
            print(json.dumps({"engine": "terrazzo-v0.2", "case": "shared_request",
                              **first, "tokens_equal_to_dense": st.tokens == st2.tokens},
                             ensure_ascii=False), flush=True)

    d = statistics.median(dense_ttft)
    p = statistics.median(paged_ttft)
    print(json.dumps({
        "engine": "terrazzo-v0.2",
        "case": "ttft_comparison",
        "dense_ttft_s": round(d, 3),
        "paged_ttft_s": round(p, 3),
        "ttft_speedup": round(d / p, 2) if p else None,
        "dense_decode_tok_s": round(statistics.median(dense_tps), 1) if dense_tps else None,
        "paged_decode_tok_s": round(statistics.median(paged_tps), 1) if paged_tps else None,
        "decode_overhead_pct": round(100 * (1 - statistics.median(paged_tps) / statistics.median(dense_tps)), 1)
        if dense_tps and paged_tps else None,
    }, ensure_ascii=False), flush=True)

    # memory: N requests share one copy instead of N
    snap = session.stats()
    print(json.dumps({
        "engine": "terrazzo-v0.2",
        "case": "memory",
        "shared_prefix_mb": round(prefix.nbytes / 1e6, 1) if prefix else 0,
        "dense_equivalent_mb_for_requests": round((prefix.nbytes * args.requests) / 1e6, 1) if prefix else 0,
        "requests_served": args.requests,
        "saved_mb": round((prefix.nbytes * (args.requests - 1)) / 1e6, 1) if prefix else 0,
        "block_used": snap["block"]["used"],
        "prefix_cache": snap["prefix_cache"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
