"""Benchmark: shared-prefix batching (v0.3).

Compares three ways to serve a prompt with a long system prefix:

  A. plain batch engine (recompute the prefix every request)
  B. batch engine + registered shared prefix (seed per-layer KV, still batches)
  C. paged session (prefix stored once and shared by reference, sequential)

Reports TTFT, decode throughput, 4-way aggregate throughput and KV accounting.

    python bench/prefix_batching.py --model mlx-community/Qwen3-4B-4bit
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo import Engine, EngineConfig  # noqa: E402
from terrazzo.paged_session import PagedSession  # noqa: E402
from terrazzo.sequence import SamplingParams  # noqa: E402

SYSTEM = (
    "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
    "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
)


def ttft(engine: Engine, prompt_ids: list[int], max_tokens: int = 8) -> float:
    seq = engine.add_request(prompt_token_ids=prompt_ids,
                             params=SamplingParams(max_tokens=max_tokens, temperature=0.0))
    t0 = time.perf_counter()
    while True:
        engine.step()
        if seq.output_token_ids or seq.status.is_finished:
            return time.perf_counter() - t0


def concurrent(engine: Engine, prompt_ids: list[int], n: int, max_tokens: int = 64) -> dict:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    seqs = [engine.add_request(prompt_token_ids=prompt_ids, params=params) for _ in range(n)]
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    wall = time.perf_counter() - t0
    total = sum(len(s.output_token_ids) for s in seqs)
    return {"n": n, "wall_s": round(wall, 2), "tokens": total,
            "aggregate_tok_s": round(total / wall, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cfg = EngineConfig(model=args.model, max_num_seqs=8)
    plain = Engine(cfg)
    tok = plain.runner.tokenizer
    sys_ids = tok.encode(SYSTEM)
    user_ids = tok.encode("用户问：KV cache 有什么好处？")
    prompt = sys_ids + user_ids
    plain.generate("你好", SamplingParams(max_tokens=4))  # warmup

    dense_ttft = [ttft(plain, prompt) for _ in range(args.repeat)]
    dense_conc = concurrent(plain, prompt, args.concurrency, args.max_tokens)

    prefixed = Engine(cfg)
    t0 = time.perf_counter()
    prefix = prefixed.register_prefix(sys_ids)
    register_s = time.perf_counter() - t0
    prefixed.generate("你好", SamplingParams(max_tokens=4))
    shared_ttft = [ttft(prefixed, prompt) for _ in range(args.repeat)]
    shared_conc = concurrent(prefixed, prompt, args.concurrency, args.max_tokens)

    d = statistics.median(dense_ttft)
    s = statistics.median(shared_ttft)
    print(json.dumps({"engine": "terrazzo-v0.3", "case": "register_prefix",
                      "system_tokens": len(sys_ids), "prefix_tokens": prefix.num_tokens if prefix else 0,
                      "register_s": round(register_s, 2),
                      "prefix_kv_mb": round(prefix.nbytes / 1e6, 1) if prefix else 0,
                      "blocks": len(prefix.block_ids) if prefix else 0}, ensure_ascii=False), flush=True)
    print(json.dumps({"engine": "terrazzo-v0.3", "case": "batch_ttft",
                      "dense_s": round(d, 3), "shared_s": round(s, 3),
                      "speedup": round(d / s, 2) if s else None,
                      "dense_prefill_tokens": len(prompt),
                      "shared_prefill_tokens": len(prompt) - (prefix.num_tokens if prefix else 0)},
                     ensure_ascii=False), flush=True)
    print(json.dumps({"engine": "terrazzo-v0.3", "case": "batch_throughput",
                      "dense": dense_conc, "shared": shared_conc,
                      "speedup": round(shared_conc["aggregate_tok_s"] / dense_conc["aggregate_tok_s"], 2)
                      if dense_conc["aggregate_tok_s"] else None}, ensure_ascii=False), flush=True)
    print(json.dumps({"engine": "terrazzo-v0.3", "case": "batch_stats",
                      "prefix_hit_tokens": prefixed.stats()["prefix_hit_tokens"],
                      "prefix_cache": prefixed.stats()["prefix_cache"],
                      "block_used": prefixed.stats()["used"]}, ensure_ascii=False), flush=True)
    plain.close()
    prefixed.close()

    # C) sequential paged session: prefix shared by reference
    from mlx_lm import load

    model, tok2 = load(args.model)
    sess = PagedSession(model, tok2, eos_token_ids=tuple(tok2.eos_token_ids))
    prefix = sess.register_prefix(sys_ids)
    paged_ttft = []
    for _ in range(args.repeat):
        st = sess.generate(prompt, max_tokens=8)
        paged_ttft.append(st.ttft_s)
    print(json.dumps({"engine": "terrazzo-v0.3", "case": "paged_ttft",
                      "dense_s": round(d, 3), "paged_s": round(statistics.median(paged_ttft), 3),
                      "speedup": round(d / statistics.median(paged_ttft), 2) if paged_ttft else None,
                      "paged_stats": sess.stats()["prefix_cache"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
