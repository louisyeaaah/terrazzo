"""Batch decode throughput + step breakdown.

Runs N concurrent greedy requests through the Engine (continuous batching inside
mlx-lm's BatchGenerator), measuring aggregate tok/s, per-step latency and how many
tokens each step produced (effective M). Compare with bench/quant_gemm chain timings:
M=1 8.4ms, M=4 27.2ms, M=16 50.6ms for the pure matmul work.

    python bench/batch_profile.py --model <mlx-dir> [--contexts 4000] [--sessions 1,4]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", type=int, default=4000)
    ap.add_argument("--sessions", default="1,4")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warm", action="store_true",
                    help="run one warmup request so the shared prefix is cached")
    args = ap.parse_args()

    from terrazzo import Engine, EngineConfig
    from terrazzo.sequence import SamplingParams

    # the manual step loop below bypasses Engine.generate(), which is what applies
    # the attention/cache patches -- install them explicitly to match server behaviour.
    from terrazzo.mask_patch import apply_mask_patch
    from terrazzo.batch_attn_patch import apply_batch_attn_patch
    apply_mask_patch()
    apply_batch_attn_patch()

    engine = Engine(EngineConfig(model=args.model, num_gpu_blocks=0,
                                 enable_prefix_cache=True,
                                 completion_batch_size=args.batch_size,
                                 prefill_batch_size=args.batch_size))
    tok = engine.tokenizer
    prompt = tok.encode((FILLER * (args.context // 20 + 60))[:])[: args.context]

    # warm the shared-prefix cache so concurrent sessions measure decode, not prefill
    if args.warm:
        wparams = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        engine.add_request(prompt_token_ids=prompt, params=wparams, request_id="warmup")
        while engine.has_unfinished():
            engine.step()

    if engine.prefix_cache is not None:
        print("prefix cache after warmup:", engine.prefix_cache.stats(), flush=True)

    for sessions in [int(s) for s in args.sessions.split(",")]:
        params = SamplingParams(max_tokens=args.tokens, temperature=0.0,
                                ignore_eos=True)
        seqs = [engine.add_request(prompt_token_ids=prompt, params=params,
                                   request_id=f"bench-{sessions}-{i}")
                for i in range(sessions)]
        rids = [s.request_id for s in seqs]
        prev = {r: 0 for r in rids}
        token_lists = {r: [] for r in rids}
        decode_times = []
        first_token: dict[str, float] = {}
        last_event: dict[str, float] = {}
        token_total = 0
        t0 = time.perf_counter()
        first_step = True
        while engine.has_unfinished():
            ts = time.perf_counter()
            outs = engine.step()
            dt = time.perf_counter() - ts
            now = time.perf_counter()
            new_tokens = 0
            for o in outs:
                if o.request_id not in prev or not o.token_ids:
                    continue
                k = len(o.token_ids)          # deltas carry exactly the new tokens
                new_tokens += k
                prev[o.request_id] += k
                token_lists[o.request_id].extend(o.token_ids)
                first_token.setdefault(o.request_id, now)
                last_event[o.request_id] = now
            if first_step:
                first_step = False
                if engine.prefix_cache is not None:
                    print(f"  [dbg] cache after first step: {engine.prefix_cache.stats()}", flush=True)
            if new_tokens:
                token_total += new_tokens
                if new_tokens <= sessions:   # decode step (all rows emit ~1)
                    decode_times.append(dt)
        total = time.perf_counter() - t0
        decode_tokens = sum(prev[r] - 1 for r in rids)
        window = max(last_event.values()) - min(first_token.values()) if first_token else 0.0
        decode_rate = decode_tokens / window if window > 0 else 0.0
        med = statistics.median(decode_times) if decode_times else 0.0
        t_start = t0
        ttft_ms = [1000 * (first_token[r] - t_start) for r in rids if r in first_token]
        per_session = [max(0.0, (last_event[r] - first_token[r])) for r in rids]
        print(f"sessions={sessions}: total={total:6.2f}s  prefetched={token_total}  "
              f"decode={decode_tokens} tok in {window:5.2f}s -> aggregate {decode_rate:6.1f} tok/s  "
              f"median_decode_step={med * 1000:5.2f}ms  "
              f"TTFT p50={statistics.median(ttft_ms) / 1000:5.2f}s  "
              f"max={max(ttft_ms) / 1000:5.2f}s  "
              f"per-session median={statistics.median(per_session) / max(1, (args.tokens - 1)) * 1000:5.1f}ms/tok  "
              f"token_agreement={len({tuple(v) for v in token_lists.values()}) == 1}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
