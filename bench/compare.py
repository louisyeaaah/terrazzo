"""Reproduce the numbers in README.md.

    python bench/compare.py --model mlx-community/Qwen3-4B-4bit
    python bench/compare.py --model <path> --baseline-batch-generator

Measures:
  1. single-request decode throughput (warmed up)
  2. 4-way and 8-way aggregate throughput through the engine
  3. prefix-cache TTFT on a repeated 400+ token prompt (cache off vs on)
  4. (optional) raw mlx-lm BatchGenerator baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo import Engine, EngineConfig  # noqa: E402
from terrazzo.sequence import SamplingParams  # noqa: E402

PROMPT = "用三句话解释 KV cache"
LONG = (
    "在大型语言模型的推理系统中，键值缓存用于保存注意力机制已经计算过的键和值，"
    "从而避免在生成每个新 token 时重复计算整个上下文。"
) * 12


def single(engine: Engine, max_tokens: int, repeat: int = 3) -> dict:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    engine.generate("你好", SamplingParams(max_tokens=8))  # warmup
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        engine.generate(PROMPT, params)
        times.append(time.perf_counter() - t0)
    best = min(times)
    return {"case": f"single_mt{max_tokens}", "best_s": round(best, 3),
            "tok_s": round(max_tokens / best, 1)}


def concurrent(engine: Engine, n: int, max_tokens: int = 128) -> dict:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    seqs = [engine.add_request(PROMPT, params) for _ in range(n)]
    t0 = time.perf_counter()
    while engine.has_unfinished():
        engine.step()
    wall = time.perf_counter() - t0
    total = sum(len(s.output_token_ids) for s in seqs)
    return {"case": f"conc_n{n}_mt{max_tokens}", "wall_s": round(wall, 2),
            "gen_tokens": total, "aggregate_tok_s": round(total / wall, 1),
            "per_req_tok_s": round(total / n / wall, 1)}


def ttft(engine: Engine, prompt: str, max_tokens: int = 8) -> float:
    seq = engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0))
    t0 = time.perf_counter()
    while True:
        engine.step()
        if seq.output_token_ids or seq.status.is_finished:
            return time.perf_counter() - t0


def prefix_cache_case(model: str) -> list[dict]:
    rows = []
    for use_cache in (False, True):
        eng = Engine(EngineConfig(model=model, enable_prefix_cache=use_cache))
        eng.generate("你好", SamplingParams(max_tokens=4))
        t1 = ttft(eng, LONG)
        t2 = ttft(eng, LONG)
        rows.append({"case": f"prefix_cache={'on' if use_cache else 'off'}",
                     "ttft_cold_s": round(t1, 3), "ttft_warm_s": round(t2, 3),
                     "speedup": round(t1 / t2, 2) if t2 else None})
        eng.close()
    return rows


def batch_generator_baseline(model: str, max_tokens: int = 128) -> dict:
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    m, tok = load(model)
    ids = tok.encode(PROMPT)
    bg = BatchGenerator(m, max_tokens=max_tokens, stop_tokens=[(t,) for t in tok.eos_token_ids],
                        completion_batch_size=8, prefill_batch_size=8, prefill_step_size=512)
    (uid,) = bg.insert_segments(segments=[[ids]], max_tokens=[max_tokens],
                                samplers=[make_sampler(temp=0.0)])
    while True:
        p, _g = bg.next()
        if any(getattr(r, "end_of_prompt", False) for r in p):
            break
    t0 = time.perf_counter()
    n = 0
    while n < max_tokens:
        _p, g = bg.next()
        n += len(g)
    wall = time.perf_counter() - t0
    return {"case": f"mlx_lm_batchgenerator_mt{max_tokens}",
            "tok_s": round(n / wall, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--baseline-batch-generator", action="store_true")
    args = ap.parse_args()

    engine = Engine(EngineConfig(model=args.model, max_num_seqs=8))
    print(json.dumps({"engine": "terrazzo", **single(engine, args.max_tokens)}), flush=True)
    for n in (4, 8):
        print(json.dumps({"engine": "terrazzo", **concurrent(engine, n, args.max_tokens)}), flush=True)
    print(json.dumps(engine.stats(), ensure_ascii=False), flush=True)
    engine.close()

    for row in prefix_cache_case(args.model):
        print(json.dumps({"engine": "terrazzo", **row}), flush=True)

    if args.baseline_batch_generator:
        print(json.dumps({"engine": "mlx-lm", **batch_generator_baseline(args.model, args.max_tokens)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
