"""Sweep prefill chunk size / batched-token budget for TTFT.

Run on an idle, cool machine: interleaves configs and repeats, because a hot M1 Max
drops to ~40% of its peak and single-shot sweeps are order-biased.

    python bench/prefill_sweep.py --model <mlx-dir> --prompt-tokens 4000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo import Engine, EngineConfig  # noqa: E402
from terrazzo.sequence import SamplingParams  # noqa: E402

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=4000)
    ap.add_argument("--chunks", default="256,512,1024,2048")
    ap.add_argument("--batched", default="2048")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    text = (FILLER * (args.prompt_tokens // 20 + 60))[:]
    configs = [(int(c), int(b)) for c in args.chunks.split(",") for b in args.batched.split(",")]
    results: dict[tuple[int, int], list[float]] = {c: [] for c in configs}

    engines = {}
    params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    for chunk, batched in configs:
        engines[(chunk, batched)] = Engine(EngineConfig(
            model=args.model, num_gpu_blocks=0, prefill_chunk_size=chunk,
            max_num_batched_tokens=batched, enable_prefix_cache=False))

    # interleave rounds so thermal drift hits every config equally
    for _ in range(args.rounds):
        for chunk, batched in configs:
            eng = engines[(chunk, batched)]
            tok = eng.runner.tokenizer
            ids = tok.encode(text)[: args.prompt_tokens] + tok.encode("请用三句话总结以上内容")
            eng.generate(prompt_token_ids=ids[:8], params=params)  # warm
            t0 = time.perf_counter()
            eng.generate(prompt_token_ids=ids, params=params)
            dt = time.perf_counter() - t0
            results[(chunk, batched)].append(len(ids) / dt)
            time.sleep(0.3)

    for (chunk, batched), runs in results.items():
        print(f"chunk={chunk:>5} batched={batched:>5}: median {statistics.median(runs):6.1f} tok/s "
              f"(best {max(runs):6.1f})", flush=True)
    for eng in engines.values():
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
