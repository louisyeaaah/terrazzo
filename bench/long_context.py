"""Long-context decode benchmark: rate vs context length, optional mask A/B.

    python bench/long_context.py --model <mlx-dir> [--ab]

`--ab` alternates the decode-mask patch on/off (3 rounds, interleaved) and checks
that the generated tokens are identical. Runs on a cool machine: sustained GPU
load throttles an M1 Max and both arms drift together, which makes single-shot
A/B numbers meaningless.
"""

from __future__ import annotations

import argparse
import queue
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo import Engine, EngineConfig  # noqa: E402
from terrazzo.sequence import SamplingParams  # noqa: E402

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"
QUESTION = "请用三句话总结以上内容"


def stream_decode(engine, ids, ntok: int):
    params = SamplingParams(max_tokens=ntok, temperature=0.0, ignore_eos=True)
    seq = engine.add_request(prompt_token_ids=ids, params=params)
    q = engine.stream(seq.request_id)
    t0 = time.perf_counter()
    t_first = None
    n = 0
    tokens: list[int] = []
    while True:
        engine.step()
        while True:
            try:
                delta = q.get_nowait()
            except queue.Empty:
                break
            for tid in (getattr(delta, "token_ids", None) or []):
                if t_first is None:
                    t_first = time.perf_counter()
                tokens.append(tid)
                n += 1
            if getattr(delta, "finished", False):
                t_end = time.perf_counter()
                rate = n / max(1e-6, t_end - max(t_first, t0))
                return tokens, rate, t_first - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="30,1000,2000,4000,8000")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--ab", action="store_true", help="A/B the decode-mask patch")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    eng = Engine(EngineConfig(model=args.model, num_gpu_blocks=0, enable_prefix_cache=False))
    tok = eng.runner.tokenizer
    from terrazzo.mask_patch import apply_mask_patch, disable_mask_patch

    for ctx in [int(c) for c in args.contexts.split(",")]:
        text = (FILLER * (ctx // 20 + 40))[:]
        ids = tok.encode(text)[:ctx] + tok.encode(QUESTION)
        stream_decode(eng, ids, 8)  # warm
        if not args.ab:
            _, rate, ttft = stream_decode(eng, ids, args.tokens)
            print(f"ctx={ctx:>5}: {rate:6.1f} tok/s  ttft={ttft:5.2f}s", flush=True)
            continue
        on, off, on_tokens, off_tokens = [], [], None, None
        for _ in range(args.rounds):
            disable_mask_patch()
            off_tokens, rate, _ = stream_decode(eng, ids, args.tokens)
            off.append(rate)
            time.sleep(0.5)
            apply_mask_patch()
            on_tokens, rate, _ = stream_decode(eng, ids, args.tokens)
            on.append(rate)
            time.sleep(0.5)
        print(f"ctx={ctx:>5}: mask_on={statistics.median(on):6.1f}  "
              f"mask_off={statistics.median(off):6.1f}  "
              f"identical={on_tokens == off_tokens}", flush=True)
    eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
