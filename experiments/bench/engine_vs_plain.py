"""Fair comparison: terrazzo engine vs plain mlx-lm generation, same protocol.

Earlier `moe_calibration.py` compared the engine's end-to-end time (prefill+decode)
with mlx-lm's decode-only rate, which inflated the gap. Here both sides use the same
two-phase protocol:

  * TTFT   = a max_tokens=1 run  (prefill + one token)
  * decode = (total - TTFT) for a max_tokens=N run, with the token count taken from
             the engine's own counters so early-EOS cannot inflate the rate.

    python experiments/bench/engine_vs_plain.py --model <dir> --max-tokens 128
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_lm import load  # noqa: E402

from terrazzo import Engine, EngineConfig  # noqa: E402
from experiments.kv_quant import generate_measured  # noqa: E402
from terrazzo.sequence import SamplingParams  # noqa: E402

PROMPT = "Write a long, detailed essay about the history of computing. Keep going for many paragraphs; do not stop early."


def engine_measured(eng, prompt_ids: list[int], max_tokens: int) -> dict:
    t0 = time.perf_counter()
    eng.generate(None, prompt_token_ids=prompt_ids, max_tokens=1)
    ttft = time.perf_counter() - t0
    s1 = eng.stats()
    t0 = time.perf_counter()
    eng.generate(None, prompt_token_ids=prompt_ids, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0
    s2 = eng.stats()
    decode_tokens = s2["decode_tokens"] - s1["decode_tokens"]
    decode_s = max(1e-6, elapsed - ttft)
    return {
        "ttft_s": round(ttft, 4),
        "decode_tok_s": round(decode_tokens / decode_s, 1),
        "decode_tokens": decode_tokens,
        "prefill_tokens": s2["prefill_tokens"] - s1["prefill_tokens"],
        "total_s": round(elapsed, 3),
        "steps": s2["steps"] - s1["steps"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    model, tok = load(args.model)
    prompt = tok.encode(PROMPT)

    generate_measured(model, prompt, max_tokens=4)  # warm both code paths
    plain = [generate_measured(model, prompt, max_tokens=args.max_tokens) for _ in range(args.repeat)]
    eng = Engine(EngineConfig(model=args.model, max_num_seqs=8, enable_prefix_cache=False))
    eng.generate("你好", SamplingParams(max_tokens=4))  # warmup
    engine = [engine_measured(eng, prompt, args.max_tokens) for _ in range(args.repeat)]
    eng.close()

    plain_decode = statistics.median(r["decode_tok_s"] for r in plain)
    engine_decode = statistics.median(r["decode_tok_s"] for r in engine)
    summary = {
        "model": Path(str(args.model)).name,
        "max_tokens": args.max_tokens,
        "plain_ttft_s": round(statistics.median(r["ttft_s"] for r in plain), 3),
        "plain_decode_tok_s": round(plain_decode, 1),
        "engine_ttft_s": round(statistics.median(r["ttft_s"] for r in engine), 3),
        "engine_decode_tok_s": round(engine_decode, 1),
        "engine_decode_tokens": [r["decode_tokens"] for r in engine],
        "engine_steps": [r["steps"] for r in engine],
        "ratio_engine_over_plain": round(engine_decode / plain_decode, 2),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
