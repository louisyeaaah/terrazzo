"""MoE calibration: measured tok/s vs the MacTurbo SLO prediction.

    python experiments/bench/moe_calibration.py --model <OLMoE dir>

Runs:
  1. SLO prediction from config.json (active params, bytes/token);
  2. plain greedy decode measurement (mlx-lm path);
  3. the terrazzo batch engine (single + 4 concurrent).
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

from experiments.kv_quant import generate_measured  # noqa: E402
from terrazzo.slo_router import capability  # noqa: E402

PROMPT = "请用三句话解释 MoE（混合专家）模型为什么在同样的显存下能跑得更快。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    cap = capability(args.model, target_tok_s=200)
    print(json.dumps({k: v for k, v in cap.items() if k != "detail"}, ensure_ascii=False), flush=True)

    model, tok = load(args.model)
    prompt = tok.encode(PROMPT)
    generate_measured(model, prompt, max_tokens=8)  # warm up kernels
    runs = [generate_measured(model, prompt, max_tokens=args.max_tokens)
            for _ in range(args.repeat)]
    decode = statistics.median(r["decode_tok_s"] for r in runs)
    med = {
        "measured_decode_tok_s": round(decode, 1),
        "predicted_tok_s": cap["predicted_tok_s"],
        "measured_over_predicted": round(decode / cap["predicted_tok_s"], 2) if cap["predicted_tok_s"] else None,
        "ttft_s": runs[0]["ttft_s"],
        "kv_cache_mb": runs[0]["kv_cache_mb"],
        "active_params_b": cap["active_params_b"],
        "bytes_per_token_gb": cap["bytes_per_token_gb"],
    }
    print(json.dumps(med, ensure_ascii=False), flush=True)

    # batch engine
    from terrazzo import Engine, EngineConfig

    eng = Engine(EngineConfig(model=args.model, max_num_seqs=8))
    from terrazzo.sequence import SamplingParams
    eng.generate("你好", SamplingParams(max_tokens=8))
    # fair single-stream decode: subtract the max_tokens=1 (prefill+1 token) run
    eng.generate(None, prompt_token_ids=prompt, max_tokens=1)
    s0 = eng.stats()
    t0 = time.perf_counter()
    eng.generate(None, prompt_token_ids=prompt, max_tokens=args.max_tokens)
    single_s = time.perf_counter() - t0
    s1 = eng.stats()
    engine_decode = s1["decode_tokens"] - s0["decode_tokens"]
    seqs = [eng.add_request(prompt_token_ids=prompt) for _ in range(4)]
    for s in seqs:
        s.sampling_params.max_tokens = args.max_tokens
    t0 = time.perf_counter()
    while eng.has_unfinished():
        eng.step()
    wall = time.perf_counter() - t0
    total = sum(len(s.output_token_ids) for s in seqs)
    print(json.dumps({
        "engine_single_tok_s": round(engine_decode / single_s, 1),
        "engine_single_note": "decode-only (prefill subtracted)",
        "engine_conc4_aggregate_tok_s": round(total / wall, 1),
        "engine_steps": eng.stats()["steps"],
    }, ensure_ascii=False), flush=True)
    eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
