"""Reproduce the KV-quantization comparison from MAC-TURBO.md §4.2.

    python experiments/bench/kv_quant.py --model mlx-community/Qwen3-4B-4bit
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlx_lm import load  # noqa: E402

from experiments.kv_quant import generate_measured  # noqa: E402

PROMPT = "请用五句话解释 KV cache 的作用，并说明它在长上下文推理中的代价。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--kv-bits", type=int, default=None, dest="kv_bits",
                    help="run a single config in this process (cleanest measurement)")
    args = ap.parse_args()

    model, tok = load(args.model)
    prompt = tok.encode(PROMPT)
    generate_measured(model, prompt, max_tokens=8)  # warm up

    if args.kv_bits is not None:
        generate_measured(model, prompt, max_tokens=8, kv_bits=args.kv_bits)
        runs = [generate_measured(model, prompt, max_tokens=args.max_tokens, kv_bits=args.kv_bits)
                for _ in range(args.repeat)]
        print(json.dumps({"model": args.model, "kv_bits": args.kv_bits,
                          "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in runs), 1),
                          "kv_cache_mb": runs[0]["kv_cache_mb"],
                          "tokens": runs[0]["tokens"]}, ensure_ascii=False), flush=True)
        return 0

    rows = {}
    for label, bits in (("fp16", None), ("8bit", 8), ("4bit", 4)):
        generate_measured(model, prompt, max_tokens=8, kv_bits=bits)  # warm this config
        runs = [generate_measured(model, prompt, max_tokens=args.max_tokens, kv_bits=bits)
                for _ in range(args.repeat)]
        rows[label] = {
            "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in runs), 1),
            "kv_cache_mb": runs[0]["kv_cache_mb"],
            "tokens": runs[0]["tokens"],
        }
    base = rows["fp16"]["tokens"]
    for label, row in rows.items():
        if label == "fp16":
            row["prefix_match_vs_fp16"] = len(base)
            continue
        n = 0
        for a, b in zip(base, row["tokens"]):
            if a != b:
                break
            n += 1
        row["prefix_match_vs_fp16"] = n
        del row["tokens"]
    del rows["fp16"]["tokens"]
    print(json.dumps({"model": args.model, "max_tokens": args.max_tokens, **rows},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
