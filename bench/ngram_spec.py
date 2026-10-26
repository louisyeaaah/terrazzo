"""A/B benchmark: n-gram speculative decode vs plain greedy decode.

    python bench/ngram_spec.py --model <mlx-dir> --contexts 1000,4000 --tokens 128

Both arms use the same greedy decode loop and cache handling; they are interleaved
per round and the generated token ids must match exactly (speculation is lossless).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FILLER = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="1000,4000")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from mlx_lm import load

    from terrazzo.ngram_spec import ngram_generate, plain_generate

    model, tok = load(args.model)
    for ctx in [int(c) for c in args.contexts.split(",")]:
        text = (FILLER * (ctx // 20 + 40))[:]
        prompt = tok.encode(text)[:ctx] + tok.encode("请继续重复上面的内容")
        # warmup both paths once (kernel compilation, allocator warm-up)
        plain_generate(model, tok, prompt, max_tokens=8)
        ngram_generate(model, tok, prompt, max_tokens=8, k=args.k, n=args.n)
        base, spec, same = [], [], True
        for _ in range(args.rounds):
            _, b = plain_generate(model, tok, prompt, max_tokens=args.tokens)
            base.append(b["decode_tok_s"])
            time.sleep(0.15)
            _, s = ngram_generate(model, tok, prompt, max_tokens=args.tokens,
                                  k=args.k, n=args.n)
            spec.append(s["decode_tok_s"])
            same = same and (b["token_ids"] == s["token_ids"])
            time.sleep(0.15)
        b_med, s_med = statistics.median(base), statistics.median(spec)
        print(f"ctx={ctx:>5}: base={b_med:6.1f}  spec={s_med:6.1f}  "
              f"gain={(s_med / b_med - 1) * 100:+.1f}%  accept={s['acceptance_rate']:.2f} "
              f"identical={same}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
