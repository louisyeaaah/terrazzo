"""Benchmark: gather-based paged attention vs the Metal paged decode kernel.

    python bench/metal_vs_gather.py --model mlx-community/Qwen3-4B-4bit

Same paged pool, same prefix, same prompts; the only difference is whether the
Qwen3 attention is patched to use the block-walking Metal kernel on decode steps.
Also reports the v0.3 dense-prefix session for context.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm import load  # noqa: E402

from terrazzo.paged_session import PagedSession  # noqa: E402
from terrazzo.pool_attn_patch import patched_qwen3_attention  # noqa: E402
from terrazzo.pool_session import PoolSession  # noqa: E402

SYSTEM = (
    "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
    "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    model, tok = load(args.model)
    eos = tuple(tok.eos_token_ids)
    sys_ids = tok.encode(SYSTEM)
    user_ids = tok.encode("用户问：KV cache 有什么好处？")
    prompt = sys_ids + user_ids

    def run(session, patched: bool):
        # warm up the kernel compilation / MLX graphs before measuring
        for _ in range(2):
            if patched:
                with patched_qwen3_attention():
                    session.generate(prompt, max_tokens=8)
            else:
                session.generate(prompt, max_tokens=8)
        ttfts, decs, toks = [], [], None
        for _ in range(args.repeat):
            if patched:
                with patched_qwen3_attention():
                    st = session.generate(prompt, max_tokens=args.max_tokens)
            else:
                st = session.generate(prompt, max_tokens=args.max_tokens)
            ttfts.append(st.ttft_s)
            decs.append(st.decode_tok_s if hasattr(st, "decode_tok_s") else st.decode_tokens / max(st.decode_s, 1e-9))
            toks = st.tokens
        return {
            "ttft_s": round(statistics.median(ttfts), 3),
            "decode_tok_s": round(statistics.median(decs), 1),
            "tokens": toks,
        }

    # gather-based pool session (no patch)
    pool_gather = PoolSession(model, tok, num_blocks=256, eos_token_ids=eos)
    pool_gather.register_prefix(sys_ids)
    gather = run(pool_gather, patched=False)

    # same session, attention patched to the Metal kernel on decode
    pool_metal = PoolSession(model, tok, num_blocks=256, eos_token_ids=eos)
    pool_metal.register_prefix(sys_ids)
    metal = run(pool_metal, patched=True)

    # v0.3 dense-prefix session for context
    dense = PagedSession(model, tok, eos_token_ids=eos)
    dense.register_prefix(sys_ids)
    st = dense.generate(prompt, max_tokens=args.max_tokens)
    dense_row = {"ttft_s": round(st.ttft_s, 3), "decode_tok_s": round(st.decode_tok_s, 1),
                 "tokens": st.tokens}

    print(json.dumps({"engine": "terrazzo-v0.4", "case": "paged_decode",
                      "gather": {k: v for k, v in gather.items() if k != "tokens"},
                      "metal": {k: v for k, v in metal.items() if k != "tokens"},
                      "dense_prefix_v03": {k: v for k, v in dense_row.items() if k != "tokens"},
                      "metal_speedup_vs_gather": round(metal["decode_tok_s"] / gather["decode_tok_s"], 2),
                      "metal_tokens_match_gather": metal["tokens"] == gather["tokens"],
                      "metal_tokens_match_dense": metal["tokens"] == dense_row["tokens"]},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
