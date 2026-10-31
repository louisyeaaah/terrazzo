"""Reproduce Qwen3-4B turn-2 cache/no-cache output difference with exact chunks."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from co_design_ablation import build_system_prompt  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from terrazzo.engine import Engine  # noqa: E402
from terrazzo.blocks import BlockManager, BlockManagerConfig  # noqa: E402
from terrazzo.paged_kv import PagedPrefixCache  # noqa: E402
from batcher_chunk_probe import (_manual_generate, _batcher, _first_diff,  # noqa: E402
                                 _choice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/yezhipeng/project/dsh-other/deepinfer-study/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25")
    ap.add_argument("--split", type=int, default=2432)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default=str(ROOT / "bench" / "multi_turn_cache_results.json"))
    args = ap.parse_args()
    model, tok = load(args.model)
    eos = set(getattr(tok, "eos_token_ids", None) or [getattr(tok, "eos_token_id", -1)])
    eos.discard(-1)
    system = build_system_prompt(2000)
    turn1 = "规范一要求所有回答必须引用条款编号。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "请用一句话总结规范一。"},
        {"role": "assistant", "content": turn1},
        {"role": "user", "content": "请再补充一句。"},
    ]
    try:
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=True,
                                           enable_thinking=False))
    except TypeError:
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=True))
    split = args.split if 0 < args.split < len(ids) else int(len(ids) * 0.94)
    suffix = ids[split:]

    def chunks(seq, step=512):
        body = seq[:-1]
        out = [body[i:i + step] for i in range(0, len(body), step)] if body else []
        out.append(seq[-1:])
        return out

    live = make_prompt_cache(model)
    model(mx.array([ids[:split]]), cache=live)
    snapshot = Engine._native_snapshot(live)
    bm = BlockManager(BlockManagerConfig(num_blocks=4096, block_size=16))
    pc = PagedPrefixCache(bm, max_entries=2, max_bytes=1 << 30)
    stored = (pc.store(ids[:split], [], native=snapshot) if snapshot is not None
              else pc.store(ids[:split],
                            [(c.keys[..., :split, :], c.values[..., :split, :])
                             for c in live]))
    restored = Engine.__new__(Engine)._seed_caches(stored, split)
    cache_ids, cache_calls = _manual_generate(model, restored, chunks(suffix), eos,
                                              args.max_new_tokens)
    nocache_ids, nocache_calls = _manual_generate(model, make_prompt_cache(model),
                                                  chunks(ids), eos, args.max_new_tokens)
    batch_cache_ids, batch_cache_calls = _batcher(
        model, eos, args.max_new_tokens, [[suffix]], [ids[:split]],
        [Engine.__new__(Engine)._seed_caches(stored, split)])
    batch_nocache_ids, batch_nocache_calls = _batcher(
        model, eos, args.max_new_tokens, [[ids]], [[]], None)
    prompt_cache_n = len(chunks(suffix))
    prompt_nocache_n = len(chunks(ids))
    diff = _first_diff(cache_ids, nocache_ids)
    cache_stopped = any(c.get("terminal") for c in cache_calls[prompt_cache_n:])
    nocache_stopped = any(c.get("terminal") for c in nocache_calls[prompt_nocache_n:])

    def gen_logits(calls, prompt_n, i):
        idx = prompt_n + i
        if 0 <= idx < len(calls):
            c = calls[idx]
            return {"input_token_ids": c.get("input_token_ids"),
                    "output_top1": c.get("output_top1"),
                    "output_logits": c.get("output_logits"),
                    "output_margin": c.get("output_margin")}
        return None

    out = {
        "model": args.model, "prompt_tokens": len(ids), "split": split,
        "suffix_tokens": len(suffix), "cache_chunks": [len(c) for c in chunks(suffix)],
        "cache_ids": cache_ids, "nocache_ids": nocache_ids,
        "batch_cache_ids": batch_cache_ids, "batch_nocache_ids": batch_nocache_ids,
        "cache_vs_nocache_first_diff_zero_based": diff,
        "cache_vs_nocache_first_diff_ordinal": (diff + 1 if diff is not None else None),
        "cache_stopped_on_eos": cache_stopped,
        "nocache_stopped_on_eos": nocache_stopped,
        "cache_tokens_excluding_eos": len(cache_ids),
        "nocache_tokens_excluding_eos": len(nocache_ids),
        "cache_tokens_including_eos": len(cache_ids) + (1 if cache_stopped else 0),
        "nocache_tokens_including_eos": len(nocache_ids) + (1 if nocache_stopped else 0),
        "request_provenance": "constructed from fixed turn-1 assistant text; full HTTP history token IDs were not archived",
        "manual_cache_matches_batch": cache_ids == batch_cache_ids,
        "manual_nocache_matches_batch": nocache_ids == batch_nocache_ids,
        "first_diff_cache_logits": (gen_logits(cache_calls, prompt_cache_n, diff)
                                    if diff is not None else None),
        "first_diff_nocache_logits": (gen_logits(nocache_calls, prompt_nocache_n, diff)
                                      if diff is not None else None),
        "first_diff_input_text": (tok.decode([cache_calls[prompt_cache_n + diff]["input_token_ids"][-1]])
                                  if diff is not None else None),
        "first_diff_cache_text": (tok.decode([cache_calls[prompt_cache_n + diff]["output_top1"]])
                                  if diff is not None else None),
        "first_diff_nocache_text": (tok.decode([nocache_calls[prompt_nocache_n + diff]["output_top1"]])
                                    if diff is not None else None),
        "cache_text": tok.decode(cache_ids),
        "nocache_text": tok.decode(nocache_ids),
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in
                      ("prompt_tokens", "split", "suffix_tokens", "cache_chunks",
                       "cache_vs_nocache_first_diff_zero_based",
                       "cache_vs_nocache_first_diff_ordinal",
                       "cache_tokens_including_eos", "nocache_tokens_including_eos",
                       "manual_cache_matches_batch",
                       "manual_nocache_matches_batch", "cache_text", "nocache_text")},
                     indent=1, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
