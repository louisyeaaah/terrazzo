"""Minimal split-prefill vs full-prefill first-divergence probe.

Loads one local MLX model, splits one chat prompt at a chosen token boundary,
and records the first position where greedy top-1 (or prompt logits) differ:
token id, both candidates' logits, top-2 margin, and the common input history.

    python bench/first_divergence_probe.py --model .hf-models/qwen3.5-0.8b
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from co_design_ablation import build_system_prompt  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


def _top(logits, k=5):
    v = logits.astype(mx.float32)
    idx = mx.argsort(v)[-k:][::-1].tolist()
    vals = [float(v[i].item()) for i in idx]
    return idx, vals


def _choice(logits):
    idx, vals = _top(logits, 2)
    margin = vals[0] - vals[1] if len(vals) > 1 else 0.0
    return idx, vals, margin


def run(model_path: str, split: int, max_new: int, system_tokens: int, question: str):
    model, tok = load(model_path)
    system = build_system_prompt(system_tokens)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": question}]
    try:
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=True,
                                           enable_thinking=False))
    except TypeError:
        ids = list(tok.apply_chat_template(messages, add_generation_prompt=True))
    if split <= 0 or split >= len(ids):
        split = int(len(ids) * 0.94)
    if split <= 0:
        raise ValueError("prompt too short")

    cache_split = make_prompt_cache(model)
    model(mx.array([ids[:split]]), cache=cache_split)
    split_prompt_logits = []
    for i, token in enumerate(ids[split:], start=split):
        out = model(mx.array([[token]]), cache=cache_split)
        mx.eval(out)
        split_prompt_logits.append(out[0, -1, :])

    cache_full = make_prompt_cache(model)
    full_out = model(mx.array([ids]), cache=cache_full)
    mx.eval(full_out)
    full_prompt_logits = [full_out[0, split + i, :] for i in range(len(ids) - split)]

    record = {
        "model": model_path, "prompt_tokens": len(ids), "split": split,
        "prompt_sha256": hashlib.sha256(
            json.dumps(ids).encode()).hexdigest()[:16],
        "system_tokens_requested": system_tokens, "question": question,
        "max_new_tokens": max_new,
    }
    # Compare the next-token distribution after every suffix prompt position.
    for j, (a, b) in enumerate(zip(split_prompt_logits, full_prompt_logits)):
        sa, va, ma = _choice(a)
        sb, vb, mb = _choice(b)
        if sa[0] != sb[0]:
            record["first_divergence"] = {
                "stage": "prompt", "position": split + j,
                "input_token_id": ids[split + j],
                "input_token": tok.decode([ids[split + j]]),
                "split_top1": sa, "full_top1": sb,
                "split_logits": va, "full_logits": vb,
                "split_margin": ma, "full_margin": mb,
                "history_token_ids": ids[: split + j + 1],
                "history_text": tok.decode(ids[: split + j + 1])[-200:],
                "split_top5": list(zip(*_top(a, 5))),
                "full_top5": list(zip(*_top(b, 5))),
            }
            break
    # If prompt logits agree, generate greedily from both caches and compare.
    if "first_divergence" not in record:
        split_last = split_prompt_logits[-1]
        full_last = full_prompt_logits[-1]
        generated = []
        s_cache, f_cache = cache_split, cache_full
        for step in range(max_new):
            sa, va, ma = _choice(split_last)
            sb, vb, mb = _choice(full_last)
            if sa[0] != sb[0]:
                record["first_divergence"] = {
                    "stage": "generation", "step": step,
                    "input_token_id": generated[-1] if generated else ids[-1],
                    "input_token": tok.decode([generated[-1]]) if generated else tok.decode([ids[-1]]),
                    "split_top1": sa, "full_top1": sb,
                    "split_logits": va, "full_logits": vb,
                    "split_margin": ma, "full_margin": mb,
                    "history_token_ids": ids + generated,
                    "history_text": tok.decode(ids + generated)[-200:],
                    "split_top5": list(zip(*_top(split_last, 5))),
                    "full_top5": list(zip(*_top(full_last, 5))),
                }
                break
            token = sa[0]
            generated.append(token)
            s_out = model(mx.array([[token]]), cache=s_cache)
            f_out = model(mx.array([[token]]), cache=f_cache)
            mx.eval(s_out, f_out)
            split_last, full_last = s_out[0, -1, :], f_out[0, -1, :]
        record["generated_ids"] = generated
        record["generated_text"] = tok.decode(generated)
    if "first_divergence" not in record:
        record["first_divergence"] = None
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / ".hf-models" / "qwen3.5-0.8b"))
    ap.add_argument("--split", type=int, default=2432)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--question", default="请用一句话总结规范二。")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "bench" / "first_divergence_results.json"))
    args = ap.parse_args()
    record = run(args.model, args.split, args.max_new_tokens,
                 args.system_tokens, args.question)
    Path(args.out).write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(json.dumps(record, indent=1, ensure_ascii=False)[:5000])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
