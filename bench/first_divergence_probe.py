"""Minimal split-prefill vs full-prefill first-divergence probe.

Uses mx.argmax exactly like mlx-lm's temperature-0 sampler; sorting is only for
displaying candidates. Prompt-internal differences are recorded separately and
generation always continues from the end of the full prompt. Generation stops on
the tokenizer's EOS ids just like the service.
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
    """Return (argmax token, top-2 margin, top-5 display).

    The sampled id must come from mx.argmax, not from the display sort: at an
    exact tie those are not guaranteed to pick the same id.
    """
    token = int(mx.argmax(logits).item())
    idx, vals = _top(logits, 5)
    margin = vals[0] - vals[1] if len(vals) > 1 else 0.0
    return token, margin, idx, vals


def _record(model, tok, split, prompt_len, prompt_first, prompt_count,
            generation_first, generated_s, generated_f, eos_ids):
    return {
        "model": model,
        "prompt_tokens": prompt_len,
        "split": split,
        "eos_token_ids": sorted(eos_ids),
        "prompt_divergence": prompt_first,
        "prompt_divergence_count": prompt_count,
        "first_output_divergence": generation_first,
        "generated_ids_split": generated_s,
        "generated_ids_full": generated_f,
        "generated_text_split": tok.decode(generated_s),
        "generated_text_full": tok.decode(generated_f),
    }


def run(model_path: str, split: int, max_new: int, system_tokens: int, question: str,
        stop_ids: list[int] | None = None):
    model, tok = load(model_path)
    eos_ids = set(stop_ids or getattr(tok, "eos_token_ids", None)
                  or [getattr(tok, "eos_token_id", -1)])
    eos_ids.discard(-1)
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
    for token in ids[split:]:
        out = model(mx.array([[token]]), cache=cache_split)
        mx.eval(out)
        split_prompt_logits.append(out[0, -1, :])

    cache_full = make_prompt_cache(model)
    full_out = model(mx.array([ids]), cache=cache_full)
    mx.eval(full_out)
    full_prompt_logits = [full_out[0, split + i, :] for i in range(len(ids) - split)]

    prompt_first = None
    prompt_count = 0
    for j, (a, b) in enumerate(zip(split_prompt_logits, full_prompt_logits)):
        sa, ma, ia, va = _choice(a)
        sb, mb, ib, vb = _choice(b)
        if sa != sb:
            prompt_count += 1
            if prompt_first is None:
                prompt_first = {
                    "stage": "prompt", "position": split + j,
                    "input_token_id": ids[split + j],
                    "input_token": tok.decode([ids[split + j]]),
                    "split_top1": sa, "full_top1": sb,
                    "split_top1_text": tok.decode([sa]),
                    "full_top1_text": tok.decode([sb]),
                    "split_logits": va, "full_logits": vb,
                    "split_margin": ma, "full_margin": mb,
                    "history_token_ids": ids[: split + j + 1],
                    "history_text": tok.decode(ids[: split + j + 1])[-200:],
                    "split_top5": list(zip(ia, va)),
                    "full_top5": list(zip(ib, vb)),
                }

    split_last, full_last = split_prompt_logits[-1], full_prompt_logits[-1]
    generated_s: list[int] = []
    generated_f: list[int] = []
    generation_first = None
    for step in range(max_new):
        st, sm, si, sv = _choice(split_last)
        ft, fm, fi, fv = _choice(full_last)
        if st != ft or ((st in eos_ids) != (ft in eos_ids)):
            generation_first = {
                "stage": "generation", "step": step,
                "input_token_id": generated_s[-1] if generated_s else ids[-1],
                "input_token": (tok.decode([generated_s[-1]]) if generated_s
                                else tok.decode([ids[-1]])),
                "split_top1": st, "full_top1": ft,
                "split_top1_text": tok.decode([st]),
                "full_top1_text": tok.decode([ft]),
                "split_logits": sv, "full_logits": fv,
                "split_margin": sm, "full_margin": fm,
                "split_stop": st in eos_ids, "full_stop": ft in eos_ids,
                "history_token_ids": ids + generated_s,
                "history_text": tok.decode(ids + generated_s)[-200:],
                "split_top5": list(zip(si, sv)),
                "full_top5": list(zip(fi, fv)),
            }
            break
        if st in eos_ids:  # both paths stop at the same token
            break
        generated_s.append(st)
        generated_f.append(ft)
        s_out = model(mx.array([[st]]), cache=cache_split)
        f_out = model(mx.array([[ft]]), cache=cache_full)
        mx.eval(s_out, f_out)
        split_last, full_last = s_out[0, -1, :], f_out[0, -1, :]

    return _record(model_path, tok, split, len(ids), prompt_first, prompt_count,
                   generation_first, generated_s, generated_f, eos_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / ".hf-models" / "qwen3.5-0.8b"))
    ap.add_argument("--split", type=int, default=2432)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--question", default="请用一句话总结规范二。")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--stop-id", type=int, action="append", default=None,
                    help="override tokenizer EOS ids (repeatable)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "first_divergence_results.json"))
    args = ap.parse_args()
    record = run(args.model, args.split, args.max_new_tokens,
                 args.system_tokens, args.question, args.stop_id)
    Path(args.out).write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(json.dumps(record, indent=1, ensure_ascii=False)[:6000])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
