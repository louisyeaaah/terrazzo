"""Narrow control: direct model calls with the exact batcher prefill chunks.

Compares a manual greedy path that reproduces BatchGenerator's chunking
(including the final single prompt token) against BatchGenerator itself, with a
recording proxy that captures each model call's logits and cache layout.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from co_design_ablation import build_system_prompt  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402
from terrazzo.engine import Engine  # noqa: E402
from terrazzo.blocks import BlockManager, BlockManagerConfig  # noqa: E402
from terrazzo.paged_kv import PagedPrefixCache  # noqa: E402


def _top(logits, k=2):
    v = logits.astype(mx.float32)
    idx = mx.argsort(v)[-k:][::-1].tolist()
    vals = [float(v[i].item()) for i in idx]
    return idx, vals


def _choice(logits):
    token = int(mx.argmax(logits).item())
    idx, vals = _top(logits, 2)
    return token, (vals[0] - vals[1] if len(vals) > 1 else 0.0), idx, vals


class Recorder:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, tokens, cache=None):
        out = self.model(tokens, cache=cache)
        mx.eval(out)
        try:
            last = out[0, -1, :]
            tok, margin, idx, vals = _choice(last)
            caches = []
            for c in cache or []:
                caches.append({"type": type(c).__name__,
                               "offset": (int(c.offset) if hasattr(c, "offset")
                                          and not isinstance(c.offset, list) else None)})
            self.calls.append({"shape": list(tokens.shape),
                               "input_ids": tokens.reshape(-1).tolist(),
                               "top1": tok, "margin": margin,
                               "top2": idx, "logits": vals,
                               "caches": caches[:4]})
        except Exception as exc:  # diagnostic only
            self.calls.append({"shape": list(tokens.shape), "error": repr(exc)})
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)


def _manual_generate(model, cache, chunks, eos, max_new):
    calls = []
    last = None
    for chunk in chunks:
        out = model(mx.array([chunk]), cache=cache)
        mx.eval(out)
        last = out[0, -1, :]
        tok, margin, idx, vals = _choice(last)
        calls.append({"chunk_len": len(chunk), "top1": tok, "margin": margin,
                      "top2": idx, "logits": vals})
    generated = []
    for _ in range(max_new):
        tok, margin, idx, vals = _choice(last)
        if tok in eos:
            break
        generated.append(tok)
        out = model(mx.array([[tok]]), cache=cache)
        mx.eval(out)
        last = out[0, -1, :]
        calls.append({"chunk_len": 1, "top1": tok, "margin": margin,
                      "top2": idx, "logits": vals, "generated": True})
    return generated, calls


def _batcher(model, eos, max_new, segments, all_tokens, caches):
    rec = Recorder(model)
    bg = BatchGenerator(rec, max_tokens=max_new,
                        stop_tokens=[(t,) for t in eos] if eos else None,
                        sampler=lambda x: mx.argmax(x, axis=-1),
                        completion_batch_size=1, prefill_batch_size=1,
                        prefill_step_size=512)
    out = []
    try:
        bg.insert_segments(segments=segments, max_tokens=[max_new],
                           caches=caches, all_tokens=all_tokens,
                           samplers=[lambda x: mx.argmax(x, axis=-1)])
        for _ in range(256):
            busy = (len(bg._unprocessed_sequences) or len(bg._currently_processing)
                    or len(bg._prompt_batch) or len(bg._generation_batch))
            if not busy:
                break
            for r in bg.next_generated():
                if r.token is not None:
                    out.append(int(r.token))
                if r.finish_reason is not None:
                    busy = False
                    break
        if out and out[-1] in eos:
            out = out[:-1]
    finally:
        bg.close()
    return out, rec.calls


def _first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def run(model_path, split, system_tokens, question, max_new):
    model, tok = load(model_path)
    eos = set(getattr(tok, "eos_token_ids", None) or [getattr(tok, "eos_token_id", -1)])
    eos.discard(-1)
    system = build_system_prompt(system_tokens)
    ids = list(tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": question}],
        add_generation_prompt=True, enable_thinking=False))
    suffix = ids[split:]
    assert suffix and split < len(ids)

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

    # Batcher chunks the suffix as [all but last] + [last]; no-cache chunks the
    # whole prompt at prefill_step_size=512, also ending with one token.
    manual_cache_chunks = [suffix[:-1], suffix[-1:]]
    manual_cache, cache_calls = _manual_generate(model, restored, manual_cache_chunks,
                                                 eos, max_new)
    prompt_body = ids[:-1]
    manual_nocache_chunks = [prompt_body[i:i + 512]
                             for i in range(0, len(prompt_body), 512)]
    manual_nocache_chunks.append(ids[-1:])
    manual_nocache, nocache_calls = _manual_generate(
        model, make_prompt_cache(model), manual_nocache_chunks, eos, max_new)

    batch_cache, batch_cache_calls = _batcher(
        model, eos, max_new, [[suffix]], [ids[:split]],
        [Engine.__new__(Engine)._seed_caches(stored, split)])
    batch_nocache, batch_nocache_calls = _batcher(model, eos, max_new, [[ids]], [[]], None)

    return {
        "model": model_path, "prompt_tokens": len(ids), "split": split,
        "manual_cache_chunks": [len(c) for c in manual_cache_chunks],
        "manual_nocache_chunks": [len(c) for c in manual_nocache_chunks],
        "manual_cache_ids": manual_cache,
        "manual_nocache_ids": manual_nocache,
        "batch_cache_ids": batch_cache,
        "batch_nocache_ids": batch_nocache,
        "cache_first_diff": _first_diff(manual_cache, batch_cache),
        "nocache_first_diff": _first_diff(manual_nocache, batch_nocache),
        "cache_matches": manual_cache == batch_cache,
        "nocache_matches": manual_nocache == batch_nocache,
        "manual_cache_calls": cache_calls,
        "batch_cache_model_calls": batch_cache_calls,
        "manual_nocache_calls": nocache_calls,
        "batch_nocache_model_calls": batch_nocache_calls,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / ".hf-models" / "qwen3.5-0.8b"))
    ap.add_argument("--split", type=int, default=2432)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--question", default="请用一句话总结规范二。")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "bench" / "batcher_chunk_results.json"))
    args = ap.parse_args()
    rec = run(args.model, args.split, args.system_tokens, args.question,
              args.max_new_tokens)
    Path(args.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in
                      ("manual_cache_chunks", "manual_nocache_chunks",
                       "cache_first_diff", "nocache_first_diff",
                       "cache_matches", "nocache_matches")}, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
