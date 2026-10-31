"""Isolate snapshot save/restore from prefill chunking.

At one fixed token boundary and with the same suffix, compare:
  A. live cache: prefill prefix, then process suffix;
  B. snapshot cache: same prefix, save via Engine._native_snapshot/layers,
     restore via Engine._seed_caches, then process the same suffix.

If A and B disagree, the difference is introduced by snapshot save/restore.
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


def _choice(logits):
    token = int(mx.argmax(logits).item())
    v = logits.astype(mx.float32)
    idx = mx.argsort(v)[-2:][::-1].tolist()
    vals = [float(v[i].item()) for i in idx]
    return token, (vals[0] - vals[1] if len(vals) > 1 else 0.0), idx, vals


def _run_batcher(model, eos, max_new, segments, all_tokens, caches):
    bg = BatchGenerator(model, max_tokens=max_new,
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
    return out


def run(model_path: str, split: int, system_tokens: int, question: str,
        max_new: int):
    model, tok = load(model_path)
    eos = set(getattr(tok, "eos_token_ids", None) or [getattr(tok, "eos_token_id", -1)])
    eos.discard(-1)
    system = build_system_prompt(system_tokens)
    ids = list(tok.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": question}],
        add_generation_prompt=True, enable_thinking=False))
    if split <= 0 or split >= len(ids):
        raise ValueError(f"split {split} outside prompt {len(ids)}")

    live = make_prompt_cache(model)
    model(mx.array([ids[:split]]), cache=live)
    snapshot = Engine._native_snapshot(live)
    bm = BlockManager(BlockManagerConfig(num_blocks=4096, block_size=16))
    pc = PagedPrefixCache(bm, max_entries=2, max_bytes=1 << 30)
    if snapshot is not None:
        stored = pc.store(ids[:split], [], native=snapshot)
    else:
        layers = [(c.keys[..., :split, :], c.values[..., :split, :]) for c in live]
        stored = pc.store(ids[:split], layers)
    if stored is None:
        raise RuntimeError("PagedPrefixCache.store returned None")
    restored = Engine.__new__(Engine)._seed_caches(stored, split)
    if not restored:
        raise RuntimeError("snapshot restore returned no caches")
    restored_for_batch = Engine.__new__(Engine)._seed_caches(stored, split)

    # Compare the actual cache state, not only generated tokens.
    def _arrays_equal(a, b):
        if isinstance(a, list):
            return len(a) == len(b) and all(_arrays_equal(x, y) for x, y in zip(a, b))
        if isinstance(a, tuple):
            return len(a) == len(b) and all(_arrays_equal(x, y) for x, y in zip(a, b))
        if isinstance(a, mx.array) or isinstance(b, mx.array):
            try:
                if a.shape != b.shape:
                    return False
                return bool(mx.all(a == b).item())
            except Exception:
                return False
        return a == b

    state_equal = all(_arrays_equal(a.state, b.state)
                      for a, b in zip(live, restored))
    meta_equal = all(a.meta_state == b.meta_state for a, b in zip(live, restored))

    # Match the service's prefill shape: process the whole suffix as one chunk.
    suffix = ids[split:]
    a = model(mx.array([suffix]), cache=live)
    b = model(mx.array([suffix]), cache=restored)
    mx.eval(a, b)
    live_last, restored_last = a[0, -1, :], b[0, -1, :]
    sa, ma, ia, va = _choice(live_last)
    sb, mb, ib, vb = _choice(restored_last)
    suffix_logits_max_diff = None
    try:
        suffix_logits_max_diff = float(mx.max(mx.abs(
            live_last.astype(mx.float32) - restored_last.astype(mx.float32))).item())
    except Exception:
        pass
    suffix_first = None
    if sa != sb:
        suffix_first = {
            "stage": "suffix_end", "position": len(ids) - 1,
            "input_token_id": ids[-1], "input_token": tok.decode([ids[-1]]),
            "live_top1": sa, "restored_top1": sb,
            "live_logits": va, "restored_logits": vb,
            "live_margin": ma, "restored_margin": mb,
            "history_token_ids": ids,
            "history_text": tok.decode(ids)[-200:],
        }

    # Continue greedy generation from the shared suffix end.
    live_gen: list[int] = []
    restored_gen: list[int] = []
    gen_first = None
    for step in range(max_new):
        sa, ma, ia, va = _choice(live_last)
        sb, mb, ib, vb = _choice(restored_last)
        if sa != sb or ((sa in eos) != (sb in eos)):
            gen_first = {
                "stage": "generation", "step": step,
                "input_token_id": live_gen[-1] if live_gen else ids[-1],
                "live_top1": sa, "restored_top1": sb,
                "live_logits": va, "restored_logits": vb,
                "live_margin": ma, "restored_margin": mb,
                "history_token_ids": ids + live_gen,
                "history_text": tok.decode(ids + live_gen)[-200:],
                "live_stop": sa in eos, "restored_stop": sb in eos,
            }
            break
        if sa in eos:
            break
        live_gen.append(sa)
        restored_gen.append(sb)
        a = model(mx.array([[sa]]), cache=live)
        b = model(mx.array([[sb]]), cache=restored)
        mx.eval(a, b)
        live_last, restored_last = a[0, -1, :], b[0, -1, :]

    # Third path: the actual BatchGenerator insert/merge path used by the server.
    batcher_ids = _run_batcher(model, eos, max_new, [[ids[split:]]],
                               [ids[:split]], [restored_for_batch])
    # Fourth path: BatchGenerator without a seeded cache, to separate cache
    # handling from BatchGenerator itself.
    batcher_nocache_ids = _run_batcher(model, eos, max_new, [[ids]], [[]], None)
    return {
        "model": model_path, "prompt_tokens": len(ids), "split": split,
        "eos_token_ids": sorted(eos),
        "snapshot_kind": "native" if snapshot is not None else "dense-kv",
        "state_equal": state_equal,
        "meta_state_equal": meta_equal,
        "suffix_logits_max_abs_diff": suffix_logits_max_diff,
        "suffix_first_divergence": suffix_first,
        "generation_first_divergence": gen_first,
        "generated_ids_live": live_gen,
        "generated_ids_restored": restored_gen,
        "generated_text_live": tok.decode(live_gen),
        "generated_text_restored": tok.decode(restored_gen),
        "batcher_generated_ids": batcher_ids,
        "batcher_matches_live": batcher_ids == live_gen,
        "batcher_generated_text": tok.decode(batcher_ids),
        "batcher_nocache_ids": batcher_nocache_ids,
        "batcher_nocache_matches_live": batcher_nocache_ids == live_gen,
        "batcher_nocache_text": tok.decode(batcher_nocache_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / ".hf-models" / "qwen3.5-0.8b"))
    ap.add_argument("--split", type=int, default=2432)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--question", default="请用一句话总结规范二。")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "bench" / "snapshot_restore_results.json"))
    args = ap.parse_args()
    rec = run(args.model, args.split, args.system_tokens, args.question,
              args.max_new_tokens)
    Path(args.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec, indent=1, ensure_ascii=False)[:5000])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
