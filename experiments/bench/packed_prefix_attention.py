"""Shared-prefix decode attention: can packing B sessions beat B x SDPA?

Agent traffic has one long shared prefix + tiny per-session suffixes. The naive
batched path runs one SDPA per session and reads the prefix B times. Variants:

  baseline    : B x SDPA over each session's full KV (current behaviour)
  packed_plain: one SDPA with q_len=B over prefix+all suffixes, NO mask
                (fast but wrong: rows see other sessions' suffixes -- upper bound only)
  packed_mask : same but with a per-row mask that hides other sessions' suffixes
                (exact; measures the cost of masking)
  manual      : matmul + softmax over [H,B,P] (exact but materialises scores)
  mma_merge   : MMA flash attention over the prefix (returning lse) + hand suffix
                attention + exact log-sum-exp merge (see terrazzo/mma_flash_attn.py)

    python experiments/bench/packed_prefix_attention.py --prefix 3400 --batch 4 --suffix 32
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

D = 128
HQ = 32
HK = 8
REP = HQ // HK


def bench(fn, n: int = 5) -> float:
    fn()
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=3400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--suffix", type=int, default=32)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    P, B, S, NL = args.prefix, args.batch, args.suffix, args.layers

    mx.random.seed(0)
    # GQA shapes as in the model: 8 KV heads, 32 query heads (SDPA broadcasts)
    kp = mx.random.normal((HK, P, D)).astype(dtype)
    vp = mx.random.normal((HK, P, D)).astype(dtype)
    ks = mx.random.normal((B, HK, S, D)).astype(dtype)
    vs = mx.random.normal((B, HK, S, D)).astype(dtype)
    q = mx.random.normal((HQ, B, D)).astype(dtype)       # packed: one row per session

    # shared KV for packed variants: prefix + all sessions' suffixes
    kkv = mx.concatenate(
        [kp[None], mx.reshape(mx.transpose(ks, (1, 0, 2, 3)), (1, HK, B * S, D))], axis=2)
    vkv = mx.concatenate(
        [vp[None], mx.reshape(mx.transpose(vs, (1, 0, 2, 3)), (1, HK, B * S, D))], axis=2)
    mask = mx.full((1, 1, B, P + B * S), -1e9, dtype=dtype)
    mask[:, :, :, :P] = 0.0
    for b in range(B):
        mask[0, 0, b, P + b * S:P + (b + 1) * S] = 0.0

    qpack = q[None]                                      # [1,HQ,B,D]

    def baseline():
        outs = []
        for b in range(B):
            k = mx.concatenate([kp, ks[b]], axis=1)
            v = mx.concatenate([vp, vs[b]], axis=1)
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, b][None, :, None, :], k[None], v[None], scale=D ** -0.5))
        return mx.concatenate(outs, axis=2)

    def packed_plain():
        return mx.fast.scaled_dot_product_attention(qpack, kkv, vkv, scale=D ** -0.5)

    def packed_mask():
        return mx.fast.scaled_dot_product_attention(qpack, kkv, vkv,
                                                    scale=D ** -0.5, mask=mask)

    def manual():
        vexp = mx.repeat(vp, REP, axis=0)                # [HQ,P,D]
        kt = mx.swapaxes(mx.repeat(kp, REP, axis=0), 1, 2)  # [HQ,D,P]
        scores = mx.matmul(q, kt) * (D ** -0.5)          # [HQ,B,P]
        probs = mx.softmax(scores.astype(mx.float32), axis=-1)
        return mx.matmul(probs.astype(dtype), vexp)      # [HQ,B,D]

    ref = baseline()
    mx.eval(ref)

    rows = []
    for name, fn, exact in (("baseline", baseline, True),
                            ("packed_plain", packed_plain, False),
                            ("packed_mask", packed_mask, True),
                            ("manual(prefix only)", manual, False)):
        out = fn()
        mx.eval(out)
        t = bench(fn)
        if exact:
            diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        else:
            diff = float("nan")
        rows.append((name, t, t * NL, diff))

    print(f"prefix={P} batch={B} suffix={S} dtype={args.dtype}, per layer / {NL} layers")
    base_ms = rows[0][1]
    for name, t, tot, diff in rows:
        d = "exact" if diff == diff and diff < 2e-2 else ("WRONG" if diff == diff else "-")
        print(f"  {name:20s}: {t:7.3f} ms/layer {tot:8.1f} ms/{NL}L  "
              f"{base_ms / t:5.2f}x  maxdiff={diff if diff == diff else -1:.4f} {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
