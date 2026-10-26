"""Packed shared-prefix decode attention with the MMA kernel + exact suffix merge.

B sessions share a long prefix and each has a tiny suffix. Instead of B x SDPA over
the full KV (prefix read B times), we run one MMA attention with the B queries packed
(q_len=B) over the shared prefix, compute each session's suffix attention by hand
(suffix is tiny), and merge both exactly with log-sum-exp.

    python experiments/bench/packed_prefix_mma.py --prefix 3400 --batch 4 --suffix 32

NOTE (2026-10-27): the MMA kernel is numerically correct but ~10x slower than SDPA
for this shape (11.4 ms vs 1.1 ms per layer), so this variant loses. Kept as a
reproduction artifact for the negative result in BENCHMARKS §43.
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


def suffix_attention_batched(q, ks, vs):
    """q [Hq,B,D], ks/vs [B,Hk,S,D] -> out [B,Hq,D], lse [B,Hq] (vectorised)."""
    kexp = mx.repeat(ks, REP, axis=1)      # [B,Hq,S,D]
    vexp = mx.repeat(vs, REP, axis=1)
    scores = mx.einsum("hbd,bhsd->bhs", q, kexp).astype(mx.float32) * (D ** -0.5)
    lse = mx.logsumexp(scores, axis=-1)                        # [B,Hq]
    probs = mx.softmax(scores, axis=-1).astype(q.dtype)
    out = mx.einsum("bhs,bhsd->bhd", probs, vexp)              # [B,Hq,D]
    return out, lse


def merge(out_p, lse_p, out_s, lse_s):
    """out_p [Hq,B,D], lse_p [Hq,B]; out_s [B,Hq,D], lse_s [B,Hq] -> [Hq,B,D]."""
    op = mx.swapaxes(out_p, 0, 1).astype(mx.float32)           # [B,Hq,D]
    lp = mx.swapaxes(lse_p, 0, 1)                              # [B,Hq]
    M = mx.maximum(lp, lse_s)
    wp = mx.exp(lp - M)[..., None]
    ws = mx.exp(lse_s - M)[..., None]
    fused = (op * wp + out_s.astype(mx.float32) * ws) / (wp + ws)
    return mx.swapaxes(fused, 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=int, default=3400)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--suffix", type=int, default=32)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    dtype = mx.bfloat16 if args.dtype == "bfloat16" else mx.float16
    P, B, S, L = args.prefix, args.batch, args.suffix, args.layers

    mx.random.seed(0)
    # one representative layer
    kp = mx.random.normal((HK, P, D)).astype(dtype)
    vp = mx.random.normal((HK, P, D)).astype(dtype)
    kt = mx.contiguous(mx.swapaxes(kp, 1, 2))
    q = mx.random.normal((HQ, B, D)).astype(dtype)                # B session queries
    ks = mx.random.normal((B, HK, S, D)).astype(dtype)
    vs = mx.random.normal((B, HK, S, D)).astype(dtype)

    from experiments.mma_flash_attn import mma_flash_attention

    # ---- reference: full attention over prefix + this session's suffix (fp32)
    def ref():
        outs = []
        for b in range(B):
            kcat = mx.concatenate([kp, ks[b]], axis=1)
            vcat = mx.concatenate([vp, vs[b]], axis=1)
            per_head = []
            for h in range(HQ):
                kh = h // REP
                sc = mx.matmul(q[h, b][None].astype(mx.float32),
                               mx.swapaxes(kcat[kh], 0, 1).astype(mx.float32)) * (D ** -0.5)
                p = mx.softmax(sc, axis=-1)
                per_head.append(mx.matmul(p, vcat[kh].astype(mx.float32))[0])
            outs.append(mx.stack(per_head))
        return mx.stack(outs, axis=1)                             # [Hq,B,D]

    def packed():
        op, lp = mma_flash_attention(q, kt, vp, causal=False, return_lse=True)
        os_, ls_ = suffix_attention_batched(q, ks, vs)
        return merge(op, lp, os_, ls_)

    ref_out = ref()
    got = packed()
    mx.eval(ref_out, got)
    diff = float(mx.max(mx.abs(got - ref_out)))

    # ---- timing per layer
    def baseline():
        outs = []
        for b in range(B):
            kcat = mx.concatenate([kp, ks[b]], axis=1)
            vcat = mx.concatenate([vp, vs[b]], axis=1)
            outs.append(mx.fast.scaled_dot_product_attention(
                q[:, b][None, :, None, :], kcat[None], vcat[None], scale=D ** -0.5))
        return mx.concatenate(outs, axis=2)

    t_base = bench(baseline)
    t_packed = bench(packed)
    print(f"prefix={P} batch={B} suffix={S} dtype={args.dtype}")
    print(f"combined maxdiff vs full reference: {diff:.4f}")
    print(f"B x SDPA(full KV): {t_base:6.2f} ms/layer -> {t_base * L:6.1f} ms/{L}L")
    print(f"packed MMA+merge:  {t_packed:6.2f} ms/layer -> {t_packed * L:6.1f} ms/{L}L  "
          f"({t_base / t_packed:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
