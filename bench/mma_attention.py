"""Can simdgroup MMA beat MLX SDPA for attention-shaped work?

Both measurements cover the same FLOPs: QK^T for `heads` query heads, `BQ` queries
each, against `KV` keys with head_dim D (GQA: q head -> kv head = head // rep).
The MMA kernel materialises the score tile (8 rows x 64 cols per threadgroup);
SDPA is measured on the same tensors (without mask) as the reference throughput.

    python bench/mma_attention.py [--kv 4000] [--bq 64] [--heads 32]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

D = 128
REP = 4   # 32 q heads / 8 kv heads

_SOURCE = """
    // one threadgroup = 1 simdgroup: 8 queries x 64 keys score tile
    const int D = dims[0];
    const int KV = dims[1];
    const int BQ = dims[2];
    const int H = dims[3];
    const int REP = dims[4];
    const int qb = threadgroup_position_in_grid.x;   // (head, 8-query block) index
    const int kb = threadgroup_position_in_grid.y;   // 64-key block index
    const int qblocks = BQ / 8;
    const int head = qb / qblocks;
    const int q0 = (qb % qblocks) * 8;
    const int k0 = kb * 64;
    if (q0 >= BQ || k0 >= KV) { return; }
    const int kv_head = head / REP;

    simdgroup_half8x8 c[8];
    for (int j = 0; j < 8; ++j) { c[j] = simdgroup_half8x8(0); }
    const device half* Qh = Q + (int(head) * BQ + q0) * D;
    for (int kk = 0; kk < D / 8; ++kk) {
        simdgroup_half8x8 a;
        simdgroup_load(a, Qh + kk * 8, D);
        for (int j = 0; j < 8; ++j) {
            const int key = k0 + j * 8;
            if (key >= KV) { continue; }
            simdgroup_half8x8 b;
            // K is stored transposed: [head, D, KV] -> fragment (d, key)
            const device half* Kh = Kt + ((int(kv_head) * D + kk * 8) * KV) + key;
            simdgroup_load(b, Kh, KV);
            simdgroup_multiply_accumulate(c[j], a, b, c[j]);
        }
    }
    for (int j = 0; j < 8; ++j) {
        const int key = k0 + j * 8;
        if (key >= KV) { continue; }
        simdgroup_store(c[j], S + ((int(head) * BQ + q0) * KV + key), KV);
    }
"""

_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mma_scores", input_names=["Q", "Kt", "dims"], output_names=["S"],
            source=_SOURCE)
    return _KERNEL


def mma_scores(q, kt, kv_heads: int) -> mx.array:
    H, BQ, _ = q.shape
    KV = kt.shape[2]
    S = _kernel()(
        inputs=[q.reshape(-1), kt.reshape(-1),
                mx.array([D, KV, BQ, H, H // kv_heads], dtype=mx.int32)],
        grid=((BQ // 8) * H * 32, (KV + 63) // 64, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(H, BQ, KV)], output_dtypes=[q.dtype])[0]
    return S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=int, default=4000)
    ap.add_argument("--bq", type=int, default=64)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    mx.random.seed(0)
    q = mx.random.normal((args.heads, args.bq, D)).astype(mx.float16)
    k = mx.random.normal((args.heads // REP, args.kv, D)).astype(mx.float16)

    krep = mx.repeat(k, REP, axis=0)          # [H, KV, D]
    kt = mx.contiguous(mx.swapaxes(k, 1, 2))     # [kv_heads, D, KV], MMA-friendly
    S = mma_scores(q, kt, args.heads // REP)
    ref = mx.matmul(q, mx.swapaxes(krep, 1, 2))
    mx.eval(S, ref)
    diff = float(mx.max(mx.abs(S.astype(mx.float32) - ref.astype(mx.float32))))
    flops = 2 * args.heads * args.bq * args.kv * D

    def bench(fn, n):
        fn()
        mx.eval(fn())
        t0 = time.perf_counter()
        for _ in range(n):
            mx.eval(fn())
        return (time.perf_counter() - t0) / n

    t_mma = bench(lambda: mma_scores(q, kt, args.heads // REP), args.reps)
    kk = mx.repeat(mx.swapaxes(k, 1, 2), REP, axis=0)
    q4, k4 = q[None], krep[None]
    t_sdpa = bench(lambda: mx.fast.scaled_dot_product_attention(
        q4, k4, k4, scale=D ** -0.5), args.reps)
    print(f"shape: heads={args.heads} bq={args.bq} kv={args.kv} D={D}")
    print(f"score maxdiff vs matmul: {diff:.4f}")
    print(f"MMA  QK^T: {t_mma * 1000:6.2f} ms  -> {flops / t_mma / 1e12:5.2f} TFLOP/s")
    print(f"SDPA same QK^T+softmax+AV: {t_sdpa * 1000:6.2f} ms  -> "
          f"{flops / t_sdpa / 1e12:5.2f} TFLOP/s (attention does ~2x these FLOPs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
