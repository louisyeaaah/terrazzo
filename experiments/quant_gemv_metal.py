"""Weight-only 4-bit GEMV: one warp per output row, dequant in registers.

Target: decode (M=1) quantized linear, where MLX's ``quantized_matmul`` measures
~205-279 GB/s on M1 Max (7.3-10 ms for Qwen3-4B's 36-layer weight stream). A GEMV
has no reuse, so the only job is to stream the 4-bit weights at full bandwidth.

Weight layout (MLX ``mx.quantize`` with group_size=64, bits=4):
    wq     [N, K/8] uint32   (8 nibbles per word, low nibble first)
    scales [N, K/64] T
    biases [N, K/64] T
    w[n, k] = q(n, k) * scales[n, k // 64] + biases[n, k // 64]

Kernel: one threadgroup of 128 threads (4 warps) handles 4 output rows, one warp per
row. x is staged in threadgroup memory (K*2 bytes; K is a compile-time constant so
the shared array is exactly sized). Each lane walks a contiguous K/32 slice, one
group's scale/bias are loaded once per packed word (8 nibbles always stay inside one
64-element group), accumulates in fp32 and finishes with one simd_sum.
"""

from __future__ import annotations

import mlx.core as mx

_KERNELS: dict = {}
ROWS_PER_TG = 4
THREADS = 128

_SOURCE = """
    const int K = {k};
    const int N = {n};
    const int G = {g};           // groups per row (K/64)
    const int GS = {gs};         // group size (64)
    const int S = K / 32;        // elements per lane (multiple of 8)
    const int WPER = S / 8;      // packed words per lane

    const uint tid = thread_position_in_threadgroup.x;
    const uint warp = tid / 32;
    const uint lane = tid % 32;
    const uint row = threadgroup_position_in_grid.x * 4 + warp;

    if (row >= uint(N)) {{ return; }}

    const device uint* wrow = wq + int(row) * (K / 8) + int(lane) * WPER;
    const device T* srow = scales + int(row) * G;
    const device T* brow = biases + int(row) * G;

    float acc = 0.0f;
    const int wvec = WPER / 2;             // WPER is even for all model shapes
    const device uint2* wvecp = (const device uint2*)(wrow);
    for (int w = 0; w < wvec; ++w) {{
        const uint2 word2 = wvecp[w];
        const int e0 = int(lane) * S + w * 16;
        const int g0 = e0 / GS;
        const int g1 = (e0 + 8) / GS;
        const float sc0 = float(srow[g0]);
        const float bi0 = float(brow[g0]);
        const float sc1 = float(srow[g1]);
        const float bi1 = float(brow[g1]);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {{
            const int q0 = int((word2.x >> (4 * j)) & 0xFu);
            acc += (float(q0) * sc0 + bi0) * float(x[e0 + j]);
        }}
        #pragma unroll
        for (int j = 0; j < 8; ++j) {{
            const int q1 = int((word2.y >> (4 * j)) & 0xFu);
            acc += (float(q1) * sc1 + bi1) * float(x[e0 + 8 + j]);
        }}
    }}
    acc = simd_sum(acc);
    if (lane == 0) {{ out[row] = T(acc); }}
"""


def _get_kernel(k: int, n: int, dtype):
    key = (k, n, dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    g = k // 64
    kernel = mx.fast.metal_kernel(
        name=f"quantized_gemv_k{k}_n{n}_{str(dtype).split('.')[-1]}",
        input_names=["x", "wq", "scales", "biases"],
        output_names=["out"],
        source=_SOURCE.format(k=k, n=n, g=g, gs=64),
    )
    _KERNELS[key] = kernel
    return kernel


def gemv_quantized(x: mx.array, wq: mx.array, scales: mx.array, biases: mx.array,
                   *, group_size: int = 64) -> mx.array:
    """y = x @ dequantize(wq, scales, biases).T, x [1,K] (or [K]) -> [1,N]."""
    if group_size != 64:
        raise ValueError("kernel is specialised for group_size=64")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[0] != 1:
        raise ValueError("GEMV handles a single token (M=1)")
    if x.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")
    n, packed_dim = wq.shape
    k = packed_dim * 8
    if x.shape[1] != k:
        raise ValueError(f"x has K={x.shape[1]} but weights are K={k}")
    if k % 32 or k // 32 % 8:
        raise ValueError("K/32 must be a multiple of 8")
    if scales.shape[1] * group_size != k:
        raise ValueError("group mismatch")
    kernel = _get_kernel(k, n, x.dtype)
    rows = ((n + ROWS_PER_TG - 1) // ROWS_PER_TG) * ROWS_PER_TG
    out = kernel(
        inputs=[x.reshape(-1), wq, scales, biases],
        template=[("T", x.dtype)],
        grid=(rows // ROWS_PER_TG * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(1, n)],
        output_dtypes=[x.dtype],
        verbose=False,
    )[0]
    return out
