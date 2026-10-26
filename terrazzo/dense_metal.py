"""Batched dense decode attention in Metal (q_len == 1).

The measured bottleneck on the agent workload is the dense SDPA call for
q_len=1 over a long, left-padded KV batch: ~8 ms per sequence per step at
L~2600 tokens, while the KV read itself is only ~0.9 ms. This kernel replaces
that call with a split-K simdgroup kernel that walks the batch's KV rows
directly.

Inputs are exactly what mlx-lm's ``BatchKVCache`` hands to SDPA during decode:
    q [B, Hq, 1, D]   k/v [B, Hkv, L, D]   lengths [B] (= cache.offset)
Rows are left-padded, so row b's real tokens are the last ``lengths[b]`` slots.

The kernel is only used when q_len == 1; prefill falls back to mlx-lm's SDPA.
"""

from __future__ import annotations

import os

import mlx.core as mx

SPLIT = int(os.environ.get("TERRAZZO_ATTN_SPLIT", "4"))

_SOURCE = """
    const int SPLIT = {split};
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint head = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const int D = {head_dim};
    const int DPL = {dims_per_lane};
    const int Hq = dims[0];
    const int Hk = dims[1];
    const int L = dims[2];
    const int rowlen = int(lengths[row]);
    const int start = L - rowlen;
    const int kv_head = int(head) / (int(Hq) / Hk);
    const float sc = scale[0];

    // one lane owns DPL contiguous dims; for DPL == 4 they are one half4 (8B) - aligned
    const int qbase = (int(row) * Hq + int(head)) * D + int(lane) * DPL;
    half4 q4 = *(const device half4*)(q + qbase);
    float4 qf = float4(q4);
    float4 acc = float4(0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    for (int p = int(sg); p < rowlen; p += SPLIT) {{
        const int pos = start + p;
        const int base = ((int(row) * Hk + kv_head) * L + pos) * D + int(lane) * DPL;
        half4 k4 = *(const device half4*)(k + base);
        float partial = metal::dot(qf, float4(k4));
        const float score = simd_sum(partial) * sc;
        const float new_m = metal::max(m, score);
        const float alpha = metal::exp(m - new_m);
        const float beta = metal::exp(score - new_m);
        l = l * alpha + beta;
        m = new_m;
        half4 v4 = *(const device half4*)(v + base);
        float4 fv = float4(v4);
        acc = acc * alpha + beta * fv;
    }}

    threadgroup float sm[{split}][32];
    threadgroup float sl[{split}][32];
    threadgroup float4 sacc[{split}][32];
    sm[sg][lane] = m;
    sl[sg][lane] = l;
    sacc[sg][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_all = -INFINITY;
    for (int s = 0; s < SPLIT; ++s) {{
        m_all = metal::max(m_all, sm[s][lane]);
    }}
    float l_all = 0.0f;
    acc = float4(0.0f);
    for (int s = 0; s < SPLIT; ++s) {{
        const float w = metal::exp(sm[s][lane] - m_all);
        l_all += sl[s][lane] * w;
        acc += sacc[s][lane] * w;
    }}
    const int obase = (int(row) * Hq + int(head)) * D + int(lane) * DPL;
    *(device half4*)(out + obase) = half4(acc / l_all);
"""

_KERNELS: dict = {}


def _get_kernel(head_dim: int, n_kv_heads: int, split: int | None = None):
    dims_per_lane = head_dim // 32
    split = split or SPLIT
    key = (head_dim, n_kv_heads, dims_per_lane, split)
    if key in _KERNELS:
        return _KERNELS[key]
    source = _SOURCE.format(split=split, head_dim=head_dim,
                            dims_per_lane=dims_per_lane, n_kv_heads=n_kv_heads)
    kernel = mx.fast.metal_kernel(
        name=f"dense_decode_attn_{head_dim}_{n_kv_heads}_s{split}",
        input_names=["q", "k", "v", "lengths", "dims", "scale"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )
    _KERNELS[key] = kernel
    return kernel


def dense_decode_attention_metal(q: mx.array, k: mx.array, v: mx.array,
                                 lengths: mx.array, scale: float,
                                 split: int | None = None) -> mx.array:
    """q [B,Hq,1,D], k/v [B,Hkv,L,D], lengths [B] -> out [B,Hq,1,D]."""
    B, Hq, _, D = q.shape
    Hk = k.shape[1]
    if D % 32 != 0:
        raise ValueError("head_dim must be a multiple of 32")
    if Hq % Hk != 0:
        raise ValueError("q heads must be a multiple of kv heads")
    split = split or SPLIT
    kernel = _get_kernel(D, Hk, split)
    out = kernel(
        inputs=[q, k, v, lengths,
                mx.array([Hq, Hk, k.shape[2]], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype)],
        grid=(Hq * 32 * split, B, 1),
        threadgroup=(32 * split, 1, 1),
        output_shapes=[(B, Hq, 1, D)],
        output_dtypes=[q.dtype],
        verbose=False,
    )[0]
    return out


def reference_attention(q, k, v, lengths, scale):
    """Dense reference with left padding, built on mlx ops (tests only)."""
    B, Hq, _, D = q.shape
    Hk = k.shape[1]
    rep = Hq // Hk
    kk = mx.repeat(k, rep, axis=1)
    vv = mx.repeat(v, rep, axis=1)
    B2, H2, L, D2 = kk.shape
    scores = (q @ kk.transpose(0, 1, 3, 2)) * scale  # [B,Hq,1,L]
    pos = mx.arange(L)
    valid = pos[None, :] >= (L - lengths[:, None])  # [B, L]
    mask = mx.where(valid[:, None, None, :], 0.0, -1e30)
    weights = mx.softmax(scores + mask, axis=-1)
    return weights @ vv
