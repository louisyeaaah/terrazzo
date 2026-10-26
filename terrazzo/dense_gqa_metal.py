"""GQA-aware decode attention in Metal: each KV head is read once per token.

The naive per-query-head kernel re-reads the same K/V row for every query head that
shares the KV head (ratio = Hq/Hk), i.e. 4x the KV bytes for Qwen3-4B. This kernel
maps one *KV head* to one threadgroup: SPLIT warps walk disjoint token ranges and all
query heads of that KV head are accumulated in registers, so K/V is read exactly once.

    q [B, Hq, 1, D], k/v [B, Hk, L, D], lengths [B] -> out [B, Hq, 1, D]

Rows are left-padded like ``BatchKVCache``: real tokens are the last ``lengths[b]``.
"""

from __future__ import annotations

import mlx.core as mx

SPLIT = 8
MAX_RATIO = 4
_KERNELS: dict = {}

_SOURCE = """
    const int SPLIT = {split};
    const int HG = {heads_per_group};
    const int RATIO = {ratio};
    const int D = {head_dim};
    const int DPL = {dims_per_lane};
    const int L = dims[0];
    const int Hk = dims[1];
    const int Hq = dims[2];
    const float sc = scale[0];

    const uint tid = thread_position_in_threadgroup.x;
    const uint warp = tid / 32;
    const uint lane = tid % 32;
    const uint group = threadgroup_position_in_grid.x;
    const uint kv_head = (group * HG) / RATIO;
    const uint qh0 = group * HG;
    const uint row = threadgroup_position_in_grid.y;
    const int rowlen = int(lengths[row]);
    const int start = L - rowlen;  // left-padded rows: valid tokens are the last rowlen slots
    const int kv_base_row = int(row) * Hk * L;

    float4 qf[4];
    for (int h = 0; h < HG; ++h) {{
        const int qoff = (int(row) * Hq + int(qh0) + h) * D + int(lane) * DPL;
        qf[h] = float4(*(const device half4*)(q + qoff));
    }}

    float m[4];
    float l[4];
    float4 acc[4];
    for (int h = 0; h < 4; ++h) {{ m[h] = -INFINITY; l[h] = 0.0f; acc[h] = float4(0.0f); }}

    const int first = start + int(warp);
    for (int p = first; p < L; p += SPLIT) {{
        const int base = ((kv_base_row + int(kv_head) * L) + p) * D + int(lane) * DPL;
        half4 k4 = *(const device half4*)(k_pool + base);
        half4 v4 = *(const device half4*)(v_pool + base);
        float4 kf = float4(k4);
        float4 vf = float4(v4);
        for (int h = 0; h < HG; ++h) {{
            const float score = simd_sum(metal::dot(qf[h], kf)) * sc;
            const float new_m = metal::max(m[h], score);
            const float alpha = metal::exp(m[h] - new_m);
            const float beta = metal::exp(score - new_m);
            l[h] = l[h] * alpha + beta;
            m[h] = new_m;
            acc[h] = acc[h] * alpha + beta * vf;
        }}
    }}

    threadgroup float sm[{split}][{heads_per_group}][32];
    threadgroup float sl[{split}][{heads_per_group}][32];
    threadgroup float4 sacc[{split}][{heads_per_group}][32];
    for (int h = 0; h < HG; ++h) {{
        sm[warp][h][lane] = m[h];
        sl[warp][h][lane] = l[h];
        sacc[warp][h][lane] = acc[h];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp == 0) {{
        for (int h = 0; h < HG; ++h) {{
            float m_all = -INFINITY;
            for (int s = 0; s < SPLIT; ++s) {{
                m_all = metal::max(m_all, sm[s][h][lane]);
            }}
            float l_all = 0.0f;
            float4 a_all = float4(0.0f);
            for (int s = 0; s < SPLIT; ++s) {{
                const float w = metal::exp(sm[s][h][lane] - m_all);
                l_all += sl[s][h][lane] * w;
                a_all += sacc[s][h][lane] * w;
            }}
            const int obase = (int(row) * Hq + int(qh0) + h) * D + int(lane) * DPL;
            *(device half4*)(out + obase) = half4(a_all / l_all);
        }}
    }}
"""


def _get_kernel(head_dim: int, ratio: int, heads_per_group: int):
    dims_per_lane = head_dim // 32
    key = (head_dim, ratio, heads_per_group, SPLIT)
    if key in _KERNELS:
        return _KERNELS[key]
    source = _SOURCE.format(split=SPLIT, ratio=ratio, heads_per_group=heads_per_group,
                            head_dim=head_dim, dims_per_lane=dims_per_lane)
    kernel = mx.fast.metal_kernel(
        name=f"dense_gqa_attn_d{head_dim}_r{ratio}_h{heads_per_group}_s{SPLIT}",
        input_names=["q", "k_pool", "v_pool", "lengths", "dims", "scale"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )
    _KERNELS[key] = kernel
    return kernel


def dense_gqa_attention_metal(q: mx.array, k: mx.array, v: mx.array,
                              lengths: mx.array, scale: float,
                              heads_per_group: int | None = None) -> mx.array:
    """q [B,Hq,1,D], k/v [B,Hk,L,D], lengths [B] -> [B,Hq,1,D].

    ``heads_per_group`` trades KV traffic (lower = more re-reads) for parallelism
    (more threadgroups): 4 = one group per KV head (8 groups on Qwen3-4B),
    2 = 16 groups, 1 = one group per query head (32 groups).
    """
    B, Hq, _, D = q.shape
    Hk = k.shape[1]
    if D != 128 or Hq % Hk:
        # the kernel loads one half4 (4 dims) per lane: 32 lanes * 4 = 128 dims
        raise ValueError("need head_dim == 128 and Hq % Hk == 0")
    ratio = Hq // Hk
    hg = heads_per_group or ratio
    if ratio > MAX_RATIO or hg > MAX_RATIO or ratio % hg:
        raise ValueError(f"bad grouping: ratio={ratio} heads_per_group={hg}")
    kernel = _get_kernel(D, ratio, hg)
    groups = Hq // hg
    out = kernel(
        inputs=[q, k, v, lengths,
                mx.array([k.shape[2], Hk, Hq], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype)],
        # grid = threads; each threadgroup of 32*SPLIT handles hg q heads
        grid=(groups * 32 * SPLIT, B, 1),
        threadgroup=(32 * SPLIT, 1, 1),
        output_shapes=[(B, Hq, 1, D)],
        output_dtypes=[q.dtype],
        verbose=False,
    )[0]
    return out
