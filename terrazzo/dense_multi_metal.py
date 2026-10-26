"""Multi-query (q_len = K) decode attention in Metal: one KV read serves K queries.

Speculative verification and chunked decode feed K tokens in one forward. MLX's SDPA
cost grows ~linearly with K at long context (measured 2.35x for K=3 at 4k, 2.60x at
8k), which is why speculative decoding loses. This kernel reads K/V once per token
and computes K query heads' worth of scores from the same loaded row.

Layout (KVCache style, right-padded buffers):

    q [B, Hq, K, D]   k/v [B, Hk, L, D]   lengths [B] (valid prefix = start + K)
    starts [B]        position of query 0 (query i sits at start + i)
    causal: query i attends positions p <= start + i

The output is [B, Hq, K, D].
"""

from __future__ import annotations

import mlx.core as mx

_KERNELS: dict = {}
SUPPORTED_K = tuple(range(1, 9))

_SOURCE = """
    const int QK = {qk};
    const int SPLIT = {split};
    const int D = {head_dim};
    const int DPL = {dims_per_lane};
    const int Hq = dims[0];
    const int Hk = dims[1];
    const int L = dims[2];
    const float sc = scale[0];

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint head = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const int rowlen = int(lengths[row]);
    const int start = int(starts[row]);
    const int kv_head = int(head) / (int(Hq) / Hk);
    const int kv_row_base = (int(row) * Hk + kv_head) * L;

    float4 qf[QK];
    for (int k = 0; k < QK; ++k) {{
        const int qoff = ((int(row) * Hq + int(head)) * QK + k) * D + int(lane) * DPL;
        vec<T, 4> q4 = *(const device vec<T, 4>*)(q + qoff);
        qf[k] = float4(q4);
    }}

    float m[QK];
    float l[QK];
    float4 acc[QK];
    for (int k = 0; k < QK; ++k) {{ m[k] = -INFINITY; l[k] = 0.0f; acc[k] = float4(0.0f); }}

    for (int p = int(sg); p < rowlen; p += SPLIT) {{
        const int base = (kv_row_base + p) * D + int(lane) * DPL;
        vec<T, 4> k4 = *(const device vec<T, 4>*)(k_pool + base);
        vec<T, 4> v4 = *(const device vec<T, 4>*)(v_pool + base);
        float4 kf = float4(k4);
        float4 vf = float4(v4);
        for (int k = 0; k < QK; ++k) {{
            if (p > start + k) {{ continue; }}  // causal
            const float score = simd_sum(metal::dot(qf[k], kf)) * sc;
            const float new_m = metal::max(m[k], score);
            const float alpha = metal::exp(m[k] - new_m);
            const float beta = metal::exp(score - new_m);
            l[k] = l[k] * alpha + beta;
            m[k] = new_m;
            acc[k] = acc[k] * alpha + beta * vf;
        }}
    }}

    threadgroup float sm[{split}][{qk}][32];
    threadgroup float sl[{split}][{qk}][32];
    threadgroup float4 sacc[{split}][{qk}][32];
    for (int k = 0; k < QK; ++k) {{
        sm[sg][k][lane] = m[k];
        sl[sg][k][lane] = l[k];
        sacc[sg][k][lane] = acc[k];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {{
        for (int k = 0; k < QK; ++k) {{
            float m_all = -INFINITY;
            for (int s = 0; s < SPLIT; ++s) {{ m_all = metal::max(m_all, sm[s][k][lane]); }}
            float l_all = 0.0f;
            float4 a_all = float4(0.0f);
            for (int s = 0; s < SPLIT; ++s) {{
                const float w = metal::exp(sm[s][k][lane] - m_all);
                l_all += sl[s][k][lane] * w;
                a_all += sacc[s][k][lane] * w;
            }}
            const int obase = ((int(row) * Hq + int(head)) * QK + k) * D + int(lane) * DPL;
            const float4 res = a_all / l_all;
            *(device vec<T, 4>*)(out + obase) = vec<T, 4>(T(res.x), T(res.y), T(res.z), T(res.w));
        }}
    }}
"""


def _split_for(qk: int) -> int:
    return 8 if qk <= 4 else 4


def _get_kernel(qk: int, head_dim: int):
    dims_per_lane = head_dim // 32
    split = _split_for(qk)
    key = (qk, head_dim, split)
    if key in _KERNELS:
        return _KERNELS[key], split
    source = _SOURCE.format(qk=qk, split=split, head_dim=head_dim,
                            dims_per_lane=dims_per_lane)
    kernel = mx.fast.metal_kernel(
        name=f"multi_query_attn_q{qk}_d{head_dim}_s{split}",
        input_names=["q", "k_pool", "v_pool", "lengths", "starts", "dims", "scale"],
        output_names=["out"],
        source=source,
        # cache slices are non-contiguous (alloc uses 256-token steps); let MLX
        # materialise row-contiguous inputs before launch
        ensure_row_contiguous=True,
    )
    _KERNELS[key] = kernel
    return kernel, split


def multi_query_attention_metal(q: mx.array, k: mx.array, v: mx.array,
                                lengths: mx.array, starts: mx.array,
                                scale: float) -> mx.array:
    """q [B,Hq,K,D], k/v [B,Hk,L,D] -> out [B,Hq,K,D] (causal per query)."""
    B, Hq, K, D = q.shape
    Hk = k.shape[1]
    if K not in SUPPORTED_K:
        raise ValueError(f"q_len {K} not supported (1..8)")
    if D != 128:
        raise ValueError("need head_dim == 128")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")
    if Hq % Hk:
        raise ValueError("Hq must be a multiple of Hk")
    kernel, split = _get_kernel(K, D)
    out = kernel(
        inputs=[q, k, v, lengths, starts,
                mx.array([Hq, Hk, k.shape[2]], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype)],
        grid=(Hq * 32 * split, B, 1),
        threadgroup=(32 * split, 1, 1),
        output_shapes=[(B, Hq, K, D)],
        output_dtypes=[q.dtype],
        verbose=False,
    )[0]
    return out
