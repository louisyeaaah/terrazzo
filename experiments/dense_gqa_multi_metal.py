"""GQA-reuse multi-query decode attention: one KV read serves RATIO*K queries.

The per-query-head multi-query kernel re-reads the shared KV head once per query
head (Qwen3-4B: 4x traffic), which made it slower than SDPA in-engine. This kernel
maps one threadgroup to a KV head and accumulates RATIO query heads x K query
positions in registers, so the KV row is read exactly once.

Layout (KVCache style, right-padded):

    q [B, Hq, K, D]   k/v [B, Hk, L, D]   lengths [B] = start + K
    starts [B]        position of query 0; query i attends p <= start + i
    out [B, Hq, K, D]
"""

from __future__ import annotations

import mlx.core as mx

_KERNELS: dict = {}
MAX_QUERIES = 8
MAX_HEADS_PER_GROUP = 8


def _split_for(q: int) -> int:
    """Threadgroup memory = SPLIT*Q*512 bytes; M1 limit is 32 KB."""
    return 4 if q <= 12 else (2 if q <= 24 else 1)


_SOURCE = """
    const int QK = {qk};
    const int RATIO = {ratio};
    const int HG = {hg};              // query heads handled per threadgroup
    const int SPLIT = {split};
    const int Q = HG * QK;             // queries accumulated per lane
    const int D = {head_dim};
    const int DPL = {dims_per_lane};
    const int Hq = dims[0];
    const int Hk = dims[1];
    const int L = dims[2];
    const float sc = scale[0];

    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint kv_head = threadgroup_position_in_grid.x;  // group index
    const uint row = threadgroup_position_in_grid.y;
    const int rowlen = int(lengths[row]);
    const int start = int(starts[row]);
    const int group = int(kv_head);
    const int kv_head_idx = (group * HG) / RATIO;
    const int qh0 = group * HG;
    const int kv_row_base = (int(row) * Hk + kv_head_idx) * L;

    float4 qf[Q];
    for (int h = 0; h < HG; ++h) {{
        for (int k = 0; k < QK; ++k) {{
            const int qoff = ((int(row) * Hq + qh0 + h) * QK + k) * D + int(lane) * DPL;
            vec<T, 4> q4 = *(const device vec<T, 4>*)(q + qoff);
            qf[h * QK + k] = float4(q4);
        }}
    }}

    float m[Q];
    float l[Q];
    float4 acc[Q];
    for (int i = 0; i < Q; ++i) {{ m[i] = -INFINITY; l[i] = 0.0f; acc[i] = float4(0.0f); }}

    for (int p = int(sg); p < rowlen; p += SPLIT) {{
        const int base = (kv_row_base + p) * D + int(lane) * DPL;
        vec<T, 4> k4 = *(const device vec<T, 4>*)(k_pool + base);
        vec<T, 4> v4 = *(const device vec<T, 4>*)(v_pool + base);
        float4 kf = float4(k4);
        float4 vf = float4(v4);
        for (int k = 0; k < QK; ++k) {{
            if (p > start + k) {{ continue; }}
            for (int h = 0; h < HG; ++h) {{
                const int i = h * QK + k;
                const float score = simd_sum(metal::dot(qf[i], kf)) * sc;
                const float new_m = metal::max(m[i], score);
                const float alpha = metal::exp(m[i] - new_m);
                const float beta = metal::exp(score - new_m);
                l[i] = l[i] * alpha + beta;
                m[i] = new_m;
                acc[i] = acc[i] * alpha + beta * vf;
            }}
        }}
    }}

    threadgroup float sm[{split}][{q}][32];
    threadgroup float sl[{split}][{q}][32];
    threadgroup half4 sacc[{split}][{q}][32];
    for (int i = 0; i < Q; ++i) {{
        sm[sg][i][lane] = m[i];
        sl[sg][i][lane] = l[i];
        sacc[sg][i][lane] = half4(acc[i]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {{
        for (int h = 0; h < HG; ++h) {{
            for (int k = 0; k < QK; ++k) {{
                const int i = h * QK + k;
                float m_all = -INFINITY;
                for (int s = 0; s < SPLIT; ++s) {{ m_all = metal::max(m_all, sm[s][i][lane]); }}
                float l_all = 0.0f;
                float4 a_all = float4(0.0f);
                for (int s = 0; s < SPLIT; ++s) {{
                    const float w = metal::exp(sm[s][i][lane] - m_all);
                    l_all += sl[s][i][lane] * w;
                    a_all += float4(sacc[s][i][lane]) * w;
                }}
                const int obase = ((int(row) * Hq + qh0 + h) * QK + k) * D + int(lane) * DPL;
                const float4 res = a_all / l_all;
                *(device vec<T, 4>*)(out + obase) = vec<T, 4>(T(res.x), T(res.y),
                                                              T(res.z), T(res.w));
            }}
        }}
    }}
"""


def _get_kernel(qk: int, ratio: int, hg: int, head_dim: int):
    q = hg * qk
    split = _split_for(q)
    dims_per_lane = head_dim // 32
    key = (qk, ratio, hg, head_dim)
    if key in _KERNELS:
        return _KERNELS[key], split
    source = _SOURCE.format(qk=qk, ratio=ratio, hg=hg, split=split, q=q,
                            head_dim=head_dim, dims_per_lane=dims_per_lane)
    kernel = mx.fast.metal_kernel(
        name=f"gqa_multi_attn_q{qk}_r{ratio}_h{hg}_d{head_dim}_s{split}",
        input_names=["q", "k_pool", "v_pool", "lengths", "starts", "dims", "scale"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=False,
    )
    _KERNELS[key] = kernel
    return kernel, split


def gqa_multi_attention_metal(q: mx.array, k: mx.array, v: mx.array,
                              lengths: mx.array, starts: mx.array,
                              scale: float, heads_per_group: int | None = None
                              ) -> mx.array:
    """q [B,Hq,K,D], k/v [B,Hk,L,D] -> out [B,Hq,K,D] (causal, right-padded).

    ``heads_per_group`` trades KV traffic (lower = more re-reads) for register
    pressure: HG=ratio reads each KV row once; HG=2 twice; HG=1 once per q head.
    """
    B, Hq, K, D = q.shape
    Hk = k.shape[1]
    if not (1 <= K <= MAX_QUERIES):
        raise ValueError(f"q_len {K} not supported (1..{MAX_QUERIES})")
    if Hq % Hk:
        raise ValueError("Hq must be a multiple of Hk")
    ratio = Hq // Hk
    hg = heads_per_group or ratio
    if hg > ratio or ratio % hg:
        raise ValueError(f"heads_per_group {hg} must divide ratio {ratio}")
    if ratio > MAX_HEADS_PER_GROUP or hg * K > 24:
        raise ValueError(f"grouping too large: hg={hg} K={K}")
    if D != 128:
        raise ValueError("need head_dim == 128")
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")
    kernel, split = _get_kernel(K, ratio, hg, D)
    out = kernel(
        inputs=[q, k, v, lengths, starts,
                mx.array([Hq, Hk, k.shape[2]], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        template=[("T", q.dtype)],
        grid=((Hq // hg) * 32 * split, B, 1),
        threadgroup=(32 * split, 1, 1),
        output_shapes=[(B, Hq, K, D)],
        output_dtypes=[q.dtype],
        verbose=False,
    )[0]
    return out
