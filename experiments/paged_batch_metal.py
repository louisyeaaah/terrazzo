"""Batched paged decode attention: one kernel launch for the whole batch.

Prototype for the "big concurrency" goal. Each row has its own block table and
length; the kernel walks the KV *pool* directly (no dense gather), so it removes
the per-layer materialisation that the dense ``BatchKVCache`` path pays.

Layouts
    q            [B, Hq, 1, D]      fp16
    k_pool       [nb, Hk, bs, D]    fp16   (per layer)
    v_pool       [nb, Hk, bs, D]    fp16
    block_table  [B, max_blocks]    int32
    lengths      [B]                int32  (real tokens per row)

Numerics are validated against a gather + reference-attention implementation in
``tests/test_paged_batch_metal.py``; the microbenchmark lives in
``bench/paged_batch_attention.py``.
"""

from __future__ import annotations

import mlx.core as mx

SPLIT = 4
_KERNELS: dict = {}

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
    const int BS = dims[2];
    const int NB = dims[3];
    const int MAXB = dims[4];
    const int rowlen = int(lengths[row]);
    const int kv_head = int(head) / (int(Hq) / Hk);
    const float sc = scale[0];
    const device int* table = block_table + int(row) * MAXB;

    const int qbase = (int(row) * Hq + int(head)) * D + int(lane) * DPL;
    half4 q4 = *(const device half4*)(q + qbase);
    float4 qf = float4(q4);
    float4 acc = float4(0.0f);

    float m = -INFINITY;
    float l = 0.0f;
    for (int p = int(sg); p < rowlen; p += SPLIT) {{
        const int blk = table[p / BS];
        const int off = p % BS;
        const int base = ((blk * Hk + kv_head) * BS + off) * D + int(lane) * DPL;
        half4 k4 = *(const device half4*)(k_pool + base);
        const float score = simd_sum(metal::dot(qf, float4(k4))) * sc;
        const float new_m = metal::max(m, score);
        const float alpha = metal::exp(m - new_m);
        const float beta = metal::exp(score - new_m);
        l = l * alpha + beta;
        m = new_m;
        half4 v4 = *(const device half4*)(v_pool + base);
        acc = acc * alpha + beta * float4(v4);
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


def _get_kernel(head_dim: int, n_kv_heads: int):
    dims_per_lane = head_dim // 32
    key = (head_dim, n_kv_heads, dims_per_lane)
    if key in _KERNELS:
        return _KERNELS[key]
    source = _SOURCE.format(split=SPLIT, head_dim=head_dim,
                            dims_per_lane=dims_per_lane)
    kernel = mx.fast.metal_kernel(
        name=f"paged_batch_attn_{head_dim}_{n_kv_heads}_s{SPLIT}",
        input_names=["q", "k_pool", "v_pool", "block_table", "lengths", "dims", "scale"],
        output_names=["out"],
        source=source,
    )
    _KERNELS[key] = kernel
    return kernel


def paged_batch_attention_metal(q: mx.array, k_pool: mx.array, v_pool: mx.array,
                                block_table: mx.array, lengths: mx.array,
                                scale: float) -> mx.array:
    """q [B,Hq,1,D], pools [nb,Hk,bs,D], table [B,max_blocks], lengths [B] -> [B,Hq,1,D]."""
    B, Hq, _, D = q.shape
    Hk = k_pool.shape[1]
    nb, _, bs, _ = k_pool.shape
    max_blocks = block_table.shape[1]
    if D % 32:
        raise ValueError("head_dim must be a multiple of 32")
    kernel = _get_kernel(D, Hk)
    out = kernel(
        inputs=[q, k_pool, v_pool, block_table, lengths,
                mx.array([Hq, Hk, bs, nb, max_blocks], dtype=mx.int32),
                mx.array([scale], dtype=mx.float32)],
        grid=(Hq * 32 * SPLIT, B, 1),
        threadgroup=(32 * SPLIT, 1, 1),
        output_shapes=[(B, Hq, 1, D)],
        output_dtypes=[q.dtype],
        verbose=False,
    )[0]
    return out


def gather_rows(k_pool: mx.array, v_pool: mx.array, block_table: mx.array,
                lengths: mx.array) -> tuple[mx.array, mx.array]:
    """Reference dense view per row: returns [B,Hk,Lmax,D] (left-aligned)."""
    B = block_table.shape[0]
    bs = k_pool.shape[2]
    Lmax = ((int(lengths.max().item()) + bs - 1) // bs) * bs if B else 0
    rows_k, rows_v = [], []
    for b in range(B):
        n = int(lengths[b].item())
        blocks = [int(x) for x in block_table[b].tolist()[: (n + k_pool.shape[2] - 1) // k_pool.shape[2]]]
        k = mx.take(k_pool, mx.array(blocks, dtype=mx.int32), axis=0)  # [nb,H,bs,D]
        v = mx.take(v_pool, mx.array(blocks, dtype=mx.int32), axis=0)
        k = mx.reshape(mx.transpose(k, (1, 0, 2, 3)), (k_pool.shape[1], len(blocks) * k_pool.shape[2], k_pool.shape[3]))
        v = mx.reshape(mx.transpose(v, (1, 0, 2, 3)), (v_pool.shape[1], len(blocks) * v_pool.shape[2], v_pool.shape[3]))
        pad = Lmax - k.shape[1]
        if pad > 0:
            k = mx.concatenate([k, mx.zeros((k.shape[0], pad, k.shape[2]), k.dtype)], axis=1)
            v = mx.concatenate([v, mx.zeros((v.shape[0], pad, v.shape[2]), v.dtype)], axis=1)
        rows_k.append(k)
        rows_v.append(v)
    return mx.stack(rows_k, 0), mx.stack(rows_v, 0)


def reference_attention(q: mx.array, k: mx.array, v: mx.array, lengths: mx.array,
                        scale: float) -> mx.array:
    """Dense reference: q [B,Hq,1,D], k/v [B,Hk,L,D], lengths [B]."""
    B, Hq, _, D = q.shape
    Hk = k.shape[1]
    rep = Hq // Hk
    kk = mx.repeat(k, rep, axis=1)
    vv = mx.repeat(v, rep, axis=1)
    L = kk.shape[2]
    scores = (q @ kk.transpose(0, 1, 3, 2)) * scale
    pos = mx.arange(L)
    valid = pos[None, :] < lengths[:, None]
    mask = mx.where(valid[:, None, None, :], 0.0, -1e30)
    return mx.softmax(scores + mask, axis=-1) @ vv
