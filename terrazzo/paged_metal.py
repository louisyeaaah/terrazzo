"""Metal kernel for paged decode attention (q_len = 1), v0.4.

Why: the gather-based path (paged_attention.py) materialises a dense
[1, H, L, D] view per layer per step, which measured -26% decode throughput.
This kernel walks the block table *inside* the attention loop, so K/V stays in the
pool and nothing dense is built.

Layout: **one simdgroup (32 lanes) per query head**, no threadgroup barriers.

  * each lane owns ``D / 32`` contiguous head dims;
  * the q·k dot product is a per-lane partial sum + one ``simd_sum``;
  * the online-softmax state (m, l, acc[dim_per_lane]) lives in registers;
  * GQA: q head -> kv head via ``head / kv_rep``.

The first version used a 128-thread threadgroup per head with a shared-memory tree
reduction (~9 barriers per token) and was **2× slower than the gather baseline**.
The simdgroup version removes every barrier; see tests/test_paged_metal.py.

Reference: ``terrazzo.paged_attention.paged_attention`` (gather path). The
tests assert allclose(1e-4) on float32 and (5e-2) on float16 storage.
"""

from __future__ import annotations

import mlx.core as mx

from .paged_pool import PagedBlockTable, PagedKVPool

_SOURCE_TEMPLATE = """
    // split-K: SPLIT simdgroups per head, each covering every SPLIT-th token
    const int SPLIT = {split};
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const uint head = threadgroup_position_in_grid.x;
    const int D = {head_dim};
    const int S = {block_size};
    const int H = {n_kv_heads};
    const int DPL = {dims_per_lane};
    const int kv_head = int(head) / kv_rep[0];
    const int n = length[0];
    const float sc = scale[0];

    float qv[{dims_per_lane}];
    float acc[{dims_per_lane}];
    const int qbase = int(head) * D + int(lane) * DPL;
    for (int j = 0; j < DPL; ++j) {{
        qv[j] = float(q[qbase + j]);
        acc[j] = 0.0f;
    }}

    float m = -INFINITY;
    float l = 0.0f;
    for (int p = int(sg); p < n; p += SPLIT) {{
        const int blk = int(block_table[p / S]);
        const int off = p % S;
        const int base = ((blk * H + kv_head) * S + off) * D + int(lane) * DPL;
        float partial = 0.0f;
        for (int j = 0; j < DPL; ++j) {{
            partial += qv[j] * float(k_pool[base + j]);
        }}
        const float score = simd_sum(partial) * sc;
        const float new_m = metal::max(m, score);
        const float alpha = metal::exp(m - new_m);
        const float beta = metal::exp(score - new_m);
        l = l * alpha + beta;
        m = new_m;
        for (int j = 0; j < DPL; ++j) {{
            acc[j] = acc[j] * alpha + beta * float(v_pool[base + j]);
        }}
    }}

    // combine the SPLIT partial softmaxes (once per launch, not per token)
    threadgroup float sm[{split}][32];
    threadgroup float sl[{split}][32];
    threadgroup float sacc[{split}][32][{dims_per_lane}];
    sm[sg][lane] = m;
    sl[sg][lane] = l;
    for (int j = 0; j < DPL; ++j) {{
        sacc[sg][lane][j] = acc[j];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_all = -INFINITY;
    for (int s = 0; s < SPLIT; ++s) {{
        m_all = metal::max(m_all, sm[s][lane]);
    }}
    float l_all = 0.0f;
    for (int j = 0; j < DPL; ++j) {{
        acc[j] = 0.0f;
    }}
    for (int s = 0; s < SPLIT; ++s) {{
        const float w = metal::exp(sm[s][lane] - m_all);
        l_all += sl[s][lane] * w;
        for (int j = 0; j < DPL; ++j) {{
            acc[j] += sacc[s][lane][j] * w;
        }}
    }}
    const int obase = int(head) * D + int(lane) * DPL;
    for (int j = 0; j < DPL; ++j) {{
        out[obase + j] = acc[j] / l_all;
    }}
"""

_KERNELS: dict[tuple, object] = {}


def metal_available() -> bool:
    return hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")


def _supported(head_dim: int) -> bool:
    return head_dim % 32 == 0 and head_dim // 32 <= 16


SPLIT = 4


def _get_kernel(head_dim: int, block_size: int, n_kv_heads: int):
    dims_per_lane = head_dim // 32
    key = (head_dim, block_size, n_kv_heads, dims_per_lane, SPLIT)
    if key in _KERNELS:
        return _KERNELS[key]
    source = _SOURCE_TEMPLATE.format(head_dim=head_dim, block_size=block_size,
                                     n_kv_heads=n_kv_heads, dims_per_lane=dims_per_lane,
                                     split=SPLIT)
    kernel = mx.fast.metal_kernel(
        name=f"paged_decode_attn_{head_dim}_{block_size}_{n_kv_heads}_s{SPLIT}",
        input_names=["q", "k_pool", "v_pool", "block_table", "length", "scale", "kv_rep"],
        output_names=["out"],
        source=source,
    )
    _KERNELS[key] = kernel
    return kernel


def paged_decode_attention_metal(
    q: mx.array,
    pool: PagedKVPool,
    table: PagedBlockTable,
    layer: int,
    scale: float,
    n_rep: int = 1,
) -> mx.array:
    """Attention for a single decode query, reading K/V straight from the pool.

    ``q`` is [1, H_q, 1, D]; returns the same shape.
    """
    if q.shape[2] != 1:
        raise ValueError("metal kernel handles decode only (q_len must be 1)")
    head_dim = int(q.shape[3])
    if not _supported(head_dim):
        raise ValueError(f"head_dim {head_dim} not supported by paged metal kernel")
    kernel = _get_kernel(head_dim, pool.block_size, pool.n_kv_heads)
    k, v = pool.layer_kv(layer)
    length = table.layer_length[layer]
    if length <= 0:
        return mx.zeros_like(q)
    heads = int(q.shape[1])
    qq = mx.contiguous(q.reshape(heads, head_dim))
    table_arr = mx.array(table.blocks, dtype=mx.int32)
    length_arr = mx.array([length], dtype=mx.int32)
    scale_arr = mx.array([scale], dtype=mx.float32)
    rep_arr = mx.array([n_rep], dtype=mx.int32)
    (out,) = kernel(
        inputs=[qq, k, v, table_arr, length_arr, scale_arr, rep_arr],
        grid=(heads * 32 * SPLIT, 1, 1),
        threadgroup=(32 * SPLIT, 1, 1),
        output_shapes=[(heads, 1, head_dim)],
        output_dtypes=[mx.float32],
        verbose=False,
    )
    return out.reshape(q.shape)
