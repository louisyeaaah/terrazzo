"""MMA flash-attention kernel (simdgroup 8x8, K stored transposed), v2.

Adds what the shared-prefix batch path needs:
  * fp16 **and** bf16 (``simdgroup_half8x8`` / ``simdgroup_bfloat8x8``, fp32 accumulate);
  * query lengths smaller than 8 (padded internally, outputs sliced back);
  * an optional per-(head, query) ``lse`` output so packed prefix attention can be
    merged *exactly* with each row's suffix attention via log-sum-exp.

Layouts: q [Hq, L, D], kt [Hk, D, KV] (transposed K), v [Hk, KV, D] -> o [Hq, L, D].
One threadgroup = 1 warp handles 8 queries of one head; KV runs in tiles of 32.
"""

from __future__ import annotations

import mlx.core as mx

D = 128
BQ = 8
BK = 32

_KERNELS: dict = {}

_TYPES = {mx.float16: ("half", "simdgroup_half8x8", "fp16"),
          mx.bfloat16: ("bfloat16_t", "simdgroup_bfloat8x8", "bf16")}

_SOURCE = """
    const int D = {d};
    const int KV = dims[0];
    const int H = dims[1];
    const int REP = dims[2];
    const int L = dims[3];                 // padded query length
    const int QBLOCKS = dims[4];
    const int causal = dims[5];
    const int plain = dims[6];             // 1: queries live at positions 0..L-1
    const float scale = sc[0];

    const uint tid = thread_position_in_threadgroup.x;   // 0..31
    const int tg = threadgroup_position_in_grid.x;
    const int head = tg / QBLOCKS;
    const int q0 = (tg % QBLOCKS) * {bq};
    const int kv_head = head / REP;

    // ---- Q fragments (8 x 128 = 16 8x8 fragments) in registers
    {frag} qf[{d8}];
    const device {t}* Qh = Q + (int(head) * L + q0) * D;
    for (int kk = 0; kk < {d8}; ++kk) {{
        simdgroup_load(qf[kk], Qh + kk * 8, D);
    }}

    threadgroup {t} Ks[{d}][{bk}];
    threadgroup {t} Vs[{bk}][{d}];
    threadgroup float Ss[{bq}][{bk}];
    threadgroup {t} Ps[{bq}][{bk}];
    threadgroup float Cs[{bq}][{d}];
    threadgroup float Os[{bq}][{d}];
    threadgroup float ml[2][{bq}];

    for (int i = tid; i < {bq} * {d}; i += 32) {{
        Os[i / {d}][i % {d}] = 0.0f;
    }}
    if (tid < {bq}) {{ ml[0][tid] = -INFINITY; ml[1][tid] = 0.0f; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device {t}* Kth = Kt + int(kv_head) * D * KV;
    const device {t}* Vh = V + int(kv_head) * KV * D;

    for (int k0 = 0; k0 < KV; k0 += {bk}) {{
        if (causal && k0 > q0 + {bq} - 1) {{ break; }}
        for (int i = tid; i < {d} * {bk}; i += 32) {{
            const int d = i / {bk};
            const int j = i % {bk};
            const int key = k0 + j;
            Ks[d][j] = (key < KV) ? Kth[d * KV + key] : ({t})0;
        }}
        for (int i = tid; i < {bk} * {d}; i += 32) {{
            const int j = i / {d};
            const int d = i % {d};
            const int key = k0 + j;
            Vs[j][d] = (key < KV) ? Vh[key * D + d] : ({t})0;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 c[4];
        for (int j = 0; j < 4; ++j) {{ c[j] = simdgroup_float8x8(0); }}
        for (int kk = 0; kk < {d8}; ++kk) {{
            for (int j = 0; j < 4; ++j) {{
                {frag} b;
                simdgroup_load(b, &Ks[kk * 8][j * 8], {bk});
                simdgroup_multiply_accumulate(c[j], qf[kk], b, c[j]);
            }}
        }}
        for (int j = 0; j < 4; ++j) {{
            simdgroup_store(c[j], &Ss[0][j * 8], {bk});
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int row = tid / 4;
        const int part = tid % 4;
        const int qpos = plain ? (q0 + row) : 0;
        float vals[8];
        float local_max = -INFINITY;
        for (int t = 0; t < 8; ++t) {{
            const int col = part * 8 + t;
            const int key = k0 + col;
            float s = Ss[row][col] * scale;
            if (causal && key > qpos) s = -INFINITY;
            vals[t] = s;
            local_max = metal::max(local_max, s);
        }}
        local_max = metal::max(local_max, simd_shuffle_xor(local_max, 1));
        local_max = metal::max(local_max, simd_shuffle_xor(local_max, 2));

        const float m_old = ml[0][row];
        const float m_new = metal::max(m_old, local_max);
        const float alpha = metal::exp(m_old - m_new);
        const float beta = (local_max == -INFINITY) ? 0.0f
                           : metal::exp(local_max - m_new);
        float local_sum = 0.0f;
        for (int t = 0; t < 8; ++t) {{
            const int col = part * 8 + t;
            const float e = (local_max == -INFINITY) ? 0.0f
                            : metal::exp(vals[t] - local_max) * beta;
            local_sum += e;
            Ps[row][col] = ({t})e;
        }}
        local_sum += simd_shuffle_xor(local_sum, 1);
        local_sum += simd_shuffle_xor(local_sum, 2);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int t = 0; t < {d} / 4; ++t) {{
            const int c = part * ({d} / 4) + t;
            Os[row][c] *= alpha;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (part == 0) {{
            ml[0][row] = m_new;
            ml[1][row] = ml[1][row] * alpha + local_sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 oc[{d8}];
        for (int j = 0; j < {d8}; ++j) {{ oc[j] = simdgroup_float8x8(0); }}
        for (int kf = 0; kf < 4; ++kf) {{
            {frag} pf;
            simdgroup_load(pf, &Ps[0][kf * 8], {bk});
            for (int j = 0; j < {d8}; ++j) {{
                {frag} b;
                simdgroup_load(b, &Vs[kf * 8][j * 8], {d});
                simdgroup_multiply_accumulate(oc[j], pf, b, oc[j]);
            }}
        }}
        for (int j = 0; j < {d8}; ++j) {{
            simdgroup_store(oc[j], &Cs[0][j * 8], {d});
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < {bq} * {d}; i += 32) {{
            Os[i / {d}][i % {d}] += Cs[i / {d}][i % {d}];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    for (int i = tid; i < {bq} * {d}; i += 32) {{
        const int r = i / {d};
        const int c = i % {d};
        const float l = ml[1][r];
        O[(int(head) * L + q0 + r) * D + c] = ({t})(Os[r][c] / l);
    }}
    if (tid < {bq}) {{
        const float l = ml[1][tid];
        LSE[(int(head) * L + q0 + tid)] = ml[0][tid] + metal::log(l);
    }}
"""


def _get_kernel(dtype):
    key = str(dtype)
    if key in _KERNELS:
        return _KERNELS[key]
    t, frag, tag = _TYPES[dtype]
    kernel = mx.fast.metal_kernel(
        name=f"mma_flash_attn_v2_{tag}",
        input_names=["Q", "Kt", "V", "dims", "sc"],
        output_names=["O", "LSE"],
        source=_SOURCE.format(d=D, bq=BQ, bk=BK, d8=D // 8, t=t, frag=frag),
    )
    _KERNELS[key] = kernel
    return kernel


def mma_flash_attention(q: mx.array, kt: mx.array, v: mx.array, *, causal: bool = True,
                        scale: float | None = None, return_lse: bool = False):
    """q [Hq, L, D], kt [Hk, D, KV], v [Hk, KV, D] -> o [Hq, L, D] (+ lse [Hq, L]).

    Rows are independent queries (one per session when used for packed prefix
    attention); pass ``plain=True`` semantics by keeping the default.
    """
    hq, l, d = q.shape
    hk = kt.shape[0]
    kv = kt.shape[2]
    if d != D:
        raise ValueError("kernel specialised for head_dim=128")
    if q.dtype not in _TYPES or kt.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/kt/v must share one dtype in (float16, bfloat16)")
    if kt.shape[1] != d:
        raise ValueError("Kt must be [Hk, D, KV]")
    lpad = ((l + BQ - 1) // BQ) * BQ
    if lpad != l:
        pad = mx.zeros((hq, lpad - l, d), dtype=q.dtype)
        q = mx.concatenate([q, pad], axis=1)
    kernel = _get_kernel(q.dtype)
    repo = hq // hk
    out, lse = kernel(
        inputs=[q, kt, v,
                mx.array([kv, hk, repo, lpad, lpad // BQ, int(causal), 1], dtype=mx.int32),
                mx.array([scale if scale is not None else d ** -0.5], dtype=mx.float32)],
        grid=((lpad // BQ) * hq * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(hq, lpad, d), (hq, lpad)],
        output_dtypes=[q.dtype, mx.float32],
        verbose=False,
    )
    out = out[:, :l, :]
    lse = lse[:, :l]
    return (out, lse) if return_lse else out
