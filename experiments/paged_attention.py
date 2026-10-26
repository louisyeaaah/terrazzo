"""Paged attention over the block pool (v0.4).

Two implementations:

* ``dense_attention``      — reference (mlx fast SDPA on a dense K/V view)
* ``paged_attention``      — gathers only the blocks the sequence owns, then runs
  the same math. Numerically identical to the dense reference when the gathered
  blocks contain the same K/V (verified in tests/test_paged_attention.py).

``paged_attention`` is the "read" half of real paging: storage is blocked and
shared, attention consumes a per-sequence gather. A Metal kernel that indexes the
pool *inside* the attention loop (no materialised gather) is the next step and
will be measured against this baseline.
"""

from __future__ import annotations

import mlx.core as mx

from terrazzo.paged_pool import PagedBlockTable, PagedKVPool


def repeat_kv(x: mx.array, n_rep: int) -> mx.array:
    """[B, H_kv, L, D] -> [B, H_kv * n_rep, L, D] (GQA expansion)."""
    if n_rep == 1:
        return x
    b, h, l, d = x.shape
    x = mx.expand_dims(x, axis=2)                     # [B, H, 1, L, D]
    x = mx.broadcast_to(x, (b, h, n_rep, l, d))
    return mx.reshape(x, (b, h * n_rep, l, d))


def dense_attention(q: mx.array, k: mx.array, v: mx.array, scale: float,
                    n_rep: int = 1, mask: str | None = "causal") -> mx.array:
    """Reference attention on dense [B, H, L, D] arrays."""
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def paged_attention(
    q: mx.array,
    pool: PagedKVPool,
    table: PagedBlockTable,
    layer: int,
    scale: float,
    n_rep: int = 1,
    mask: str | None = "causal",
) -> mx.array:
    """Attention where K/V are read from the block pool via ``table``.

    ``q`` is [1, H_q, N, D]; K/V are gathered [1, H_kv, L, D] and expanded.
    """
    k, v = table.gather(layer)
    # decode steps have q_len == 1 against a length-L cache; prefill has q_len == L
    if q.shape[2] not in (1, k.shape[2]):
        raise ValueError(f"query length {q.shape[2]} != cached length {k.shape[2]}")
    return dense_attention(q, k, v, scale, n_rep=n_rep, mask=mask)


def attention_with_prefix(
    q: mx.array,
    pool: PagedKVPool,
    prefix_blocks: list[int],
    prefix_len: int,
    suffix_k: mx.array,
    suffix_v: mx.array,
    layer: int,
    scale: float,
    n_rep: int = 1,
) -> mx.array:
    """Attention for a query over a shared prefix + local suffix (no table yet)."""
    bs = pool.block_size
    k_pre, v_pre = pool.gather(layer, prefix_blocks, prefix_len)
    k = mx.concatenate([k_pre, suffix_k], axis=2)
    v = mx.concatenate([v_pre, suffix_v], axis=2)
    return dense_attention(q, k, v, scale, n_rep=n_rep, mask="causal")
