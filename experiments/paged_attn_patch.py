"""Route paged-cache decode attention straight into the block pool.

``PoolKVCache`` normally *gathers* the sequence's blocks into a dense K/V view on
every layer of every step, which measured ~3x slower decode than the dense engine.
With this patch:

* ``PoolKVCache.update_and_fetch`` skips the gather for q_len == 1 (it only appends
  the new K/V and returns a tiny placeholder, which SDPA never sees);
* ``qwen3.scaled_dot_product_attention`` detects a pool-backed cache and calls
  ``PagedKVPool``-indexed ``paged_decode_attention_metal`` instead of SDPA.

Prefill chunks (q_len > 1) keep the gather path until a paged prefill kernel lands.
``TERRAZZO_PAGED_METAL=0`` disables.
"""

from __future__ import annotations

import os

import mlx.core as mx

_PATCHED = False
_ORIGINAL_SDPA = None
_ORIGINAL_UPDATE = None


def _is_pool_cache(cache) -> bool:
    return (
        cache is not None
        and hasattr(cache, "metal_attention")
        and hasattr(cache, "table")
        and hasattr(cache, "pool")
    )


def _pool_decode_attention(cache, queries, scale):
    """q_len=1 attention straight out of the block pool."""
    if os.environ.get("TERRAZZO_PAGED_KERNEL", "gqa") == "perhead":
        n_rep = queries.shape[1] // cache.pool.n_kv_heads
        return cache.metal_attention(queries, scale, n_rep=n_rep).astype(queries.dtype)
    from .paged_gqa_metal import paged_gqa_attention_metal

    layer = cache.layer_idx
    rowlen = int(cache.table.layer_length[layer])
    if rowlen <= 0:
        return mx.zeros_like(queries)
    k, v = cache.pool.layer_kv(layer)
    dtype = k.dtype
    hq = queries.shape[1]
    hk = cache.pool.n_kv_heads
    ratio = hq // hk
    hg = min(2, ratio)  # 2 is the best trade-off measured for K=1 (16 groups)
    blocks = mx.array(cache.table.blocks, dtype=mx.int32)
    out = paged_gqa_attention_metal(
        queries.astype(dtype), k, v,
        mx.array([rowlen], dtype=mx.int32),
        mx.array([rowlen - 1], dtype=mx.int32),
        blocks, scale, heads_per_group=hg)
    return out.astype(queries.dtype)


def apply_paged_attn_patch() -> bool:
    global _PATCHED, _ORIGINAL_SDPA, _ORIGINAL_UPDATE
    if _PATCHED:
        return True
    if os.environ.get("TERRAZZO_PAGED_METAL", "1") == "0":
        return False
    try:
        import mlx_lm.models.qwen3 as qwen3

        from terrazzo.pool_cache import PoolKVCache
    except Exception:  # pragma: no cover - mlx-lm missing
        return False

    _ORIGINAL_SDPA = qwen3.scaled_dot_product_attention
    _ORIGINAL_UPDATE = PoolKVCache.update_and_fetch
    original_sdpa = _ORIGINAL_SDPA
    original_update = _ORIGINAL_UPDATE

    def patched_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        if (sinks is None and queries is not None and queries.shape[2] == 1
                and _is_pool_cache(cache) and queries.shape[3] == 128):
            try:
                return _pool_decode_attention(cache, queries, scale)
            except Exception:
                pass
        return original_sdpa(queries, keys, values, cache=cache, scale=scale,
                             mask=mask, sinks=sinks)

    def patched_update(self, keys, values):
        if keys.shape[2] == 1 and keys.dtype in (mx.float16, mx.bfloat16):
            self.append(keys, values)
            dummy = getattr(self, "_decode_dummy", None)
            if dummy is None:
                dummy = (mx.zeros((1, self.pool.n_kv_heads, 1, self.pool.head_dim),
                                  dtype=keys.dtype),
                         mx.zeros((1, self.pool.n_kv_heads, 1, self.pool.head_dim),
                                  dtype=keys.dtype))
                self._decode_dummy = dummy
            return dummy
        return original_update(self, keys, values)

    qwen3.scaled_dot_product_attention = patched_sdpa
    PoolKVCache.update_and_fetch = patched_update
    _PATCHED = True
    return True


def disable_paged_attn_patch() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    import mlx_lm.models.qwen3 as qwen3

    from terrazzo.pool_cache import PoolKVCache

    if _ORIGINAL_SDPA is not None:
        qwen3.scaled_dot_product_attention = _ORIGINAL_SDPA
    if _ORIGINAL_UPDATE is not None:
        PoolKVCache.update_and_fetch = _ORIGINAL_UPDATE
    _PATCHED = False
