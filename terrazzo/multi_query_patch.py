"""Route q_len=2..8 decode forwards to the multi-query Metal attention kernel.

The model's SDPA calls for speculative verification (K draft tokens in one forward)
cost ~K x a single decode forward on MLX; our kernel reads K/V once for all K
queries. Only plain (right-padded, unquantized) ``KVCache`` caches and fp16 are
patched; prefill chunks (q_len > 8) and padded batch caches keep MLX's SDPA.

``TERRAZZO_MULTI_Q=0`` disables, ``TERRAZZO_MULTI_Q_MAX`` caps the query length.
"""

from __future__ import annotations

import os

import mlx.core as mx

_PATCHED = False
_ORIGINAL = None


def _eligible(cache, queries, keys, values) -> bool:
    if cache is None or queries is None or keys is None or values is None:
        return False
    if not (hasattr(cache, "offset") and isinstance(cache.offset, int)):
        return False
    if hasattr(cache, "left_padding") or hasattr(cache, "bits"):
        return False
    return True


def apply_multi_query_patch() -> bool:
    global _PATCHED, _ORIGINAL
    if _PATCHED:
        return True
    # Default OFF: the current kernel maps one threadgroup per query head and
    # re-reads the shared KV head 4x (per-q-head mapping). Measured 1.2-1.3x
    # SLOWER than SDPA in-model. Enable to experiment; the GQA-reuse variant is
    # the next step (one KV read serves ratio * K queries).
    if os.environ.get("TERRAZZO_MULTI_Q", "0") != "1":
        return False
    import mlx_lm.models.qwen3 as qwen3

    from .dense_multi_metal import SUPPORTED_K, multi_query_attention_metal

    original = qwen3.scaled_dot_product_attention
    _ORIGINAL = original
    max_k = int(os.environ.get("TERRAZZO_MULTI_Q_MAX", "8"))

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        K = queries.shape[2] if queries is not None else 0
        if (sinks is None and 1 < K <= min(max_k, max(SUPPORTED_K))
                and isinstance(keys, mx.array)
                and queries.dtype in (mx.float16, mx.bfloat16)
                and queries.shape[3] == 128 and queries.shape[1] % keys.shape[1] == 0
                and _eligible(cache, queries, keys, values)):
            try:
                rowlen = int(cache.offset)
                lengths = mx.array([rowlen], dtype=mx.int32)
                starts = mx.array([rowlen - K], dtype=mx.int32)
                if rowlen >= K:
                    return multi_query_attention_metal(
                        queries, keys, values, lengths, starts, scale)
            except Exception:
                pass
        return original(queries, keys, values, cache=cache, scale=scale, mask=mask,
                        sinks=sinks)

    qwen3.scaled_dot_product_attention = patched
    _PATCHED = True
    return True


def disable_multi_query_patch() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    import mlx_lm.models.qwen3 as qwen3

    if _ORIGINAL is not None:
        qwen3.scaled_dot_product_attention = _ORIGINAL
    _PATCHED = False
