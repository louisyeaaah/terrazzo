"""Patch mlx-lm's Qwen3 attention to use the batched dense decode kernel.

mlx-lm calls ``scaled_dot_product_attention`` from ``mlx_lm.models.qwen3``; during
batch decode (q_len == 1, ``BatchKVCache``) that call is the measured long-context
bottleneck. The patch swaps it for ``dense_decode_attention_metal`` and falls back
to the original SDPA everywhere else (prefill, quantized caches, unsupported dtypes).
"""

from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx

from .dense_metal import dense_decode_attention_metal


def _eligible(cache, queries, keys, values) -> bool:
    if cache is None or queries is None:
        return False
    if not (hasattr(cache, "offset") and hasattr(cache, "left_padding")):
        return False
    if queries.shape[2] != 1 or keys is None or keys.shape[2] <= 1:
        return False
    if queries.dtype != mx.float16 or keys.dtype != mx.float16:
        return False
    return queries.shape[3] % 32 == 0 and queries.shape[1] % keys.shape[1] == 0


@contextmanager
def patched_qwen3_batch_attention():
    import mlx_lm.models.qwen3 as qwen3

    original = qwen3.scaled_dot_product_attention

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        if sinks is None and _eligible(cache, queries, keys, values):
            try:
                lengths = cache.offset.astype(mx.int32)
                if keys.shape[2] < 3072:
                    # medium context: sharing a KV head across 2 query heads wins
                    from .dense_gqa_metal import dense_gqa_attention_metal

                    return dense_gqa_attention_metal(queries, keys, values, lengths, scale,
                                                     heads_per_group=2)
                split = 8 if keys.shape[2] >= 6144 else 4  # more parallelism on long KV
                return dense_decode_attention_metal(queries, keys, values, lengths, scale,
                                                     split=split)
            except Exception:
                pass
        return original(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    qwen3.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        qwen3.scaled_dot_product_attention = original


_PATCHED = False
_ORIGINAL = None


def apply_batch_attn_patch(min_tokens: int | None = None) -> bool:
    """Install the auto-dispatch patch (long KV -> Metal kernel, else SDPA).

    ``TERRAZZO_BATCH_ATTN=0`` disables; ``TERRAZZO_BATCH_ATTN_MIN_TOKENS`` sets the
    context threshold (default 4096).
    """
    global _PATCHED, _ORIGINAL
    if _PATCHED:
        return True
    import os

    if os.environ.get("TERRAZZO_BATCH_ATTN", "1") == "0":
        return False
    threshold = min_tokens
    if threshold is None:
        threshold = int(os.environ.get("TERRAZZO_BATCH_ATTN_MIN_TOKENS", "512"))
    import mlx_lm.models.qwen3 as qwen3

    original = qwen3.scaled_dot_product_attention
    _ORIGINAL = original

    def patched(queries, keys, values, cache, scale, mask, sinks=None):
        # quantized caches hand SDPA tuples, not arrays: keep the original path
        if (sinks is None and isinstance(keys, mx.array) and keys.shape[2] >= threshold
                and _eligible(cache, queries, keys, values)):
            try:
                lengths = cache.offset.astype(mx.int32)
                if keys.shape[2] < 3072:
                    # medium context: sharing a KV head across 2 query heads wins
                    from .dense_gqa_metal import dense_gqa_attention_metal

                    return dense_gqa_attention_metal(queries, keys, values, lengths, scale,
                                                     heads_per_group=2)
                split = 8 if keys.shape[2] >= 6144 else 4  # more parallelism on long KV
                return dense_decode_attention_metal(queries, keys, values, lengths, scale,
                                                     split=split)
            except Exception:
                pass
        return original(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)

    qwen3.scaled_dot_product_attention = patched
    _PATCHED = True
    return True


def disable_batch_attn_patch() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    import mlx_lm.models.qwen3 as qwen3

    if _ORIGINAL is not None:
        qwen3.scaled_dot_product_attention = _ORIGINAL
    _PATCHED = False
