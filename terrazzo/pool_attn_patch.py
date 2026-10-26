"""Patch Qwen3 attention to use the paged pool + Metal decode kernel (v0.4).

mlx-lm's attention always calls ``cache.update_and_fetch`` and then a dense SDPA,
so a paged cache is forced to materialise a dense K/V view. This patch keeps the
model code intact but swaps the attention *implementation* when the cache is a
``PoolKVCache``:

  * prefill (L > 1): scatter K/V into the pool, gather the sequence's blocks and run
    the same SDPA the model would have run;
  * decode  (L == 1): scatter K/V, then run ``paged_decode_attention_metal`` which
    reads the blocks directly — no dense view, no per-step copy.

Used as a context manager so it can be measured against the unpatched baseline:

    with patched_qwen3_attention():
        session.generate(...)
"""

from __future__ import annotations

import contextlib

import mlx.core as mx
from mlx_lm.models.qwen3 import Attention

from .paged_metal import metal_available
from .pool_cache import PoolKVCache

_ORIGINAL = Attention.__call__


def _patched_call(self, x: mx.array, mask=None, cache=None):
    if not isinstance(cache, PoolKVCache):
        return _ORIGINAL(self, x, mask, cache)

    B, L, D = x.shape
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
    keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    queries = self.rope(queries, offset=cache.offset)
    keys = self.rope(keys, offset=cache.offset)
    cache.append(keys, values)

    n_rep = self.n_heads // self.n_kv_heads
    if L == 1 and metal_available():
        out = cache.metal_attention(queries, self.scale, n_rep=n_rep)
        out = out.astype(queries.dtype)
    else:
        k, v = cache.gather()
        out = mx.fast.scaled_dot_product_attention(queries, k, v, scale=self.scale, mask=mask)
    output = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


@contextlib.contextmanager
def patched_qwen3_attention():
    Attention.__call__ = _patched_call
    try:
        yield
    finally:
        Attention.__call__ = _ORIGINAL
