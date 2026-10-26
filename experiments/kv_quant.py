"""KV-cache quantization for the MacTurbo stack.

Formula level: KV bytes are *not* part of the per-token weight read, so KV
quantization does not raise single-request decode speed. It buys:
  * concurrency: the same RAM holds more sequences / longer contexts;
  * long-context decode (attention reads scale with KV size);
  * prefix-cache capacity in the paged pool (v0.4 stores fp16 today).

This module provides the building block + a measured comparison:
``quantized_prompt_cache`` builds a per-layer cache list where every plain
``KVCache`` is replaced by mlx-lm's ``QuantizedKVCache`` (group 64, 4 or 8 bit).
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache


def quantized_prompt_cache(model, bits: int = 4, group_size: int = 64) -> tuple[list, int]:
    """Per-layer cache list with quantized KV; returns (caches, num_replaced)."""
    caches = make_prompt_cache(model)
    replaced = 0
    for i, c in enumerate(caches):
        if isinstance(c, KVCache):
            caches[i] = QuantizedKVCache(group_size=group_size, bits=bits)
            replaced += 1
    return caches, replaced


def cache_bytes(caches) -> int:
    total = 0
    for c in caches:
        try:
            total += int(c.nbytes)
        except Exception:
            pass
    return total


def _prefill(model, tokens: list[int], caches, chunk: int = 1024):
    logits = None
    for s in range(0, len(tokens), chunk):
        logits = model(mx.array([tokens[s:s + chunk]]), cache=caches)
        mx.eval(logits)
    return logits


def generate_measured(model, prompt_ids: list[int], *, max_tokens: int = 64,
                      kv_bits: int | None = None, group_size: int = 64) -> dict:
    """Greedy generation with an optional quantized KV cache; returns metrics."""
    import time

    if kv_bits:
        caches, replaced = quantized_prompt_cache(model, kv_bits, group_size)
    else:
        caches, replaced = make_prompt_cache(model), 0

    t0 = time.perf_counter()
    logits = _prefill(model, prompt_ids, caches)
    token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    tokens = [token]
    ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    while len(tokens) < max_tokens:
        logits = model(mx.array([[token]]), cache=caches)
        mx.eval(logits)
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        tokens.append(token)
    decode_s = time.perf_counter() - t1
    return {
        "kv_bits": kv_bits,
        "layers_quantized": replaced,
        "output_tokens": len(tokens),
        "ttft_s": round(ttft, 4),
        "decode_tok_s": round((len(tokens) - 1) / decode_s, 1) if decode_s else 0.0,
        "kv_cache_mb": round(cache_bytes(caches) / 1e6, 1),
        "tokens": tokens,
    }
