"""Keep the KV cache contiguous and mask the unused tail instead of slicing.

``BatchKVCache`` allocates K/V in 256-token steps and hands SDPA a *slice*
``keys[..., :idx, :]``. That slice is not contiguous (stride between head rows is
the allocation length), and MLX's SDPA falls back to a path ~3.5x slower than for
contiguous inputs: measured 10.3 ms vs 2.74 ms per 36 layers at 4k context.

This patch keeps the cached buffers full and contiguous and instead extends the
attention mask over the unused tail, so the fast contiguous kernel is used with a
(masked) path that costs ~3.9 ms per 36 layers at 4k — still much better than 10.3.

``TERRAZZO_CONTIG_KV=0`` disables it.
"""

from __future__ import annotations

import os

import mlx.core as mx

_PATCHED = False
_ORIGINAL_UPDATE = None
_ORIGINAL_MASK = None


def _tail_mask(cache, N: int):
    """Mask over the full allocation: causal for valid tokens, False for the tail."""
    alloc = int(cache.keys.shape[2])
    offset = int(cache._idx)
    linds = mx.arange(offset, offset + N)[:, None]
    rinds = mx.arange(alloc)[None]
    mask = linds >= rinds                      # causal: query i sees keys <= its position
    lp = getattr(cache, "left_padding", None)
    if lp is not None:
        mask = mask & (mx.expand_dims(lp, (1, 2, 3)) <= rinds)
    return mask


def apply_contig_kv_patch() -> bool:
    global _PATCHED, _ORIGINAL_UPDATE, _ORIGINAL_MASK
    if _PATCHED:
        return True
    # Default OFF: in-engine A/B measured it slower than the sliced cache
    # (1k -6.8%, 4k -3.8%, 8k -8.5%); kept opt-in for future experiments.
    if os.environ.get("TERRAZZO_CONTIG_KV", "0") != "1":
        return False
    from mlx_lm.models.cache import BatchKVCache

    if not hasattr(BatchKVCache, "update_and_fetch"):
        return False
    original_update = BatchKVCache.update_and_fetch
    original_mask = BatchKVCache.make_mask
    _ORIGINAL_UPDATE, _ORIGINAL_MASK = original_update, original_mask

    # The original update builds the padded (B, Hk, alloc, D) buffer and writes the
    # new slice in place; it returns sliced views. Wrap it: run the original, then
    # return the full buffers (contiguous).
    def _ensure_capacity(self, needed: int) -> None:
        """Grow the buffer so the mask length matches the buffers the model sees."""
        if self.keys is None:
            return
        alloc = int(self.keys.shape[2])
        if needed <= alloc:
            return
        step = max(1, int(getattr(type(self), "step", 256)))
        grow = ((needed - alloc + step - 1) // step) * step
        B, H, _, D = self.keys.shape
        extra_k = mx.zeros((B, H, grow, D), self.keys.dtype)
        extra_v = mx.zeros((B, H, grow, self.values.shape[3]), self.values.dtype)
        self.keys = mx.concatenate([self.keys, extra_k], axis=2)
        self.values = mx.concatenate([self.values, extra_v], axis=2)

    def update_and_fetch(self, keys, values):
        _ensure_capacity(self, int(self._idx) + int(keys.shape[2]))
        original_update(self, keys, values)  # in-place write + bookkeeping
        return self.keys, self.values

    def make_mask(self, N, return_array=False, **kwargs):
        if self.keys is None:
            return original_mask(self, N, return_array=return_array, **kwargs)
        _ensure_capacity(self, int(self._idx) + N)
        return _tail_mask(self, N)

    BatchKVCache.update_and_fetch = update_and_fetch
    BatchKVCache.make_mask = make_mask
    _PATCHED = True
    return True


def disable_contig_kv_patch() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    from mlx_lm.models.cache import BatchKVCache

    if _ORIGINAL_UPDATE is not None:
        BatchKVCache.update_and_fetch = _ORIGINAL_UPDATE
    if _ORIGINAL_MASK is not None:
        BatchKVCache.make_mask = _ORIGINAL_MASK
    _PATCHED = False
