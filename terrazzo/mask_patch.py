"""Skip the causal mask at single-token decode when the batch has no padding.

mlx-lm's ``BatchKVCache.make_mask`` always materialises a causal mask array, and
``scaled_dot_product_attention`` then takes the masked (slower) kernel. During
decode (N == 1) the mask is all-true unless some row is left-padded, so it can be
dropped:

    L=4096: masked 3.87 ms vs plain 2.73 ms per 36 layers
    L=8192: masked 7.59 ms vs plain 4.83 ms

Rows with real left padding still get the original mask, so correctness is
unchanged. ``TERRAZZO_DECODE_MASK=0`` disables the patch.
"""

from __future__ import annotations

import os

import mlx.core as mx

_PATCHED = False
_ORIGINAL_MAKE_MASK = None


def _padding_present(cache) -> bool | None:
    """Cached 'does this batch have left padding' flag (None when unknown)."""
    return getattr(cache, "_dsh_has_padding", None)


def _compute_padding(cache) -> bool | None:
    lp = getattr(cache, "left_padding", None)
    try:
        return bool(lp is not None and int(mx.max(lp).item()) > 0)
    except Exception:  # pragma: no cover - defensive
        return None


def _install_padding_tracking(cls) -> None:
    original_prepare = cls.prepare
    original_finalize = cls.finalize

    def _refresh(cache) -> None:
        lp = getattr(cache, "left_padding", None)
        try:
            cache._dsh_has_padding = bool(lp is not None and int(mx.max(lp).item()) > 0)
        except Exception:  # pragma: no cover - defensive
            cache._dsh_has_padding = None

    def prepare(self, *args, **kwargs):
        out = original_prepare(self, *args, **kwargs)
        _refresh(self)
        return out

    def finalize(self, *args, **kwargs):
        out = original_finalize(self, *args, **kwargs)
        _refresh(self)
        return out

    cls.prepare = prepare
    cls.finalize = finalize

    # extend()/filter() change left_padding/batch layout without going through
    # prepare()/finalize(); invalidate and recompute the flag on those paths too.
    for name in ("extend", "filter"):
        original = getattr(cls, name, None)
        if original is None:
            continue

        def make_wrapper(original):
            def wrapper(self, *args, **kwargs):
                out = original(self, *args, **kwargs)
                _refresh(self)
                return out
            return wrapper

        setattr(cls, name, make_wrapper(original))


def apply_mask_patch() -> bool:
    """Install the patch once; returns True when installed/enabled."""
    global _PATCHED, _ORIGINAL_MAKE_MASK
    if _PATCHED:
        return True
    if os.environ.get("TERRAZZO_DECODE_MASK", "1") == "0":
        return False
    try:
        from mlx_lm.models.cache import BatchKVCache
    except Exception:  # pragma: no cover - mlx-lm not installed
        return False

    _install_padding_tracking(BatchKVCache)
    original_make_mask = BatchKVCache.make_mask

    def make_mask(self, N, return_array=False, **kwargs):
        if N == 1 and not kwargs.get("window_size"):
            has_padding = _padding_present(self)
            if has_padding is None:  # cache never went through prepare/finalize
                has_padding = _compute_padding(self)
                self._dsh_has_padding = has_padding
            if has_padding is False:
                return None
        return original_make_mask(self, N, return_array=return_array, **kwargs)

    BatchKVCache.make_mask = make_mask
    _ORIGINAL_MAKE_MASK = original_make_mask
    _PATCHED = True
    return True


def disable_mask_patch() -> None:
    """Restore mlx-lm's original make_mask (for A/B benchmarks)."""
    global _PATCHED
    if not _PATCHED:
        return
    from mlx_lm.models.cache import BatchKVCache

    if _ORIGINAL_MAKE_MASK is not None:
        BatchKVCache.make_mask = _ORIGINAL_MAKE_MASK
    _PATCHED = False
_ORIGINAL_MAKE_MASK = None
