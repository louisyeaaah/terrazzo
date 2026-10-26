"""PoolKVCache: an mlx-lm-compatible cache backed by the paged block pool (v0.4).

Drop-in replacement for ``mlx_lm.models.cache.KVCache``:

  * ``update_and_fetch(k, v)`` scatters the new K/V into the sequence's blocks and
    returns the gathered dense view for the model's attention;
  * the shared prefix is just a set of referenced blocks, so nothing is copied
    when several sequences attach to it.

Compared with v0.2's ``PagedKVCache`` (frozen dense prefix arrays), this owns the
KV per block and can therefore also page the *suffix*, trim, and share at block
granularity.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import create_attention_mask

from .paged_pool import PagedBlockTable, PagedKVPool


class PoolKVCache:
    def __init__(self, layer_idx: int, pool: PagedKVPool, table: PagedBlockTable,
                 auto_release: bool = True) -> None:
        self.layer_idx = layer_idx
        self.pool = pool
        self.table = table
        self.auto_release = auto_release
        self._nbytes_written = 0

    # ------------------------------------------------------------------ protocol
    @property
    def offset(self) -> int:
        # per-layer position: layers are appended one after another within a step
        return self.table.layer_length[self.layer_idx]

    def size(self) -> int:
        return self.table.length

    def empty(self) -> bool:
        return self.table.length == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return self.table.trim(n)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self.append(keys, values)
        return self.gather()

    # ------------------------------------------------------ paging-aware split
    def append(self, keys: mx.array, values: mx.array) -> None:
        """Scatter K/V into the pool without materialising a dense view."""
        self.table.append(self.layer_idx, keys, values)
        self._nbytes_written += keys.nbytes + values.nbytes

    def gather(self) -> tuple[mx.array, mx.array]:
        return self.table.gather(self.layer_idx)

    def metal_attention(self, q: mx.array, scale: float, n_rep: int = 1) -> mx.array:
        """Paged decode attention straight from the pool (q_len must be 1)."""
        from .paged_metal import paged_decode_attention_metal

        return paged_decode_attention_metal(q, self.pool, self.table, self.layer_idx,
                                            scale, n_rep=n_rep)

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None):
        return create_attention_mask(N, offset=self.offset, return_array=return_array,
                                     window_size=window_size)

    @property
    def nbytes(self) -> int:
        """Bytes this sequence added (the shared prefix is not attributed)."""
        return self._nbytes_written

    # ------------------------------------------------------------------- extras
    def detach(self) -> None:
        """Release this sequence's blocks (shared prefix refs included)."""
        self.table.release()

    def __del__(self) -> None:  # pragma: no cover - best effort
        if not getattr(self, "auto_release", True):
            return
        try:
            self.detach()
        except Exception:
            pass


def make_pool_caches(n_layers: int, pool: PagedKVPool, table: PagedBlockTable,
                     auto_release: bool = True) -> list[PoolKVCache]:
    """One cache per layer, all sharing the same block table."""
    return [PoolKVCache(i, pool, table, auto_release=auto_release) for i in range(n_layers)]
