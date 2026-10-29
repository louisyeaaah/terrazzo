"""Real paged KV storage with cross-request shared prefixes (v0.2).

v0.1 shipped a *logical* block manager and an exact-match prompt cache: the KV
itself lived in per-request MLX cache objects, so two requests with the same
1K-token system prompt still computed and stored that prefix twice.

This module closes that gap:

  * ``SharedPrefix``      a frozen, block-aligned KV prefix kept once in memory
  * ``PagedPrefixCache``  block-manager-backed cache of shared prefixes with
                          ref counting, byte budget and LRU eviction
  * ``PagedKVCache``      a drop-in replacement for ``mlx_lm``'s ``KVCache`` that
                          serves reads as ``concat(shared_prefix, local_suffix)``
  * ``PagedSession``      register a system prefix once, then every request that
                          starts with it only prefills its own suffix

Sharing is exact: identical prefix tokens produce identical KV, so a request that
attaches to a shared prefix computes the same logits as a request that prefilled
from scratch (verified in ``tests/test_paged_kv.py``).

Honest limitation: MLX has no paged-attention kernel, so each step pays one
``mx.concatenate`` of the prefix and the suffix per layer. That is O(context)
copy per step — measurable but small against attention's own memory traffic
(see ``bench/paged_kv.py``). A true kernel would gather from the block pool
inside attention and remove the copy.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import mlx.core as mx

from mlx_lm.models.cache import create_attention_mask

from .blocks import BlockManager


# --------------------------------------------------------------------------- #
# shared prefix
# --------------------------------------------------------------------------- #
def aligned_prefix_length(num_tokens: int, block_size: int,
                          min_tokens: int = 0) -> int:
    """Block-aligned prefix length that can actually be registered."""
    n = (int(num_tokens) // int(block_size)) * int(block_size)
    return n if n >= int(min_tokens) else 0


def _state_nbytes(value) -> int:
    if isinstance(value, mx.array):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return sum(_state_nbytes(v) for v in value)
    return 0


def _clone_state(value):
    """Detach a cache state from its live owner.

    MLX arrays are immutable from Python's perspective; adding zero forces a new
    array for the snapshot while preserving dtype/shape.
    """
    if isinstance(value, mx.array):
        return value + mx.zeros_like(value)
    if isinstance(value, list):
        return [_clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(v) for v in value)
    return value
@dataclass
class SharedPrefix:
    """A frozen KV prefix shared by any number of sequences."""

    tokens: tuple[int, ...]
    num_tokens: int
    # per layer: (keys, values) shaped [1, n_kv_heads, num_tokens, head_dim]
    layers: list[tuple[mx.array, mx.array]]
    # optional native per-layer cache snapshots for non-dense-KV caches:
    # list of (cache_class, state, meta_state). When present, layers may be empty.
    native: list | None = None
    block_ids: list[int] = field(default_factory=list)
    block_hashes: tuple[int, ...] = ()
    ref_count: int = 1          # the cache itself holds one reference
    active: int = 0             # sequences currently attached
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    @property
    def nbytes(self) -> int:
        if self.native:
            return _state_nbytes([st for _cls, st, _meta in self.native])
        return sum(k.nbytes + v.nbytes for k, v in self.layers)


class PagedPrefixCache:
    """Block-manager-backed cache of shared prefixes."""

    def __init__(
        self,
        block_manager: BlockManager,
        *,
        max_bytes: int = 4 << 30,
        max_entries: int = 8,
        min_tokens: int | None = None,
    ) -> None:
        self.block_manager = block_manager
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.min_tokens = min_tokens or block_manager.block_size
        self.entries: "OrderedDict[tuple[int, ...], SharedPrefix]" = OrderedDict()
        # block hash -> (prefix, block index): lets us match a *partial* prefix
        self.block_index: dict[int, tuple[SharedPrefix, int]] = {}
        self.bytes_used = 0
        self.num_hits = 0
        self.num_misses = 0
        self.num_stores = 0
        self.num_evictions = 0

    # ------------------------------------------------------------------ lookup
    def _match(self, prompt_tokens: list[int]) -> tuple[SharedPrefix, int] | None:
        """Longest shareable block prefix, without mutating LRU/stats."""
        bs = self.block_manager.block_size
        hashes = self.block_manager.hash_chain(prompt_tokens)
        best: tuple[SharedPrefix, int] | None = None
        for idx, h in enumerate(hashes):
            if h is None:
                break
            found = self.block_index.get(h)
            if found is None:
                break
            entry, entry_idx = found
            if entry_idx != idx:  # same hash must sit at the same block position
                break
            best = (entry, (idx + 1) * bs)
        return best

    def peek(self, prompt_tokens: list[int]) -> tuple[SharedPrefix, int] | None:
        """Inspect a potential hit for the cost model without counting it."""
        return self._match(prompt_tokens)

    def lookup(self, prompt_tokens: list[int]) -> tuple[SharedPrefix, int] | None:
        """Longest *shareable block prefix* of ``prompt_tokens``.

        Unlike v0.2 this does not require the prompt to contain a whole cached
        prefix: it walks the block hash chain and stops at the first block that is
        not in the cache, so a prompt that diverges after 3 of 8 cached blocks
        still shares those 3 blocks (``hit_len`` is returned separately).
        """
        best = self._match(prompt_tokens)
        if best is None:
            self.num_misses += 1
            return None
        self.entries.move_to_end(best[0].tokens)
        best[0].hits += 1
        self.num_hits += 1
        return best

    # ------------------------------------------------------------------- store
    def store(self, prompt_tokens: list[int], layers: list[tuple[mx.array, mx.array]],
              native: list | None = None) -> SharedPrefix | None:
        """Register a block-aligned prefix (mutates nothing if it cannot fit)."""
        bs = self.block_manager.block_size
        n = aligned_prefix_length(len(prompt_tokens), bs, self.min_tokens)
        if n <= 0:
            return None
        tokens = tuple(prompt_tokens[:n])
        if tokens in self.entries:
            return None
        num_blocks = n // bs
        if not self.block_manager.can_allocate(num_blocks):
            return None
        block_ids = self.block_manager.allocate(num_blocks, token_ids=list(tokens))
        hashes = tuple(self.block_manager.hash_chain(list(tokens)))
        # detach from the generation buffer: slices keep their parent alive
        frozen = [(mx.contiguous(k[..., :n, :]), mx.contiguous(v[..., :n, :])) for k, v in layers]
        entry = SharedPrefix(
            tokens=tokens, num_tokens=n, layers=frozen, native=native,
            block_ids=block_ids, block_hashes=hashes,
        )
        self.entries[tokens] = entry
        for idx, h in enumerate(entry.block_hashes):
            if h is not None:
                self.block_index[h] = (entry, idx)
        self.bytes_used += entry.nbytes
        self.num_stores += 1
        self._evict()
        return entry

    # --------------------------------------------------------------- ref-count
    def acquire(self, prefix: SharedPrefix) -> None:
        prefix.ref_count += 1
        prefix.active += 1
        self.block_manager.ref(prefix.block_ids)

    def release(self, prefix: SharedPrefix) -> None:
        if prefix.active > 0:
            prefix.active -= 1
        prefix.ref_count = max(1, prefix.ref_count - 1)  # cache keeps its own ref

    # -------------------------------------------------------------------- evict
    def _evict(self) -> None:
        while (
            len(self.entries) > self.max_entries or self.bytes_used > self.max_bytes
        ) and self.entries:
            victim_key = None
            for key, entry in self.entries.items():
                if entry.active == 0:  # never evict a prefix a request is using
                    victim_key = key
                    break
            if victim_key is None:
                break
            entry = self.entries.pop(victim_key)
            for h in entry.block_hashes:
                if h is not None and self.block_index.get(h, (None, -1))[0] is entry:
                    del self.block_index[h]
            self.bytes_used -= entry.nbytes
            self.block_manager.free(entry.block_ids)
            self.num_evictions += 1

    def clear(self) -> None:
        for entry in self.entries.values():
            self.block_manager.free(entry.block_ids)
        self.entries.clear()
        self.block_index.clear()
        self.bytes_used = 0

    # -------------------------------------------------------------------- stats
    def stats(self) -> dict:
        total = self.num_hits + self.num_misses
        return {
            "entries": len(self.entries),
            "active": sum(e.active for e in self.entries.values()),
            "nbytes": self.bytes_used,
            "hits": self.num_hits,
            "misses": self.num_misses,
            "stores": self.num_stores,
            "evictions": self.num_evictions,
            "hit_rate": round(self.num_hits / total, 4) if total else 0.0,
        }


# --------------------------------------------------------------------------- #
# paged cache
# --------------------------------------------------------------------------- #
class _GrowingBuffer:
    """Minimal contiguous KV buffer (same growth policy as mlx-lm's KVCache)."""

    __slots__ = ("keys", "values", "offset", "step")

    def __init__(self, step: int = 256) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.step = step

    def append(self, keys: mx.array, values: mx.array) -> None:
        new = keys.shape[2]
        if self.keys is None or self.offset + new > self.keys.shape[2]:
            B, H, _, D = keys.shape
            n_steps = (self.step + new - 1) // self.step
            k = mx.zeros((B, H, n_steps * self.step, D), keys.dtype)
            v = mx.zeros((B, H, n_steps * self.step, values.shape[3]), values.dtype)
            if self.keys is not None:
                if self.offset % self.step != 0:
                    self.keys = self.keys[..., : self.offset, :]
                    self.values = self.values[..., : self.offset, :]
                self.keys = mx.concatenate([self.keys, k], axis=2)
                self.values = mx.concatenate([self.values, v], axis=2)
            else:
                self.keys, self.values = k, v
        self.keys[..., self.offset : self.offset + new, :] = keys
        self.values[..., self.offset : self.offset + new, :] = values
        self.offset += new

    def view(self) -> tuple[mx.array, mx.array]:
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def nbytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n


class PagedKVCache:
    """Drop-in ``KVCache`` that can prepend a frozen shared prefix."""

    def __init__(self, layer_idx: int = 0, prefix: SharedPrefix | None = None,
                 prefix_cache: PagedPrefixCache | None = None,
                 prefix_len: int | None = None) -> None:
        self.layer_idx = layer_idx
        self.prefix = prefix
        self.prefix_cache = prefix_cache
        # a request may share only the first ``prefix_len`` tokens of a longer prefix
        self.prefix_len = (
            prefix.num_tokens if prefix is not None and prefix_len is None
            else (prefix_len or 0)
        )
        self.buf = _GrowingBuffer()
        if prefix is not None and prefix_cache is not None:
            prefix_cache.acquire(prefix)

    # --------------------------------------------------------------- protocol
    @property
    def offset(self) -> int:
        return (self.prefix_len if self.prefix else 0) + self.buf.offset

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        # the shared prefix is immutable; only the local suffix can be trimmed
        return self.buf.trim(n)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self.buf.append(keys, values)
        k, v = self.buf.view()
        if self.prefix is not None:
            pk, pv = self.prefix.layers[self.layer_idx]
            if self.prefix_len < pk.shape[2]:
                pk = pk[..., : self.prefix_len, :]
                pv = pv[..., : self.prefix_len, :]
            k = mx.concatenate([pk, k], axis=2)
            v = mx.concatenate([pv, v], axis=2)
        return k, v

    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None):
        return create_attention_mask(N, offset=self.offset, return_array=return_array,
                                     window_size=window_size)

    @property
    def nbytes(self) -> int:
        """Bytes owned by this sequence (the shared prefix is not counted)."""
        return self.buf.nbytes

    def detach(self) -> None:
        if self.prefix is not None and self.prefix_cache is not None:
            self.prefix_cache.release(self.prefix)
            self.prefix = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.detach()
        except Exception:
            pass
