"""True paged KV storage for MLX (v0.4).

v0.2/v0.3 shared *frozen dense arrays*: a prefix was one `mx.array` per layer that
every sequence concatenated against. This module replaces that with a block pool:

    k_pool: [num_blocks, n_kv_heads, block_size, head_dim]   (one per layer)
    v_pool: same

A sequence holds a **block table** (list of block ids), so:

  * new tokens are **scattered** into blocks at (block, offset) — no per-sequence
    contiguous buffer, no copy-on-grow;
  * a shared prefix is just a list of block ids referenced by several sequences
    (ref-counted), so the KV exists **once** however many requests use it;
  * reading is a gather over only the blocks the sequence owns.

Notes / limits (documented in DESIGN.md):
  * prefixes attach at block granularity and are treated as full, immutable blocks.
    Two sequences sharing a *partially filled* block would need copy-on-write; that
    is not needed for the system-prompt use case and is not implemented.
  * `gather` materialises a dense [1, H, L, D] view for attention. A Metal kernel
    that indexes the pool inside attention is the next step (see paged_attention).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class PoolStats:
    num_blocks: int = 0
    block_size: int = 0
    used_blocks: int = 0
    shared_blocks: int = 0
    peak_blocks: int = 0
    nbytes: int = 0
    layers_allocated: int = 0


class PagedKVPool:
    """Per-layer block pool with ref-counted blocks."""

    def __init__(
        self,
        num_layers: int,
        *,
        num_blocks: int = 512,
        block_size: int = 16,
        n_kv_heads: int = 8,
        head_dim: int = 128,
        dtype=mx.float16,
        kv_bits: int | None = None,
        group_size: int = 64,
    ) -> None:
        if kv_bits not in (None, 4, 8):
            raise ValueError("kv_bits must be None, 4 or 8")
        if kv_bits is not None and head_dim % group_size:
            raise ValueError(f"head_dim {head_dim} must be divisible by group_size {group_size}")
        self.kv_bits = kv_bits
        self.group_size = group_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        # layer pools are allocated lazily so unit tests can stay tiny
        self._k: list[mx.array | None] = [None] * num_layers
        self._v: list[mx.array | None] = [None] * num_layers
        self._kq: list[dict | None] = [None] * num_layers
        self._vq: list[dict | None] = [None] * num_layers
        self.ref_count = [0] * num_blocks
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self._peak = 0

    # ------------------------------------------------------------------ memory
    @property
    def _elements_per_int(self) -> int:
        return 32 // (self.kv_bits or 32)

    def _quant_shapes(self) -> tuple[tuple, tuple]:
        epi = self._elements_per_int
        packed = (self.num_blocks, self.n_kv_heads, self.block_size,
                  self.head_dim // epi)
        meta = (self.num_blocks, self.n_kv_heads, self.block_size,
                self.head_dim // self.group_size)
        return packed, meta

    def _empty_quant(self) -> dict:
        packed, meta = self._quant_shapes()
        return {"wq": mx.zeros(packed, mx.uint32),
                "scales": mx.zeros(meta, self.dtype),
                "biases": mx.zeros(meta, self.dtype)}

    def _ensure_layer(self, layer: int) -> None:
        if self.kv_bits is not None:
            if self._kq[layer] is not None:
                return
            self._kq[layer] = self._empty_quant()
            self._vq[layer] = self._empty_quant()
            return
        if self._k[layer] is not None:
            return
        shape = (self.num_blocks, self.n_kv_heads, self.block_size, self.head_dim)
        self._k[layer] = mx.zeros(shape, self.dtype)
        self._v[layer] = mx.zeros(shape, self.dtype)

    @property
    def k(self) -> list[mx.array | None]:
        return self._k

    @property
    def v(self) -> list[mx.array | None]:
        return self._v

    def layer_kv(self, layer: int) -> tuple[mx.array, mx.array]:
        if self.kv_bits is not None:
            raise NotImplementedError(
                "quantized pool has no dense layer KV: use gather()/paged_attention "
                "(the Metal kernel needs fp16 layer arrays)")
        self._ensure_layer(layer)
        return self._k[layer], self._v[layer]

    # -------------------------------------------------------------- allocation
    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def num_shared(self) -> int:
        return sum(1 for r in self.ref_count if r > 1)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self.free_blocks)

    def allocate(self, n: int) -> list[int]:
        if n > len(self.free_blocks):
            raise ValueError(f"cannot allocate {n} blocks, only {len(self.free_blocks)} free")
        ids = [self.free_blocks.pop() for _ in range(n)]
        for b in ids:
            self.ref_count[b] = 1
        self._peak = max(self._peak, self.num_used)
        return ids

    def ref(self, blocks: list[int]) -> None:
        for b in blocks:
            if self.ref_count[b] <= 0:
                raise ValueError(f"block {b} is free, cannot ref")
            self.ref_count[b] += 1

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if self.ref_count[b] <= 0:
                continue
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free_blocks.append(b)

    def clone_block(self, src: int) -> int:
        """Allocate a private copy of ``src`` (all layers) for copy-on-write."""
        (dst,) = self.allocate(1)
        for layer in range(self.num_layers):
            self._ensure_layer(layer)
            if self.kv_bits is not None:
                for store in (self._kq[layer], self._vq[layer]):
                    for name in ("wq", "scales", "biases"):
                        store[name][dst] = store[name][src]
            else:
                self._k[layer][dst] = self._k[layer][src]
                self._v[layer][dst] = self._v[layer][src]
        return dst

    # ------------------------------------------------------------------- write
    def write(self, layer: int, block_ids: list[int], start: int,
              k: mx.array, v: mx.array) -> None:
        """Scatter ``n`` new tokens at positions ``[start, start+n)``.

        ``k``/``v`` have shape [1, H, n, D]; ``block_ids`` covers every block the
        sequence owns (so index ``p // block_size`` is valid).
        """
        self._ensure_layer(layer)
        bs = self.block_size
        n = k.shape[2]
        pos = 0
        if self.kv_bits is not None:
            while pos < n:
                global_pos = start + pos
                block_idx = global_pos // bs
                offset = global_pos % bs
                take = min(bs - offset, n - pos)
                dest = block_ids[block_idx]
                for store, dense in ((self._kq[layer], k), (self._vq[layer], v)):
                    wq, scales, biases = mx.quantize(
                        dense[0, :, pos:pos + take, :],
                        group_size=self.group_size, bits=self.kv_bits)
                    store["wq"][dest, :, offset:offset + take, :] = wq
                    store["scales"][dest, :, offset:offset + take, :] = scales
                    store["biases"][dest, :, offset:offset + take, :] = biases
                pos += take
            return
        while pos < n:
            global_pos = start + pos
            block_idx = global_pos // bs
            offset = global_pos % bs
            take = min(bs - offset, n - pos)
            dest = block_ids[block_idx]
            self._k[layer][dest, :, offset:offset + take, :] = k[0, :, pos:pos + take, :]
            self._v[layer][dest, :, offset:offset + take, :] = v[0, :, pos:pos + take, :]
            pos += take

    # ------------------------------------------------------------------- read
    def gather(self, layer: int, block_ids: list[int], length: int) -> tuple[mx.array, mx.array]:
        """Dense [1, H, length, D] view of the first ``length`` tokens."""
        self._ensure_layer(layer)
        if not block_ids:
            shape = (1, self.n_kv_heads, 0, self.head_dim)
            return mx.zeros(shape, self.dtype), mx.zeros(shape, self.dtype)
        idx = mx.array(block_ids, dtype=mx.int32)
        if self.kv_bits is not None:
            return self._gather_quant(layer, idx, length)
        k = mx.take(self._k[layer], idx, axis=0)   # [nb, H, bs, D]
        v = mx.take(self._v[layer], idx, axis=0)
        nb = k.shape[0]
        # [nb, H, bs, D] -> [H, nb*bs, D] -> [1, H, nb*bs, D]
        k = mx.reshape(mx.transpose(k, (1, 0, 2, 3)),
                       (self.n_kv_heads, nb * self.block_size, self.head_dim))
        v = mx.reshape(mx.transpose(v, (1, 0, 2, 3)),
                       (self.n_kv_heads, nb * self.block_size, self.head_dim))
        k = mx.expand_dims(k, 0)[:, :, :length, :]
        v = mx.expand_dims(v, 0)[:, :, :length, :]
        return k, v

    def _gather_quant(self, layer: int, idx: mx.array, length: int) -> tuple[mx.array, mx.array]:
        nb = len(idx)
        epi = self._elements_per_int
        packed_dim = self.head_dim // epi
        meta_dim = self.head_dim // self.group_size
        total = nb * self.block_size

        def flatten(store: dict, name: str, last: int) -> mx.array:
            x = mx.take(store[name], idx, axis=0)          # [nb, H, bs, last]
            x = mx.transpose(x, (1, 0, 2, 3))              # [H, nb, bs, last]
            return mx.reshape(x, (self.n_kv_heads, total, last))

        outs = []
        for store in (self._kq[layer], self._vq[layer]):
            wq = flatten(store, "wq", packed_dim)
            sc = flatten(store, "scales", meta_dim)
            bi = flatten(store, "biases", meta_dim)
            dense = mx.dequantize(wq, sc, bi, group_size=self.group_size, bits=self.kv_bits)
            outs.append(mx.expand_dims(dense, 0)[:, :, :length, :])
        return outs[0], outs[1]

    # ------------------------------------------------------------------- stats
    def stats(self) -> PoolStats:
        nbytes = 0
        layers = 0
        for i, (lk, lv) in enumerate(zip(self._k, self._v)):
            if self.kv_bits is not None:
                if self._kq[i] is None:
                    continue
                layers += 1
                for store in (self._kq[i], self._vq[i]):
                    nbytes += sum(store[name].nbytes for name in ("wq", "scales", "biases"))
                continue
            if lk is None:
                continue
            layers += 1
            nbytes += lk.nbytes + lv.nbytes
        return PoolStats(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            used_blocks=self.num_used,
            shared_blocks=self.num_shared,
            peak_blocks=self._peak,
            nbytes=nbytes,
            layers_allocated=layers,
        )


class PagedBlockTable:
    """Block table for one sequence: optional shared prefix + private suffix blocks."""

    __slots__ = ("pool", "blocks", "length", "_prefix_len", "layer_length")

    def __init__(self, pool: PagedKVPool, *, prefix_blocks: list[int] | None = None,
                 prefix_len: int = 0) -> None:
        self.pool = pool
        self.blocks: list[int] = []
        self.length = 0
        self._prefix_len = 0
        # one logical length per layer: the model calls update_and_fetch once per
        # layer for the same tokens, so the table must not advance per layer call
        self.layer_length = [0] * pool.num_layers
        if prefix_blocks:
            self.pool.ref(prefix_blocks)
            self.blocks.extend(prefix_blocks)
            self._prefix_len = prefix_len
            self.length = prefix_len
            self.layer_length = [prefix_len] * pool.num_layers

    # ------------------------------------------------------------------ growth
    def ensure_capacity(self, total_tokens: int) -> None:
        """Make sure the table can address ``total_tokens`` positions."""
        need = (total_tokens + self.pool.block_size - 1) // self.pool.block_size
        if need > len(self.blocks):
            self.blocks.extend(self.pool.allocate(need - len(self.blocks)))
        self.length = total_tokens

    def _ensure_private(self, start: int) -> None:
        """Copy-on-write: never write into a block another sequence still holds.

        Only needed when a prefix was attached at a non-block boundary (the last
        shared block is partially filled). Block-aligned prefixes always write
        into freshly allocated blocks.
        """
        idx = start // self.pool.block_size
        if idx >= len(self.blocks):
            return
        bid = self.blocks[idx]
        if self.pool.ref_count[bid] > 1:
            new_bid = self.pool.clone_block(bid)
            self.pool.free([bid])
            self.blocks[idx] = new_bid

    def append(self, layer: int, k: mx.array, v: mx.array) -> None:
        """Write new tokens for ``layer`` at that layer's current position."""
        n = k.shape[2]
        start = self.layer_length[layer]
        self.ensure_capacity(start + n)
        self._ensure_private(start)
        self.pool.write(layer, self.blocks, start, k, v)
        self.layer_length[layer] = start + n
        self.length = max(self.length, start + n)

    def gather(self, layer: int) -> tuple[mx.array, mx.array]:
        # each layer has its own logical length: the model calls update_and_fetch
        # layer by layer, so a layer must only see positions it has written.
        return self.pool.gather(layer, self.blocks, self.layer_length[layer])

    def trim(self, n: int) -> int:
        """Drop the last ``n`` positions (all layers), releasing unused blocks."""
        n = min(n, self.length - self._prefix_len)
        if n <= 0:
            return 0
        self.length -= n
        self.layer_length = [max(0, l - n) for l in self.layer_length]
        keep = (self.length + self.pool.block_size - 1) // self.pool.block_size
        if keep < len(self.blocks):
            self.pool.free(self.blocks[keep:])
            self.blocks = self.blocks[:keep]
        return n

    @property
    def prefix_len(self) -> int:
        return self._prefix_len

    def release(self) -> None:
        self.pool.free(self.blocks)
        self.blocks = []
        self.length = 0
        self._prefix_len = 0

    def __len__(self) -> int:
        return self.length
