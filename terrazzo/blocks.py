"""Paged KV block manager.

Port of DeepInfer's ``csrc/engine/block_manager.hpp`` semantics:
  - a fixed pool of blocks, each holding ``block_size`` token slots
  - ``ref_count`` per block; a block returns to the free list at ref_count == 0
  - block 0 is reserved as the "empty" sentinel (same trick as DeepInfer)
  - allocation is all-or-nothing (``can_allocate`` then ``allocate``)

Added on top (DeepInfer's open-source tree has no content-hash prefix cache --
its hit length logic lives in the un-opened KVCache store):
  - per-block ``block_hash`` chained over token ids (vLLM-style hash chain)
  - ``PrefixCache`` in prefix_cache.py uses it to find the longest cached prefix

On Apple Silicon the physical KV tensors live inside MLX cache objects; this
class is the capacity/accounting layer that decides what can be admitted and
what must be preempted. That mirrors the role the C++ block manager plays in
DeepInfer: the scheduler asks it "do we have room for this?", the executor
materialises the KV.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class BlockManagerConfig:
    num_blocks: int = 512
    block_size: int = 16
    # DeepInfer keeps one block of slack for decode-time appends.
    reserve_blocks: int = 1


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    block_hash: int | None = None
    token_ids: list[int] = field(default_factory=list)

    @property
    def is_free(self) -> bool:
        return self.ref_count == 0


def hash_block_tokens(parent_hash: int | None, token_ids: Iterable[int]) -> int:
    """Chained block hash: H(parent_hash, tokens).

    Identical prefixes produce identical hashes regardless of where the block
    sits in the pool, which is what makes the prefix cache a hash lookup.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(b"\x00" if parent_hash is None else parent_hash.to_bytes(8, "little"))
    h.update(bytes(str(tuple(token_ids)), "utf-8"))
    return int.from_bytes(h.digest()[:8], "little")


class BlockManager:
    """Fixed-size paged KV allocator with ref counting."""

    def __init__(self, cfg: BlockManagerConfig | None = None) -> None:
        self.cfg = cfg or BlockManagerConfig()
        self.block_size = self.cfg.block_size
        self.num_blocks = self.cfg.num_blocks
        # Block 0 is the sentinel, exactly like DeepInfer's block_manager.hpp.
        self.blocks: list[Block] = [Block(0, ref_count=1)]
        self.blocks += [Block(i) for i in range(1, self.num_blocks)]
        self.free_blocks: list[int] = list(range(self.num_blocks - 1, 0, -1))
        # hash -> block_id (only blocks referenced by the prefix cache)
        self.hash_to_block: dict[int, int] = {}
        self._peak_used = 0

    # ------------------------------------------------------------------ stats
    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - 1 - len(self.free_blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks) - self.cfg.reserve_blocks

    @property
    def usage(self) -> float:
        return self.num_used_blocks / max(1, self.num_blocks - 1)

    @property
    def peak_usage(self) -> float:
        return self._peak_used / max(1, self.num_blocks - 1)

    # ------------------------------------------------------------- allocation
    def blocks_for(self, num_tokens: int) -> int:
        return max(1, -(-num_tokens // self.block_size))

    def can_allocate(self, num_blocks: int) -> bool:
        return num_blocks <= self.num_free_blocks

    def allocate(self, num_blocks: int, *, token_ids: list[int] | None = None,
                 block_offset: int = 0) -> list[int]:
        """Allocate ``num_blocks`` blocks for the token range starting at
        ``block_offset * block_size`` of ``token_ids`` (hashes stay chained)."""
        if not self.can_allocate(num_blocks):
            raise ValueError(
                f"cannot allocate {num_blocks} blocks, only {self.num_free_blocks} free"
            )
        chain = self.hash_chain(token_ids) if token_ids is not None else []
        block_ids = [self.free_blocks.pop() for _ in range(num_blocks)]
        for i, bid in enumerate(block_ids):
            blk = self.blocks[bid]
            blk.ref_count = 1
            if token_ids is not None:
                idx = block_offset + i
                start = idx * self.block_size
                blk.token_ids = token_ids[start:start + self.block_size]
                if idx < len(chain) and chain[idx] is not None:
                    blk.block_hash = chain[idx]
                    self.hash_to_block[blk.block_hash] = bid
        self._peak_used = max(self._peak_used, self.num_used_blocks)
        return block_ids

    def free(self, block_ids: Iterable[int]) -> None:
        for bid in block_ids:
            if bid == 0:
                continue
            blk = self.blocks[bid]
            if blk.ref_count == 0:
                continue
            blk.ref_count -= 1
            if blk.ref_count == 0:
                if blk.block_hash is not None and self.hash_to_block.get(blk.block_hash) == bid:
                    del self.hash_to_block[blk.block_hash]
                blk.block_hash = None
                blk.token_ids = []
                self.free_blocks.append(bid)

    def ref(self, block_ids: Iterable[int]) -> None:
        for bid in block_ids:
            self.blocks[bid].ref_count += 1

    # ------------------------------------------------------------ prefix hash
    def hash_chain(self, token_ids: list[int]) -> list[int | None]:
        """Block-aligned hash chain for ``token_ids`` (last partial block -> None)."""
        hashes: list[int | None] = []
        parent: int | None = None
        for start in range(0, len(token_ids), self.block_size):
            chunk = token_ids[start:start + self.block_size]
            if len(chunk) < self.block_size:
                hashes.append(None)
                break
            parent = hash_block_tokens(parent, chunk)
            hashes.append(parent)
        return hashes

    def lookup_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Longest prefix (in tokens) present in the cache, plus its block ids."""
        matched_blocks: list[int] = []
        parent: int | None = None
        n = 0
        for start in range(0, len(token_ids), self.block_size):
            chunk = token_ids[start:start + self.block_size]
            if len(chunk) < self.block_size:
                break
            parent = hash_block_tokens(parent, chunk)
            bid = self.hash_to_block.get(parent)
            if bid is None:
                break
            matched_blocks.append(bid)
            n += len(chunk)
        return n, matched_blocks

    # -------------------------------------------------------------- debugging
    def snapshot(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used": self.num_used_blocks,
            "free": self.num_free_blocks,
            "usage": round(self.usage, 4),
            "peak_usage": round(self.peak_usage, 4),
            "cached_hashes": len(self.hash_to_block),
        }
