"""Unit tests for paged KV / shared prefixes (no model required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.blocks import BlockManager, BlockManagerConfig  # noqa: E402
from terrazzo.paged_kv import PagedKVCache, PagedPrefixCache  # noqa: E402


def fake_layers(n_tokens: int, layers: int = 2, heads: int = 2, dim: int = 4):
    return [
        (mx.ones((1, heads, n_tokens, dim)), mx.ones((1, heads, n_tokens, dim)))
        for _ in range(layers)
    ]


class PagedPrefixCacheTest(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(BlockManagerConfig(num_blocks=128, block_size=16))
        self.pc = PagedPrefixCache(self.bm, max_entries=4, max_bytes=1 << 30)

    def test_store_allocates_blocks(self):
        p = self.pc.store(list(range(64)), fake_layers(64))
        self.assertIsNotNone(p)
        self.assertEqual(p.num_tokens, 64)
        self.assertEqual(len(p.block_ids), 4)
        self.assertEqual(self.bm.num_used_blocks, 4)
        self.assertEqual(self.pc.stats()["entries"], 1)

    def test_store_truncates_to_block_boundary(self):
        p = self.pc.store(list(range(70)), fake_layers(70))
        self.assertEqual(p.num_tokens, 64)  # 70 -> 64 (4 blocks)
        self.assertEqual(len(p.block_ids), 4)

    def test_lookup_finds_longest_prefix(self):
        a = self.pc.store(list(range(64)), fake_layers(64))
        b_tokens = list(range(80))
        b = self.pc.store(b_tokens, fake_layers(80))
        hit = self.pc.lookup(b_tokens + [1, 2, 3])
        self.assertIsNotNone(hit)
        prefix, n = hit
        self.assertEqual(n, 80)  # longest wins
        self.assertIs(prefix, b)
        # a shorter query still hits 64 tokens (the entry may be the longer one:
        # its arrays are sliced down to the matched length)
        prefix2, n2 = self.pc.lookup(list(range(64)) + [999])
        self.assertEqual(n2, 64)
        self.assertIn(prefix2, (a, b))

    def test_partial_prefix_when_prompt_diverges(self):
        bs = self.bm.block_size
        full = list(range(80))
        self.pc.store(full, fake_layers(80))
        # diverges after 48 tokens (3 blocks) into a different continuation
        prompt = full[:48] + [999] * 32
        hit = self.pc.lookup(prompt)
        self.assertIsNotNone(hit)
        entry, n = hit
        self.assertEqual(n, 48)
        self.assertEqual(n % bs, 0)
        # the shared entry can still serve this shorter hit
        self.assertGreaterEqual(entry.num_tokens, n)

    def test_partial_hit_at_every_block_boundary(self):
        bs = self.bm.block_size
        full = list(range(96))
        self.pc.store(full, fake_layers(96))
        for k in range(1, 96 // bs + 1):
            prompt = full[: k * bs] + [12345]
            hit = self.pc.lookup(prompt)
            self.assertIsNotNone(hit, f"no hit at {k} blocks")
            self.assertEqual(hit[1], k * bs)

    def test_no_partial_hit_inside_a_block(self):
        full = list(range(64))
        self.pc.store(full, fake_layers(64))
        # changing a token inside the first block breaks the chain there
        prompt = full[:8] + [7] + full[9:]
        self.assertIsNone(self.pc.lookup(prompt))

    def test_lookup_misses(self):
        self.pc.store(list(range(64)), fake_layers(64))
        self.assertIsNone(self.pc.lookup([999] * 32))
        self.assertEqual(self.pc.stats()["misses"], 1)

    def test_refcount_keeps_blocks_alive(self):
        p = self.pc.store(list(range(64)), fake_layers(64))
        used = self.bm.num_used_blocks
        self.pc.acquire(p)
        self.pc.release(p)
        self.assertEqual(self.bm.num_used_blocks, used)  # cache still holds its ref
        self.assertEqual(p.ref_count, 1)

    def test_eviction_respects_active(self):
        self.pc.max_entries = 2
        p1 = self.pc.store(list(range(64)), fake_layers(64))
        p2 = self.pc.store([100] * 64, fake_layers(64))
        self.pc.acquire(p1)  # p1 is active, must not be evicted
        self.pc.store([200] * 64, fake_layers(64))
        self.assertIn(tuple(range(64)), self.pc.entries)
        self.assertNotIn(tuple([100] * 64), self.pc.entries)
        self.pc.release(p1)

    def test_clear_frees_blocks(self):
        self.pc.store(list(range(64)), fake_layers(64))
        self.pc.clear()
        self.assertEqual(self.bm.num_used_blocks, 0)
        self.assertEqual(len(self.pc.entries), 0)


class PagedKVCacheTest(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(BlockManagerConfig(num_blocks=128, block_size=16))
        self.pc = PagedPrefixCache(self.bm)
        self.prefix = self.pc.store(list(range(64)), fake_layers(64))

    def test_offset_and_concat(self):
        c = PagedKVCache(0, self.prefix, self.pc)
        self.assertEqual(c.offset, 64)
        k = mx.zeros((1, 2, 3, 4))
        v = mx.zeros((1, 2, 3, 4))
        out_k, out_v = c.update_and_fetch(k, v)
        self.assertEqual(c.offset, 67)
        self.assertEqual(out_k.shape, (1, 2, 67, 4))
        self.assertEqual(out_v.shape, (1, 2, 67, 4))

    def test_trim_only_touches_suffix(self):
        c = PagedKVCache(0, self.prefix, self.pc)
        c.update_and_fetch(mx.zeros((1, 2, 5, 4)), mx.zeros((1, 2, 5, 4)))
        self.assertEqual(c.trim(3), 3)
        self.assertEqual(c.offset, 66)  # 64 shared + 2 suffix
        self.assertGreaterEqual(c.trim(10), 2)  # cannot trim into the shared prefix
        self.assertEqual(c.offset, 64)

    def test_mask_reflects_offset(self):
        c = PagedKVCache(0, self.prefix, self.pc)
        self.assertEqual(c.make_mask(1), None)  # single-token decode needs no mask
        c.update_and_fetch(mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))
        self.assertEqual(c.make_mask(4), "causal")

    def test_detach_releases_active(self):
        c = PagedKVCache(0, self.prefix, self.pc)
        self.assertEqual(self.prefix.active, 1)
        c.detach()
        self.assertEqual(self.prefix.active, 0)
        self.assertEqual(self.bm.num_used_blocks, 4)  # cache ref remains

    def test_shared_prefix_is_shared_not_copied(self):
        c1 = PagedKVCache(0, self.prefix, self.pc)
        c2 = PagedKVCache(0, self.prefix, self.pc)
        k1, _ = c1.update_and_fetch(mx.zeros((1, 2, 4, 4)), mx.zeros((1, 2, 4, 4)))
        k2, _ = c2.update_and_fetch(mx.zeros((1, 2, 2, 4)), mx.zeros((1, 2, 2, 4)))
        pk, _pv = self.prefix.layers[0]
        # both sequences read the same frozen prefix block
        self.assertTrue(mx.array_equal(k1[..., :64, :], pk))
        self.assertTrue(mx.array_equal(k2[..., :64, :], pk))
        # the sequence only owns its own growing suffix
        self.assertGreater(c1.nbytes, 0)
        self.assertEqual(self.prefix.active, 2)


if __name__ == "__main__":
    unittest.main()
