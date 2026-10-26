"""Unit tests for the true paged KV pool (v0.4)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.paged_pool import PagedBlockTable, PagedKVPool  # noqa: E402

H, D, BS = 2, 8, 4


def make_pool(layers=2, blocks=32):
    return PagedKVPool(num_layers=layers, num_blocks=blocks, block_size=BS,
                       n_kv_heads=H, head_dim=D, dtype=mx.float32)


def rand_kv(n: int):
    k = mx.random.normal((1, H, n, D))
    v = mx.random.normal((1, H, n, D))
    return k, v


class PoolWriteReadTest(unittest.TestCase):
    def test_write_gather_matches_dense(self):
        pool = make_pool()
        table = PagedBlockTable(pool)
        chunks = []
        total = 0
        for n in (1, 5, 4, 7):
            k, v = rand_kv(n)
            table.append(0, k, v)
            chunks.append((k, v))
            total += n
        got_k, got_v = table.gather(0)
        want_k = mx.concatenate([k for k, _ in chunks], axis=2)
        want_v = mx.concatenate([v for _, v in chunks], axis=2)
        self.assertEqual(table.length, total)
        self.assertEqual(got_k.shape, (1, H, total, D))
        self.assertTrue(mx.allclose(got_k, want_k, atol=1e-6))
        self.assertTrue(mx.allclose(got_v, want_v, atol=1e-6))

    def test_positions_across_block_boundaries(self):
        pool = make_pool()
        table = PagedBlockTable(pool)
        k = mx.arange(H * 20 * D, dtype=mx.float32).reshape(1, H, 20, D)
        table.append(0, k, k)
        got_k, _ = table.gather(0)
        self.assertEqual(got_k.shape, (1, H, 20, D))
        self.assertTrue(mx.array_equal(got_k, k))
        # 20 tokens over BS=4 -> exactly 5 blocks
        self.assertEqual(len(table.blocks), 5)

    def test_gather_truncates_to_length(self):
        pool = make_pool()
        table = PagedBlockTable(pool)
        k = mx.ones((1, H, 5, D))
        v = mx.zeros((1, H, 5, D))
        table.append(0, k, v)
        got_k, got_v = table.gather(0)
        self.assertEqual(got_k.shape[2], 5)
        self.assertTrue(mx.allclose(got_k, k))
        self.assertFalse(bool(mx.any(got_v)))

    def test_layers_are_independent(self):
        pool = make_pool(layers=3)
        tables = [PagedBlockTable(pool) for _ in range(3)]
        for layer, t in enumerate(tables):
            k = mx.full((1, H, 3, D), float(layer + 1))
            t.append(layer, k, k)
        for layer, t in enumerate(tables):
            got, _ = t.gather(layer)
            self.assertTrue(mx.allclose(got, mx.full((1, H, 3, D), float(layer + 1))))


class SharedPrefixTest(unittest.TestCase):
    def build_shared(self):
        pool = make_pool(blocks=64)
        prefix = PagedBlockTable(pool)
        k, v = rand_kv(16)  # 4 blocks
        prefix.append(0, k, v)
        shared_k, shared_v = prefix.gather(0)
        return pool, prefix, shared_k, shared_v

    def test_ref_count_on_attach(self):
        pool, prefix, _, _ = self.build_shared()
        n_blocks = len(prefix.blocks)
        self.assertEqual(pool.ref_count[prefix.blocks[0]], 1)
        a = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=16)
        b = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=16)
        self.assertEqual(pool.ref_count[prefix.blocks[0]], 3)  # prefix + a + b
        self.assertEqual(pool.num_shared, n_blocks)
        # two sequences share exactly the same blocks
        self.assertEqual(a.blocks, b.blocks)

    def test_shared_prefix_survives_suffix_writes(self):
        pool, prefix, shared_k, shared_v = self.build_shared()
        a = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=16)
        b = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=16)
        ka, va = rand_kv(3)
        kb, vb = rand_kv(5)
        a.append(0, ka, va)
        b.append(0, kb, vb)
        # each sequence reads the same shared prefix, then its own suffix
        self.assertTrue(mx.allclose(a.gather(0)[0][..., :16, :], shared_k))
        self.assertTrue(mx.allclose(b.gather(0)[0][..., :16, :], shared_k))
        self.assertTrue(mx.allclose(a.gather(0)[0][..., 16:, :], ka))
        self.assertTrue(mx.allclose(b.gather(0)[0][..., 16:, :], kb))
        # prefix table itself is untouched
        self.assertTrue(mx.allclose(prefix.gather(0)[0], shared_k))

    def test_release_returns_blocks(self):
        pool, prefix, _, _ = self.build_shared()
        a = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=16)
        a.append(0, *rand_kv(2))
        used_after_append = pool.num_used
        a.release()
        self.assertLess(pool.num_used, used_after_append)
        self.assertEqual(pool.ref_count[prefix.blocks[0]], 1)  # only the prefix table
        prefix.release()
        self.assertEqual(pool.num_used, 0)

    def test_concurrent_sequences_share_prefix_without_copy(self):
        """Three sequences attached at once: the prefix costs one copy of blocks."""
        pool, prefix, shared_k, _ = self.build_shared()
        prefix_blocks = len(prefix.blocks)
        tables = []
        for i in range(3):
            t = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=16)
            t.append(0, *rand_kv(2 + i))  # different suffix lengths
            tables.append(t)
        # all three point at the same prefix block ids
        for t in tables:
            self.assertEqual(t.blocks[:prefix_blocks], prefix.blocks)
        self.assertEqual(pool.num_shared, prefix_blocks)
        # used blocks = prefix (once) + the three suffixes
        suffix_blocks = sum(len(t.blocks) - prefix_blocks for t in tables)
        self.assertEqual(pool.num_used, prefix_blocks + suffix_blocks)
        # each sequence still reads the shared prefix correctly
        for t in tables:
            self.assertTrue(mx.allclose(t.gather(0)[0][..., :16, :], shared_k))

    def test_copy_on_write_for_non_aligned_prefix(self):
        """Attaching a partially filled prefix block must not corrupt the sharer."""
        pool = make_pool(blocks=64)
        prefix = PagedBlockTable(pool)
        k = mx.arange(H * 10 * D, dtype=mx.float32).reshape(1, H, 10, D)
        prefix.append(0, k, k)          # 10 tokens, BS=4 -> blocks [.., partial]
        prefix_k, _ = prefix.gather(0)
        # attach with a non-block-aligned prefix length
        t = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=10)
        new_k = mx.full((1, H, 2, D), -1.0)
        t.append(0, new_k, new_k)
        # the shared prefix is unchanged
        self.assertTrue(mx.array_equal(prefix.gather(0)[0], prefix_k))
        # the writer sees prefix + its own writes
        got, _ = t.gather(0)
        self.assertEqual(got.shape[2], 12)
        self.assertTrue(mx.array_equal(got[..., :10, :], prefix_k))
        self.assertTrue(mx.allclose(got[..., 10:, :], new_k))
        # CoW allocated a private block for the writer
        self.assertNotEqual(t.blocks[-1], prefix.blocks[-1])

    def test_double_free_is_idempotent(self):
        pool = make_pool()
        t = PagedBlockTable(pool)
        t.append(0, *rand_kv(4))
        blocks = list(t.blocks)
        t.release()
        t.release()
        self.assertEqual(pool.num_used, 0)
        self.assertTrue(pool.can_allocate(len(blocks)))


class AllocationTest(unittest.TestCase):
    def test_cannot_over_allocate(self):
        pool = make_pool(blocks=2)
        t = PagedBlockTable(pool)
        t.append(0, *rand_kv(BS * 2))  # exactly 2 blocks
        self.assertEqual(pool.num_used, 2)
        self.assertFalse(pool.can_allocate(1))
        with self.assertRaises(ValueError):
            pool.allocate(1)

    def test_stats(self):
        pool = make_pool(blocks=8)
        t = PagedBlockTable(pool)
        t.append(0, *rand_kv(9))  # 3 blocks
        st = pool.stats()
        self.assertEqual(st.used_blocks, 3)
        self.assertEqual(st.block_size, BS)
        self.assertEqual(st.layers_allocated, 1)
        self.assertGreater(st.nbytes, 0)


if __name__ == "__main__":
    unittest.main()
