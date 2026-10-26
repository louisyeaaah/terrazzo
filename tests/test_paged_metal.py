"""Numerical + microbenchmark tests for the Metal paged decode kernel."""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from experiments.paged_attention import paged_attention  # noqa: E402
from terrazzo.paged_metal import metal_available, paged_decode_attention_metal  # noqa: E402
from terrazzo.paged_pool import PagedBlockTable, PagedKVPool  # noqa: E402

LONG = [10, 33, 64, 129]


@unittest.skipUnless(metal_available(), "MLX Metal kernels not available")
class MetalPagedDecodeTest(unittest.TestCase):
    def build(self, n_kv=2, n_q=4, d=32, bs=4, length=10, dtype=mx.float32):
        pool = PagedKVPool(num_layers=1, num_blocks=64, block_size=bs,
                           n_kv_heads=n_kv, head_dim=d, dtype=dtype)
        table = PagedBlockTable(pool)
        k = mx.random.normal((1, n_kv, length, d)).astype(dtype)
        v = mx.random.normal((1, n_kv, length, d)).astype(dtype)
        table.append(0, k, v)
        q = mx.random.normal((1, n_q, 1, d))
        return pool, table, q

    def test_matches_gather_reference(self):
        for length in LONG:
            with self.subTest(length=length):
                mx.random.seed(length)
                pool, table, q = self.build(length=length)
                scale = 32 ** -0.5
                ref = paged_attention(q, pool, table, 0, scale, n_rep=2, mask=None)
                got = paged_decode_attention_metal(q, pool, table, 0, scale, n_rep=2)
                self.assertEqual(got.shape, ref.shape)
                self.assertTrue(mx.allclose(ref, got, atol=1e-4),
                                f"maxdiff={float(mx.max(mx.abs(ref - got)))}")

    def test_gqa_variants(self):
        for n_kv, n_q in ((2, 2), (2, 4), (2, 8), (4, 4)):
            with self.subTest(n_kv=n_kv, n_q=n_q):
                mx.random.seed(n_kv * 10 + n_q)
                pool, table, q = self.build(n_kv=n_kv, n_q=n_q, d=64, length=17)
                scale = 64 ** -0.5
                ref = paged_attention(q, pool, table, 0, scale, n_rep=n_q // n_kv, mask=None)
                got = paged_decode_attention_metal(q, pool, table, 0, scale, n_rep=n_q // n_kv)
                self.assertTrue(mx.allclose(ref, got, atol=1e-4))

    def test_block_sizes(self):
        for bs in (1, 2, 4, 16):
            with self.subTest(block_size=bs):
                mx.random.seed(bs)
                pool, table, q = self.build(bs=bs, length=21)
                scale = 32 ** -0.5
                ref = paged_attention(q, pool, table, 0, scale, n_rep=2, mask=None)
                got = paged_decode_attention_metal(q, pool, table, 0, scale, n_rep=2)
                self.assertTrue(mx.allclose(ref, got, atol=1e-4))

    def test_shared_prefix_tables(self):
        mx.random.seed(11)
        pool = PagedKVPool(num_layers=1, num_blocks=64, block_size=4,
                           n_kv_heads=2, head_dim=32, dtype=mx.float32)
        prefix = PagedBlockTable(pool)
        prefix.append(0, *[mx.random.normal((1, 2, 8, 32)) for _ in range(2)])
        a = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=8)
        a.append(0, mx.random.normal((1, 2, 3, 32)), mx.random.normal((1, 2, 3, 32)))
        q = mx.random.normal((1, 4, 1, 32))
        scale = 32 ** -0.5
        ref = paged_attention(q, pool, a, 0, scale, n_rep=2, mask=None)
        got = paged_decode_attention_metal(q, pool, a, 0, scale, n_rep=2)
        self.assertTrue(mx.allclose(ref, got, atol=1e-4))

    def test_float16_pool(self):
        mx.random.seed(12)
        pool, table, q = self.build(dtype=mx.float16, length=25)
        scale = 32 ** -0.5
        ref = paged_attention(q, pool, table, 0, scale, n_rep=2, mask=None)
        got = paged_decode_attention_metal(q, pool, table, 0, scale, n_rep=2)
        # fp16 storage: compare against the fp32 view of the same pool
        self.assertLess(float(mx.max(mx.abs(ref - got))), 5e-2)

    def test_metal_is_not_slower_than_gather(self):
        """Sanity: for the real model's shape the kernel must not be a regression."""
        mx.random.seed(13)
        H, Hq, D, BS, L = 8, 32, 128, 16, 400
        pool = PagedKVPool(num_layers=1, num_blocks=64, block_size=BS,
                           n_kv_heads=H, head_dim=D, dtype=mx.float16)
        table = PagedBlockTable(pool)
        table.append(0, mx.random.normal((1, H, L, D)).astype(mx.float16),
                     mx.random.normal((1, H, L, D)).astype(mx.float16))
        q = mx.random.normal((1, Hq, 1, D))
        scale = D ** -0.5

        def timeit(fn, n=20):
            mx.eval(fn())
            t0 = time.perf_counter()
            for _ in range(n):
                mx.eval(fn())
            return (time.perf_counter() - t0) / n * 1000

        gather_ms = timeit(lambda: paged_attention(q, pool, table, 0, scale, n_rep=4, mask=None))
        metal_ms = timeit(lambda: paged_decode_attention_metal(q, pool, table, 0, scale, n_rep=4))
        self.assertLess(metal_ms, gather_ms * 1.5,
                        f"metal {metal_ms:.3f}ms vs gather {gather_ms:.3f}ms")


if __name__ == "__main__":
    unittest.main()
