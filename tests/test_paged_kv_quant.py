"""Tests for KV-quantized paged pools (8-bit / 4-bit)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.paged_pool import PagedBlockTable, PagedKVPool  # noqa: E402

H, D, BS = 8, 128, 16
LAYERS = 2
BLOCKS = 32


def make_pool(bits):
    return PagedKVPool(num_layers=LAYERS, num_blocks=BLOCKS, block_size=BS,
                       n_kv_heads=H, head_dim=D, kv_bits=bits)


def fill(pool, tokens=40):
    table = PagedBlockTable(pool)
    k = mx.random.normal((1, H, tokens, D)).astype(mx.float16)
    v = mx.random.normal((1, H, tokens, D)).astype(mx.float16)
    for layer in range(LAYERS):
        table.append(layer, k, v)
    return table, k, v


class QuantizedPoolTest(unittest.TestCase):
    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            PagedKVPool(num_layers=1, num_blocks=4, block_size=BS, n_kv_heads=H,
                        head_dim=D, kv_bits=3)

    def test_8bit_close_to_fp16(self):
        base = make_pool(None)
        t0, k, v = fill(base)
        q = make_pool(8)
        t1 = PagedBlockTable(q)
        t1.append(0, k, v)
        t1.append(1, k, v)
        gk0, _ = t0.gather(0)
        gk1, _ = t1.gather(0)
        diff = float(mx.max(mx.abs(gk0.astype(mx.float32) - gk1.astype(mx.float32))))
        self.assertLess(diff, 0.05, f"8-bit maxdiff {diff}")
        self.assertEqual(gk1.shape, gk0.shape)

    def test_memory_ratios(self):
        pools = {}
        for bits in (None, 8, 4):
            p = make_pool(bits)
            fill(p)
            pools[bits] = p.stats().nbytes
        self.assertLess(pools[8], pools[None] * 0.6)
        self.assertLess(pools[4], pools[None] * 0.4)
        self.assertLess(pools[4], pools[8])

    def test_4bit_is_lossy_but_shaped(self):
        base = make_pool(None)
        t0, k, v = fill(base)
        q = make_pool(4)
        t1 = PagedBlockTable(q)
        t1.append(0, k, v)
        t1.append(1, k, v)
        diff = float(mx.max(mx.abs(t0.gather(0)[0] - t1.gather(0)[0])))
        self.assertGreater(diff, 0.1, "4-bit quantization should visibly change values")

    def test_shared_prefix_in_quantized_pool(self):
        pool = make_pool(8)
        prefix = PagedBlockTable(pool)
        k = mx.random.normal((1, H, 2 * BS, D)).astype(mx.float16)
        prefix.append(0, k, k)
        prefix.append(1, k, k)
        a = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=2 * BS)
        b = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=2 * BS)
        for table in (a, b):
            table.append(0, mx.zeros((1, H, 3, D), dtype=mx.float16),
                         mx.zeros((1, H, 3, D), dtype=mx.float16))
        self.assertEqual(pool.num_shared, 2)
        gk, _ = a.gather(0)
        self.assertEqual(gk.shape[2], 2 * BS + 3)

    def test_clone_block_copies_quantized_data(self):
        pool = make_pool(8)
        table, k, v = fill(pool)
        src = table.blocks[0]
        dst = pool.clone_block(src)
        for layer in range(LAYERS):
            for store in (pool._kq[layer], pool._vq[layer]):
                for name in ("wq", "scales", "biases"):
                    self.assertTrue(mx.array_equal(store[name][dst], store[name][src]))

    def test_layer_kv_rejected_for_quantized_pool(self):
        pool = make_pool(8)
        fill(pool)
        with self.assertRaises(NotImplementedError):
            pool.layer_kv(0)


if __name__ == "__main__":
    unittest.main()
