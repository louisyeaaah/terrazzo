"""Unit tests for paged attention (no model required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.paged_attention import (  # noqa: E402
    attention_with_prefix,
    dense_attention,
    paged_attention,
    repeat_kv,
)
from terrazzo.paged_pool import PagedBlockTable, PagedKVPool  # noqa: E402

H, Hq, D, BS = 2, 4, 8, 4  # n_rep = 2


def make_pool(blocks=32):
    return PagedKVPool(num_layers=1, num_blocks=blocks, block_size=BS,
                       n_kv_heads=H, head_dim=D, dtype=mx.float32)


class RepeatKVTest(unittest.TestCase):
    def test_repeat(self):
        x = mx.arange(2 * 2 * 3 * 4, dtype=mx.float32).reshape(2, 2, 3, 4)
        y = repeat_kv(x, 2)
        self.assertEqual(y.shape, (2, 4, 3, 4))
        self.assertTrue(mx.array_equal(y[:, 0], x[:, 0]))
        self.assertTrue(mx.array_equal(y[:, 1], x[:, 0]))
        self.assertTrue(mx.array_equal(y[:, 2], x[:, 1]))


class PagedAttentionTest(unittest.TestCase):
    def test_matches_dense_prefill(self):
        mx.random.seed(1)
        pool = make_pool()
        table = PagedBlockTable(pool)
        L = 12
        k = mx.random.normal((1, H, L, D))
        v = mx.random.normal((1, H, L, D))
        table.append(0, k, v)
        q = mx.random.normal((1, Hq, L, D))
        scale = D ** -0.5
        ref = dense_attention(q, k, v, scale, n_rep=2)
        got = paged_attention(q, pool, table, 0, scale, n_rep=2)
        self.assertTrue(mx.allclose(ref, got, atol=1e-5))

    def test_matches_dense_with_shared_prefix(self):
        mx.random.seed(2)
        pool = make_pool()
        prefix = PagedBlockTable(pool)
        kp = mx.random.normal((1, H, 8, D))
        vp = mx.random.normal((1, H, 8, D))
        prefix.append(0, kp, vp)
        # two sequences attach to the same prefix, different suffixes
        a = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=8)
        b = PagedBlockTable(pool, prefix_blocks=prefix.blocks, prefix_len=8)
        ka = mx.random.normal((1, H, 3, D)); va = mx.random.normal((1, H, 3, D))
        kb = mx.random.normal((1, H, 5, D)); vb = mx.random.normal((1, H, 5, D))
        a.append(0, ka, va); b.append(0, kb, vb)
        scale = D ** -0.5
        for table, ks, vs in ((a, ka, va), (b, kb, vb)):
            q = mx.random.normal((1, Hq, table.length, D))
            got = paged_attention(q, pool, table, 0, scale, n_rep=2)
            k_dense = mx.concatenate([kp, ks], axis=2)
            v_dense = mx.concatenate([vp, vs], axis=2)
            ref = dense_attention(q, k_dense, v_dense, scale, n_rep=2)
            self.assertTrue(mx.allclose(ref, got, atol=1e-5))

    def test_attention_with_prefix_helper(self):
        mx.random.seed(3)
        pool = make_pool()
        prefix = PagedBlockTable(pool)
        kp = mx.random.normal((1, H, 8, D)); vp = mx.random.normal((1, H, 8, D))
        prefix.append(0, kp, vp)
        ks = mx.random.normal((1, H, 2, D)); vs = mx.random.normal((1, H, 2, D))
        q = mx.random.normal((1, Hq, 10, D))
        got = attention_with_prefix(q, pool, prefix.blocks, 8, ks, vs, 0, D ** -0.5, n_rep=2)
        ref = dense_attention(q, mx.concatenate([kp, ks], axis=2),
                              mx.concatenate([vp, vs], axis=2), D ** -0.5, n_rep=2)
        self.assertTrue(mx.allclose(got, ref, atol=1e-5))

    def test_decode_step_matches_dense(self):
        mx.random.seed(4)
        pool = make_pool()
        table = PagedBlockTable(pool)
        k5 = mx.random.normal((1, H, 5, D)); v5 = mx.random.normal((1, H, 5, D))
        table.append(0, k5, v5)
        # the new decode token is written into the cache before attention runs
        k6 = mx.random.normal((1, H, 1, D)); v6 = mx.random.normal((1, H, 1, D))
        table.append(0, k6, v6)
        q = mx.random.normal((1, Hq, 1, D))
        k = mx.concatenate([k5, k6], axis=2)
        v = mx.concatenate([v5, v6], axis=2)
        scale = D ** -0.5
        ref = dense_attention(q, k, v, scale, n_rep=2, mask=None)
        got = paged_attention(q, pool, table, 0, scale, n_rep=2, mask=None)
        self.assertTrue(mx.allclose(ref, got, atol=1e-5))

    def test_length_mismatch_raises(self):
        pool = make_pool()
        table = PagedBlockTable(pool)
        table.append(0, mx.zeros((1, H, 4, D)), mx.zeros((1, H, 4, D)))
        with self.assertRaises(ValueError):
            paged_attention(mx.zeros((1, Hq, 8, D)), pool, table, 0, 0.5, n_rep=2)


if __name__ == "__main__":
    unittest.main()
