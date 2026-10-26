"""Numerics for the GQA-reuse multi-query attention kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.dense_gqa_multi_metal import gqa_multi_attention_metal  # noqa: E402

D, Hq, Hk = 128, 32, 8


def reference(q, k, v, start, K):
    outs = []
    for i in range(K):
        n = start + i + 1
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, i:i + 1, :], k[:, :, :n, :], v[:, :, :n, :], scale=D ** -0.5))
    return mx.concatenate(outs, 2)


class GqaMultiKernelTest(unittest.TestCase):
    def _check(self, rowlen, K, hg, dtype=mx.float16, atol=2e-2):
        mx.random.seed(0)
        q = mx.random.normal((1, Hq, K, D)).astype(dtype)
        k = mx.random.normal((1, Hk, rowlen, D)).astype(dtype)
        v = mx.random.normal((1, Hk, rowlen, D)).astype(dtype)
        out = gqa_multi_attention_metal(
            q, k, v, mx.array([rowlen], dtype=mx.int32),
            mx.array([rowlen - K], dtype=mx.int32), D ** -0.5, heads_per_group=hg)
        ref = reference(q, k, v, rowlen - K, K)
        mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"hg={hg} K={K} {dtype} maxdiff={diff}")
        self.assertEqual(out.dtype, dtype)

    def test_heads_per_group(self):
        for hg in (1, 2, 4):
            for K in (1, 2, 3):
                self._check(128, K, hg)

    def test_bf16(self):
        for hg in (1, 2, 4):
            self._check(256, 2, hg, dtype=mx.bfloat16)

    def test_long_context(self):
        self._check(4096, 3, hg=1)

    def test_rejects_bad_grouping(self):
        q = mx.zeros((1, Hq, 3, D), dtype=mx.float16)
        k = mx.zeros((1, Hk, 8, D), dtype=mx.float16)
        with self.assertRaises(ValueError):
            gqa_multi_attention_metal(q, k, k, mx.array([8], dtype=mx.int32),
                                      mx.array([5], dtype=mx.int32), 0.1,
                                      heads_per_group=3)
        with self.assertRaises(ValueError):
            gqa_multi_attention_metal(q, k, k, mx.array([8], dtype=mx.int32),
                                      mx.array([5], dtype=mx.int32), 0.1,
                                      heads_per_group=8)


if __name__ == "__main__":
    unittest.main()
