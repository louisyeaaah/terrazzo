"""Numerics for the multi-query (q_len=K) decode attention kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.dense_multi_metal import (  # noqa: E402
    SUPPORTED_K,
    multi_query_attention_metal,
)

Hq, Hk, D = 32, 8, 128


def reference(q, k, v, start, K):
    outs = []
    for i in range(K):
        n = start + i + 1
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, i:i + 1, :], k[:, :, :n, :], v[:, :, :n, :], scale=D ** -0.5))
    return mx.concatenate(outs, 2)


class MultiQueryKernelTest(unittest.TestCase):
    def _check(self, rowlen, K, dtype=mx.float16, atol=2e-2):
        mx.random.seed(0)
        q = mx.random.normal((1, Hq, K, D)).astype(dtype)
        k = mx.random.normal((1, Hk, rowlen, D)).astype(dtype)
        v = mx.random.normal((1, Hk, rowlen, D)).astype(dtype)
        out = multi_query_attention_metal(
            q, k, v, mx.array([rowlen], dtype=mx.int32),
            mx.array([rowlen - K], dtype=mx.int32), D ** -0.5)
        ref = reference(q, k, v, rowlen - K, K)
        mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"{dtype} rowlen={rowlen} K={K} maxdiff={diff}")
        self.assertEqual(out.dtype, dtype)

    def test_all_supported_query_lengths(self):
        for K in SUPPORTED_K:
            self._check(64, K)

    def test_bf16(self):
        for K in (1, 2, 3, 4, 8):
            self._check(128, K, dtype=mx.bfloat16)

    def test_long_context(self):
        self._check(4096, 3)

    def test_causal_mask(self):
        rowlen, K = 16, 4
        q = mx.ones((1, Hq, K, D), dtype=mx.float16)
        k = mx.ones((1, Hk, rowlen, D), dtype=mx.float16)
        v = mx.zeros((1, Hk, rowlen, D), dtype=mx.float16)
        v[0, :, rowlen - 1, :] = 1.0
        out = multi_query_attention_metal(
            q, k, v, mx.array([rowlen], dtype=mx.int32),
            mx.array([rowlen - K], dtype=mx.int32), D ** -0.5)
        mx.eval(out)
        vals = [round(float(out[0, 0, i, 0]), 3) for i in range(K)]
        self.assertEqual(vals[:K - 1], [0.0] * (K - 1))
        self.assertGreater(vals[-1], 0.0)

    def test_rejects_bad_dtype_and_head_dim(self):
        with self.assertRaises(ValueError):
            multi_query_attention_metal(
                mx.zeros((1, Hq, 2, 128), dtype=mx.float32),
                mx.zeros((1, Hk, 8, 128), dtype=mx.float32),
                mx.zeros((1, Hk, 8, 128), dtype=mx.float32),
                mx.array([8], dtype=mx.int32), mx.array([6], dtype=mx.int32), 0.1)
        with self.assertRaises(ValueError):
            multi_query_attention_metal(
                mx.zeros((1, Hq, 2, 96)), mx.zeros((1, Hk, 8, 96)),
                mx.zeros((1, Hk, 8, 96)),
                mx.array([8], dtype=mx.int32), mx.array([6], dtype=mx.int32), 0.1)


if __name__ == "__main__":
    unittest.main()
