"""Numerics for the GQA-aware decode kernel (each KV head read once)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.dense_gqa_metal import dense_gqa_attention_metal  # noqa: E402

Hq, Hk, D = 32, 8, 128


def reference(q, k, v, lens, L):
    out = []
    for b, n in enumerate(lens):
        out.append(mx.fast.scaled_dot_product_attention(
            q[b:b + 1], k[b:b + 1][:, :, L - n:, :], v[b:b + 1][:, :, L - n:, :],
            scale=D ** -0.5))
    return mx.concatenate(out, 0)


class GqaKernelTest(unittest.TestCase):
    def _check(self, L, lens, atol=2e-2):
        mx.random.seed(0)
        B = len(lens)
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
        k = mx.random.normal((B, Hk, L, D)).astype(mx.float16)
        v = mx.random.normal((B, Hk, L, D)).astype(mx.float16)
        out = dense_gqa_attention_metal(q, k, v, mx.array(lens, dtype=mx.int32), D ** -0.5)
        ref = reference(q, k, v, lens, L)
        mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"maxdiff {diff} for L={L} lens={lens}")

    def test_full_length(self):
        self._check(1024, [1024])

    def test_left_padded(self):
        self._check(1024, [1000])

    def test_padded_batch(self):
        self._check(512, [17, 1, 512])

    def test_long_context(self):
        self._check(4096, [4000])

    def test_head_dim_guard(self):
        with self.assertRaises(ValueError):
            dense_gqa_attention_metal(
                mx.zeros((1, 4, 1, 96)), mx.zeros((1, 2, 8, 96)),
                mx.zeros((1, 2, 8, 96)), mx.array([8], dtype=mx.int32), 0.1)


if __name__ == "__main__":
    unittest.main()
