"""Numerics for the paged GQA-reuse attention kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.paged_gqa_metal import paged_gqa_attention_metal  # noqa: E402

D, Hq, Hk, BS = 128, 32, 8, 16


def reference(q, k, v, blocks, start, K):
    # k/v are pool layer buffers [nb, Hk, bs, D]
    idx = mx.array(blocks, dtype=mx.int32)
    kk = mx.take(k, idx, axis=0)
    vv = mx.take(v, idx, axis=0)
    kk = mx.transpose(kk, (1, 0, 2, 3)).reshape(Hk, -1, D)[None]
    vv = mx.transpose(vv, (1, 0, 2, 3)).reshape(Hk, -1, D)[None]
    outs = []
    for i in range(K):
        n = start + i + 1
        outs.append(mx.fast.scaled_dot_product_attention(
            q[:, :, i:i + 1, :], kk[:, :, :n, :], vv[:, :, :n, :], scale=D ** -0.5))
    return mx.concatenate(outs, 2)


class PagedGqaKernelTest(unittest.TestCase):
    def _check(self, blocks, rowlen, K, hg, dtype=mx.float16, atol=2e-2):
        mx.random.seed(0)
        nb = max(blocks) + 2
        k = mx.random.normal((nb, Hk, BS, D)).astype(dtype)
        v = mx.random.normal((nb, Hk, BS, D)).astype(dtype)
        q = mx.random.normal((1, Hq, K, D)).astype(dtype)
        out = paged_gqa_attention_metal(
            q, k, v, mx.array([rowlen], dtype=mx.int32),
            mx.array([rowlen - K], dtype=mx.int32),
            mx.array(blocks, dtype=mx.int32), D ** -0.5, heads_per_group=hg)
        ref = reference(q, k, v, blocks, rowlen - K, K)
        mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"K={K} hg={hg} {dtype} diff={diff}")

    def test_decode_and_verify(self):
        blocks = list(range(24))
        self._check(blocks, 64, 1, hg=2)
        self._check(blocks, 64, 1, hg=1)
        self._check(blocks, 62, 3, hg=2)

    def test_bf16(self):
        blocks = list(range(16))
        self._check(blocks, 32, 1, hg=2, dtype=mx.bfloat16)


if __name__ == "__main__":
    unittest.main()
