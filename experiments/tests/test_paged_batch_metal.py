"""Numerics for the batched paged decode kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.paged_batch_metal import (  # noqa: E402
    gather_rows,
    paged_batch_attention_metal,
    reference_attention,
)

Hq, Hk, D, BS = 32, 8, 128, 16


def build(batch_lengths, seed=0):
    mx.random.seed(seed)
    B = len(batch_lengths)
    maxb = (max(batch_lengths) + BS - 1) // BS
    nb = sum((n + BS - 1) // BS for n in batch_lengths)
    k_pool = mx.random.normal((nb, Hk, BS, D)).astype(mx.float16)
    v_pool = mx.random.normal((nb, Hk, BS, D)).astype(mx.float16)
    used, table = 0, []
    for n in batch_lengths:
        need = (n + BS - 1) // BS
        table.append(list(range(used, used + need)) + [0] * (maxb - need))
        used += need
    block_table = mx.array(table, dtype=mx.int32)
    lengths = mx.array(batch_lengths, dtype=mx.int32)
    q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
    return q, k_pool, v_pool, block_table, lengths


class PagedBatchKernelTest(unittest.TestCase):
    def _check(self, lens, atol=2e-2):
        q, k_pool, v_pool, table, lengths = build(lens)
        out = paged_batch_attention_metal(q, k_pool, v_pool, table, lengths, D ** -0.5)
        kk, vv = gather_rows(k_pool, v_pool, table, lengths)
        ref = reference_attention(q, kk, vv, lengths, D ** -0.5)
        mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"maxdiff {diff} for {lens}")

    def test_padded_batch(self):
        self._check([300, 17, 256, 1])

    def test_single_row(self):
        self._check([128])

    def test_long_context(self):
        self._check([2600, 1500])

    def test_multi_block_gqa(self):
        self._check([16 * 5, 16 * 3 + 4, 16 * 7])

    def test_bad_head_dim(self):
        with self.assertRaises(ValueError):
            paged_batch_attention_metal(
                mx.zeros((1, 4, 1, 100)), mx.zeros((4, 2, BS, 100)),
                mx.zeros((4, 2, BS, 100)),
                mx.array([[0]], dtype=mx.int32), mx.array([8], dtype=mx.int32), 0.1)


if __name__ == "__main__":
    unittest.main()
