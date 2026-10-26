"""simdgroup MMA basics and the transposed-K score kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402


class MmaMatmulTest(unittest.TestCase):
    def test_8x8_matmul(self):
        src = """
            simdgroup_half8x8 a, b, c = simdgroup_half8x8(0);
            simdgroup_load(a, A, 8);
            simdgroup_load(b, B, 8);
            simdgroup_multiply_accumulate(c, a, b, c);
            simdgroup_store(c, C, 8);
        """
        kern = mx.fast.metal_kernel(name="mma_unit", input_names=["A", "B"],
                                    output_names=["C"], source=src)
        mx.random.seed(0)
        a = mx.random.normal((8, 8)).astype(mx.float16)
        b = mx.random.normal((8, 8)).astype(mx.float16)
        c = kern(inputs=[a, b], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                 output_shapes=[(8, 8)], output_dtypes=[mx.float16])[0]
        ref = a @ b
        mx.eval(c, ref)
        diff = float(mx.max(mx.abs(c.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, 0.05)


class MmaScoresTest(unittest.TestCase):
    def test_scores_match_matmul(self):
        from bench.mma_attention import mma_scores

        mx.random.seed(1)
        heads, bq, kv, d = 2, 16, 64, 128
        q = mx.random.normal((heads, bq, d)).astype(mx.float16)
        k = mx.random.normal((heads, kv, d)).astype(mx.float16)
        kt = mx.contiguous(mx.swapaxes(k, 1, 2))
        s = mma_scores(q, kt, heads)
        ref = mx.clip(mx.matmul(q, mx.swapaxes(k, 1, 2)), -65504, 65504)
        mx.eval(s, ref)
        diff = float(mx.max(mx.abs(s.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, 0.5, f"maxdiff={diff}")


if __name__ == "__main__":
    unittest.main()
