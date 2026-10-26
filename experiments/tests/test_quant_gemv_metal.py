"""Numerics for the weight-only 4-bit GEMV kernel."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.quant_gemv_metal import gemv_quantized  # noqa: E402


class QuantGemvTest(unittest.TestCase):
    def _check(self, n: int, k: int, dtype=mx.float16, atol=2e-2):
        mx.random.seed(0)
        w = (mx.random.normal((n, k)) * 0.02).astype(dtype)
        wq = mx.quantize(w, group_size=64, bits=4)
        x = mx.random.normal((1, k)).astype(dtype)
        y = gemv_quantized(x, *wq)
        ref = mx.quantized_matmul(x, *wq, transpose=True, group_size=64, bits=4)
        mx.eval(y, ref)
        diff = float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"N={n} K={k} {dtype} maxdiff={diff}")
        self.assertEqual(y.shape, (1, n))

    def test_model_shapes(self):
        for n, k in ((256, 2560), (6144, 2560), (2560, 4096), (9728, 2560),
                     (2560, 9728)):
            self._check(n, k)

    def test_lm_head_shape(self):
        self._check(151936 // 64, 2560)

    def test_bf16(self):
        self._check(1024, 2560, dtype=mx.bfloat16)

    def test_rejects_multi_token(self):
        w = mx.zeros((64, 2560), dtype=mx.float16)
        wq = mx.quantize(w, group_size=64, bits=4)
        with self.assertRaises(ValueError):
            gemv_quantized(mx.zeros((2, 2560), dtype=mx.float16), *wq)


if __name__ == "__main__":
    unittest.main()
