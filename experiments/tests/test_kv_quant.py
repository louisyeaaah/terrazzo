"""Tests for KV-cache quantization."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.kv_quant import cache_bytes, quantized_prompt_cache  # noqa: E402

MODEL = os.environ.get("TERRAZZO_MAC_MODEL", "")


class CacheSizeTest(unittest.TestCase):
    def test_quantized_cache_is_smaller(self):
        from mlx_lm.models.cache import KVCache, QuantizedKVCache

        L, H, D = 64, 8, 128
        k = mx.random.normal((1, H, L, D)).astype(mx.float16)
        v = mx.random.normal((1, H, L, D)).astype(mx.float16)
        sizes = {}
        c = KVCache(); c.update_and_fetch(k, v); sizes["fp16"] = cache_bytes([c])
        for bits in (8, 4):
            q = QuantizedKVCache(group_size=64, bits=bits)
            q.update_and_fetch(k, v)
            sizes[bits] = cache_bytes([q])
            self.assertLess(sizes[bits], sizes["fp16"])
        self.assertLess(sizes[8], sizes["fp16"])
        self.assertLess(sizes[4], sizes[8])
        # 4-bit should be roughly a quarter of fp16 (allow unpacking overhead)
        self.assertLess(sizes[4], sizes["fp16"] * 0.4)


@unittest.skipUnless(MODEL and Path(MODEL).is_dir(), "set TERRAZZO_MAC_MODEL")
class ModelKvQuantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlx_lm import load

        cls.model, cls.tok = load(MODEL)
        cls.prompt = cls.tok.encode("请用五句话解释 KV cache 的作用。")

    def _first_token(self, bits: int | None) -> int:
        if bits:
            caches, replaced = quantized_prompt_cache(self.model, bits)
            self.assertGreater(replaced, 0)
        else:
            from mlx_lm.models.cache import make_prompt_cache

            caches = make_prompt_cache(self.model)
        logits = self.model(mx.array([self.prompt]), cache=caches)
        mx.eval(logits)
        return int(mx.argmax(logits[:, -1, :], axis=-1).item())

    def test_8bit_matches_fp16_first_token(self):
        self.assertEqual(self._first_token(8), self._first_token(None))

    def test_4bit_is_lossy_for_this_model(self):
        # documented finding: 4-bit KV changes the distribution enough to flip the
        # first token for Qwen3-4B (mlx-lm's own kv_bits=4 path behaves the same)
        fp16 = self._first_token(None)
        q4 = self._first_token(4)
        self.assertNotEqual(fp16, q4)


if __name__ == "__main__":
    unittest.main()
