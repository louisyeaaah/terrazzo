"""Fused decode forward: weight concat must be exact and greedy tokens identical."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402


class ConcatWeightsTest(unittest.TestCase):
    def test_concat_matches_individual(self):
        from experiments.decode_fused import _cat_weights

        mx.random.seed(0)
        ws = []
        for n, k in ((128, 256), (64, 256), (64, 256)):
            w = (mx.random.normal((n, k)) * 0.02).astype(mx.float16)
            ws.append(mx.quantize(w, group_size=64, bits=4))
        cat = _cat_weights([type("M", (), {"weight": wq, "scales": sc, "biases": bi})()
                            for wq, sc, bi in ws])
        x = mx.random.normal((1, 256)).astype(mx.float16)
        fused = mx.quantized_matmul(x, *cat, transpose=True, group_size=64, bits=4)
        parts = [mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=4)
                 for w in ws]
        ref = mx.concatenate(parts, axis=-1)
        mx.eval(fused, ref)
        diff = float(mx.max(mx.abs(fused.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, 1e-4)


class FusedDecodeModelTest(unittest.TestCase):
    def test_tokens_match_baseline(self):
        model_path = os.environ.get("TERRAZZO_MAC_MODEL")
        if not model_path or not Path(model_path).is_dir():
            self.skipTest("set TERRAZZO_MAC_MODEL")
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        from experiments.decode_fused import build_fused, fused_decode_step

        model, tok = load(model_path)
        fused = build_fused(model)
        ids = tok.encode("知识库条目：KV cache 缓存注意力键值对。" * 8)
        caches = make_prompt_cache(model)
        mx.eval(model(mx.array([ids[:-1]]), cache=caches))
        token = ids[-1]
        base_tokens, fused_tokens = [], []
        for _ in range(4):
            logits = model(mx.array([[token]]), cache=caches)
            mx.eval(logits)
            token = int(mx.argmax(logits[0, -1]))
            base_tokens.append(token)
        caches2 = make_prompt_cache(model)
        mx.eval(model(mx.array([ids[:-1]]), cache=caches2))
        token = ids[-1]
        for _ in range(4):
            logits = fused_decode_step(model, fused, token, caches2)
            mx.eval(logits)
            token = int(mx.argmax(logits[0, -1]))
            fused_tokens.append(token)
        self.assertEqual(base_tokens, fused_tokens)


if __name__ == "__main__":
    unittest.main()
