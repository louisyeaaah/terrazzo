"""Tests for the batched dense Metal decode attention kernel."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.dense_metal import (  # noqa: E402
    dense_decode_attention_metal,
    reference_attention,
)

MODEL = os.environ.get("TERRAZZO_MAC_MODEL", "")


class KernelNumericsTest(unittest.TestCase):
    def _check(self, B, Hq, Hk, L, D, lengths, atol=5e-3):
        mx.random.seed(0)
        q = mx.random.normal((B, Hq, 1, D)).astype(mx.float16)
        k = mx.random.normal((B, Hk, L, D)).astype(mx.float16)
        v = mx.random.normal((B, Hk, L, D)).astype(mx.float16)
        lens = mx.array(lengths, dtype=mx.int32)
        out = dense_decode_attention_metal(q, k, v, lens, D**-0.5)
        ref = reference_attention(q, k, v, lens, D**-0.5)
        mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"maxdiff={diff} for {B,Hq,Hk,L,lengths}")

    def test_single_row(self):
        self._check(1, 8, 8, 32, 128, [32])

    def test_left_padding(self):
        self._check(1, 8, 8, 32, 128, [17])

    def test_single_valid_token(self):
        self._check(1, 8, 8, 32, 128, [1])

    def test_gqa_and_batch_padding(self):
        self._check(4, 32, 8, 300, 128, [300, 17, 256, 1])

    def test_long_context(self):
        self._check(2, 32, 8, 2600, 128, [2600, 1500])

    def test_bad_head_dim(self):
        with self.assertRaises(ValueError):
            dense_decode_attention_metal(
                mx.zeros((1, 4, 1, 100)), mx.zeros((1, 4, 8, 100)),
                mx.zeros((1, 4, 8, 100)), mx.array([8], dtype=mx.int32), 0.1)


@unittest.skipUnless(MODEL and Path(MODEL).is_dir(), "set TERRAZZO_MAC_MODEL")
class PatchIntegrationTest(unittest.TestCase):
    def test_patched_attention_matches_original_tokens(self):
        """Greedy decode with the patched kernel matches the unpatched path."""
        from mlx_lm import load

        from terrazzo import Engine, EngineConfig
        from terrazzo.batch_attn_patch import patched_qwen3_batch_attention

        model, tok = load(MODEL)
        sys_ids = tok.encode("背景资料：" * 200)[:1200]

        def run(patched: bool):
            eng = Engine(EngineConfig(model=MODEL, num_gpu_blocks=2048, block_size=16,
                                      max_num_seqs=8))
            eng.register_prefix(sys_ids)
            seqs = [eng.add_request(prompt_token_ids=sys_ids + tok.encode(f"问题{i}"))
                    for i in range(3)]
            for s in seqs:
                s.sampling_params.max_tokens = 24
            ctx = patched_qwen3_batch_attention() if patched else None
            if ctx:
                ctx.__enter__()
            try:
                for _ in range(200):
                    if not eng.has_unfinished():
                        break
                    eng.step()
            finally:
                if ctx:
                    ctx.__exit__(None, None, None)
            out = [tuple(s.output_token_ids) for s in seqs]
            eng.close()
            return out

        base = run(False)
        patched = run(True)
        self.assertEqual(base, patched)


if __name__ == "__main__":
    unittest.main()
