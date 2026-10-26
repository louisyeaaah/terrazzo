"""N-gram speculative decoding must be lossless for greedy sampling."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class NgramSpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = os.environ.get("TERRAZZO_MAC_MODEL")
        if not model or not Path(model).is_dir():
            raise unittest.SkipTest("set TERRAZZO_MAC_MODEL")
        from mlx_lm import load

        cls.model, cls.tok = load(model)

    def _run(self, k, n, max_tokens=16):
        from terrazzo.ngram_spec import ngram_generate, plain_generate

        prompt = self.tok.encode("知识库条目：KV cache 缓存注意力键值对，避免重复计算。" * 3 + "请继续")
        _, base = plain_generate(self.model, self.tok, prompt, max_tokens=max_tokens)
        _, spec = ngram_generate(self.model, self.tok, prompt, max_tokens=max_tokens, k=k, n=n)
        return base, spec

    def test_greedy_tokens_identical(self):
        base, spec = self._run(k=2, n=3, max_tokens=16)
        self.assertEqual(base["token_ids"], spec["token_ids"])

    def test_zero_drafts_matches_baseline(self):
        base, spec = self._run(k=0, n=3, max_tokens=12)
        self.assertEqual(base["token_ids"], spec["token_ids"])


if __name__ == "__main__":
    unittest.main()
