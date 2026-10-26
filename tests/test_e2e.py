"""End-to-end test with a real MLX model.

Skipped unless TERRAZZO_MAC_MODEL points at a local model directory (or a HF id).
Run it explicitly:

    TERRAZZO_MAC_MODEL=mlx-community/Qwen3-4B-4bit \
      python -m unittest tests.test_e2e -v
"""

import os
import unittest

MODEL = os.environ.get("TERRAZZO_MAC_MODEL")


@unittest.skipUnless(MODEL, "set TERRAZZO_MAC_MODEL to run the e2e test")
class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terrazzo import Engine, EngineConfig

        cls.engine = Engine(EngineConfig(model=MODEL, max_num_seqs=4))

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_generate_returns_text(self):
        from terrazzo.sequence import SamplingParams

        text = self.engine.generate("你好", SamplingParams(max_tokens=8, temperature=0.0))
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_concurrent_requests_finish(self):
        from terrazzo.sequence import SamplingParams

        params = SamplingParams(max_tokens=16, temperature=0.0)
        seqs = [self.engine.add_request(f"数到{i}", params) for i in range(3)]
        self.engine.run_until(timeout=120)
        for seq in seqs:
            self.assertGreaterEqual(len(seq.output_token_ids), 1)
            self.assertTrue(seq.status.is_finished)

    def test_prefix_cache_hits_on_repeat(self):
        from terrazzo.sequence import SamplingParams

        params = SamplingParams(max_tokens=4, temperature=0.0)
        prompt = "这是一个用于测试前缀缓存的前缀，重复多次应当命中。"
        self.engine.generate(prompt, params)
        self.engine.generate(prompt, params)
        self.assertGreater(self.engine.prefix_cache.num_hits, 0)

    def test_stats_shape(self):
        st = self.engine.stats()
        for key in ("steps", "used", "free", "usage", "decode_tokens"):
            self.assertIn(key, st)


if __name__ == "__main__":
    unittest.main()


class AutoPrefixStoreTest(unittest.TestCase):
    """Regression: stored prefix tokens and KV snapshot must have the same length."""

    def test_store_length_matches_snapshot(self):
        import os
        from pathlib import Path

        model = os.environ.get("TERRAZZO_MAC_MODEL")
        if not model or not Path(model).is_dir():
            self.skipTest("set TERRAZZO_MAC_MODEL")
        from mlx_lm import load

        from terrazzo import Engine, EngineConfig

        _, tok = load(model)
        system = "背景资料：" * 220  # ~660 tokens, longer than the 512 store floor
        ids = tok.encode(system)
        eng = Engine(EngineConfig(model=model, num_gpu_blocks=2048, block_size=16,
                                  max_num_seqs=2, prefix_cache_max_tokens=512))
        seq = eng.add_request(prompt_token_ids=ids + tok.encode("问题A"))
        seq.sampling_params.max_tokens = 4
        while eng.has_unfinished():
            eng.step()
        entries = list(eng.prefix_cache.entries.values())
        self.assertTrue(entries, "expected a stored prefix entry")
        entry = entries[0]
        self.assertEqual(entry.num_tokens, 512)
        k = entry.layers[0][0]
        self.assertEqual(k.shape[-2], entry.num_tokens)
        hit = eng.prefix_cache.lookup(ids + tok.encode("问题B"))
        self.assertIsNotNone(hit)
        self.assertEqual(hit[1], 512)
        eng.close()


class PrefixDedupeTest(unittest.TestCase):
    """Agent traffic: same system prompt + different user turns must store ONE snapshot."""

    def test_shared_system_prompt_stored_once(self):
        import os
        from pathlib import Path

        model = os.environ.get("TERRAZZO_MAC_MODEL")
        if not model or not Path(model).is_dir():
            self.skipTest("set TERRAZZO_MAC_MODEL")
        from mlx_lm import load

        from terrazzo import Engine, EngineConfig

        _, tok = load(model)
        system = "背景资料：" * 200  # ~600 tokens, over the dedupe floor
        ids = tok.encode(system)
        eng = Engine(EngineConfig(model=model, num_gpu_blocks=2048, block_size=16,
                                  max_num_seqs=4))
        def run(q):
            seq = eng.add_request(prompt_token_ids=ids + tok.encode(q))
            seq.sampling_params.max_tokens = 4
            while eng.has_unfinished():
                eng.step()
        run("问题一")
        run("问题二")
        cache = eng.prefix_cache
        self.assertEqual(cache.num_stores, 1, "second prompt should be deduped")
        self.assertEqual(len(cache.entries), 1)
        self.assertGreaterEqual(cache.num_hits, 1)
        eng.close()
