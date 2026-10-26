"""End-to-end tests for shared-prefix *batching* (v0.3).

Skipped unless TERRAZZO_MAC_MODEL is set:

    TERRAZZO_MAC_MODEL=mlx-community/Qwen3-4B-4bit \
      python -m unittest tests.test_engine_prefix_e2e -v
"""

import os
import unittest

MODEL = os.environ.get("TERRAZZO_MAC_MODEL")


@unittest.skipUnless(MODEL, "set TERRAZZO_MAC_MODEL to run the engine prefix tests")
class EnginePrefixEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terrazzo import Engine, EngineConfig

        cls.engine = Engine(EngineConfig(model=MODEL, max_num_seqs=8))
        cls.tok = cls.engine.runner.tokenizer
        cls.system = (
            "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
            "以下是背景资料：" + "知识库条目：KV cache 用于缓存注意力键值。" * 15
        )
        cls.system_ids = cls.tok.encode(cls.system)
        cls.prefix = cls.engine.register_prefix(cls.system_ids)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_prefix_registered(self):
        self.assertIsNotNone(self.prefix)
        self.assertGreater(self.prefix.num_tokens, 0)
        bs = self.engine.block_manager.block_size
        self.assertEqual(self.prefix.num_tokens % bs, 0)

    def test_request_skips_prefix_prefill(self):
        from terrazzo.sequence import SamplingParams

        prompt = self.system_ids + self.tok.encode("用户问：KV cache 有什么好处？")
        before = self.engine.stats()["prefix_hit_tokens"]
        self.engine.generate(None, SamplingParams(max_tokens=8, temperature=0.0),
                             prompt_token_ids=prompt)
        after = self.engine.stats()["prefix_hit_tokens"]
        self.assertGreater(after - before, 0)

    def test_concurrent_requests_with_shared_prefix(self):
        from terrazzo.sequence import SamplingParams

        params = SamplingParams(max_tokens=16, temperature=0.0)
        prompt = self.system_ids + self.tok.encode("请用一句话解释分页缓存")
        seqs = [self.engine.add_request(prompt_token_ids=prompt, params=params) for _ in range(3)]
        self.engine.run_until(timeout=120)
        outputs = [list(s.output_token_ids) for s in seqs]
        for out, seq in zip(outputs, seqs):
            self.assertGreaterEqual(len(out), 1)
            self.assertTrue(seq.status.is_finished)
        # greedy decoding on identical prompts must agree across the batched requests
        self.assertTrue(all(out == outputs[0] for out in outputs))

    def test_prefill_accounting_reflects_cache_hit(self):
        from terrazzo.sequence import SamplingParams

        prompt = self.system_ids + self.tok.encode("另一条问题：KV cache 什么时候失效？")
        st_before = self.engine.stats()["prefill_tokens"]
        self.engine.generate(None, SamplingParams(max_tokens=4, temperature=0.0),
                             prompt_token_ids=prompt)
        st_after = self.engine.stats()["prefill_tokens"]
        self.assertLess(st_after - st_before, len(prompt))


if __name__ == "__main__":
    unittest.main()
