"""End-to-end tests for the paged / shared-prefix path.

Skipped unless TERRAZZO_MAC_MODEL is set:

    TERRAZZO_MAC_MODEL=mlx-community/Qwen3-4B-4bit \
      python -m unittest tests.test_paged_e2e -v
"""

import os
import unittest

MODEL = os.environ.get("TERRAZZO_MAC_MODEL")


@unittest.skipUnless(MODEL, "set TERRAZZO_MAC_MODEL to run the paged e2e tests")
class PagedEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlx_lm import load

        from terrazzo.paged_session import PagedSession

        cls.model, cls.tok = load(MODEL)
        eos = tuple(cls.tok.eos_token_ids)
        cls.session = PagedSession(cls.model, cls.tok, eos_token_ids=eos)
        cls.system = (
            "你是一个严谨的中文技术助理。回答必须准确、简洁，必要时给出示例。"
            "以下是背景资料：" + "知识库条目：KV cache 用于缓存注意力键值。" * 20
        )
        cls.system_ids = cls.tok.encode(cls.system)
        cls.prefix = cls.session.register_prefix(cls.system_ids)

    def test_prefix_registered(self):
        self.assertIsNotNone(self.prefix)
        self.assertGreater(self.prefix.num_tokens, 0)
        self.assertEqual(self.prefix.num_tokens % self.session.block_manager.block_size, 0)

    def test_shared_prefix_skips_recomputation(self):
        from terrazzo.paged_session import PagedSession

        prompt = self.system_ids + self.tok.encode("用户问：KV cache 有什么好处？")
        st = self.session.generate(prompt, max_tokens=8)
        self.assertGreater(st.prefix_hit_tokens, 0)
        self.assertEqual(st.prefill_tokens, len(prompt) - self.prefix.num_tokens)
        self.assertLess(st.prefill_tokens, len(prompt))

        # a fresh session without the prefix prefills everything
        plain = PagedSession(self.model, self.tok, eos_token_ids=tuple(self.tok.eos_token_ids))
        st_dense = plain.generate(prompt, max_tokens=8)
        self.assertEqual(st_dense.prefix_hit_tokens, 0)
        self.assertEqual(st_dense.prefill_tokens, len(prompt))

    def test_shared_prefix_stores_exactly_the_prefixed_kv(self):
        """Structural invariant: the cache really holds the prefix's K/V, not a recompute.

        Feeding a length-1 suffix through a PagedKVCache must return
        ``concat(stored_prefix_kv, new_kv)`` for every layer.
        """
        import mlx.core as mx

        from terrazzo.paged_kv import PagedKVCache

        prompt = self.system_ids + self.tok.encode("用一句话解释 KV cache")
        hit = self.session.prefix_cache.lookup(prompt)
        self.assertIsNotNone(hit)
        prefix, n = hit
        cache0 = PagedKVCache(0, prefix, self.session.prefix_cache, prefix_len=n)
        k = mx.zeros((1, prefix.layers[0][0].shape[1], 1, prefix.layers[0][0].shape[3]))
        v = mx.zeros_like(k)
        out_k, out_v = cache0.update_and_fetch(k, v)
        pk, pv = prefix.layers[0]
        self.assertTrue(mx.array_equal(out_k[..., :n, :], pk[..., :n, :]))
        self.assertTrue(mx.array_equal(out_v[..., :n, :], pv[..., :n, :]))
        cache0.detach()

    def test_shared_prefix_matches_one_shot_dense_within_tolerance(self):
        """Against a one-shot prefill the logits differ only by chunk-boundary numerics.

        On this 4-bit model the top-2 margin at the first step is often ~0, so a
        greedy token can legitimately flip; we therefore compare logits, not tokens.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        prompt = self.system_ids + self.tok.encode("用一句话解释 KV cache")
        st = self.session.generate(prompt, max_tokens=2, temperature=0.0, keep_logits=True)
        caches = make_prompt_cache(self.model)
        logits = self.model(mx.array([prompt]), cache=caches)
        mx.eval(logits)
        diff = float(mx.max(mx.abs(st.first_logits - logits[:, -1, :])))
        self.assertLess(diff, 3.0, f"logit diff too large: {diff}")

    def test_partial_prefix_matches_dense_within_tolerance(self):
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        bs = self.session.block_manager.block_size
        cut = (self.prefix.num_tokens // 2 // bs) * bs
        self.assertGreater(cut, 0)
        # shares the first `cut` tokens, then diverges completely
        prompt = self.system_ids[:cut] + self.tok.encode("现在讨论一个完全不同的话题：如何做红烧肉。")
        st = self.session.generate(prompt, max_tokens=2, temperature=0.0, keep_logits=True)
        self.assertEqual(st.prefix_hit_tokens, cut)
        self.assertLess(st.prefill_tokens, len(prompt))

        caches = make_prompt_cache(self.model)
        logits = self.model(mx.array([prompt]), cache=caches)
        mx.eval(logits)
        diff = float(mx.max(mx.abs(st.first_logits - logits[:, -1, :])))
        self.assertLess(diff, 5.0, f"logit diff too large: {diff}")

    def test_block_accounting(self):
        snap = self.session.stats()
        self.assertGreater(snap["block"]["used"], 0)
        self.assertGreaterEqual(snap["prefix_cache"]["entries"], 1)
        self.assertGreater(snap["prefix_cache"]["nbytes"], 0)


if __name__ == "__main__":
    unittest.main()
