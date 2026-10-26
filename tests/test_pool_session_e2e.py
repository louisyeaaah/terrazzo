"""End-to-end tests for the paged-pool session (v0.4).

Skipped unless TERRAZZO_MAC_MODEL is set:

    TERRAZZO_MAC_MODEL=mlx-community/Qwen3-4B-4bit \
      python -m unittest tests.test_pool_session_e2e -v
"""

import os
import unittest

MODEL = os.environ.get("TERRAZZO_MAC_MODEL")


@unittest.skipUnless(MODEL, "set TERRAZZO_MAC_MODEL to run the pool e2e tests")
class PoolSessionEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlx_lm import load

        from terrazzo.pool_session import PoolSession

        cls.model, cls.tok = load(MODEL)
        eos = tuple(cls.tok.eos_token_ids)
        cls.session = PoolSession(cls.model, cls.tok, num_blocks=256, eos_token_ids=eos)
        cls.system = (
            "你是一个严谨的中文技术助理。回答必须准确、简洁。背景资料："
            + "知识库条目：KV cache 用于缓存注意力键值。" * 15
        )
        cls.sys_ids = cls.tok.encode(cls.system)
        cls.prefix = cls.session.register_prefix(cls.sys_ids)

    def test_prefix_registered_in_pool(self):
        self.assertIsNotNone(self.prefix)
        st = self.session.stats()["pool"]
        self.assertGreaterEqual(st["used_blocks"], len(self.prefix.blocks))
        self.assertEqual(st["shared_blocks"], len(self.prefix.blocks))

    def test_request_reuses_blocks_without_copy(self):
        prompt = self.sys_ids + self.tok.encode("用户问：KV cache 有什么好处？")
        before = self.session.stats()["pool"]
        st = self.session.generate(prompt, max_tokens=4)
        after = self.session.stats()["pool"]
        self.assertEqual(st.prefix_hit_tokens, self.prefix.num_tokens)
        # suffix blocks are released when the request finishes: the pool must not
        # grow by a copy of the prefix
        self.assertLessEqual(after["used_blocks"], before["used_blocks"] + 4)
        self.assertEqual(after["shared_blocks"], len(self.prefix.blocks))

    def test_three_requests_do_not_copy_prefix(self):
        prompt = self.sys_ids + self.tok.encode("再来一个问题：分页缓存是什么？")
        for _ in range(3):
            self.session.generate(prompt, max_tokens=4)
        pool = self.session.stats()["pool"]
        # the prefix exists once (its blocks), not once per request
        self.assertLess(pool["used_blocks"], len(self.prefix.blocks) * 2)
        self.assertEqual(pool["shared_blocks"], len(self.prefix.blocks))

    def test_logits_match_dense_session(self):
        import mlx.core as mx

        from terrazzo.paged_session import PagedSession

        prompt = self.sys_ids + self.tok.encode("用一句话解释 KV cache")
        st = self.session.generate(prompt, max_tokens=4, keep_logits=True)
        dense = PagedSession(self.model, self.tok, eos_token_ids=tuple(self.tok.eos_token_ids))
        ref = dense.generate(prompt, max_tokens=4, keep_logits=True)
        diff = float(mx.max(mx.abs(st.first_logits - ref.first_logits)))
        self.assertLess(diff, 5.0, f"logit diff too large: {diff}")

    def test_metal_patch_matches_gather_path(self):
        """The Metal decode kernel must produce the same greedy stream as the gather path."""
        from terrazzo.pool_attn_patch import patched_qwen3_attention
        from terrazzo.pool_session import PoolSession

        prompt = self.sys_ids + self.tok.encode("用一句话解释 KV cache")
        base = PoolSession(self.model, self.tok, num_blocks=256,
                           eos_token_ids=tuple(self.tok.eos_token_ids))
        base.register_prefix(self.sys_ids)
        st_gather = base.generate(prompt, max_tokens=16)

        patched = PoolSession(self.model, self.tok, num_blocks=256,
                              eos_token_ids=tuple(self.tok.eos_token_ids))
        patched.register_prefix(self.sys_ids)
        with patched_qwen3_attention():
            st_metal = patched.generate(prompt, max_tokens=16)

        mismatches = sum(1 for a, b in zip(st_gather.tokens, st_metal.tokens) if a != b)
        # near-tie greedy flips are documented; allow at most one
        self.assertLessEqual(mismatches, 1,
                             f"too many token mismatches: {st_gather.tokens} vs {st_metal.tokens}")

    def test_trim_releases_blocks(self):
        from terrazzo.paged_pool import PagedBlockTable

        pool = self.session.pool
        table = PagedBlockTable(pool)
        cache_like_k = __import__("mlx.core", fromlist=["core"]).zeros((1, 8, 40, 128))
        table.append(0, cache_like_k, cache_like_k)
        used = pool.num_used
        self.assertEqual(table.trim(32), 32)
        self.assertLess(pool.num_used, used)


if __name__ == "__main__":
    unittest.main()
