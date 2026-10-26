"""Long-conversation prefix caching: the snapshot cap must cover the history."""

import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AutoPrefixBudgetTest(unittest.TestCase):
    def test_auto_budget(self):
        from terrazzo.cli import _auto_prefix_budget

        cap, budget = _auto_prefix_budget()
        self.assertIn(cap, (4096, 8192))
        self.assertGreaterEqual(budget, 1 << 30)


class LongChatPrefixTest(unittest.TestCase):
    def test_second_turn_reuses_long_history(self):
        model = os.environ.get("TERRAZZO_MAC_MODEL")
        if not model or not Path(model).is_dir():
            self.skipTest("set TERRAZZO_MAC_MODEL")
        from terrazzo import Engine, EngineConfig
        from terrazzo.sequence import SamplingParams

        engine = Engine(EngineConfig(model=model, num_gpu_blocks=0,
                                     prefix_cache_max_tokens=8192,
                                     prefix_cache_bytes=2_000_000_000))
        tok = engine.runner.tokenizer
        filler = "知识库条目：KV cache 缓存注意力键值对，避免重复计算。" * 300
        history = tok.encode(filler)[:4000]

        def turn(extra, ntok=8):
            ids = list(history) + tok.encode(extra)
            params = SamplingParams(max_tokens=ntok, temperature=0.0, ignore_eos=True)
            seq = engine.add_request(prompt_token_ids=ids, params=params)
            q = engine.stream(seq.request_id)
            t0 = time.perf_counter()
            first = None
            while True:
                engine.step()
                while True:
                    try:
                        delta = q.get_nowait()
                    except Exception:
                        break
                    if getattr(delta, "token_ids", None) and first is None:
                        first = time.perf_counter()
                    if getattr(delta, "finished", False):
                        return (first or time.perf_counter()) - t0

        turn("请用三句话总结以上内容")
        ttft2 = turn("再用一句话总结")
        stored = max((len(e.tokens) for e in engine.prefix_cache.entries.values()), default=0)
        self.assertGreaterEqual(stored, 4000, "snapshot should cover the 4k history")
        self.assertLess(ttft2, 2.0, f"second-turn TTFT {ttft2:.2f}s should hit the prefix")
        engine.close()


if __name__ == "__main__":
    unittest.main()
