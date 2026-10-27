"""Pure-Python tests for the cache/scheduler cost model (no MLX required)."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from terrazzo.blocks import BlockManager, BlockManagerConfig
from terrazzo.cost import CostModel, kv_bytes_per_token
from terrazzo.scheduler import Scheduler, SchedulerConfig
from terrazzo.sequence import Sequence, SamplingParams


def _write_model_config(root: Path) -> None:
    (root / "config.json").write_text(json.dumps({
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 64,
        "head_dim": 8,
        "vocab_size": 128,
    }))


class KvBytesTest(unittest.TestCase):
    def test_kv_bytes_from_config(self):
        with tempfile.TemporaryDirectory() as d:
            _write_model_config(Path(d))
            # 4 layers * 2 (K,V) * 2 kv_heads * 8 head_dim * 2 bytes
            self.assertEqual(kv_bytes_per_token(d), 256)

    def test_missing_config_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(kv_bytes_per_token(d))


class CostModelTest(unittest.TestCase):
    def test_large_cached_prefix_is_reused(self):
        m = CostModel(prefill_tok_s=450.0, copy_tok_s=8000.0)
        c = m.estimate(prompt_tokens=2544, cached_tokens=2544)
        self.assertTrue(c.reuse)
        self.assertEqual(c.reason, "copy-cheaper")
        self.assertEqual(c.uncached_tokens, 0)

    def test_auto_recomputes_when_copy_is_expensive(self):
        # a fast prefill makes copying the cached span the expensive option
        m = CostModel(prefill_tok_s=200_000.0, copy_tok_s=1000.0,
                      copy_overhead_ms=5.0)
        c = m.estimate(prompt_tokens=2000, cached_tokens=16)
        self.assertFalse(c.reuse)
        self.assertEqual(c.reason, "recompute-cheaper")

    def test_policy_overrides(self):
        m = CostModel(prefill_tok_s=200_000.0, copy_tok_s=1000.0,
                      copy_overhead_ms=5.0)
        self.assertTrue(m.estimate(2000, 16, policy="always").reuse)
        self.assertFalse(m.estimate(2000, 2000, policy="never").reuse)
        self.assertEqual(m.estimate(2000, 0).reason, "no-cache-hit")

    def test_calibration_moves_rates(self):
        m = CostModel(prefill_tok_s=100.0)
        m.observe_step(prefill_tokens=1000, seconds=1.0)
        self.assertGreater(m.prefill_tok_s, 100.0)
        self.assertEqual(m.prefill_observations, 1)
        self.assertIn("prefill_tok_s", m.stats())


def _seq(rid: str, prompt_len: int, uncached: int, arrived: float) -> Sequence:
    s = Sequence(request_id=rid, prompt_token_ids=list(range(prompt_len)),
                 sampling_params=SamplingParams(max_tokens=4))
    s.estimated_uncached_tokens = uncached
    s.arrival_time = arrived
    return s


class SchedulerPolicyTest(unittest.TestCase):
    def _scheduler(self, policy: str) -> Scheduler:
        bm = BlockManager(BlockManagerConfig(num_blocks=256, block_size=16))
        return Scheduler(SchedulerConfig(max_num_batched_tokens=2048,
                                         prefill_chunk_size=512,
                                         max_num_seqs=8,
                                         policy=policy,
                                         aging_tokens_per_s=0.0), bm)

    def test_fifo_preserves_arrival_order(self):
        sch = self._scheduler("fifo")
        now = time.perf_counter()
        for i, cost in enumerate((100, 10, 500)):
            sch.add(_seq(f"r{i}", 64, cost, now + i * 0.01))
        out = sch.schedule()
        self.assertEqual([c.seq.request_id for c in out.chunks], ["r0", "r1", "r2"])

    def test_cost_policy_admits_cheapest_first(self):
        sch = self._scheduler("cost")
        now = time.perf_counter()
        for i, cost in enumerate((100, 10, 500)):
            sch.add(_seq(f"r{i}", 64, cost, now))
        out = sch.schedule()
        self.assertEqual([c.seq.request_id for c in out.chunks], ["r1", "r0", "r2"])

    def test_aging_prevents_starvation(self):
        sch = self._scheduler("cost")
        sch.cfg.aging_tokens_per_s = 100.0
        now = time.perf_counter()
        old = _seq("old", 64, 500, now - 10.0)   # 500 - 1000 = negative -> first
        new = _seq("new", 64, 10, now)
        sch.add(old)
        sch.add(new)
        out = sch.schedule()
        self.assertEqual([c.seq.request_id for c in out.chunks], ["old", "new"])


if __name__ == "__main__":
    unittest.main()


class PredictionRangeTest(unittest.TestCase):
    def test_out_of_calibration_range_is_not_confident(self):
        m = CostModel(prefill_points=[(100, 100.0)])  # 100 tokens -> 100 ms
        p = m.predict_prefill(1000)
        self.assertFalse(p.confident)
        self.assertEqual(p.source, "above-calibration")
        self.assertLess(p.low_ms, p.high_ms)

    def test_auto_reuse_requires_non_overlapping_ranges(self):
        # copy is cheap and confident; recompute is expensive -> reuse
        m = CostModel(prefill_tok_s=100.0, copy_tok_s=10_000.0)
        c = m.estimate(2000, 1000)
        self.assertTrue(c.reuse)
        # make copy cost huge and prefill instant -> conservative recompute
        m2 = CostModel(prefill_tok_s=1_000_000.0, copy_tok_s=10.0,
                       copy_overhead_ms=20.0)
        self.assertFalse(m2.estimate(2000, 1000).reuse)


class MemoryCostSchedulerTest(unittest.TestCase):
    def test_predicted_kv_tokens_break_ties(self):
        from terrazzo.scheduler import Scheduler, SchedulerConfig
        from terrazzo.blocks import BlockManager, BlockManagerConfig

        sch = Scheduler(SchedulerConfig(max_num_batched_tokens=2048,
                                        prefill_chunk_size=512,
                                        max_num_seqs=8,
                                        policy="cost",
                                        aging_tokens_per_s=0.0,
                                        memory_cost_weight=1.0),
                        BlockManager(BlockManagerConfig(num_blocks=256, block_size=16)))
        now = time.perf_counter()
        # equal uncached work; the second request predicts more KV growth
        a = _seq("small-kv", 64, 64, now)
        a.estimated_kv_tokens = 10
        b = _seq("big-kv", 64, 64, now)
        b.estimated_kv_tokens = 200
        sch.add(a)
        sch.add(b)
        out = sch.schedule()
        self.assertEqual([c.seq.request_id for c in out.chunks], ["small-kv", "big-kv"])
