"""Unit tests for the DeepInfer-on-Mac engine (no MLX required)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.blocks import (  # noqa: E402
    BlockManager,
    BlockManagerConfig,
    hash_block_tokens,
)
from terrazzo.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from terrazzo.sequence import SamplingParams, Sequence, SequenceStatus  # noqa: E402


def make_seq(rid: str, n_prompt: int, max_tokens: int = 16) -> Sequence:
    return Sequence(
        request_id=rid,
        prompt_token_ids=list(range(n_prompt)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
    )


class BlockManagerTest(unittest.TestCase):
    def setUp(self):
        self.bm = BlockManager(BlockManagerConfig(num_blocks=33, block_size=16))

    def test_block_zero_is_reserved(self):
        self.assertEqual(self.bm.blocks[0].ref_count, 1)
        self.assertEqual(self.bm.num_free_blocks, 31)  # 32 usable - 1 reserve

    def test_allocate_and_free(self):
        ids = self.bm.allocate(4)
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(self.bm.num_free_blocks, 27)
        self.bm.free(ids)
        self.assertEqual(self.bm.num_free_blocks, 31)

    def test_ref_count_holds_block(self):
        ids = self.bm.allocate(2)
        self.bm.ref(ids)
        self.bm.free(ids)
        self.assertEqual(self.bm.num_free_blocks, 29)  # still referenced
        self.bm.free(ids)
        self.assertEqual(self.bm.num_free_blocks, 31)

    def test_cannot_over_allocate(self):
        self.assertTrue(self.bm.can_allocate(31))
        self.assertFalse(self.bm.can_allocate(32))
        with self.assertRaises(ValueError):
            self.bm.allocate(40)

    def test_blocks_for(self):
        self.assertEqual(self.bm.blocks_for(0), 1)
        self.assertEqual(self.bm.blocks_for(16), 1)
        self.assertEqual(self.bm.blocks_for(17), 2)
        self.assertEqual(self.bm.blocks_for(64), 4)

    def test_hash_chain_matches_across_managers(self):
        tokens = list(range(64))
        bm2 = BlockManager(BlockManagerConfig(num_blocks=33, block_size=16))
        bm1 = self.bm
        bm1.allocate(4, token_ids=tokens)
        hit, blocks = bm2.lookup_prefix(tokens)
        # different manager: hashes must agree structurally, but its hash map is empty
        self.assertEqual(hit, 0)
        hit, blocks = bm1.lookup_prefix(tokens)
        self.assertEqual(hit, 64)
        self.assertEqual(len(blocks), 4)

    def test_partial_prefix_hit(self):
        tokens = list(range(64))
        self.bm.allocate(4, token_ids=tokens)
        hit, _ = self.bm.lookup_prefix(tokens + [999, 1000, 1001])
        self.assertEqual(hit, 64)
        hit2, _ = self.bm.lookup_prefix(tokens[:20])
        self.assertEqual(hit2, 16)  # only the full first block counts

    def test_hash_is_chained(self):
        a = hash_block_tokens(None, [1, 2, 3])
        b = hash_block_tokens(a, [4, 5, 6])
        self.assertNotEqual(a, b)
        self.assertEqual(a, hash_block_tokens(None, [1, 2, 3]))


class SchedulerTest(unittest.TestCase):
    def build(self, **kw):
        bm = BlockManager(BlockManagerConfig(num_blocks=kw.pop("num_blocks", 64), block_size=16))
        cfg = SchedulerConfig(**kw)
        return Scheduler(cfg, bm), bm

    def test_fifo_admission(self):
        sch, _ = self.build(max_num_seqs=2)
        a, b = make_seq("a", 10), make_seq("b", 10)
        sch.add(a)
        sch.add(b)
        out = sch.schedule()
        self.assertEqual([c.seq.request_id for c in out.chunks], ["a", "b"])
        self.assertTrue(all(c.is_prefill for c in out.chunks))

    def test_chunked_prefill(self):
        sch, _ = self.build(prefill_chunk_size=32, max_num_batched_tokens=1024)
        seq = make_seq("long", 100)
        sch.add(seq)
        # same request comes back next step until the prompt is consumed
        seen = 0
        for _ in range(10):
            out = sch.schedule()
            if out.num_prefill_tokens == 0:
                break
            seen += out.num_prefill_tokens
            seq.num_computed_prompt_tokens += out.num_prefill_tokens
            if seq.remaining_prompt_tokens == 0:
                break
        self.assertEqual(seen, 100)

    def test_token_budget_limits_admission(self):
        sch, _ = self.build(max_num_batched_tokens=40, prefill_chunk_size=40, max_num_seqs=8)
        for i in range(4):
            sch.add(make_seq(f"s{i}", 40))
        out = sch.schedule()
        self.assertLessEqual(out.num_prefill_tokens, 40)
        self.assertEqual(len(out.chunks), 1)

    def test_lcfs_preemption_when_blocks_run_out(self):
        sch, bm = self.build(max_num_seqs=4, num_blocks=8)
        # two running sequences holding most of the pool
        for i in range(2):
            seq = make_seq(f"run{i}", 32)
            seq.num_computed_prompt_tokens = 32
            seq.status = SequenceStatus.RUNNING
            seq.block_table = bm.allocate(2, token_ids=seq.all_token_ids)
            sch.running.append(seq)
        # a new request that cannot fit
        sch.add(make_seq("new", 64))
        out = sch.schedule()
        self.assertEqual(len(out.preempted), 1)
        self.assertEqual(sch.num_preemptions, 1)
        # victim is the newest / least progress
        self.assertEqual(out.preempted[0].request_id, "run0")
        self.assertEqual(out.preempted[0].status, SequenceStatus.WAITING)

    def test_stop_preempted_request_drops_instead_of_requeue(self):
        sch, bm = self.build(max_num_seqs=4, num_blocks=8, stop_preempted_request=True)
        seq = make_seq("run", 32)
        seq.num_computed_prompt_tokens = 32
        seq.status = SequenceStatus.RUNNING
        seq.block_table = bm.allocate(2, token_ids=seq.all_token_ids)
        sch.running.append(seq)
        sch.add(make_seq("new", 64))
        out = sch.schedule()
        self.assertEqual(len(out.preempted), 1)
        self.assertEqual(out.preempted[0].status, SequenceStatus.FINISHED_PREEMPTED)
        self.assertNotIn(seq, sch.waiting)

    def test_finish_frees_blocks(self):
        sch, bm = self.build(num_blocks=16)
        seq = make_seq("x", 32)
        sch.add(seq)
        sch.schedule()
        used = bm.num_used_blocks
        self.assertGreater(used, 0)
        sch.finish(seq)
        self.assertEqual(bm.num_used_blocks, 0)

    def test_abort_removes_from_waiting(self):
        sch, _ = self.build()
        seq = make_seq("x", 32)
        sch.add(seq)
        self.assertTrue(sch.abort("x"))
        self.assertEqual(sch.num_unfinished, 0)
        self.assertEqual(seq.status, SequenceStatus.FINISHED_ABORTED)

    def test_too_long_prompt_is_ignored(self):
        sch, _ = self.build(max_model_len=32)
        seq = make_seq("long", 64)
        sch.add(seq)
        self.assertEqual(seq.status, SequenceStatus.FINISHED_IGNORED)
        self.assertEqual(len(sch.finished), 1)

    def test_round_robin_spreads_prefill_budget(self):
        """One long prompt must not monopolise a step (16-session TTFT fix)."""
        sch, _ = self.build(prefill_chunk_size=32, max_num_batched_tokens=96,
                            max_num_seqs=8, num_blocks=64)
        seqs = [make_seq(f"r{i}", 100) for i in range(3)]
        for seq in seqs:
            sch.add(seq)
        out = sch.schedule()
        self.assertEqual(len(out.chunks), 3)
        self.assertEqual({c.seq.request_id for c in out.chunks}, {"r0", "r1", "r2"})
        self.assertEqual(out.num_prefill_tokens, 96)
        # partially prefilled requests rotate to the back: the next step continues
        # all three rather than finishing r0 first
        for seq in seqs:
            seq.num_computed_prompt_tokens += 32
        out2 = sch.schedule()
        self.assertEqual({c.seq.request_id for c in out2.chunks}, {"r0", "r1", "r2"})

    def test_preemption_step_defers_new_admissions(self):
        """A preempted step must not admit newcomers (avoids preemption cascades)."""
        sch, bm = self.build(max_num_seqs=4, num_blocks=8)
        for i in range(2):
            seq = make_seq(f"run{i}", 32)
            seq.num_computed_prompt_tokens = 32
            seq.status = SequenceStatus.RUNNING
            seq.block_table = bm.allocate(2, token_ids=seq.all_token_ids)
            sch.running.append(seq)
        sch.add(make_seq("new", 64))
        out = sch.schedule()
        self.assertEqual([s.request_id for s in out.preempted], ["run0"])
        self.assertFalse(any(c.is_prefill for c in out.chunks))


if __name__ == "__main__":
    unittest.main()


class AgentProfileTest(unittest.TestCase):
    def test_profile_bumps_defaults_only(self):
        import argparse

        from terrazzo.cli import apply_agent_profile

        a = argparse.Namespace(profile="agent", max_num_seqs=8)
        apply_agent_profile(a)
        self.assertEqual(a.max_num_seqs, 16)
        import os
        self.assertEqual(os.environ.get("TERRAZZO_ADMIT_BATCH"), "16")

        b = argparse.Namespace(profile="agent", max_num_seqs=4)   # explicit wins
        apply_agent_profile(b)
        self.assertEqual(b.max_num_seqs, 4)

        c = argparse.Namespace(profile="default", max_num_seqs=8)
        apply_agent_profile(c)
        self.assertEqual(c.max_num_seqs, 8)
