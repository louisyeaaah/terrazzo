"""Continuous-batching scheduler.

Port of the parts of DeepInfer's ``csrc/engine/scheduler.hpp`` that make sense on
a single Apple GPU:

  * FIFO ``waiting`` deque; preempted requests are pushed back to the *front*
    (``scheduler.hpp:_preempt`` -> ``waiting.push_front``)
  * chunked prefill under a per-step token budget (``_schedule_prefill``)
  * decode append-slot: a sequence needs a fresh block when it crosses a block
    boundary (``_append_slot``)
  * LCFS preemption: when the pool is exhausted, victims are taken from the
    *shortest-output* end of the running set (``_schedule_decode``)
  * ``stop_preempted_request``: drop instead of recompute when set

Deliberate deviations (documented in DESIGN.md):
  * no DP all-reduce step-type voting -- DeepInfer needs it because one MoE EP
    group must issue the same step type on every rank; a single Mac has no ranks
  * a step may mix prompt and decode work -- again, no EP constraint, and mixing
    lowers TTFT under load
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .blocks import BlockManager
from .sequence import Sequence, SequenceStatus


@dataclass
class SchedulerConfig:
    # per-step budget of prompt tokens (DeepInfer: max_num_batched_tokens)
    max_num_batched_tokens: int = 2048
    # how many prompt tokens one request may consume in a single step
    prefill_chunk_size: int = 512
    # max concurrent sequences
    max_num_seqs: int = 8
    # DeepInfer rejects prompts longer than this
    max_model_len: int = 32768
    # drop instead of recompute on preemption (DeepInfer: stop_preempted_request)
    stop_preempted_request: bool = False
    # "fifo" keeps the DeepInfer queue order; "cost" orders by estimated remaining
    # prefill work with aging so low-cost requests cannot starve long prompts
    policy: str = "fifo"
    aging_tokens_per_s: float = 100.0
    # predicted memory increment contributes this much to the cost key
    memory_cost_weight: float = 0.1


@dataclass
class ScheduledChunk:
    seq: Sequence
    num_tokens: int          # prompt tokens to prefill this step
    is_prefill: bool


@dataclass
class SchedulerOutputs:
    chunks: list[ScheduledChunk] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    finished: list[Sequence] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    block_usage: float = 0.0
    free_blocks: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.chunks and not self.finished

    @property
    def has_prompt(self) -> bool:
        return any(c.is_prefill for c in self.chunks)


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, block_manager: BlockManager) -> None:
        self.cfg = cfg
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        # rolling stats
        self.num_preemptions = 0
        self.num_admitted = 0
        self.num_steps = 0

    # ---------------------------------------------------------------- requests
    def add(self, seq: Sequence) -> None:
        if seq.prompt_len == 0:
            seq.finish("ignored")
            self.finished.append(seq)
            return
        if seq.prompt_len > self.cfg.max_model_len:
            seq.finish("ignored")
            self.finished.append(seq)
            return
        self.waiting.append(seq)

    def abort(self, request_id: str) -> bool:
        for seq in list(self.waiting):
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                seq.finish("abort")
                self.finished.append(seq)
                return True
        for seq in list(self.running):
            if seq.request_id == request_id:
                self.running.remove(seq)
                self.block_manager.free(seq.block_table + seq.prefix_blocks)
                seq.block_table, seq.prefix_blocks = [], []
                seq.finish("abort")
                self.finished.append(seq)
                return True
        return False

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    # ---------------------------------------------------------------- schedule
    def schedule(self) -> SchedulerOutputs:
        self.num_steps += 1
        out = SchedulerOutputs()

        # 1) preemption first: make room before admitting anything
        self._maybe_preempt(out)
        preempted_ids = {s.request_id for s in out.preempted}

        # 2) cost-aware admission order. FIFO is the default and preserves the
        # historical DeepInfer behaviour; "cost" only changes *who goes first*,
        # not the chunking/preemption mechanisms below.
        if self.cfg.policy == "cost" and len(self.waiting) > 1:
            now = time.perf_counter()
            aging = self.cfg.aging_tokens_per_s
            ordered = sorted(
                self.waiting,
                key=lambda s: (
                    max(0.0, getattr(s, "estimated_uncached_tokens", s.prompt_len)
                        + self.cfg.memory_cost_weight
                        * getattr(s, "estimated_kv_tokens", 0)
                        - aging * max(0.0, now - s.arrival_time)),
                    s.arrival_time,
                ),
            )
            self.waiting = deque(ordered)

        # 2) chunked prefill admission.
        #
        # Round-robin over waiting requests instead of always taking waiting[0]:
        # the old code put a partially-prefilled request back at the *front* and
        # broke, so one long prompt monopolised the step budget and every other
        # request waited for it to finish (16 sessions x 2.5k shared prefix -> 8-11 s
        # TTFT). Chunks are now spread across requests under the same per-step
        # budget; partially prefilled requests rotate to the back of the queue.
        budget = self.cfg.max_num_batched_tokens
        active = len(self.running)  # sequences resident or admitted this step
        # after a preemption, do not admit anything new this step: the victim was
        # pushed back to the front and admitting more work would cascade preemptions
        candidates = 0 if out.preempted else len(self.waiting)
        deferred: list[Sequence] = []
        blocked = False
        for _ in range(candidates):
            if budget <= 0 or active >= self.cfg.max_num_seqs or blocked:
                deferred.extend(self.waiting)
                self.waiting.clear()
                break
            seq = self.waiting.popleft()
            if seq.request_id in preempted_ids:
                # never re-admit a request preempted in this same step; keep it queued
                deferred.append(seq)
                continue
            want = min(seq.remaining_prompt_tokens, self.cfg.prefill_chunk_size, budget)
            if want <= 0:
                # prompt already computed (e.g. fully cached): promote it
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                active += 1
                continue
            if not self._ensure_blocks(seq, seq.num_computed_prompt_tokens + want):
                # no room for this request's next chunk: skip it (keeps FIFO fairness)
                deferred.append(seq)
                blocked = True
                continue
            seq.status = SequenceStatus.RUNNING
            out.chunks.append(ScheduledChunk(seq, want, is_prefill=True))
            out.num_prefill_tokens += want
            budget -= want
            active += 1
            self.num_admitted += 1
            if seq.remaining_prompt_tokens > want:
                deferred.append(seq)      # continue in a later step (round-robin)
            else:
                self.running.append(seq)
        self.waiting.extend(deferred)

        # 3) decode step for everything that already finished prefill
        for seq in list(self.running):
            if seq.status != SequenceStatus.RUNNING:
                continue
            if seq.remaining_prompt_tokens > 0:
                continue  # still prefilling (chunked), gets a chunk above
            if not self._ensure_blocks(seq, seq.num_tokens + 1):
                self._preempt(seq, out)
                continue
            out.chunks.append(ScheduledChunk(seq, 1, is_prefill=False))
            out.num_decode_tokens += 1

        out.block_usage = self.block_manager.usage
        out.free_blocks = self.block_manager.num_free_blocks
        return out

    # ------------------------------------------------------------------ blocks
    def _blocks_needed(self, seq: Sequence, total_tokens: int) -> int:
        return self.block_manager.blocks_for(total_tokens) - len(seq.block_table)

    def _ensure_blocks(self, seq: Sequence, total_tokens: int) -> bool:
        need = self._blocks_needed(seq, total_tokens)
        if need <= 0:
            return True
        if not self.block_manager.can_allocate(need):
            return False
        seq.block_table.extend(
            self.block_manager.allocate(
                need, token_ids=seq.all_token_ids, block_offset=len(seq.block_table)
            )
        )
        return True

    # -------------------------------------------------------------- preemption
    def _maybe_preempt(self, out: SchedulerOutputs) -> None:
        # nothing to do unless the next waiting request cannot fit
        if not self.waiting:
            return
        seq = self.waiting[0]
        need = self.block_manager.blocks_for(max(seq.prompt_len, 1))
        if self.block_manager.can_allocate(need):
            return
        if not self.running:
            return
        # LCFS: the shortest-output sequence is the most recently admitted one
        victim = min(self.running, key=lambda s: s.output_len)
        self._preempt(victim, out)

    def _preempt(self, seq: Sequence, out: SchedulerOutputs) -> None:
        self.num_preemptions += 1
        self.running.remove(seq)
        self.block_manager.free(seq.block_table + seq.prefix_blocks)
        seq.block_table, seq.prefix_blocks = [], []
        seq.num_computed_prompt_tokens = 0
        seq.preempt_count += 1
        if self.cfg.stop_preempted_request:
            seq.finish("preempted")
            self.finished.append(seq)
        else:
            # DeepInfer recomputes: reset to WAITING and jump the queue.
            seq.status = SequenceStatus.WAITING
            self.waiting.appendleft(seq)
        out.preempted.append(seq)

    # ---------------------------------------------------------------- lifecycle
    def finish(self, seq: Sequence) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self.block_manager.free(seq.block_table + seq.prefix_blocks)
        seq.block_table, seq.prefix_blocks = [], []
        self.finished.append(seq)

    def drain_finished(self) -> list[Sequence]:
        out, self.finished = self.finished, []
        return out

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        return {
            "steps": self.num_steps,
            "waiting": len(self.waiting),
            "running": len(self.running),
            "admitted": self.num_admitted,
            "preemptions": self.num_preemptions,
            **self.block_manager.snapshot(),
        }
