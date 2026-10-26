"""Sequence (request) state machine.

Port of DeepInfer's ``csrc/engine/sequence.hpp`` status machine, trimmed to the
states that exist on a single device (no PD-disaggregation / VE pipeline).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()          # queued, blocks not yet assigned
    RUNNING = enum.auto()          # prefill done (or in progress), decoding
    FINISHED_STOPPED = enum.auto()  # EOS or stop token
    FINISHED_LENGTH = enum.auto()   # max_tokens reached
    FINISHED_ABORTED = enum.auto()  # client gone
    FINISHED_PREEMPTED = enum.auto()  # dropped instead of recomputed (stop_preempted_request)
    FINISHED_IGNORED = enum.auto()  # prompt longer than max_model_len

    @property
    def is_finished(self) -> bool:
        return self not in (SequenceStatus.WAITING, SequenceStatus.RUNNING)


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()


@dataclass
class Sequence:
    """One request."""

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: SequenceStatus = SequenceStatus.WAITING

    output_token_ids: list[int] = field(default_factory=list)
    # tokens whose KV has actually been computed (prompt prefill progress)
    num_computed_prompt_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    # blocks owed to the prefix cache (ref counted; released when the seq ends)
    prefix_blocks: list[int] = field(default_factory=list)

    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finished_time: float | None = None
    finish_reason: str | None = None

    # bookkeeping
    preempt_count: int = 0
    num_scheduled_tokens: int = 0
    cached_prompt_tokens: int = 0

    # ------------------------------------------------------------- properties
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def last_token_id(self) -> int:
        return (
            self.output_token_ids[-1]
            if self.output_token_ids
            else self.prompt_token_ids[-1]
        )

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_prefill_done(self) -> bool:
        return self.num_computed_prompt_tokens >= self.prompt_len

    @property
    def remaining_prompt_tokens(self) -> int:
        return self.prompt_len - self.num_computed_prompt_tokens

    # ------------------------------------------------------------- transition
    def append_token(self, token_id: int, max_tokens: int) -> None:
        self.output_token_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        if len(self.output_token_ids) >= max_tokens:
            self.finish("length")

    def finish(self, reason: str) -> None:
        self.finish_reason = reason
        self.finished_time = time.perf_counter()
        self.status = {
            "stop": SequenceStatus.FINISHED_STOPPED,
            "length": SequenceStatus.FINISHED_LENGTH,
            "abort": SequenceStatus.FINISHED_ABORTED,
            "preempted": SequenceStatus.FINISHED_PREEMPTED,
            "ignored": SequenceStatus.FINISHED_IGNORED,
        }[reason]

    # ----------------------------------------------------------------- metrics
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def decode_speed(self) -> float | None:
        if self.first_token_time is None or self.output_len < 2:
            return None
        elapsed = time.perf_counter() - self.first_token_time
        return (self.output_len - 1) / max(elapsed, 1e-9)
