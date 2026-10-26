"""Paged session: shared-prefix generation on MLX (v0.2).

Registers a long prefix (system prompt, tool schema, doc header) once, then every
request that starts with it only prefills its own suffix. Unlike v0.1's
exact-match cache, two *different* requests can share the same prefix KV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache

from .blocks import BlockManager, BlockManagerConfig
from .paged_kv import PagedKVCache, PagedPrefixCache, SharedPrefix

EOS_FALLBACK = ()


@dataclass
class GenerateStats:
    prompt_tokens: int = 0
    first_logits: Any = None  # last-position logits of the prompt, for equivalence checks
    prefix_hit_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    ttft_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    tokens: list[int] = field(default_factory=list)

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s else 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "ttft_s": round(self.ttft_s, 4),
            "prefill_s": round(self.prefill_s, 4),
            "decode_tok_s": round(self.decode_tok_s, 1),
        }


class PagedSession:
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        block_size: int = 16,
        num_blocks: int = 4096,
        max_prefix_bytes: int = 4 << 30,
        max_prefix_entries: int = 8,
        prefill_chunk_size: int = 1024,
        eos_token_ids: tuple[int, ...] = (),
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.block_manager = BlockManager(
            BlockManagerConfig(num_blocks=num_blocks, block_size=block_size)
        )
        self.prefix_cache = PagedPrefixCache(
            self.block_manager,
            max_bytes=max_prefix_bytes,
            max_entries=max_prefix_entries,
        )
        self.n_layers = len(make_prompt_cache(model))
        self.prefill_chunk_size = prefill_chunk_size
        self.eos_token_ids = tuple(eos_token_ids)

    # ------------------------------------------------------------------ caches
    def _new_caches(self, prefix: SharedPrefix | None, prefix_len: int | None = None) -> list[PagedKVCache]:
        return [
            PagedKVCache(i, prefix, self.prefix_cache, prefix_len=prefix_len)
            for i in range(self.n_layers)
        ]

    def release(self, caches: list[PagedKVCache]) -> None:
        for c in caches:
            c.detach()

    # ----------------------------------------------------------------- prefill
    def _prefill(self, tokens: list[int], caches: list[PagedKVCache]):
        logits = None
        for start in range(0, len(tokens), self.prefill_chunk_size):
            chunk = tokens[start : start + self.prefill_chunk_size]
            logits = self.model(mx.array([chunk]), cache=caches)
            mx.eval(logits)
        return logits

    def register_prefix(self, token_ids: list[int]) -> SharedPrefix | None:
        """Compute and freeze a shared prefix; returns ``None`` if it doesn't fit."""
        caches = self._new_caches(None)
        try:
            if token_ids:
                self._prefill(token_ids, caches)
            layers = [c.buf.view() for c in caches]
            return self.prefix_cache.store(token_ids, layers)
        finally:
            self.release(caches)
            for c in caches:
                c.buf.keys = c.buf.values = None

    # ------------------------------------------------------------------ decode
    def _sample(self, logits, temperature: float) -> int:
        if temperature <= 0:
            return int(mx.argmax(logits, axis=-1).item())
        probs = mx.softmax(logits / temperature, axis=-1)
        return int(mx.random.categorical(mx.log(probs)).item())

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop_on_eos: bool = True,
        keep_logits: bool = False,
        on_token=None,
    ) -> GenerateStats:
        stats = GenerateStats(prompt_tokens=len(prompt_token_ids))
        hit = self.prefix_cache.lookup(prompt_token_ids)
        prefix, hit_len = hit if hit else (None, 0)
        stats.prefix_hit_tokens = hit_len
        caches = self._new_caches(prefix, hit_len)
        suffix = prompt_token_ids[hit_len:]
        stats.prefill_tokens = len(suffix)
        t0 = time.perf_counter()
        try:
            if suffix:
                logits = self._prefill(suffix, caches)
            else:
                # nothing left to prefill: replay the final token so sampling can start
                logits = self.model(mx.array([prompt_token_ids[-1:]]), cache=caches)
                mx.eval(logits)
            stats.prefill_s = time.perf_counter() - t0
            next_logits = logits[:, -1, :]
            if keep_logits:
                stats.first_logits = next_logits
            token = self._sample(next_logits, temperature)
            stats.ttft_s = time.perf_counter() - t0
            stats.tokens.append(token)
            if on_token is not None:
                on_token(token)
            decode_start = time.perf_counter()
            while len(stats.tokens) < max_tokens:
                if stop_on_eos and self.eos_token_ids and token in self.eos_token_ids:
                    break
                logits = self.model(mx.array([[token]]), cache=caches)
                token = self._sample(logits[:, -1, :], temperature)
                stats.tokens.append(token)
                if on_token is not None:
                    on_token(token)
            # remove a trailing EOS from the reported output
            if stop_on_eos and self.eos_token_ids and stats.tokens and stats.tokens[-1] in self.eos_token_ids:
                stats.tokens = stats.tokens[:-1]
            stats.decode_s = time.perf_counter() - decode_start
            stats.decode_tokens = max(0, len(stats.tokens) - 1)
        finally:
            self.release(caches)
        return stats

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "block": self.block_manager.snapshot(),
            "prefix_cache": self.prefix_cache.stats(),
        }

    def describe(self) -> str:
        st = self.stats()
        return (
            f"layers={st['n_layers']} blocks_used={st['block']['used']}/{st['block']['num_blocks']} "
            f"prefix_entries={st['prefix_cache']['entries']} "
            f"prefix_bytes={st['prefix_cache']['nbytes'] / 1e6:.1f}MB"
        )
