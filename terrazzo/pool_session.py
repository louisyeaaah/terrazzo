"""PoolSession: generation over the paged block pool (v0.4).

Like ``PagedSession`` (v0.2) but the KV lives in a real block pool instead of
frozen dense prefix arrays:

  * every token's K/V is written once, into the sequence's own blocks
  * a registered prefix is a set of blocks that later requests *reference*
    (ref-counted), so nothing is copied
  * ``pool.stats()`` reports real block usage / shared blocks / bytes
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from .paged_pool import PagedBlockTable, PagedKVPool
from .pool_cache import make_pool_caches


def _model_dims(model) -> tuple[int, int]:
    """(n_kv_heads, head_dim) from the model config (``model.args``)."""
    args = getattr(model, "args", None)
    if args is None:
        raise RuntimeError("model has no .args; cannot infer KV shape")
    n_kv_heads = getattr(args, "num_key_value_heads", None)
    head_dim = getattr(args, "head_dim", None)
    if head_dim is None:
        hidden = getattr(args, "hidden_size", None)
        n_heads = getattr(args, "num_attention_heads", None)
        if hidden is not None and n_heads:
            head_dim = hidden // n_heads
    if n_kv_heads is None or head_dim is None:
        raise RuntimeError(f"cannot infer n_kv_heads/head_dim from args: {args}")
    return int(n_kv_heads), int(head_dim)


@dataclass
class PoolStats:
    prompt_tokens: int = 0
    prefix_hit_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    ttft_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    tokens: list[int] = field(default_factory=list)
    first_logits: Any = None

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "ttft_s": round(self.ttft_s, 4),
            "prefill_s": round(self.prefill_s, 4),
            "decode_tok_s": round(self.decode_tokens / self.decode_s, 1) if self.decode_s else 0.0,
        }


@dataclass
class SharedBlocks:
    tokens: tuple[int, ...]
    num_tokens: int
    blocks: list[int]
    refs: int = 1
    hits: int = 0


class PoolSession:
    def __init__(
        self,
        model,
        tokenizer=None,
        *,
        block_size: int = 16,
        num_blocks: int = 512,
        prefill_chunk_size: int = 1024,
        eos_token_ids: tuple[int, ...] = (),
        max_prefixes: int = 8,
        kv_bits: int | None = None,
        pool_dtype: "mx.Dtype" = mx.float16,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        n_kv_heads, head_dim = _model_dims(model)
        self.n_layers = len(model.layers)
        self.pool = PagedKVPool(
            self.n_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=pool_dtype,
            kv_bits=kv_bits,
        )
        self.prefixes: "OrderedDict[tuple[int, ...], SharedBlocks]" = OrderedDict()
        self.max_prefixes = max_prefixes
        self.prefill_chunk_size = prefill_chunk_size
        self.eos_token_ids = tuple(eos_token_ids)
        self.num_requests = 0

    # -------------------------------------------------------------- prefixes
    def register_prefix(self, token_ids: list[int]) -> SharedBlocks | None:
        """Prefill a prefix into its own blocks and keep them for sharing."""
        bs = self.pool.block_size
        n = (len(token_ids) // bs) * bs
        if n < bs:
            return None
        table = PagedBlockTable(self.pool)
        # the prefix entry keeps these blocks; the temporary caches must not free them
        caches = make_pool_caches(self.n_layers, self.pool, table, auto_release=False)
        tokens = token_ids[:n]
        for start in range(0, n, self.prefill_chunk_size):
            piece = tokens[start:start + self.prefill_chunk_size]
            self.model(mx.array([piece]), cache=caches)
        if self.pool.kv_bits is None:
            mx.eval([self.pool.layer_kv(i)[0] for i in range(self.n_layers)])
        else:
            mx.eval([self.pool.gather(i, table.blocks, n)[0] for i in range(self.n_layers)])
        entry = SharedBlocks(tokens=tuple(tokens), num_tokens=n, blocks=list(table.blocks))
        self.prefixes[tuple(tokens)] = entry
        self._evict_prefixes()
        return entry

    def lookup_prefix(self, token_ids: list[int]) -> SharedBlocks | None:
        best = None
        for key, entry in self.prefixes.items():
            if len(key) > len(token_ids) or tuple(token_ids[: len(key)]) != key:
                continue
            if best is None or entry.num_tokens > best.num_tokens:
                best = entry
        if best is not None:
            best.hits += 1
            self.prefixes.move_to_end(best.tokens)
        return best

    def _evict_prefixes(self) -> None:
        while len(self.prefixes) > self.max_prefixes:
            _tokens, entry = self.prefixes.popitem(last=False)
            if entry.refs <= 1:
                self.pool.free(entry.blocks)
            else:
                # still in use: put it back and stop evicting
                self.prefixes[_tokens] = entry
                break

    # ---------------------------------------------------------------- generate
    def _new_table(self, prefix: SharedBlocks | None) -> PagedBlockTable:
        if prefix is None:
            return PagedBlockTable(self.pool)
        self.pool.ref(prefix.blocks)
        return PagedBlockTable(self.pool, prefix_blocks=prefix.blocks,
                               prefix_len=prefix.num_tokens)

    def _sample(self, logits, temperature: float) -> int:
        if temperature <= 0:
            return int(mx.argmax(logits, axis=-1).item())
        probs = mx.softmax(logits / temperature, axis=-1)
        return int(mx.random.categorical(mx.log(probs)).item())

    def _prefill(self, tokens: list[int], caches):
        logits = None
        for start in range(0, len(tokens), self.prefill_chunk_size):
            piece = tokens[start:start + self.prefill_chunk_size]
            logits = self.model(mx.array([piece]), cache=caches)
            mx.eval(logits)
        return logits

    def generate(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop_on_eos: bool = True,
        keep_logits: bool = False,
        on_token=None,
    ) -> PoolStats:
        stats = PoolStats(prompt_tokens=len(prompt_token_ids))
        self.num_requests += 1
        prefix = self.lookup_prefix(prompt_token_ids)
        hit = prefix.num_tokens if prefix else 0
        stats.prefix_hit_tokens = hit
        table = self._new_table(prefix)
        caches = make_pool_caches(self.n_layers, self.pool, table)
        suffix = prompt_token_ids[hit:]
        stats.prefill_tokens = len(suffix)
        t0 = time.perf_counter()
        try:
            if suffix:
                logits = self._prefill(suffix, caches)
            else:
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
            if stop_on_eos and self.eos_token_ids and stats.tokens and stats.tokens[-1] in self.eos_token_ids:
                stats.tokens = stats.tokens[:-1]
            stats.decode_s = time.perf_counter() - decode_start
            stats.decode_tokens = max(0, len(stats.tokens) - 1)
        finally:
            table.release()
        return stats

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        st = self.pool.stats()
        return {
            "num_layers": self.n_layers,
            "prefixes": [{"tokens": e.num_tokens, "blocks": len(e.blocks), "refs": e.refs,
                          "hits": e.hits} for e in self.prefixes.values()],
            "pool": {
                "num_blocks": st.num_blocks, "block_size": st.block_size,
                "used_blocks": st.used_blocks, "shared_blocks": st.shared_blocks,
                "peak_blocks": st.peak_blocks, "nbytes": st.nbytes,
                "layers_allocated": st.layers_allocated,
            },
        }
