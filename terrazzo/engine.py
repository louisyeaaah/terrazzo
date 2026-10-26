"""Engine: the step() loop that ties scheduler + runner together.

Mirrors DeepInfer's ``terrazzo/engine/llm_engine.py``:

    while unfinished:
        outputs = scheduler.schedule()
        outputs = run_workers(outputs)          # executed asynchronously
        process_outputs(outputs)                # may be deferred by one step (ZOS)

Differences on Apple Silicon (documented in DESIGN.md):
  * MLX already evaluates asynchronously (``mx.async_eval`` inside mlx-lm), so
    DeepInfer's "move post-process one step later" trick is not needed for the
    GPU to stay busy. We keep the call structure so the code reads like the
    original, but post-process happens in the same step.
  * one process, one GPU: no DP ranks, no ``_run_workers`` RPC.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

from .blocks import BlockManager, BlockManagerConfig
from .paged_kv import PagedPrefixCache, SharedPrefix
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutputs
from .sequence import SamplingParams, Sequence, SequenceStatus


@dataclass
class EngineConfig:
    model: str = "mlx-community/Qwen3-4B-4bit"
    # scheduler
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 2048
    prefill_chunk_size: int = 512
    max_model_len: int = 32768
    stop_preempted_request: bool = False
    # KV budget (logical paging; MLX owns the physical tensors).
    # 0 = derive from (memory available for KV, model KV bytes/token, block size).
    num_gpu_blocks: int = 4096
    block_size: int = 16
    # prefix cache
    enable_prefix_cache: bool = True
    prefix_cache_entries: int = 16
    prefix_cache_bytes: int = 1 << 30  # 1 GB: one 4k-token prefix snapshot is ~370 MB
    prefix_cache_max_tokens: int = 4096  # cap a stored prefix snapshot (covers 2k-4k system prompts)
    # skip storing a snapshot when an existing entry already shares this many tokens
    # (agent traffic: every session sends the same system prompt, different user turn)
    prefix_dedupe_min_tokens: int = 1024
    # executor
    completion_batch_size: int = 8
    prefill_batch_size: int = 8
    warmup: bool = True


@dataclass
class RequestOutput:
    request_id: str
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finished: bool = False
    finish_reason: str | None = None
    ttft: float | None = None
    num_prompt_tokens: int = 0
    num_output_tokens: int = 0


def auto_num_gpu_blocks(model: str, block_size: int, *, kv_fraction: float = 0.35) -> int:
    """Blocks that fit a KV budget of ``kv_fraction`` × physical RAM.

    Uses the model config only (no weights): KV bytes/token =
    layers × 2 × n_kv_heads × head_dim × 2 bytes (fp16).
    """
    import os

    from .model_profile import load_config

    cfg = load_config(model)
    layers = cfg.get("num_hidden_layers")
    heads = cfg.get("num_kv_heads") or cfg.get("num_key_value_heads")
    hidden = cfg.get("hidden_size")
    attn_heads = cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    if head_dim is None and hidden and attn_heads:
        head_dim = hidden // attn_heads
    if not (layers and heads and head_dim):
        return 4096
    per_token = layers * 2 * heads * head_dim * 2
    total_mb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e6
    budget = kv_fraction * total_mb * 1e6
    blocks = int(budget / (per_token * block_size))
    return max(512, min(32768, blocks))


class Engine:
    def __init__(self, cfg: EngineConfig | None = None, runner=None) -> None:
        from .mask_patch import apply_mask_patch

        apply_mask_patch()
        from .contig_kv_patch import apply_contig_kv_patch

        self._contig_kv = apply_contig_kv_patch()
        if not self._contig_kv:
            # the Metal kernels assume left-padded rows; with the contiguous cache
            # the real tokens are at the *front*, so keep MLX's SDPA there.
            from .batch_attn_patch import apply_batch_attn_patch

            apply_batch_attn_patch()
        from .multi_query_patch import apply_multi_query_patch

        apply_multi_query_patch()  # env-gated (TERRAZZO_MULTI_Q=1); default OFF
        self.cfg = cfg or EngineConfig()
        if self.cfg.num_gpu_blocks <= 0:
            computed = auto_num_gpu_blocks(self.cfg.model, self.cfg.block_size)
            print(f"[terrazzo] auto num_gpu_blocks={computed} "
                  f"({computed * self.cfg.block_size} tokens, 35% of RAM for KV)", flush=True)
            self.cfg.num_gpu_blocks = computed
        self.block_manager = BlockManager(
            BlockManagerConfig(
                num_blocks=self.cfg.num_gpu_blocks, block_size=self.cfg.block_size
            )
        )
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_num_batched_tokens=self.cfg.max_num_batched_tokens,
                prefill_chunk_size=self.cfg.prefill_chunk_size,
                max_num_seqs=self.cfg.max_num_seqs,
                max_model_len=self.cfg.max_model_len,
                stop_preempted_request=self.cfg.stop_preempted_request,
            ),
            self.block_manager,
        )
        self.prefix_cache = (
            PagedPrefixCache(
                self.block_manager,
                max_entries=self.cfg.prefix_cache_entries,
                max_bytes=self.cfg.prefix_cache_bytes,
            )
            if self.cfg.enable_prefix_cache
            else None
        )
        if runner is None:
            from .runner import MLXRunner

            runner = MLXRunner(
                self.cfg.model,
                completion_batch_size=self.cfg.completion_batch_size,
                prefill_batch_size=self.cfg.prefill_batch_size,
                prefill_step_size=self.cfg.prefill_chunk_size,
            )
        self.runner = runner
        self.tokenizer = getattr(runner, "tokenizer", None)

        self._inserted: dict[str, int] = {}   # request_id -> uid
        self._request_id = 0
        self._lock = threading.RLock()
        self._streams: dict[str, "queue.Queue"] = {}
        self._outputs: dict[str, RequestOutput] = {}

        # metrics
        self.num_steps = 0
        self.prefix_hit_tokens = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.step_times: list[float] = []
        self.start_time = time.perf_counter()

    # ---------------------------------------------------------------- requests
    def add_request(
        self,
        prompt: str | list[int] | None = None,
        params: SamplingParams | None = None,
        *,
        prompt_token_ids: list[int] | None = None,
        request_id: str | None = None,
    ) -> Sequence:
        with self._lock:
            if prompt_token_ids is not None:
                ids = list(prompt_token_ids)
            else:
                if prompt is None:
                    raise ValueError("prompt or prompt_token_ids required")
                ids = self.runner.encode(prompt)
            params = params or SamplingParams()
            self._request_id += 1
            rid = request_id or f"req-{self._request_id}"
            seq = Sequence(
                request_id=rid,
                prompt_token_ids=ids,
                sampling_params=params,
            )
            self._outputs[rid] = RequestOutput(
                request_id=rid, num_prompt_tokens=len(ids)
            )
            self._streams[rid] = queue.Queue()
            self.scheduler.add(seq)
            return seq

    def stream(self, request_id: str) -> "queue.Queue":
        return self._streams[request_id]

    def abort(self, request_id: str) -> bool:
        with self._lock:
            return self.scheduler.abort(request_id)

    # -------------------------------------------------------------------- step
    def step(self) -> list[RequestOutput]:
        with self._lock:
            t0 = time.perf_counter()
            outs = self.scheduler.schedule()
            self._apply_preemptions(outs)
            self._insert_new(outs)

            if self.scheduler.num_unfinished == 0:
                self._collect_finished()
                return []

            prompt_responses, generation_responses = self.runner.step()
            deltas = self._process_prompt_responses(prompt_responses)
            deltas += self._process_generation_responses(generation_responses)
            self._collect_finished()

            self.num_steps += 1
            self.step_times.append(time.perf_counter() - t0)
            return deltas

    def _apply_preemptions(self, outs: SchedulerOutputs) -> None:
        if not outs.preempted:
            return
        for seq in outs.preempted:
            uid = self._inserted.pop(seq.request_id, None)
            if uid is not None:
                try:
                    self.runner.bg.remove([uid])
                except Exception:
                    pass

    def _insert_new(self, outs: SchedulerOutputs) -> None:
        for chunk in outs.chunks:
            seq = chunk.seq
            if not chunk.is_prefill or seq.request_id in self._inserted:
                continue
            cache, cached_tokens = None, 0
            if self.prefix_cache is not None:
                hit = self.prefix_cache.lookup(seq.prompt_token_ids)
                if hit is not None:
                    prefix, cached_tokens = hit
                    cache = self._seed_caches(prefix, cached_tokens)
                    seq.cached_prompt_tokens = cached_tokens
                    seq.prefix_blocks = []  # blocks are owned by the shared prefix
                    self.prefix_hit_tokens += cached_tokens
            try:
                uid = self.runner.insert(seq, cache=cache, cached_tokens=cached_tokens)
            except Exception:
                if cache is not None:
                    # shared-prefix reuse failed: retry without it rather than dropping
                    self.prefix_cache.num_hits -= 1
                    self.prefix_hit_tokens -= cached_tokens
                    uid = self.runner.insert(seq)
                    seq.cached_prompt_tokens = 0
                else:
                    raise
            self._inserted[seq.request_id] = uid

    def _seed_caches(self, prefix: SharedPrefix, hit_len: int) -> list:
        """Per-layer MLX KVCache objects pre-seeded with the shared prefix.

        mlx-lm's continuous batcher works on dense per-layer caches, so this makes
        the prefix usable *inside* the batching path at the cost of a copy on the
        first growth step (see DESIGN.md: memory sharing survives only in the
        sequential paged path).
        """
        from mlx_lm.models.cache import KVCache

        caches = []
        for k, v in prefix.layers:
            c = KVCache()
            c.keys = k[..., :hit_len, :]
            c.values = v[..., :hit_len, :]
            c.offset = hit_len
            caches.append(c)
        return caches

    def register_prefix(self, token_ids: list[int]) -> SharedPrefix | None:
        """Precompute a shared prefix so later requests skip prefilling it."""
        if self.prefix_cache is None:
            return None
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        caches = make_prompt_cache(self.runner.model)
        # registration blocks admission, so use the largest chunk the step budget
        # allows (fewer forwards) instead of the small in-flight prefill chunk
        chunk = max(self.cfg.prefill_chunk_size,
                    min(len(token_ids), self.cfg.max_num_batched_tokens))
        logits = None
        for start in range(0, len(token_ids), chunk):
            piece = token_ids[start:start + chunk]
            logits = self.runner.model(mx.array([piece]), cache=caches)
            mx.eval(logits)
        layers = [(c.keys[..., : c.offset, :], c.values[..., : c.offset, :]) for c in caches]
        return self.prefix_cache.store(token_ids, layers)

    # -------------------------------------------------------------- responses
    def _process_prompt_responses(self, responses) -> list[RequestOutput]:
        deltas: list[RequestOutput] = []
        for r in responses:
            seq = self.runner.uid_to_seq.get(getattr(r, "uid", -1))
            if seq is None:
                continue
            processed, total = getattr(r, "progress", (0, 0))
            seq.num_computed_prompt_tokens = max(seq.num_computed_prompt_tokens, processed)
            if getattr(r, "end_of_prompt", False):
                seq.num_computed_prompt_tokens = seq.prompt_len
                self.num_prefill_tokens += seq.prompt_len - seq.cached_prompt_tokens
                self._maybe_store_prefix(seq, self.runner.extract_cache(seq))
            else:
                # `processed` is cumulative for the request, so count only the delta
                prev = seq.num_computed_prompt_tokens
                seq.num_computed_prompt_tokens = max(prev, processed)
                self.num_prefill_tokens += max(0, processed - prev)
        return deltas

    def _maybe_store_prefix(self, seq: Sequence, prompt_cache=None) -> None:
        """Populate the shared-prefix cache from a finished request's KV."""
        if self.prefix_cache is None or seq.cached_prompt_tokens > 0:
            return
        if not prompt_cache:
            return
        bs = self.block_manager.block_size
        limit = min(seq.prompt_len, max(0, self.cfg.prefix_cache_max_tokens))
        n = (limit // bs) * bs  # store a bounded, block-aligned prefix snapshot
        if n < bs:
            return
        if self._shares_cached_prefix(seq.prompt_token_ids):
            return
        try:
            layers = []
            for c in prompt_cache:
                if not hasattr(c, "keys") or c.keys is None:
                    return
                layers.append((c.keys[..., :n, :], c.values[..., :n, :]))
            # tokens and layers must cover the SAME length: store() derives its
            # block hashes from the token list, so passing the full prompt while
            # handing over a trimmed KV snapshot corrupts the entry.
            self.prefix_cache.store(list(seq.prompt_token_ids[:n]), layers)
        except Exception:
            return

    def _shares_cached_prefix(self, prompt_tokens: list[int]) -> bool:
        """True when an entry already covers this prompt's leading tokens.

        Storing per-user prompts would otherwise create one ~700 MB snapshot per
        session in the cold batch and evict each other. Session traffic shares the
        system prompt, so one entry is enough.
        """
        min_shared = max(1, self.cfg.prefix_dedupe_min_tokens)
        cache = self.prefix_cache
        if cache is None:
            return False
        for entry in cache.entries.values():
            shared = 0
            for a, b in zip(entry.tokens, prompt_tokens):
                if a != b:
                    break
                shared += 1
                if shared >= min_shared:
                    return True
            if len(entry.tokens) >= min_shared and shared >= min(len(entry.tokens), len(prompt_tokens)):
                return True
        return False

    def _process_generation_responses(self, responses) -> list[RequestOutput]:
        deltas: list[RequestOutput] = []
        for r in responses:
            seq = self.runner.uid_to_seq.get(r.uid)
            if seq is None:
                continue
            token = int(r.token)
            seq.append_token(token, seq.sampling_params.max_tokens)
            self.num_decode_tokens += 1
            finish_reason = r.finish_reason
            if finish_reason == "stop" and seq.status == SequenceStatus.RUNNING:
                seq.finish("stop")
            if seq.status.is_finished or finish_reason:
                if not seq.status.is_finished:
                    seq.finish(finish_reason or "stop")
                self._maybe_store_prefix(seq, getattr(r, "prompt_cache", None))
                self.scheduler.finish(seq)
                self._inserted.pop(seq.request_id, None)
                self.runner.uid_to_seq.pop(r.uid, None)
            out = self._outputs[seq.request_id]
            out.token_ids.append(token)
            out.num_output_tokens = len(seq.output_token_ids)
            out.ttft = seq.ttft()
            delta = RequestOutput(
                request_id=seq.request_id,
                token_ids=[token],
                text="",
                finished=seq.status.is_finished,
                finish_reason=seq.finish_reason,
            )
            deltas.append(delta)
            self._push(seq.request_id, delta)
        return deltas

    def _push(self, request_id: str, delta: RequestOutput) -> None:
        q = self._streams.get(request_id)
        if q is not None:
            q.put(delta)

    def _collect_finished(self) -> None:
        for seq in self.scheduler.drain_finished():
            out = self._outputs.get(seq.request_id)
            if out is None:
                continue
            out.finished = True
            out.finish_reason = seq.finish_reason
            out.token_ids = list(seq.output_token_ids)
            out.ttft = seq.ttft()
            out.num_output_tokens = len(seq.output_token_ids)
            self._push(seq.request_id, out)

    # ------------------------------------------------------------- convenience
    def has_unfinished(self) -> bool:
        return self.scheduler.num_unfinished > 0

    def generate(
        self,
        prompt: str | list[str] | None = None,
        params: SamplingParams | None = None,
        *,
        prompts: list[str] | None = None,
        prompt_token_ids: list[int] | list[list[int]] | None = None,
        max_tokens: int | None = None,
        use_tqdm: bool = False,
        spec_ngram: bool = False,
        spec_k: int = 4,
        spec_n: int = 3,
        json_schema: dict | None = None,
    ) -> str | list[str]:
        """Blocking convenience path (like DeepInfer's offline ``LLM.generate``)."""
        params = params or SamplingParams()
        if max_tokens is not None:
            params = SamplingParams(**{**params.__dict__, "max_tokens": max_tokens})
        items: list[tuple[str | None, list[int] | None]] = []
        if prompt_token_ids is not None:
            if prompt_token_ids and isinstance(prompt_token_ids[0], (list, tuple)):
                items = [(None, list(ids)) for ids in prompt_token_ids]  # type: ignore[arg-type]
            else:
                items = [(None, list(prompt_token_ids))]  # type: ignore[arg-type]
        elif prompts is not None:
            items = [(p, None) for p in prompts]
        elif isinstance(prompt, list):
            items = [(p, None) for p in prompt]
        else:
            items = [(prompt, None)]

        if json_schema is not None and len(items) == 1:
            from .json_schema import json_generate

            prompt_text, ids = items[0]
            if ids is None:
                ids = self.runner.encode(prompt_text or "")
            text, stats = json_generate(
                self.runner.model, self.runner.tokenizer, list(ids), json_schema,
                max_tokens=params.max_tokens)
            self.last_json_stats = stats
            return text
        if spec_ngram and len(items) == 1:
            # n-gram/prompt-lookup speculative decode (single request, greedy)
            from .ngram_spec import ngram_generate

            prompt_text, ids = items[0]
            if ids is None:
                ids = self.runner.encode(prompt_text or "")
            text, stats = ngram_generate(
                self.runner.model, self.runner.tokenizer, list(ids),
                max_tokens=params.max_tokens, k=spec_k, n=spec_n,
                temperature=params.temperature, ignore_eos=params.ignore_eos)
            self.last_spec_stats = stats
            self._last_spec_text = text
            return text
        self.last_spec_stats = None
        seqs = [self.add_request(p, params, prompt_token_ids=ids) for p, ids in items]
        while self.has_unfinished():
            self.step()
        results = []
        for seq in seqs:
            results.append(self.runner.decode(seq.output_token_ids))
        return results if len(results) > 1 else results[0]

    def run_until(self, timeout: float | None = None) -> None:
        t0 = time.perf_counter()
        while self.has_unfinished():
            self.step()
            if timeout is not None and time.perf_counter() - t0 > timeout:
                break

    # ------------------------------------------------------------------ stats
    def stats(self) -> dict:
        st = self.scheduler.stats()
        if self.step_times:
            st["mean_step_ms"] = round(1000 * sum(self.step_times) / len(self.step_times), 2)
        st["prefill_tokens"] = self.num_prefill_tokens
        st["prefix_hit_tokens"] = self.prefix_hit_tokens
        st["decode_tokens"] = self.num_decode_tokens
        st["uptime_s"] = round(time.perf_counter() - self.start_time, 2)
        if self.prefix_cache is not None:
            st["prefix_cache"] = self.prefix_cache.stats()
        return st

    def close(self) -> None:
        self.runner.close()
