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
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .blocks import BlockManager, BlockManagerConfig
from .paged_kv import (PagedPrefixCache, SharedPrefix, _clone_state,
                       aligned_prefix_length)
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
    # auto: copy only when cheaper than recomputing the cached span (see cost.py)
    prefix_reuse: str = "auto"          # auto | always | never
    scheduler_policy: str = "fifo"      # fifo | cost
    scheduler_aging_tokens_per_s: float = 100.0
    memory_cost_weight: float = 0.1     # cost policy: predicted KV tokens weight
    cost_calibration: bool = True       # update the predicted-cost model from steps
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
        from .compat import require_tested_mlx_lm

        self._mlx_lm_version = require_tested_mlx_lm()
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
                policy=self.cfg.scheduler_policy,
                aging_tokens_per_s=self.cfg.scheduler_aging_tokens_per_s,
                memory_cost_weight=self.cfg.memory_cost_weight,
            ),
            self.block_manager,
        )
        from .cost import CostModel, kv_bytes_per_token

        self.cost_model = CostModel(
            kv_bytes_per_token=kv_bytes_per_token(self.cfg.model) or 0)
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
        # Streams/outputs have their own lock: the hang watchdog must be able to
        # fail requests while the worker thread holds _lock inside runner.step().
        # Lock order: _lock -> _stream_lock is allowed; never acquire _lock while
        # holding _stream_lock (fail_streams/stream/_push only take the stream lock).
        self._stream_lock = threading.RLock()
        self._streams: dict[str, "queue.Queue"] = {}
        self._outputs: dict[str, RequestOutput] = {}

        # metrics
        self.num_steps = 0
        self.request_stages = {"added": 0, "first_token": 0,
                               "finished": 0, "aborted": 0, "failed": 0}
        self.concurrency_changes: list[dict] = []
        self._executor_broken = False
        self.prefix_hit_tokens = 0
        self.prefix_recompute_skips = 0
        self.prefix_reuse_autos = 0
        # Phase-0 instrumentation: scheduler decision vs executor reality.
        self._exec_trace: deque[dict] = deque(maxlen=512)
        self._first_prefill_steps: list[dict] = []
        # bounded request -> predicted cost (for explaining scheduling decisions)
        self._request_costs: OrderedDict[str, dict] = OrderedDict()
        self._step_prompt_tokens = 0
        self._step_decode_tokens = 0
        self._control_totals = {
            "steps": 0, "scheduled_prefill_tokens": 0, "actual_prompt_tokens": 0,
            "scheduled_decode_tokens": 0, "actual_decode_tokens": 0,
            "scheduled_prefill_requests": 0, "inserted_prefill_requests": 0,
            "steps_with_prefill": 0, "steps_with_mismatch": 0,
        }
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
            if getattr(self, "_executor_broken", False):
                raise RuntimeError("executor is broken after a failed reset; restart the server")
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
            with self._stream_lock:
                self._outputs[rid] = RequestOutput(
                    request_id=rid, num_prompt_tokens=len(ids)
                )
                self._streams[rid] = queue.Queue()
            self.request_stages["added"] += 1
            cached_est, snapshot_bytes = 0, 0
            if self.prefix_cache is not None:
                hit = self.prefix_cache.peek(ids)
                if hit is not None:
                    prefix_est, cached_est = hit
                    snapshot_bytes = prefix_est.nbytes
            cost = self.cost_model.estimate(
                len(ids), cached_est, snapshot_bytes, policy=self.cfg.prefix_reuse)
            seq.estimated_uncached_tokens = cost.uncached_tokens
            seq.estimated_kv_bytes = int(len(ids) * self.cost_model.kv_bytes_per_token)
            # cached prefix blocks are shared; only the suffix + generated tokens
            # are new KV for this request
            new_tokens = (len(ids) - cached_est if cost.reuse else len(ids))
            seq.estimated_kv_tokens = int(new_tokens + params.max_tokens)
            self._request_costs[rid] = {
                "prompt_tokens": len(ids),
                "cached_estimate": int(cached_est),
                "uncached_estimate": int(cost.uncached_tokens),
                "kv_tokens": int(seq.estimated_kv_tokens),
                "reuse_estimate": bool(cost.reuse),
            }
            if len(self._request_costs) > 256:
                self._request_costs.popitem(last=False)
            self.scheduler.add(seq)
            return seq

    def stream(self, request_id: str) -> "queue.Queue":
        with self._stream_lock:
            return self._streams[request_id]

    def set_max_num_seqs(self, n: int) -> int:
        """Change the admission cap for *new* requests (running requests continue)."""
        n = max(1, int(n))
        with self._lock:
            self.cfg.max_num_seqs = n
            self.scheduler.cfg.max_num_seqs = n
            self.concurrency_changes.append({"at": time.time(), "max_num_seqs": n})
            if len(self.concurrency_changes) > 64:
                self.concurrency_changes = self.concurrency_changes[-64:]
        return n

    def abort(self, request_id: str) -> bool:
        with self._lock:
            uid = self._inserted.pop(request_id, None)
            seq = None
            for candidate in list(self.scheduler.waiting) + list(self.scheduler.running):
                if candidate.request_id == request_id:
                    seq = candidate
                    break
            if seq is None and uid is not None:
                seq = self.runner.uid_to_seq.get(uid)
            ok = self.scheduler.abort(request_id)
            if ok:
                self.request_stages["aborted"] += 1
            # Remove from the real executor too. MLXRunner.remove() takes Sequence
            # objects (it resolves uids itself) and frees the BatchGenerator entry;
            # scheduler.abort() only handles the logical blocks.
            remove_error = None
            if seq is not None:
                try:
                    self.runner.remove([seq])
                except Exception as exc:
                    remove_error = exc
            if uid is not None:
                self._inserted.pop(request_id, None)
            if remove_error is not None:
                # Do not report success when the executor still owns the request.
                # Treat it as a fatal step failure: abort the remaining requests,
                # rebuild the batcher if possible, or mark the executor broken.
                try:
                    self.handle_failure(remove_error)
                except Exception:
                    self._executor_broken = True
                return False
        if ok:
            # terminate a handler waiting on this stream as well, then release it
            with self._stream_lock:
                q = self._streams.pop(request_id, None)
                self._outputs.pop(request_id, None)
            if q is not None:
                try:
                    q.put({"request_id": request_id, "token_ids": [], "finished": True,
                           "finish_reason": "abort"})
                except Exception:
                    pass
        return ok

    def fail_streams(self, exc: BaseException) -> int:
        """Terminate every waiting stream without touching the execution lock.

        Used by the hang watchdog: the worker thread may be stuck inside MLX and
        holding ``_lock``.  This path therefore only takes ``_stream_lock`` (brief,
        never held across a GPU call), so terminal events are still delivered and the
        caller can proceed to its restart path.
        """
        with getattr(self, "_stream_lock", self._lock):
            ids = list(self._streams)
            for rid in ids:
                queue_ = self._streams.pop(rid, None)
                self._outputs.pop(rid, None)
                if queue_ is not None:
                    try:
                        queue_.put({"request_id": rid, "token_ids": [], "finished": True,
                                    "finish_reason": "error",
                                    "error": f"{type(exc).__name__}: {exc}"})
                    except Exception:
                        pass
        stages = getattr(self, "request_stages", None)
        if stages is not None:
            stages["failed"] += len(ids)
        return len(ids)

    def handle_failure(self, exc: BaseException) -> int:
        """Fatal step error: drop scheduler + executor state before serving again.

        Scheduler aborts free logical blocks; the BatchGenerator is then rebuilt so
        stale sequences/uid mappings cannot receive new work. If the rebuild itself
        fails, the engine refuses new requests until the process is restarted.
        """
        with self._lock:
            stream_lock = getattr(self, "_stream_lock", self._lock)
            with stream_lock:
                ids = list(self._streams)
            # cover requests that are in the scheduler but not (or no longer) in
            # the HTTP stream map as well
            waiting = getattr(self.scheduler, "waiting", ()) or ()
            running = getattr(self.scheduler, "running", ()) or ()
            for seq in list(waiting) + list(running):
                try:
                    self.scheduler.abort(seq.request_id)
                except Exception:
                    pass
            for rid in ids:
                try:
                    self.scheduler.abort(rid)
                except Exception:
                    pass
            runner = getattr(self, "runner", None)
            self._executor_broken = False
            try:
                if runner is not None and hasattr(runner, "reset"):
                    runner.reset()
                elif runner is not None:
                    runner.remove(list(runner.uid_to_seq.values()))
            except Exception:
                self._executor_broken = True
            self._inserted.clear()
            if runner is not None:
                uid_map = getattr(runner, "uid_to_seq", None)
                if uid_map is not None:
                    uid_map.clear()
        return self.fail_streams(exc)

    # -------------------------------------------------------------------- step
    def step(self) -> list[RequestOutput]:
        with self._lock:
            t0 = time.perf_counter()
            outs = self.scheduler.schedule()
            self._apply_preemptions(outs)
            self._step_prompt_tokens = 0
            self._step_decode_tokens = 0
            inserted, cache_before = self._insert_new(outs)

            if self.scheduler.num_unfinished == 0:
                self._collect_finished()
                return []

            prompt_responses, generation_responses = self.runner.step()
            deltas = self._process_prompt_responses(prompt_responses)
            deltas += self._process_generation_responses(generation_responses)
            self._collect_finished()

            elapsed = time.perf_counter() - t0
            self.num_steps += 1
            self.step_times.append(elapsed)
            if self.cfg.cost_calibration:
                self.cost_model.observe_step(outs.num_prefill_tokens, elapsed)
            self._record_exec_trace(outs, inserted, cache_before, elapsed)
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

    def _insert_new(self, outs: SchedulerOutputs):
        cache_before = self._cache_counters()
        inserted: list[str] = []
        for chunk in outs.chunks:
            seq = chunk.seq
            if not chunk.is_prefill or seq.request_id in self._inserted:
                continue
            cache, cached_tokens = None, 0
            prefix = None
            lookup_happened = False
            if self.prefix_cache is not None:
                hit = self.prefix_cache.peek(seq.prompt_token_ids)
                if hit is not None:
                    prefix, cached_tokens = hit
                    # Native snapshots can only be restored on a full registered
                    # prefix hit, and only when the request still has a suffix.
                    # If the *entire request* is cached (no suffix), the runner
                    # would have to recompute (no reliable rollback): skip the
                    # lookup so it is not counted as a cache hit.
                    native_no_suffix = (prefix.native is not None
                                        and cached_tokens == seq.prompt_len)
                    native_partial = (prefix.native is not None
                                      and cached_tokens != prefix.num_tokens)
                    decision = self.cost_model.estimate(
                        seq.prompt_len, cached_tokens, prefix.nbytes,
                        policy=self.cfg.prefix_reuse)
                    if decision.reuse and not native_partial and not native_no_suffix:
                        confirmed = self.prefix_cache.lookup(seq.prompt_token_ids)
                        if confirmed is not None:
                            prefix, cached_tokens = confirmed
                        lookup_happened = True
                        cache = self._seed_caches(prefix, cached_tokens)
                        seq.cached_prompt_tokens = cached_tokens
                        seq.prefix_blocks = []  # blocks owned by the shared prefix
                    else:
                        self.prefix_recompute_skips += 1
                        cached_tokens = 0
            try:
                uid = self.runner.insert(seq, cache=cache, cached_tokens=cached_tokens)
            except Exception:
                if cache is not None:
                    # shared-prefix reuse failed: retry without it rather than dropping
                    if lookup_happened:
                        self.prefix_cache.undo_hit(prefix)
                        lookup_happened = False
                    uid = self.runner.insert(seq)
                    seq.cached_prompt_tokens = 0
                    cached_tokens = 0
                else:
                    raise
            actual = int(getattr(seq, "cached_prompt_tokens", 0) or 0)
            if lookup_happened:
                if actual <= 0:
                    self.prefix_cache.undo_hit(prefix)
                else:
                    # count what the executor actually reused, not what was found
                    self.prefix_hit_tokens += actual
                    self.prefix_reuse_autos += 1
            self._inserted[seq.request_id] = uid
            inserted.append(seq.request_id)
        return inserted, cache_before

    def _cache_counters(self) -> dict:
        c = self.prefix_cache
        if c is None:
            return {"hits": 0, "stores": 0, "skips": 0, "hit_tokens": 0}
        return {"hits": c.num_hits, "stores": c.num_stores,
                "skips": self.prefix_recompute_skips,
                "hit_tokens": self.prefix_hit_tokens}

    def _record_exec_trace(self, outs: SchedulerOutputs, inserted: list[str],
                           cache_before: dict, elapsed: float) -> None:
        cache_now = self._cache_counters()
        prefill_chunks = [c for c in outs.chunks if c.is_prefill]
        actual_prompt = self._step_prompt_tokens
        actual_decode = self._step_decode_tokens
        t = {
            "step": self.num_steps,
            "scheduled_prefill_tokens": outs.num_prefill_tokens,
            "scheduled_decode_tokens": outs.num_decode_tokens,
            "scheduled_prefill_requests": len(prefill_chunks),
            "inserted_prefill_requests": len(inserted),
            "actual_prompt_tokens": actual_prompt,
            "actual_decode_tokens": actual_decode,
            "step_ms": round(elapsed * 1000, 2),
            "cache_hits": cache_now["hits"] - cache_before["hits"],
            "cache_stores": cache_now["stores"] - cache_before["stores"],
            "cache_skips": cache_now["skips"] - cache_before["skips"],
            "scheduled_ids": [c.seq.request_id for c in prefill_chunks],
            "inserted_ids": list(inserted),
        }
        self._exec_trace.append(t)
        if prefill_chunks and len(self._first_prefill_steps) < 12:
            self._first_prefill_steps.append(t)
        c = self._control_totals
        c["steps"] += 1
        c["scheduled_prefill_tokens"] += outs.num_prefill_tokens
        c["actual_prompt_tokens"] += actual_prompt
        c["scheduled_decode_tokens"] += outs.num_decode_tokens
        c["actual_decode_tokens"] += actual_decode
        c["scheduled_prefill_requests"] += len(prefill_chunks)
        c["inserted_prefill_requests"] += len(inserted)
        if prefill_chunks:
            c["steps_with_prefill"] += 1
            if actual_prompt != outs.num_prefill_tokens:
                c["steps_with_mismatch"] += 1

    def _seed_caches(self, prefix: SharedPrefix, hit_len: int) -> list:
        """Per-layer caches pre-seeded with the shared prefix.

        Dense K/V entries support partial block-aligned hits. Native snapshots
        (ArraysCache / RotatingKVCache / ...) are restored only on a full prefix
        hit; arbitrary truncation of hybrid/sliding caches is not supported yet.
        """
        from mlx_lm.models.cache import KVCache

        if prefix.native:
            if hit_len != prefix.num_tokens:
                return []
            restored = []
            for cls, state, meta in prefix.native:
                restored.append(cls.from_state(_clone_state(state), meta))
            return restored
        caches = []
        for k, v in prefix.layers:
            c = KVCache()
            c.keys = k[..., :hit_len, :]
            c.values = v[..., :hit_len, :]
            c.offset = hit_len
            caches.append(c)
        return caches

    @staticmethod
    def _native_snapshot(caches) -> list | None:
        """Capture native cache state only when the model is not plain KVCache."""
        from mlx_lm.models.cache import KVCache

        if all(isinstance(c, KVCache) for c in caches):
            return None
        snapshots = []
        for c in caches:
            try:
                snapshots.append((type(c), _clone_state(c.state), c.meta_state))
            except Exception:
                return None
        return snapshots

    def register_prefix(self, token_ids: list[int]) -> SharedPrefix | None:
        """Precompute a shared prefix so later requests skip prefilling it."""
        if self.prefix_cache is None:
            return None
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        # Decide the cache boundary BEFORE running any prefill. Native snapshots
        # cannot be truncated after the fact, so only process the block-aligned
        # token span that will actually be registered.
        bs = self.prefix_cache.block_manager.block_size
        n = aligned_prefix_length(len(token_ids), bs, self.prefix_cache.min_tokens)
        if n <= 0:
            return None
        token_ids = token_ids[:n]
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
        native = self._native_snapshot(caches)
        if native is not None:
            return self.prefix_cache.store(token_ids, [], native=native)
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
            prev = seq.num_computed_prompt_tokens
            end = bool(getattr(r, "end_of_prompt", False))
            new = seq.prompt_len if end else max(prev, processed)
            delta = max(0, new - prev)
            seq.num_computed_prompt_tokens = new
            self._step_prompt_tokens += delta
            if end:
                self.num_prefill_tokens += seq.prompt_len - seq.cached_prompt_tokens
                self._maybe_store_prefix(seq, self.runner.extract_cache(seq))
            else:
                self.num_prefill_tokens += delta
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
            first = seq.first_token_time is None
            seq.append_token(token, seq.sampling_params.max_tokens)
            if first:
                self.request_stages["first_token"] += 1
            self.num_decode_tokens += 1
            self._step_decode_tokens += 1
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
            out = self._outputs.get(seq.request_id)
            if out is None:  # already released by a watchdog/abort
                continue
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
        with self._stream_lock:
            q = self._streams.get(request_id)
            if q is not None:
                q.put(delta)

    def _collect_finished(self) -> None:
        for seq in self.scheduler.drain_finished():
            self.request_stages["finished"] += 1
            with self._stream_lock:
                out = self._outputs.get(seq.request_id)
                if out is not None:
                    out.finished = True
                    out.finish_reason = seq.finish_reason
                    out.token_ids = list(seq.output_token_ids)
                    out.ttft = seq.ttft()
                    out.num_output_tokens = len(seq.output_token_ids)
                    q = self._streams.get(seq.request_id)
                    if q is not None:
                        try:
                            q.put(out)
                        except Exception:
                            pass
                # the HTTP handler owns the queue and reads the final event; the
                # engine must not retain completed requests forever
                self._streams.pop(seq.request_id, None)
                self._outputs.pop(seq.request_id, None)

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
        st["mlx_lm_version"] = getattr(self, "_mlx_lm_version", None)
        if self.prefix_cache is not None:
            st["prefix_cache"] = self.prefix_cache.stats()
            st["prefix_cache"]["reuse_policy"] = self.cfg.prefix_reuse
            st["prefix_cache"]["recompute_skips"] = self.prefix_recompute_skips
            st["prefix_cache"]["reuse_autos"] = self.prefix_reuse_autos
        else:
            st["prefix_cache"] = {"enabled": False,
                                  "reuse_policy": self.cfg.prefix_reuse}
        st["scheduler_policy"] = self.cfg.scheduler_policy
        st["max_num_seqs"] = self.scheduler.cfg.max_num_seqs
        st["executor"] = {"completion_batch_size": self.cfg.completion_batch_size,
                          "prefill_batch_size": self.cfg.prefill_batch_size,
                          "prefill_step_size": self.cfg.prefill_chunk_size}
        st["concurrency_changes"] = list(self.concurrency_changes)
        st["request_stages"] = dict(self.request_stages)
        st["cost_model"] = self.cost_model.stats()
        c = dict(self._control_totals)
        c["actual_to_scheduled_prompt_ratio"] = round(
            c["actual_prompt_tokens"] / max(1, c["scheduled_prefill_tokens"]), 3)
        c["last_steps"] = list(self._exec_trace)[-20:]
        c["first_prefill_steps"] = list(self._first_prefill_steps)
        c["requests"] = list(self._request_costs.items())[-64:]
        st["execution_control"] = c
        return st

    def close(self) -> None:
        self.runner.close()
