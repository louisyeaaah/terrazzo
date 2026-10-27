"""OpenAI-compatible HTTP server with two runtimes.

Two ways to serve on one Mac:

* **batch** (default): :class:`Engine` — continuous batching via mlx-lm's
  BatchGenerator. Requests can share a registered prefix (seeded per-layer KV),
  but the prefix is copied per sequence on first growth.
* **paged** (`--paged`): :class:`PagedSession` — sequential execution, but the
  prefix K/V is stored **once** and shared by reference across all requests.

Both expose the same OpenAI-ish surface, so you can point a client at either.
"""

from __future__ import annotations

import faulthandler
import json
import os
import queue
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .sequence import SamplingParams

TokenDelta = dict


def _mlx_memory() -> dict:
    try:
        import mlx.core as mx

        return {
            "mlx_active_mb": round(mx.get_active_memory() / 1e6, 1),
            "mlx_peak_mb": round(mx.get_peak_memory() / 1e6, 1),
            "mlx_cache_mb": round(mx.get_cache_memory() / 1e6, 1),
        }
    except Exception:
        return {}


def apply_mlx_memory_limit(cap_mb: float | None) -> dict:
    """Enforce the memory cap inside MLX itself (allocator-level), not just by polling.

    ``mx.set_memory_limit`` makes an over-budget allocation raise instead of pushing
    the process/system into swap; the runtime's memory guard then answers 503.
    ``TERRAZZO_MLX_MEMORY_LIMIT=0`` disables the allocator limit.
    """
    if cap_mb is None:
        return {"enabled": False}
    if os.environ.get("TERRAZZO_MLX_MEMORY_LIMIT", "1") == "0":
        return {"enabled": False}
    import mlx.core as mx

    cap = int(cap_mb * 1e6)
    cache = int(cap * 0.25)
    try:
        mx.set_memory_limit(cap)
        mx.set_cache_limit(cache)
    except Exception as exc:  # pragma: no cover - platform dependent
        return {"enabled": False, "error": str(exc)}
    return {"enabled": True, "memory_limit_mb": round(cap / 1e6, 1),
            "cache_limit_mb": round(cache / 1e6, 1)}


class BaseRuntime:
    name = "base"

    def _memory_stats(self) -> dict:
        return _mlx_memory()

    tokenizer = None
    template_kwargs: dict = {}
    default_system: str | None = None
    default_params: dict | None = None
    model_path: str | None = None

    def submit(self, prompt_ids: list[int], params: SamplingParams,
               request_id: str | None = None,
               timeout: float | None = None) -> tuple[str, "queue.Queue"]:
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def abort(self, request_id: str) -> bool:
        """Best-effort request cancellation; subclasses override where possible."""
        return False

    def prewarm(self, system_text: str) -> dict:
        """Pre-register a system prompt as a shared prefix (Ollama-style "load")."""
        raise NotImplementedError("this runtime has no prefix cache")

    def concurrency(self) -> int | None:
        return None

    def set_concurrency(self, n: int) -> int:
        raise NotImplementedError("this runtime has no adjustable admission cap")

    # ---------------------------------------------------------------- watchdog
    def _start_watchdog(self) -> None:
        """Start the hang watchdog; worker loops must update ``_last_progress``."""
        self.hang_timeout_s = float(os.environ.get("TERRAZZO_HANG_TIMEOUT_S", "120"))
        self._last_progress = time.perf_counter()
        self._busy = False
        self._hung = False
        self._watchdog = _WorkerWatchdog(
            name=self.name,
            is_busy=lambda: bool(getattr(self, "_busy", False)),
            progress_age=lambda: time.perf_counter() - getattr(
                self, "_last_progress", time.perf_counter()),
            hang_timeout_s=self.hang_timeout_s,
            on_hang=self._on_worker_hang,
        )
        self._watchdog.start()

    def _worker_stats(self) -> dict:
        return {
            "busy": bool(getattr(self, "_busy", False)),
            "unhealthy": bool(getattr(self, "_unhealthy", False)),
            "hung": bool(getattr(self, "_hung", False)),
            "last_progress_age_s": round(
                time.perf_counter() - getattr(self, "_last_progress", time.perf_counter()), 1),
            "hang_timeout_s": getattr(self, "hang_timeout_s", None),
        }

    def _fail_outstanding(self, exc: BaseException) -> None:
        """Fail everything the runtime has queued/in flight (subclass hook)."""

    def _on_worker_hang(self) -> None:
        if getattr(self, "_unhealthy", False):
            return
        self._unhealthy = True
        self._hung = True
        exc = TimeoutError(
            f"MLX worker made no progress for {getattr(self, 'hang_timeout_s', 0):.0f}s")
        try:
            self._fail_outstanding(exc)
        except Exception:  # pragma: no cover - best effort only
            pass
        print(f"[watchdog] {self.name} worker hung; failing outstanding requests "
              f"and rejecting new ones", flush=True)
        if os.environ.get("TERRAZZO_HANG_RESTART") == "1":
            # process supervisor (launchd/systemd) restarts us; a hung Metal call
            # cannot be preempted from Python
            os._exit(70)

    # tokenizer helpers
    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, tokens: list[int]) -> str:
        try:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        except TypeError:
            return self.tokenizer.decode(tokens)

    def apply_chat_template(self, messages: list[dict], tools=None,
                            enable_thinking: bool | None = None) -> list[int]:
        kwargs = dict(self.template_kwargs)
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        if tools:
            kwargs["tools"] = tools
            try:
                return list(self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_dict=False, **kwargs))
            except TypeError as exc:
                raise ValueError(
                    "this model's chat template does not support tools; "
                    "use a tool-aware template (e.g. Qwen3)") from exc
        try:
            return list(self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False, **kwargs))
        except TypeError:
            return list(self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_dict=False))


def auto_prefix_cap(cfg: "EngineConfig") -> int:
    """Max tokens the server may auto-register as a shared prefix on first sight.

    Mirrors the configured snapshot cap so a 8k agent system prompt is registered
    whole; ``TERRAZZO_AUTO_PREFIX_MAX_TOKENS`` overrides it explicitly.
    """
    env_cap = os.environ.get("TERRAZZO_AUTO_PREFIX_MAX_TOKENS")
    if env_cap:
        return int(env_cap)
    return max(4096, int(cfg.prefix_cache_max_tokens))


# --------------------------------------------------------------------------- #
# batching runtime
# --------------------------------------------------------------------------- #
class EngineRuntime(BaseRuntime):
    name = "batch"

    def __init__(self, cfg: EngineConfig, system_text: str | None = None) -> None:
        self.engine: Engine | None = None
        self.prefix = None
        self._cfg = cfg
        self._system_text = system_text
        self.default_system = system_text
        self._pending: list[dict] = []
        self._pending_lock = threading.Lock()
        # experimental: auto-register a long prompt's prefix on first sight.
        # Default OFF: the block-hash entry currently matches only part of the
        # registered prefix (~896 of 2048 tokens in our measurement), so it can be
        # slower than `--system-file` pre-registration. Enable explicitly.
        # Auto-prefix: on the first batch, pre-register the prefix shared by the queued
        # requests (one prefill) so the whole batch only pays the suffix. Paid once;
        # `TERRAZZO_AUTO_PREFIX=0` disables it and relies on store-after-finish only.
        self.auto_prefix = os.environ.get("TERRAZZO_AUTO_PREFIX", "1") != "0"
        self.auto_prefix_min_tokens = 512
        # Follow the configured snapshot cap (auto: 8192 tokens on >=48 GB machines)
        # instead of a hard-coded 4096: an 8k agent system prompt used to be truncated
        # to 4096 tokens here, so the warm request still had to prefill the rest.
        self.auto_prefix_max_tokens = auto_prefix_cap(cfg)
        self._auto_prefix_stats = {"registrations": 0, "last_ms": None, "errors": []}
        # admission cadence: joining a running batch forces mlx-lm to merge prompt
        # caches, so admit arrivals in waves instead of one-by-one
        self._steps_since_admit = 0
        self.admit_batch_size = int(os.environ.get("TERRAZZO_ADMIT_BATCH", "4"))
        self.admit_wait_steps = int(os.environ.get("TERRAZZO_ADMIT_WAIT", "12"))
        # collect near-simultaneous arrivals before admitting them as one batch, so a
        # cost-aware scheduler actually sees the whole wave (0 keeps legacy behaviour)
        self.admit_settle_ms = float(os.environ.get("TERRAZZO_ADMIT_SETTLE_MS", "0"))
        self._pending_since = 0.0
        # load-aware cap selector (experimental): only changes admissions
        from .concurrency import AutoCap, AutoCapConfig

        self.auto_cap = None
        if os.environ.get("TERRAZZO_AUTO_CAP") == "1":
            cfg = AutoCapConfig(
                min_hold_s=float(os.environ.get("TERRAZZO_AUTO_MIN_HOLD", "6")),
                hit_high=float(os.environ.get("TERRAZZO_AUTO_HIT_HIGH", "0.75")),
                hit_low=float(os.environ.get("TERRAZZO_AUTO_HIT_LOW", "0.55")),
            )
            self.auto_cap = AutoCap(cfg, current=cfg.high_cap)
            self._auto_last_eval = 0.0
            self._auto_inputs = {"hit_ratio": 0.0, "queue": 0, "memory_ratio": 0.0}
        # memory guard: reject instead of letting Metal run out of memory
        import mlx.core as mxc
        self._mx = mxc
        cap_env = os.environ.get("TERRAZZO_MEMORY_CAP_MB", "")
        if cap_env:
            self.memory_cap_mb = float(cap_env)
        else:
            # default: 70% of physical RAM, the point where a runaway prefix cache or
            # KV pool should be contained instead of letting Metal die
            try:
                total_mb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e6
                self.memory_cap_mb = 0.7 * total_mb
            except Exception:
                self.memory_cap_mb = None
        self._json_jobs: list[dict] = []
        self._json_lock = threading.Lock()
        self._mlx_limits: dict = {}
        self._over_memory = False
        self._memory_rejections = 0
        self._memory_evictions = 0
        self._mem_peak_mb = 0.0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._control: "queue.Queue" = queue.Queue()
        # MLX streams are thread-local: the model must be created *and* stepped on
        # the same thread, so the whole engine lives in this worker thread.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=300):
            raise TimeoutError("engine worker did not initialise within 300s")
        if self._init_error is not None:
            raise self._init_error
        self._active_failures = 0
        self._unhealthy = False
        self._start_watchdog()

    def _loop(self) -> None:
        try:
            from .engine import Engine

            self.engine = Engine(self._cfg)
            self.tokenizer = self.engine.runner.tokenizer
            self._baseline_mb = self._mx.get_active_memory() / 1e6
            if self.memory_cap_mb is not None and self.memory_cap_mb < self._baseline_mb + 256:
                print(f"[memory-guard] cap {self.memory_cap_mb:.0f} MB is below model baseline "
                      f"{self._baseline_mb:.0f} MB; raising it to baseline+256 MB", flush=True)
                self.memory_cap_mb = self._baseline_mb + 256
            self._mlx_limits = apply_mlx_memory_limit(self.memory_cap_mb)
            if self._mlx_limits.get("enabled"):
                print(f"[memory-guard] MLX allocator limit {self._mlx_limits['memory_limit_mb']:.0f} MB "
                      f"(cache {self._mlx_limits['cache_limit_mb']:.0f} MB)", flush=True)
            if self._system_text:
                ids = self.apply_chat_template(
                    [{"role": "system", "content": self._system_text}])
                self.prefix = self.engine.register_prefix(ids)
        except BaseException as exc:  # surface init failures to the caller
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._last_mem_check = 0.0
        while not self._stop.is_set():
            while True:
                try:
                    control_job = self._control.get_nowait()
                except queue.Empty:
                    break
                try:
                    control_job()
                except BaseException:
                    pass
            self._last_progress = time.perf_counter()
            self._busy = bool(self.engine.has_unfinished())
            if time.time() - self._last_mem_check >= 1.0:
                self._last_mem_check = time.time()
                self._memory_check()
            if self.auto_cap is not None and time.time() - self._auto_last_eval >= 1.0:
                self._auto_last_eval = time.time()
                try:
                    self._auto_cap_tick()
                except Exception:
                    pass
            if self._pending_count():
                idle = not self.engine.has_unfinished()
                if (idle and self.admit_settle_ms > 0
                        and self._pending_count() < self.admit_batch_size):
                    now = time.monotonic()
                    if self._pending_since == 0.0:
                        self._pending_since = now
                    if now - self._pending_since < self.admit_settle_ms / 1000.0:
                        time.sleep(0.005)
                        continue
                if (idle or self._pending_count() >= self.admit_batch_size
                        or self._steps_since_admit >= self.admit_wait_steps):
                    slots = self._drain_pending()
                    if idle:
                        self._maybe_auto_register_batch(slots)
                    for slot in slots:
                        self._admit(slot)
                    self._steps_since_admit = 0
                    self._pending_since = 0.0
                    continue
            if not self.engine.has_unfinished() and self._pending_count() == 0:
                job = None
                with self._json_lock:
                    if self._json_jobs:
                        job = self._json_jobs.pop(0)
                if job is not None:
                    self._run_json_job(job)
                    continue
            if self.engine.has_unfinished():
                try:
                    self.engine.step()
                except RuntimeError as exc:
                    # Metal OOM and friends: drop caches, fail the in-flight requests
                    # explicitly (otherwise their HTTP handlers wait forever), and keep
                    # serving. After repeated failures the runtime refuses new work.
                    self._mx.clear_cache()
                    self._over_memory = True
                    self._memory_evictions += 1
                    self._active_failures += 1
                    failed = 0
                    try:
                        failed = self.engine.handle_failure(exc)
                    except Exception:
                        pass
                    with self._pending_lock:
                        pending, self._pending = self._pending, []
                    for slot in pending:
                        slot["error"] = exc
                        slot["event"].set()
                    failed += len(pending)
                    print(f"[engine] step failed ({self._active_failures}): {exc}; "
                          f"terminated {failed} request(s)", flush=True)
                    if self._active_failures >= 3:
                        self._unhealthy = True
                        print("[engine] too many consecutive step failures; new requests "
                              "will be rejected until restart", flush=True)
                else:
                    # reset only when step() returned successfully; the old code
                    # cleared the counter right after the exception branch, so it
                    # could never reach the three-consecutive-failure threshold
                    self._active_failures = 0
                self._steps_since_admit += 1
            else:
                time.sleep(0.002)

    def _fail_outstanding(self, exc: BaseException) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, []
        for slot in pending:
            slot["error"] = exc
            slot["event"].set()
        engine = getattr(self, "engine", None)
        if engine is not None:
            try:
                engine.fail_streams(exc)
            except Exception:
                pass

    def _run_json_job(self, job: dict) -> None:
        """Constrained JSON generation runs on the MLX worker thread while idle."""
        try:
            from .json_schema import json_generate

            text, stats = json_generate(
                self.engine.runner.model, self.engine.runner.tokenizer,
                job["prompt_ids"], job["schema"], max_tokens=job["max_tokens"])
            job["result"] = (text, stats)
        except BaseException as exc:  # surface to the waiting HTTP handler
            job["error"] = exc
        finally:
            job["event"].set()

    def generate_json(self, prompt_ids: list[int], schema: dict, *,
                      max_tokens: int = 512, timeout: float = 600.0):
        job = {"prompt_ids": list(prompt_ids), "schema": schema,
               "max_tokens": max_tokens, "event": threading.Event(),
               "result": None, "error": None}
        with self._json_lock:
            self._json_jobs.append(job)
        if not job["event"].wait(timeout):
            raise TimeoutError("json generation timed out")
        if job["error"] is not None:
            raise job["error"]
        return job["result"]

    def _memory_check(self) -> None:
        if self.memory_cap_mb is None:
            return
        active = self._mx.get_active_memory() / 1e6
        self._mem_peak_mb = max(self._mem_peak_mb, active)
        if active > self.memory_cap_mb:
            # shed what we can own: MLX cache + shared prefix snapshots
            self._mx.clear_cache()
            if self.engine is not None and self.engine.prefix_cache is not None:
                self.engine.prefix_cache.clear()
                self._memory_evictions += 1
            active = self._mx.get_active_memory() / 1e6
        over = active > self.memory_cap_mb
        if over and not self._over_memory:
            print(f"[memory-guard] over cap: active {active:.0f} MB > {self.memory_cap_mb:.0f} MB; "
                  f"rejecting new requests until it drops", flush=True)
        elif not over and self._over_memory:
            print(f"[memory-guard] recovered: active {active:.0f} MB", flush=True)
        self._over_memory = over

    def _pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    def _drain_pending(self) -> list[dict]:
        with self._pending_lock:
            slots, self._pending = self._pending, []
            return slots

    def _admit(self, slot: dict) -> None:
        """Runs on the MLX worker thread: add the request to the engine."""
        try:
            seq = self.engine.add_request(
                prompt_token_ids=slot["prompt_ids"], params=slot["params"],
                request_id=slot["rid"])
            slot["stream"] = self.engine.stream(seq.request_id)
        except BaseException as exc:  # never leave the HTTP handler hanging
            slot["error"] = exc
        finally:
            slot["event"].set()

    def _maybe_auto_register_batch(self, slots: list[dict]) -> None:
        """Register the prefix shared by this batch (agent sessions share the system prompt)."""
        prompts = [s["prompt_ids"] for s in slots if s.get("prompt_ids")]
        if not prompts:
            return
        common = prompts[0]
        for other in prompts[1:]:
            n = 0
            for a, b in zip(common, other):
                if a != b:
                    break
                n += 1
            common = common[:n]
            if len(common) < self.auto_prefix_min_tokens:
                return
        self._maybe_auto_register(common)

    def _maybe_auto_register(self, prompt_ids: list[int]) -> None:
        """Register a long prompt's prefix once so later sessions share it (agent traffic)."""
        if not self.auto_prefix or self.engine is None:
            return
        if len(prompt_ids) < self.auto_prefix_min_tokens:
            return
        cache = self.engine.prefix_cache
        if cache is None:
            return
        if cache.lookup(prompt_ids) is not None:
            return
        if self.engine._shares_cached_prefix(prompt_ids):
            return
        bs = self.engine.block_manager.block_size
        n = min(len(prompt_ids) - 1, self.auto_prefix_max_tokens)
        n = (n // bs) * bs
        if n < bs:
            return
        t0 = time.perf_counter()
        try:
            self.prefix = self.engine.register_prefix(prompt_ids[:n])
            self._auto_prefix_stats["registrations"] += 1
            self._auto_prefix_stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as exc:  # pragma: no cover - surfaced via /stats
            self._auto_prefix_stats["errors"].append(f"{type(exc).__name__}: {exc}")

    def submit(self, prompt_ids, params, request_id=None, timeout=None):
        if getattr(self, "_unhealthy", False):
            raise MemoryError("engine worker is unhealthy after repeated step failures; "
                              "restart the server")
        if self._over_memory:
            self._memory_rejections += 1
            raise MemoryError(
                f"server over memory cap ({self.memory_cap_mb:.0f} MB); retry later")
        rid = request_id or f"req-{uuid.uuid4().hex[:10]}"
        slot = {"rid": rid, "prompt_ids": prompt_ids, "params": params,
                "event": threading.Event(), "stream": None, "error": None}
        max_pending = int(os.environ.get("TERRAZZO_MAX_PENDING", "256"))
        with self._pending_lock:
            if len(self._pending) >= max_pending:
                raise MemoryError(
                    f"pending request queue is full ({max_pending} requests); retry later")
            self._pending.append(slot)
        wait_s = 900.0 if timeout is None else max(0.001, float(timeout))
        if not slot["event"].wait(timeout=wait_s):
            with self._pending_lock:
                try:
                    self._pending.remove(slot)
                except ValueError:
                    pass  # already drained by the worker; abort it below
            engine = getattr(self, "engine", None)
            if engine is not None:
                try:
                    engine.abort(rid)
                except Exception:
                    pass
            raise TimeoutError(f"request {rid} was not admitted within {wait_s:.1f}s")
        if slot.get("error") is not None:
            raise slot["error"]
        if slot.get("stream") is None:
            raise RuntimeError(f"request {rid} was admitted without a stream")
        return rid, slot["stream"]

    def _run_on_worker(self, fn, timeout: float = 900.0):
        ev = threading.Event()
        box: dict = {}

        def job():
            try:
                box["result"] = fn()
            except BaseException as exc:  # pragma: no cover - surfaced to caller
                box["error"] = exc
            finally:
                ev.set()

        self._control.put(job)
        if not ev.wait(timeout):
            raise TimeoutError("engine worker did not run the control job in time")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def concurrency(self) -> int | None:
        engine = getattr(self, "engine", None)
        return engine.scheduler.cfg.max_num_seqs if engine is not None else None

    def set_concurrency(self, n: int) -> int:
        return self._run_on_worker(lambda: self.engine.set_max_num_seqs(n))

    def _auto_cap_tick(self) -> None:
        engine = getattr(self, "engine", None)
        if engine is None or self.auto_cap is None:
            return
        # Recent admitted requests are biased by the current cap: with cap4 a mixed
        # phase can show only cached requests while unique ones wait outside the
        # engine. Include the pending queue so the decision sees the real mix.
        costs = list(engine._request_costs.values())[-self.auto_cap.cfg.window:]
        try:
            with self._pending_lock:
                slots = list(self._pending)
            cache = getattr(engine, "prefix_cache", None)
            for slot in slots[:self.auto_cap.cfg.window]:
                ids = slot.get("prompt_ids") or []
                if not ids:
                    continue
                cached = 0
                if cache is not None:
                    hitp = cache.peek(ids)
                    if hitp is not None:
                        cached = hitp[1]
                costs.append({"prompt_tokens": len(ids),
                              "cached_estimate": cached,
                              "uncached_estimate": len(ids) - cached})
        except Exception:
            pass
        hit, avg_uncached, samples = self.auto_cap.observe_inputs(costs)
        queue = len(engine.scheduler.waiting) + self._pending_count()
        cap_mb = self.memory_cap_mb or 1.0
        mem_ratio = 0.0
        try:
            mem_ratio = (self._mx.get_active_memory() / 1e6) / max(cap_mb, 1.0)
        except Exception:
            pass
        self._auto_inputs = {"hit_ratio": round(hit, 3), "avg_uncached": avg_uncached,
                             "samples": samples, "queue": queue,
                             "memory_ratio": round(mem_ratio, 3)}
        target = self.auto_cap.decide(now=time.monotonic(), hit_ratio=hit,
                                      avg_uncached=avg_uncached, queue=queue,
                                      memory_ratio=mem_ratio, samples=samples)
        if target is not None:
            engine.set_max_num_seqs(target)
            print(f"[auto-cap] -> {target} (hit={hit:.2f} uncached={avg_uncached} "
                  f"queue={queue} mem={mem_ratio:.2f})", flush=True)

    def prewarm(self, system_text: str) -> dict:
        """Register a system prompt now so the first request already hits the cache."""
        if not system_text or not system_text.strip():
            raise ValueError("system prompt is empty")

        def job():
            ids = self.apply_chat_template([{"role": "system", "content": system_text}])
            t0 = time.perf_counter()
            prefix = self.engine.register_prefix(ids)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            if prefix is None:
                return {"registered": False, "tokens": 0, "ms": ms}
            self.prefix = prefix
            return {"registered": True, "tokens": prefix.num_tokens, "ms": ms}

        return self._run_on_worker(job)

    def stats(self) -> dict:
        st = self.engine.stats()
        st["runtime"] = self.name
        st["prefix_registered_tokens"] = self.prefix.num_tokens if self.prefix else 0
        st["auto_prefix"] = {"enabled": self.auto_prefix,
                             "pending": len(self._pending), **self._auto_prefix_stats}
        st["memory_guard"] = {"cap_mb": self.memory_cap_mb, "over_memory": self._over_memory,
                              "rejections": self._memory_rejections,
                              "evictions": self._memory_evictions,
                              "peak_active_mb": round(self._mem_peak_mb, 1),
                              "mlx_allocator_limit": self._mlx_limits}
        st["worker"] = self._worker_stats()
        engine = getattr(self, "engine", None)
        if engine is not None:
            st["completion_batch_size"] = engine.cfg.completion_batch_size
            st["admission"] = {"batch_size": self.admit_batch_size,
                               "wait_steps": self.admit_wait_steps,
                               "settle_ms": self.admit_settle_ms}
        if self.auto_cap is not None:
            st["auto_concurrency"] = {**self.auto_cap.stats(),
                                      "inputs": dict(self._auto_inputs)}
        else:
            st["auto_concurrency"] = {"enabled": False}
        st.update(self._memory_stats())
        return st

    def abort(self, request_id: str) -> bool:
        if self.engine is None:
            return False
        try:
            return bool(self.engine.abort(request_id))
        except Exception:
            return False

    def close(self) -> None:
        self._stop.set()
        watchdog = getattr(self, "_watchdog", None)
        if watchdog is not None:
            watchdog.stop()
        self._thread.join(timeout=2)
        self.engine.close()


# --------------------------------------------------------------------------- #
# paged runtime (shared prefix by reference, sequential)
# --------------------------------------------------------------------------- #
class PagedRuntime(BaseRuntime):
    name = "paged"

    def __init__(self, cfg: EngineConfig, system_text: str | None = None) -> None:
        self.tokenizer = None
        self.session = None
        self.prefix = None
        self._cfg = cfg
        self._system_text = system_text
        self.default_system = system_text
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._control: "queue.Queue" = queue.Queue()
        self.num_requests = 0
        self.num_tokens = 0
        # MLX streams are thread-local: load + generate on the same thread.
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=300):
            raise TimeoutError("engine worker did not initialise within 300s")
        if self._init_error is not None:
            raise self._init_error
        self._active_failures = 0
        self._unhealthy = False
        self._active = None
        self._start_watchdog()

    def _init(self) -> None:
        from mlx_lm import load as mlx_load

        from .paged_session import PagedSession

        model, tok = mlx_load(self._cfg.model)
        self.tokenizer = tok
        self.session = PagedSession(
            model, tok,
            block_size=self._cfg.block_size,
            num_blocks=self._cfg.num_gpu_blocks,
            eos_token_ids=tuple(tok.eos_token_ids),
        )
        if self._system_text:
            ids = self.apply_chat_template(
                [{"role": "system", "content": self._system_text}])
            self.prefix = self.session.register_prefix(ids)

    def _fail_outstanding(self, exc: BaseException) -> None:
        active = getattr(self, "_active", None)
        if active is not None:
            request_id, q = active
            try:
                q.put({"request_id": request_id, "token_ids": [], "finished": True,
                       "finish_reason": "error", "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _loop(self) -> None:
        try:
            self._init()
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            request_id, prompt_ids, params, q = item
            self.num_requests += 1
            self._last_progress = time.perf_counter()
            self._busy = True
            self._active = (request_id, q)

            def on_token(tok_id: int, q=q, request_id=request_id) -> None:
                self.num_tokens += 1
                self._last_progress = time.perf_counter()
                q.put({"request_id": request_id, "token_ids": [tok_id], "finished": False,
                       "finish_reason": None})

            try:
                stats = self.session.generate(
                    prompt_ids,
                    max_tokens=params.max_tokens,
                    temperature=params.temperature,
                    on_token=on_token,
                )
                q.put({"request_id": request_id, "token_ids": [], "finished": True,
                       "finish_reason": "stop" if stats.decode_tokens else "length",
                       "stats": stats.as_dict()})
                self._active_failures = 0
            except Exception as exc:  # keep the server alive
                q.put({"request_id": request_id, "token_ids": [], "finished": True,
                       "finish_reason": "error", "error": str(exc)})
                self._active_failures += 1
                if self._active_failures >= 3:
                    self._unhealthy = True
            finally:
                self._active = None
                self._busy = False
                self._last_progress = time.perf_counter()

    def submit(self, prompt_ids, params, request_id=None, timeout=None):
        if getattr(self, "_unhealthy", False):
            raise MemoryError(f"{self.name} runtime is unhealthy (worker failure/hang); "
                              f"restart the server")
        request_id = request_id or f"req-{uuid.uuid4().hex[:10]}"
        q: "queue.Queue" = queue.Queue()
        self._queue.put((request_id, prompt_ids, params, q))
        return request_id, q

    def stats(self) -> dict:
        st = self.session.stats()
        st["runtime"] = self.name
        st["requests"] = getattr(self, "num_requests", 0)
        st["tokens"] = getattr(self, "num_tokens", 0)
        st["queue_depth"] = self._queue.qsize()
        st["auto_prefix"] = {"enabled": self.auto_prefix, **self._auto_prefix_stats}
        st["prefix_registered_tokens"] = self.prefix.num_tokens if self.prefix else 0
        st["worker"] = self._worker_stats()
        engine = getattr(self, "engine", None)
        if engine is not None:
            st["completion_batch_size"] = engine.cfg.completion_batch_size
            st["admission"] = {"batch_size": self.admit_batch_size,
                               "wait_steps": self.admit_wait_steps,
                               "settle_ms": self.admit_settle_ms}
        if self.auto_cap is not None:
            st["auto_concurrency"] = {**self.auto_cap.stats(),
                                      "inputs": dict(self._auto_inputs)}
        else:
            st["auto_concurrency"] = {"enabled": False}
        st.update(self._memory_stats())
        return st

    def close(self) -> None:
        self._stop.set()
        watchdog = getattr(self, "_watchdog", None)
        if watchdog is not None:
            watchdog.stop()
        self._queue.put(None)
        self._thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# pool runtime (v0.4: true paged KV, shared prefix by block reference)
# --------------------------------------------------------------------------- #
class PoolRuntime(BaseRuntime):
    name = "pool"

    def __init__(self, cfg: EngineConfig, system_text: str | None = None,
                 use_metal: bool = False) -> None:
        self.tokenizer = None
        self.session = None
        self.prefix = None
        self.use_metal = use_metal
        self._cfg = cfg
        self._system_text = system_text
        self.default_system = system_text
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._init_error: BaseException | None = None
        self._control: "queue.Queue" = queue.Queue()
        self.num_requests = 0
        self.num_tokens = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=300):
            raise TimeoutError("engine worker did not initialise within 300s")
        if self._init_error is not None:
            raise self._init_error
        self._active_failures = 0
        self._unhealthy = False
        self._active = None
        self._start_watchdog()

    def _init(self) -> None:
        from mlx_lm import load as mlx_load

        from .pool_session import PoolSession

        model, tok = mlx_load(self._cfg.model)
        self.tokenizer = tok
        self.session = PoolSession(
            model, tok,
            block_size=self._cfg.block_size,
            num_blocks=self._cfg.num_gpu_blocks,
            eos_token_ids=tuple(tok.eos_token_ids),
        )
        if self.use_metal:
            from .pool_attn_patch import patched_qwen3_attention

            self._patch = patched_qwen3_attention()
            self._patch.__enter__()
        if self._system_text:
            ids = self.apply_chat_template(
                [{"role": "system", "content": self._system_text}])
            self.prefix = self.session.register_prefix(ids)

    def _fail_outstanding(self, exc: BaseException) -> None:
        active = getattr(self, "_active", None)
        if active is not None:
            request_id, q = active
            try:
                q.put({"request_id": request_id, "token_ids": [], "finished": True,
                       "finish_reason": "error", "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _loop(self) -> None:
        try:
            self._init()
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
            return
        self._ready.set()
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            request_id, prompt_ids, params, q = item
            self.num_requests += 1
            self._last_progress = time.perf_counter()
            self._busy = True
            self._active = (request_id, q)

            def on_token(tok_id: int, q=q, request_id=request_id) -> None:
                self.num_tokens += 1
                self._last_progress = time.perf_counter()
                q.put({"request_id": request_id, "token_ids": [tok_id], "finished": False,
                       "finish_reason": None})

            try:
                stats = self.session.generate(
                    prompt_ids, max_tokens=params.max_tokens,
                    temperature=params.temperature, on_token=on_token,
                )
                q.put({"request_id": request_id, "token_ids": [], "finished": True,
                       "finish_reason": "stop" if stats.decode_tokens else "length",
                       "stats": stats.as_dict()})
                self._active_failures = 0
            except Exception as exc:
                q.put({"request_id": request_id, "token_ids": [], "finished": True,
                       "finish_reason": "error", "error": str(exc)})
                self._active_failures += 1
                if self._active_failures >= 3:
                    self._unhealthy = True
            finally:
                self._active = None
                self._busy = False
                self._last_progress = time.perf_counter()

    def submit(self, prompt_ids, params, request_id=None, timeout=None):
        if getattr(self, "_unhealthy", False):
            raise MemoryError(f"{self.name} runtime is unhealthy (worker failure/hang); "
                              f"restart the server")
        request_id = request_id or f"req-{uuid.uuid4().hex[:10]}"
        q: "queue.Queue" = queue.Queue()
        self._queue.put((request_id, prompt_ids, params, q))
        return request_id, q

    def stats(self) -> dict:
        st = self.session.stats() if self.session is not None else {}
        st["runtime"] = self.name
        st["metal"] = self.use_metal
        st["requests"] = getattr(self, "num_requests", 0)
        st["tokens"] = getattr(self, "num_tokens", 0)
        st["queue_depth"] = self._queue.qsize()
        st["auto_prefix"] = {"enabled": self.auto_prefix, **self._auto_prefix_stats}
        st["prefix_registered_tokens"] = self.prefix.num_tokens if self.prefix else 0
        st["worker"] = self._worker_stats()
        engine = getattr(self, "engine", None)
        if engine is not None:
            st["completion_batch_size"] = engine.cfg.completion_batch_size
            st["admission"] = {"batch_size": self.admit_batch_size,
                               "wait_steps": self.admit_wait_steps,
                               "settle_ms": self.admit_settle_ms}
        if self.auto_cap is not None:
            st["auto_concurrency"] = {**self.auto_cap.stats(),
                                      "inputs": dict(self._auto_inputs)}
        else:
            st["auto_concurrency"] = {"enabled": False}
        st.update(self._memory_stats())
        return st

    def close(self) -> None:
        self._stop.set()
        watchdog = getattr(self, "_watchdog", None)
        if watchdog is not None:
            watchdog.stop()
        self._queue.put(None)
        self._thread.join(timeout=2)
        if getattr(self, "_patch", None) is not None:
            try:
                self._patch.__exit__(None, None, None)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
def _sampling_from_body(body: dict, defaults: dict | None = None) -> SamplingParams:
    source = dict(defaults or {})
    source.update({k: v for k, v in body.items() if v is not None})
    return SamplingParams(
        max_tokens=int(source.get("max_tokens") or source.get("max_completion_tokens") or 128),
        temperature=float(source.get("temperature", 0.0)),
        top_p=float(source.get("top_p", 1.0)),
        top_k=int(source.get("top_k", 0)),
    )


class _WorkerWatchdog(threading.Thread):
    """Detect a worker thread that is busy but makes no progress (hung Metal call)."""

    def __init__(self, name: str, is_busy, progress_age, hang_timeout_s: float,
                 on_hang, interval: float = 1.0):
        super().__init__(name=f"watchdog-{name}", daemon=True)
        self._is_busy = is_busy
        self._progress_age = progress_age
        self._hang_timeout_s = float(hang_timeout_s)
        self._on_hang = on_hang
        self._interval = interval
        self._stopped = threading.Event()

    def run(self) -> None:
        while not self._stopped.wait(self._interval):
            try:
                if self._is_busy() and self._progress_age() > self._hang_timeout_s:
                    self._on_hang()
                    return
            except Exception:  # pragma: no cover - watchdog must never crash
                pass

    def stop(self) -> None:
        self._stopped.set()


class PayloadTooLarge(Exception):
    def __init__(self, size: int):
        self.size = size
        super().__init__(f"request body too large: {size} bytes")


def make_handler(runtime: BaseRuntime, *, api_key: str | None = None,
                 rate_limit_rps: float = 0.0, request_timeout_s: float = 600.0,
                 max_body_bytes: int = 1 << 20, idle_timeout_s: float = 30.0):
    bucket = {"tokens": float(rate_limit_rps or 0.0), "last": time.time()}
    bucket_lock = threading.Lock()

    def rate_ok() -> bool:
        if not rate_limit_rps:
            return True
        with bucket_lock:
            now = time.time()
            bucket["tokens"] = min(rate_limit_rps,
                                   bucket["tokens"] + (now - bucket["last"]) * rate_limit_rps)
            bucket["last"] = now
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
        return False

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # socket timeout for the whole request (covers both blocking reads and writes,
        # e.g. slowloris clients and clients that stop reading a stream)
        timeout = idle_timeout_s

        def log_message(self, fmt, *args):
            pass

        # ------------------------------------------------------------ helpers
        def _authorized(self) -> bool:
            if not api_key:
                return True
            import hmac

            header = self.headers.get("Authorization", "")
            return hmac.compare_digest(header.encode(), f"Bearer {api_key}".encode())

        def handle(self) -> None:
            try:
                super().handle()
            except (TimeoutError, ConnectionResetError, BrokenPipeError):
                # socket timeout (idle/slow client) or client went away mid-response
                self.close_connection = True

        def _json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse(self, obj: dict) -> None:
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        @staticmethod
        def _norm(delta) -> dict:
            """Runtime queues may carry dicts (paged) or RequestOutput (batch)."""
            if isinstance(delta, dict):
                return delta
            return {
                "token_ids": list(getattr(delta, "token_ids", []) or []),
                "finished": bool(getattr(delta, "finished", False)),
                "finish_reason": getattr(delta, "finish_reason", None),
            }

        def _deadline_reset(self) -> None:
            self._deadline = time.perf_counter() + request_timeout_s

        def _remaining(self) -> float:
            deadline = getattr(self, "_deadline", None)
            if deadline is None:
                return request_timeout_s
            return deadline - time.perf_counter()

        def _submit(self, prompt_ids, params, request_id):
            return runtime.submit(prompt_ids, params, request_id=request_id,
                                  timeout=max(0.001, self._remaining()))

        def _qget(self, q, request_id):
            remaining = self._remaining()
            if remaining <= 0:
                raise TimeoutError(f"request {request_id} timed out")
            try:
                return q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"request {request_id} exceeded the {request_timeout_s:.0f}s deadline")

        def _read_body(self) -> dict:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("chunked request bodies are not supported")
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                raise ValueError("Content-Length required")
            try:
                n = int(raw_len)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if n < 0:
                raise ValueError("invalid Content-Length")
            if n > max_body_bytes:
                raise PayloadTooLarge(n)
            data = self.rfile.read(n) or b"{}"
            return json.loads(data)

        # ------------------------------------------------------- Ollama API ---
        def _start_ndjson(self) -> None:
            if getattr(self, "_ndjson_started", False):
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self._ndjson_started = True

        def _nd(self, obj: dict) -> None:
            self._start_ndjson()
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        def _ollama_name(self, body: dict) -> str:
            path = getattr(runtime, "model_path", None) or "terrazzo"
            return body.get("model") or Path(path.rstrip("/")).name or "terrazzo"

        def _ollama_error(self, code: int, message: str) -> None:
            self._json(code, {"error": message})

        @staticmethod
        def _iso_now() -> str:
            from datetime import datetime, timezone

            return datetime.now(timezone.utc).isoformat()

        def _ollama_timing(self, t0: float, prompt_ids, tokens) -> dict:
            dur = int((time.perf_counter() - t0) * 1e9)
            return {
                "total_duration": dur,
                "load_duration": 0,
                "prompt_eval_count": len(prompt_ids),
                "prompt_eval_duration": 0,
                "eval_count": len(tokens),
                "eval_duration": dur,
            }

        def _ollama_emit(self, body: dict, prompt_ids, *, chat: bool, params, stop,
                         json_schema, tools=None) -> None:
            from .tools import parse_tool_calls

            model = self._ollama_name(body)
            stream = body.get("stream", True) is not False
            think = body.get("think")
            if think is None:
                think = body.get("thinking")
            think = bool(think)
            t0 = time.perf_counter()
            stop = [s for s in (stop or []) if s]

            def frame(done: bool = False, **kw) -> dict:
                out = {"model": model, "created_at": self._iso_now(), "done": done}
                out.update(kw)
                return out

            if json_schema is not None:
                try:
                    text, stats = runtime.generate_json(
                        prompt_ids, json_schema, max_tokens=params.max_tokens)
                except MemoryError as exc:
                    return self._ollama_error(503, str(exc))
                except Exception as exc:
                    return self._ollama_error(500, f"generation failed: {exc}")
                n = int(stats.get("steps", 0))
                body_msg = ({"message": {"role": "assistant", "content": text}} if chat
                            else {"response": text})
                if stream:
                    self._nd(frame(**body_msg))
                    self._nd(frame(done=True, done_reason="stop", eval_count=n,
                                   eval_duration=int((time.perf_counter() - t0) * 1e9),
                                   total_duration=int((time.perf_counter() - t0) * 1e9),
                                   prompt_eval_count=len(prompt_ids)))
                else:
                    self._json(200, frame(done=True, done_reason="stop", eval_count=n, **body_msg))
                return
            try:
                _rid, q = self._submit(prompt_ids, params, body.get("request_id"))
            except MemoryError as exc:
                return self._ollama_error(503, str(exc))
            except TimeoutError as exc:
                return self._ollama_error(504, str(exc))
            except Exception as exc:
                return self._ollama_error(500, f"submit failed: {exc}")

            if tools:
                # tool calls are emitted as one message after a buffered generation
                tokens: list[int] = []
                try:
                    while True:
                        d = self._norm(self._qget(q, _rid))
                        if d.get("error"):
                            raise RuntimeError(str(d["error"]))
                        ids = d.get("token_ids") or []
                        if d.get("finished") and len(ids) > 1:
                            tokens = list(ids)
                        else:
                            tokens.extend(ids)
                        if d.get("finished"):
                            break
                except TimeoutError as exc:
                    runtime.abort(_rid)
                    return self._ollama_error(504, str(exc))
                except RuntimeError as exc:
                    runtime.abort(_rid)
                    return self._ollama_error(500, str(exc))
                from .tools import split_thinking

                thinking_text, answer_text = split_thinking(runtime.decode(tokens))
                cleaned, calls = parse_tool_calls(answer_text or runtime.decode(tokens))
                msg = {"role": "assistant", "content": cleaned}
                if calls:
                    msg["tool_calls"] = calls
                if think and thinking_text:
                    msg["thinking"] = thinking_text
                timing = self._ollama_timing(t0, prompt_ids, tokens)
                if stream:
                    if cleaned or calls:
                        self._nd(frame(message=msg))
                    self._nd(frame(done=True, done_reason="stop", **timing))
                else:
                    self._json(200, frame(done=True, done_reason="stop", message=msg, **timing))
                return

            from .tools import split_thinking

            out: list[int] = []
            printed = 0
            printed_think = 0
            stopped = False
            try:
                while True:
                    d = self._norm(self._qget(q, _rid))
                    if d.get("error"):
                        raise RuntimeError(str(d["error"]))
                    ids = d.get("token_ids") or []
                    if d.get("finished") and len(ids) > 1:
                        out = list(ids)
                    else:
                        out.extend(ids)
                    text = runtime.decode(out)
                    safe = text.rstrip("\ufffd")
                    thinking_text, answer_text = split_thinking(safe)
                    # hold back a partial "<think>" tag so it never leaks as content
                    for n in range(6, 0, -1):
                        if answer_text.endswith("<think>"[:n]) and not answer_text.endswith("<think>"):
                            answer_text = answer_text[:-n]
                    delta = answer_text[printed:]
                    if stop:
                        cut = len(answer_text)
                        hit = False
                        for marker in stop:
                            idx = answer_text.find(marker)
                            if idx >= 0:
                                cut = min(cut, idx)
                                hit = True
                        if cut < len(answer_text):
                            delta = answer_text[printed:cut]
                        if hit:
                            stopped = True
                    think_delta = thinking_text[printed_think:] if think else ""
                    if delta and stream:
                        if chat:
                            self._nd(frame(message={"role": "assistant", "content": delta}))
                        else:
                            self._nd(frame(response=delta))
                        printed = len(answer_text) if not stopped else printed + len(delta)
                    elif delta:
                        printed = len(answer_text)
                    if think_delta and stream:
                        if chat:
                            self._nd(frame(message={"role": "assistant", "content": "",
                                                    "thinking": think_delta}))
                        else:
                            self._nd(frame(thinking=think_delta))
                    printed_think = len(thinking_text)
                    if d.get("finished") or stopped:
                        break
            except (TimeoutError, RuntimeError) as exc:
                runtime.abort(_rid)
                if stream:
                    self._nd(frame(done=True, done_reason="error", error=str(exc)))
                else:
                    self._ollama_error(504 if isinstance(exc, TimeoutError) else 500,
                                       str(exc))
                return
            timing = self._ollama_timing(t0, prompt_ids, out)
            done_reason = "stop"
            if stream:
                self._nd(frame(done=True, done_reason=done_reason, **timing))
            else:
                thinking_text, text = split_thinking(runtime.decode(out))
                if stop:
                    for marker in stop:
                        idx = text.find(marker)
                        if idx >= 0:
                            text = text[:idx]
                if chat:
                    msg = {"role": "assistant", "content": text}
                    if think and thinking_text:
                        msg["thinking"] = thinking_text
                    payload = {"message": msg}
                else:
                    payload = {"response": text}
                    if think and thinking_text:
                        payload["thinking"] = thinking_text
                self._json(200, frame(done=True, done_reason=done_reason, **payload, **timing))

        def _ollama_options(self, body: dict):
            options = dict(body.get("options") or {})
            params = _sampling_from_body(options, defaults=getattr(runtime, "default_params", None))
            stop = options.get("stop", body.get("stop")) or []
            if isinstance(stop, str):
                stop = [stop]
            return params, [str(s) for s in stop]

        def _ollama_format(self, body: dict):
            from .json_schema import SchemaError, compile_schema

            fmt = body.get("format")
            if fmt is None:
                return None
            if fmt in ("json", "json_object"):
                return {}
            if isinstance(fmt, dict):
                try:
                    return compile_schema(fmt)
                except SchemaError as exc:
                    raise ValueError(f"invalid format schema: {exc}") from exc
            raise ValueError(f"unsupported format: {fmt!r}")

        def _ollama_chat(self, body: dict) -> None:
            from .tools import (apply_tool_choice, normalize_messages,
                                validate_tools)

            messages = normalize_messages(body.get("messages") or [])
            if not messages:
                return self._ollama_error(400, "messages are required")
            default_system = getattr(runtime, "default_system", None)
            if default_system and not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": default_system}] + messages
            tools = None
            extra: list[dict] = []
            if body.get("tools"):
                try:
                    tools = validate_tools(body["tools"])
                    tools, extra = apply_tool_choice(tools, body.get("tool_choice"))
                except ValueError as exc:
                    return self._ollama_error(400, str(exc))
            if extra:
                messages = messages + extra
            think = body.get("think")
            if think is None:
                think = body.get("thinking")
            thinking = bool(think) if isinstance(think, bool) else False
            try:
                prompt_ids = runtime.apply_chat_template(
                    messages, tools=tools, enable_thinking=thinking)
            except ValueError as exc:
                return self._ollama_error(400, str(exc))
            except Exception:
                prompt_ids = runtime.encode(
                    "\n".join(str(m.get("content", "")) for m in messages))
            try:
                params, stop = self._ollama_options(body)
                json_schema = self._ollama_format(body)
            except ValueError as exc:
                return self._ollama_error(400, str(exc))
            self._ollama_emit(body, prompt_ids, chat=True, params=params, stop=stop,
                              json_schema=json_schema, tools=tools or None)

        def _ollama_generate(self, body: dict) -> None:
            if body.get("template"):
                return self._ollama_error(
                    501, "Modelfile TEMPLATE is not supported: MLX models use their "
                         "own chat template")
            prompt = body.get("prompt") or ""
            if body.get("raw"):
                prompt_ids = runtime.encode(prompt)
            else:
                messages = []
                system = body.get("system") or getattr(runtime, "default_system", None)
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                think_flag = body.get("think")
                if think_flag is None:
                    think_flag = body.get("thinking")
                think_flag = bool(think_flag) if isinstance(think_flag, bool) else False
                try:
                    prompt_ids = runtime.apply_chat_template(
                        messages, enable_thinking=think_flag)
                except Exception:
                    prompt_ids = runtime.encode(prompt)
            try:
                params, stop = self._ollama_options(body)
                json_schema = self._ollama_format(body)
            except ValueError as exc:
                return self._ollama_error(400, str(exc))
            self._ollama_emit(body, prompt_ids, chat=False, params=params, stop=stop,
                              json_schema=json_schema)

        def _ollama_show(self, body: dict) -> None:
            from .models import show as store_show

            name = body.get("model") or body.get("name")
            if not name:
                return self._ollama_error(400, "model is required")
            try:
                info = store_show(name)
            except FileNotFoundError as exc:
                return self._ollama_error(404, str(exc))
            recipe = info.get("recipe") or {}
            params = recipe.get("parameters") or {}
            modelfile = info.get("modelfile") or f"FROM {info['path']}"
            cfg = info.get("config") or {}
            self._json(200, {
                "modelfile": modelfile,
                "parameters": "\n".join(f"{k} {v}" for k, v in params.items()),
                "template": "",
                "details": {"format": "mlx",
                            "family": cfg.get("model_type") or "mlx",
                            "parameter_size": "",
                            "quantization_level": ""},
                "model_info": cfg,
            })

        def _ollama_api(self, path: str, body: dict) -> None:
            from .models import copy as store_copy
            from .models import create as store_create
            from .models import remove as store_remove

            if path == "/api/chat":
                return self._ollama_chat(body)
            if path == "/api/generate":
                return self._ollama_generate(body)
            if path == "/api/show":
                return self._ollama_show(body)
            if path in ("/api/delete", "/api/rm"):
                name = body.get("model") or body.get("name")
                if not name:
                    return self._ollama_error(400, "model is required")
                try:
                    ok = store_remove(name)
                except Exception as exc:
                    return self._ollama_error(400, str(exc))
                if not ok:
                    return self._ollama_error(404, f"model {name!r} not found in the store")
                return self._json(200, {})
            if path == "/api/copy":
                src = body.get("source") or body.get("from")
                dst = body.get("destination") or body.get("to")
                if not src or not dst:
                    return self._ollama_error(400, "source and destination are required")
                try:
                    store_copy(src, dst)
                except FileExistsError as exc:
                    return self._ollama_error(409, str(exc))
                except Exception as exc:
                    return self._ollama_error(400, str(exc))
                return self._json(200, {})
            if path == "/api/create":
                name = body.get("model") or body.get("name")
                if not name:
                    return self._ollama_error(400, "model is required")
                from .modelfile import parse_modelfile

                try:
                    if body.get("modelfile"):
                        recipe = parse_modelfile(str(body["modelfile"]))
                    else:
                        params = dict(body.get("parameters") or {})
                        recipe = {"base": body.get("from") or body.get("base"),
                                  "system": body.get("system"),
                                  "parameters": params}
                        if not recipe["base"]:
                            return self._ollama_error(400, "from (base model) is required")
                    store_create(name, recipe)
                except FileExistsError as exc:
                    return self._ollama_error(409, str(exc))
                except Exception as exc:
                    return self._ollama_error(400, str(exc))
                return self._json(200, {"status": "success"})
            if path in ("/api/pull", "/api/push"):
                return self._ollama_error(
                    501, "this runtime has no registry: use `terrazzo pull "
                         "<hugging-face-repo-id>` (pull) or publish on Hugging Face (push)")
            if path in ("/api/embed", "/api/embeddings"):
                return self._ollama_error(
                    501, "embeddings are not supported yet (generation models only)")
            if path == "/api/version":
                from . import __version__

                return self._json(200, {"version": __version__})
            return self._ollama_error(404, "not found")


        # --------------------------------------------------------------- GET
        def do_GET(self):  # noqa: N802
            if self.path not in ("/health", "/v1/health") and not self._authorized():
                self._json(401, {"error": {"type": "unauthorized",
                                           "message": "missing or invalid Bearer token"}})
                return
            path_only = self.path.split("?", 1)[0]
            if self.path in ("/health", "/v1/health"):
                self._json(200, {"status": "ok", "runtime": runtime.name})
            elif path_only == "/api/version":
                from . import __version__

                self._json(200, {"version": __version__})
            elif path_only == "/api/show":
                from urllib.parse import parse_qs, urlparse

                q = parse_qs(urlparse(self.path).query)
                body = {"model": (q.get("name") or q.get("model") or [None])[0]}
                self._ollama_show(body)
            elif self.path in ("/stats", "/metrics"):
                self._json(200, runtime.stats())
            elif self.path == "/v1/models":
                entry = {"id": "terrazzo", "object": "model", "owned_by": "local"}
                cap = getattr(runtime, "capability", None)
                if cap:
                    entry.update({
                        "predicted_tok_s": cap["predicted_tok_s"],
                        "target_tok_s": cap.get("target_tok_s"),
                        "meets_target": cap.get("meets_target"),
                        "bytes_per_token_gb": cap["bytes_per_token_gb"],
                        "is_moe": cap["is_moe"],
                        "active_params_b": cap["active_params_b"],
                        "chip": cap["chip"],
                    })
                self._json(200, {"object": "list", "data": [entry]})
            elif self.path in ("/api/tags", "/api/ps"):
                from .models import list_models

                model_path = getattr(getattr(runtime, "_cfg", None), "model", "terrazzo")
                model_name = Path(model_path).name if model_path else "terrazzo"
                entries = []
                found = False
                target = str(Path(model_path).resolve()) if model_path else ""
                for row in list_models():
                    if target and row["path"] == target:
                        found = True
                    entries.append({
                        "name": row["name"], "model": row["path"],
                        "modified_at": row["modified_at"], "size": row["size_bytes"],
                        "details": {"format": "mlx", "architecture": row.get("architecture")},
                    })
                if not found and model_path:
                    size = 0
                    try:
                        size = sum(p.stat().st_size for p in Path(model_path).rglob("*") if p.is_file())
                    except OSError:
                        pass
                    entries.insert(0, {
                        "name": model_name, "model": model_path,
                        "modified_at": "-", "size": size,
                        "details": {"format": "mlx", "loaded": True},
                    })
                if self.path == "/api/tags":
                    self._json(200, {"models": entries})
                else:
                    st = runtime.stats()
                    self._json(200, {"models": [{
                        "name": model_name, "model": model_path,
                        "size": entries[0]["size"] if entries else 0,
                        "runtime": runtime.name,
                        "uptime_s": st.get("uptime_s"),
                        "steps": st.get("steps"),
                        "details": {"format": "mlx", "loaded": True},
                    }]})
            elif self.path == "/capability":
                cap = getattr(runtime, "capability", None)
                self._json(200, cap or {"error": "capability unavailable"})
            else:
                self._json(404, {"error": "not found"})

        # -------------------------------------------------------------- POST
        def do_POST(self):  # noqa: N802
            # auth first: never read an untrusted request body before authentication
            if not self._authorized():
                self._json(401, {"error": {"type": "unauthorized",
                                           "message": "missing or invalid Bearer token"}})
                return
            self._deadline_reset()
            try:
                body = self._read_body()
            except PayloadTooLarge as exc:
                self._json(413, {"error": {"type": "payload_too_large",
                                           "message": str(exc),
                                           "max_bytes": max_body_bytes}})
                return
            except Exception:
                self._json(400, {"error": "invalid json"})
                return
            path_only = self.path.split("?", 1)[0]
            if path_only.startswith("/api/"):
                self._ollama_api(path_only, body)
                return
            if self.path == "/v1/abort":
                rid = body.get("request_id")
                ok = runtime.abort(rid) if isinstance(rid, str) else False
                self._json(200, {"aborted": ok, "request_id": rid})
                return
            if self.path == "/v1/prewarm":
                system = body.get("system")
                if not isinstance(system, str):
                    msgs = body.get("messages")
                    if isinstance(msgs, list):
                        system = "".join(
                            str(m.get("content") or "") for m in msgs
                            if isinstance(m, dict) and m.get("role") == "system")
                try:
                    result = runtime.prewarm(system or "")
                except NotImplementedError as exc:
                    self._json(501, {"error": {"type": "unsupported", "message": str(exc)}})
                    return
                except ValueError as exc:
                    self._json(400, {"error": {"type": "invalid_request", "message": str(exc)}})
                    return
                self._json(200, {"status": "prewarmed", **result})
                return
            if self.path == "/v1/concurrency":
                if os.environ.get("TERRAZZO_ENABLE_CONTROL") != "1":
                    self._json(404, {"error": {"type": "control_disabled",
                                               "message": "set TERRAZZO_ENABLE_CONTROL=1"}})
                    return
                try:
                    n = int(body.get("max_num_seqs"))
                except Exception:
                    self._json(400, {"error": {"type": "invalid_request",
                                               "message": "max_num_seqs must be an integer"}})
                    return
                if not 1 <= n <= 256:
                    self._json(400, {"error": {"type": "invalid_request",
                                               "message": "max_num_seqs out of range"}})
                    return
                try:
                    applied = runtime.set_concurrency(n)
                except NotImplementedError as exc:
                    self._json(501, {"error": {"type": "unsupported", "message": str(exc)}})
                    return
                except TimeoutError as exc:
                    self._json(504, {"error": {"type": "timeout", "message": str(exc)}})
                    return
                self._json(200, {"max_num_seqs": applied,
                                 "current": runtime.concurrency()})
                return
            if not rate_ok():
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "1")
                payload = json.dumps({"error": {"type": "rate_limited",
                                                "message": "too many requests"}}).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path == "/v1/chat/completions":
                json_schema = None
                rf = body.get("response_format")
                if rf:
                    kind = rf.get("type") if isinstance(rf, dict) else None
                    if kind == "json_object":
                        json_schema = {}
                    elif kind == "json_schema":
                        inner = rf.get("json_schema") or {}
                        schema = inner.get("schema") if isinstance(inner, dict) else None
                        if not isinstance(schema, dict):
                            self._json(400, {"error": {
                                "type": "invalid_request",
                                "message": "json_schema.response_format requires json_schema.schema"}})
                            return
                        try:
                            from .json_schema import SchemaError, compile_schema

                            json_schema = compile_schema(schema)
                        except SchemaError as exc:
                            self._json(400, {"error": {"type": "invalid_request",
                                                       "message": f"invalid json schema: {exc}"}})
                            return
                    else:
                        self._json(400, {"error": {
                            "type": "invalid_request",
                            "message": f"unsupported response_format type: {kind}"}})
                        return
                tools = None
                extra_messages: list[dict] = []
                if body.get("tools"):
                    try:
                        from .tools import (apply_tool_choice, normalize_messages,
                                            validate_tools)

                        tools = validate_tools(body["tools"])
                        tools, extra_messages = apply_tool_choice(
                            tools, body.get("tool_choice"))
                    except ValueError as exc:
                        self._json(400, {"error": {"type": "invalid_request",
                                                   "message": str(exc)}})
                        return
                messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]
                from .tools import normalize_messages as _normalize_messages

                messages = _normalize_messages(messages)
                default_system = getattr(runtime, "default_system", None)
                if default_system and not any(
                        isinstance(m, dict) and m.get("role") == "system" for m in messages):
                    messages = [{"role": "system", "content": default_system}] + messages
                if extra_messages:
                    messages = messages + extra_messages
                # tool calling is unusable while <think> eats the token budget;
                # callers can opt back in with {"thinking": true}
                thinking = body.get("thinking")
                if not isinstance(thinking, bool):
                    thinking = False if body.get("tools") else None
                try:
                    prompt_ids = runtime.apply_chat_template(
                        messages, tools=tools, enable_thinking=thinking)
                except ValueError as exc:
                    self._json(400, {"error": {"type": "invalid_request", "message": str(exc)}})
                    return
                except Exception:
                    prompt_ids = runtime.encode(
                        "\n".join(str(m.get("content", "")) for m in messages))
                self._serve(prompt_ids, body, chat=True, json_schema=json_schema,
                            tools=tools, tools_requested=bool(body.get("tools")))
            elif self.path == "/v1/completions":
                prompt = body.get("prompt", "")
                if isinstance(prompt, list):
                    prompt = prompt[0]
                self._serve(runtime.encode(prompt), body, chat=False)
            else:
                self._json(404, {"error": "not found"})

        def do_DELETE(self):  # noqa: N802
            if not self._authorized():
                self._json(401, {"error": {"type": "unauthorized",
                                           "message": "missing or invalid Bearer token"}})
                return
            self._deadline_reset()
            try:
                body = self._read_body()
            except PayloadTooLarge as exc:
                self._json(413, {"error": {"type": "payload_too_large",
                                           "message": str(exc)}})
                return
            except Exception:
                body = {}
            path_only = self.path.split("?", 1)[0]
            if path_only.startswith("/api/"):
                self._ollama_api(path_only, body)
                return
            self._json(404, {"error": "not found"})

        # ---------------------------------------------------------- streaming
        def _serve_json(self, prompt_ids: list[int], body: dict, chat: bool,
                        json_schema: dict, params) -> None:
            model = body.get("model") or "terrazzo"
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            try:
                text, stats = runtime.generate_json(
                    prompt_ids, json_schema, max_tokens=params.max_tokens,
                    timeout=max(0.001, self._remaining()))
            except MemoryError as exc:
                self._json(503, {"error": {"type": "overloaded", "message": str(exc)}})
                return
            except TimeoutError as exc:
                self._json(504, {"error": {"type": "timeout", "message": str(exc)}})
                return
            except Exception as exc:
                self._json(500, {"error": {"type": "generation_failed", "message": str(exc)}})
                return
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                self._json(500, {"error": {"type": "invalid_json_output",
                                           "message": f"constraint failed: {exc}"}})
                return
            finish = "stop" if stats.get("complete") else "length"
            if bool(body.get("stream")):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                         "model": model,
                         "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                      "finish_reason": None}]}
                self._sse(chunk)
                done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                self._sse(done)
                opts = body.get("stream_options") or {}
                if isinstance(opts, dict) and opts.get("include_usage"):
                    self._sse({"id": cid, "object": "chat.completion.chunk",
                               "created": created, "model": model, "choices": [],
                               "usage": {"prompt_tokens": len(prompt_ids),
                                         "completion_tokens": int(stats.get("steps", 0)),
                                         "total_tokens": len(prompt_ids) + int(stats.get("steps", 0))}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": stats.get("steps", 0),
                     "total_tokens": len(prompt_ids) + stats.get("steps", 0)}
            if chat:
                self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                 "model": model,
                                 "choices": [{"index": 0, "finish_reason": finish,
                                              "message": {"role": "assistant", "content": text}}],
                                 "usage": usage, "json_stats": stats})
            else:
                self._json(200, {"id": cid, "object": "text_completion", "created": created,
                                 "model": model,
                                 "choices": [{"index": 0, "finish_reason": finish, "text": text}],
                                 "usage": usage, "json_stats": stats})

        def _serve_tools(self, prompt_ids: list[int], body: dict, params, tools: list[dict]) -> None:
            """Buffered generation + tool-call parsing (OpenAI tools format)."""
            from .tools import parse_tool_calls, strip_thinking

            model = body.get("model") or "terrazzo"
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            request_id = body.get("request_id")
            try:
                request_id, q = self._submit(prompt_ids, params, request_id)
            except MemoryError as exc:
                self._json(503, {"error": {"type": "overloaded", "message": str(exc)}})
                return
            except TimeoutError as exc:
                self._json(504, {"error": {"type": "timeout", "message": str(exc)}})
                return
            except Exception as exc:
                self._json(500, {"error": {"type": "submit_failed", "message": str(exc)}})
                return
            tokens: list[int] = []
            finish = "stop"
            try:
                while True:
                    delta = self._norm(self._qget(q, request_id))
                    ids = delta.get("token_ids") or []
                    if delta.get("finished") and len(ids) > 1:
                        tokens = ids
                    else:
                        tokens.extend(ids)
                    if delta.get("finished"):
                        finish = delta.get("finish_reason") or "stop"
                        break
            except TimeoutError as exc:
                runtime.abort(request_id)
                self._json(504, {"error": {"type": "timeout", "message": str(exc)}})
                return
            text = runtime.decode(tokens)
            if tools:
                cleaned, tool_calls = parse_tool_calls(text)
            else:
                # tools were offered but explicitly disabled (tool_choice=none):
                # never turn tag-like text into tool calls
                cleaned, tool_calls = strip_thinking(text), []
            usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
                     "total_tokens": len(prompt_ids) + len(tokens)}
            message = {"role": "assistant", "content": cleaned or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
                finish = "tool_calls"
            if not bool(body.get("stream")):
                self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                 "model": model,
                                 "choices": [{"index": 0, "finish_reason": finish,
                                              "message": message}],
                                 "usage": usage})
                return
            # streaming: buffered content chunk, then a tool_calls chunk (if any)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if cleaned:
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model,
                           "choices": [{"index": 0, "delta": {"role": "assistant",
                                                              "content": cleaned},
                                        "finish_reason": None}]})
            if tool_calls:
                delta_calls = [{"index": i, "id": c["id"], "type": "function",
                                "function": c["function"]}
                               for i, c in enumerate(tool_calls)]
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model,
                           "choices": [{"index": 0, "delta": {"tool_calls": delta_calls},
                                        "finish_reason": None}]})
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            opts = body.get("stream_options") or {}
            if isinstance(opts, dict) and opts.get("include_usage"):
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model, "choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _serve(self, prompt_ids: list[int], body: dict, chat: bool,
                   json_schema: dict | None = None, tools: list[dict] | None = None,
                   tools_requested: bool = False) -> None:
            stream = bool(body.get("stream"))
            params = _sampling_from_body(body)
            if json_schema is not None:
                self._serve_json(prompt_ids, body, chat, json_schema, params)
                return
            if tools or tools_requested:
                self._serve_tools(prompt_ids, body, params, tools or [])
                return
            request_id = body.get("request_id")
            try:
                request_id, q = self._submit(prompt_ids, params, request_id)
            except TimeoutError as exc:
                self._json(504, {"error": {"type": "timeout", "message": str(exc)}})
                return
            except MemoryError as exc:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "5")
                body = json.dumps({"error": {"type": "overloaded", "message": str(exc)}}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            except Exception as exc:
                self._json(500, {"error": {"type": "submit_failed", "message": str(exc)}})
                return
            model = body.get("model") or "terrazzo"
            cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if not stream:
                tokens: list[int] = []
                finish = None
                try:
                    while True:
                        delta = self._norm(self._qget(q, request_id))
                        if delta.get("error"):
                            raise RuntimeError(str(delta["error"]))
                        ids = delta.get("token_ids") or []
                        if delta.get("finished") and len(ids) > 1:
                            tokens = ids  # final event carries the full output
                        else:
                            tokens.extend(ids)
                        if delta.get("finished"):
                            finish = delta.get("finish_reason") or "stop"
                            break
                except TimeoutError as exc:
                    runtime.abort(request_id)
                    self._json(504, {"error": {"type": "timeout", "message": str(exc)}})
                    return
                except RuntimeError as exc:
                    runtime.abort(request_id)
                    self._json(500, {"error": {"type": "generation_failed",
                                               "message": str(exc)}})
                    return
                text = runtime.decode(tokens)
                usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(tokens),
                         "total_tokens": len(prompt_ids) + len(tokens)}
                if chat:
                    self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                     "model": model,
                                     "choices": [{"index": 0, "finish_reason": finish,
                                                  "message": {"role": "assistant", "content": text}}],
                                     "usage": usage})
                else:
                    self._json(200, {"id": cid, "object": "text_completion", "created": created,
                                     "model": model,
                                     "choices": [{"index": 0, "finish_reason": finish, "text": text}],
                                     "usage": usage})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            streamed_ids: list[int] = []
            printed = 0
            try:
                while True:
                    delta = self._norm(self._qget(q, request_id))
                    if delta.get("error"):
                        raise RuntimeError(str(delta["error"]))
                    pieces = delta.get("token_ids") or []
                    for tok_id in pieces:
                        streamed_ids.append(tok_id)
                        # a token may split a multi-byte UTF-8 character: decode the
                        # whole prefix and stream only the safe delta
                        text = runtime.decode(streamed_ids)
                        safe = text.rstrip("\ufffd")
                        if len(safe) <= printed:
                            continue
                        piece = safe[printed:]
                        printed = len(safe)
                        if chat:
                            self._sse({"id": cid, "object": "chat.completion.chunk",
                                       "created": created, "model": model,
                                       "choices": [{"index": 0, "delta": {"content": piece},
                                                    "finish_reason": None}]})
                        else:
                            self._sse({"id": cid, "object": "text_completion",
                                       "created": created, "model": model,
                                       "choices": [{"index": 0, "text": piece,
                                                    "finish_reason": None}]})
                    if delta.get("finished"):
                        reason = delta.get("finish_reason") or "stop"
                        if chat:
                            self._sse({"id": cid, "object": "chat.completion.chunk",
                                       "created": created, "model": model,
                                       "choices": [{"index": 0, "delta": {},
                                                    "finish_reason": reason}]})
                        else:
                            self._sse({"id": cid, "object": "text_completion",
                                       "created": created, "model": model,
                                       "choices": [{"index": 0, "text": "",
                                                    "finish_reason": reason}]})
                        opts = body.get("stream_options") or {}
                        if isinstance(opts, dict) and opts.get("include_usage"):
                            self._sse({
                                "id": cid,
                                "object": "chat.completion.chunk" if chat else "text_completion",
                                "created": created, "model": model, "choices": [],
                                "usage": {"prompt_tokens": len(prompt_ids),
                                          "completion_tokens": len(streamed_ids),
                                          "total_tokens": len(prompt_ids) + len(streamed_ids)},
                            })
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        break
            except (TimeoutError, RuntimeError) as exc:
                # headers are already out: terminate the SSE stream explicitly
                runtime.abort(request_id)
                try:
                    self._sse({"id": cid,
                               "object": "chat.completion.chunk" if chat else "text_completion",
                               "created": created, "model": model,
                               "choices": [{"index": 0,
                                            "delta": {} if chat else {},
                                            "finish_reason": "error"}],
                               "error": {"type": "timeout", "message": str(exc)}})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception:
                    pass
            except (BrokenPipeError, ConnectionResetError):
                # the client went away: free its KV now rather than decoding to the end
                try:
                    runtime.abort(request_id)
                except Exception:
                    pass

    return Handler


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a cap: excess connections get an immediate 503.

    Unbounded threads are a trivial DoS on a shared machine: a slow client (or a
    slow generation) can pile up threads. The limit is enforced before the request
    is parsed; queued inference work has its own cap in the runtime.
    """

    daemon_threads = True

    def __init__(self, server_address, handler_class, *, max_concurrent: int = 64):
        super().__init__(server_address, handler_class)
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            body = json.dumps({"error": {"type": "overloaded",
                                         "message": "too many concurrent requests"}}).encode()
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Type: application/json\r\nRetry-After: 1\r\n"
                    b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                    b"Connection: close\r\n\r\n" + body)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def serve(cfg: EngineConfig, host: str = "127.0.0.1", port: int = 8080,
          *, mode: str = "batch", system_text: str | None = None,
          default_params: dict | None = None,
          use_metal: bool = False, target_tok_s: float | None = None,
          allow_below_target: bool = False, api_key: str | None = None,
          rate_limit_rps: float = 0.0, no_thinking: bool = False,
          tls_cert: str | None = None, tls_key: str | None = None) -> None:
    # MacTurbo admission: refuse to start when the requested SLO is unreachable
    capability = None
    try:
        from .slo_router import capability as compute_capability, enforce

        capability = compute_capability(cfg.model, target_tok_s=target_tok_s)
        ok, message = enforce(capability, allow_below_target=allow_below_target)
        print(message, flush=True)
        if not ok:
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[MacTurbo] capability check skipped: {exc}", flush=True)
    if mode == "pool":
        runtime: BaseRuntime = PoolRuntime(cfg, system_text, use_metal=use_metal)
    elif mode == "paged":
        runtime = PagedRuntime(cfg, system_text)
    else:
        runtime = EngineRuntime(cfg, system_text)
    runtime.capability = capability
    runtime.model_path = cfg.model
    runtime.default_params = default_params
    if no_thinking:
        runtime.template_kwargs = {"enable_thinking": False}
    max_concurrent = int(os.environ.get("TERRAZZO_MAX_CONCURRENT", "64"))
    request_timeout_s = float(os.environ.get("TERRAZZO_REQUEST_TIMEOUT_S", "600"))
    max_body_bytes = int(float(os.environ.get("TERRAZZO_MAX_BODY_MB", "1")) * (1 << 20))
    idle_timeout_s = float(os.environ.get("TERRAZZO_IDLE_TIMEOUT_S", "30"))
    tls_cert = tls_cert or os.environ.get("TERRAZZO_TLS_CERT")
    tls_key = tls_key or os.environ.get("TERRAZZO_TLS_KEY")
    scheme = "https" if tls_cert else "http"
    if os.environ.get("TERRAZZO_STACK_DUMP") == "1":
        # diagnostics for hangs: `kill -USR1 <pid>` dumps every thread's stack to
        # stderr (captured in the server log) without killing the process
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

    httpd = BoundedThreadingHTTPServer(
        (host, port),
        make_handler(runtime, api_key=api_key, rate_limit_rps=rate_limit_rps,
                     request_timeout_s=request_timeout_s,
                     max_body_bytes=max_body_bytes,
                     idle_timeout_s=idle_timeout_s),
        max_concurrent=max_concurrent)
    if tls_cert:
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=tls_cert, keyfile=tls_key)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f"[terrazzo] runtime={runtime.name} model={cfg.model}"
          + (" metal=on" if getattr(runtime, "use_metal", False) else ""), flush=True)
    if system_text:
        n = runtime.prefix.num_tokens if runtime.prefix else 0
        print(f"[terrazzo] system prefix registered: {n} tokens", flush=True)
    print(f"[terrazzo] listening on {scheme}://{host}:{port}  (OpenAI compatible)"
          + ("  auth=on" if api_key else "") + (f"  rate_limit={rate_limit_rps}/s"
                                               if rate_limit_rps else "")
          + f"  max_concurrent={max_concurrent}"
          + f"  request_timeout={request_timeout_s:.0f}s", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        runtime.close()
