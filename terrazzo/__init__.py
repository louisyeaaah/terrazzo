"""Terrazzo: local LLM inference for Apple Silicon.

Terrazzo is an inference runtime and server for macOS/Apple Silicon (MLX), built
around paged-KV scheduling, continuous batching and shared-prefix caching. Its
scheduler semantics are a port of DeepInfer's design (see DESIGN.md).

What is ported from DeepInfer (engine semantics):
  - paged KV block manager with ref-counting and a free list (csrc/engine/block_manager.hpp)
  - chunked prefill + continuous batching scheduler (csrc/engine/scheduler.hpp)
  - LCFS preemption with push-front requeue (scheduler.hpp:_preempt)
  - step() loop, per-step metrics, streaming outputs (engine/llm_engine.py)

What is adapted (because Apple Silicon has no CUDA/NCCL/FP4):
  - compute backend: mlx-lm model + MLX kernels instead of FlashMLA/DeepGEMM/DeepEP
  - KV lives in MLX cache objects; the block manager is the capacity/budget layer
  - no EP/EPLB/PD-disaggregation (single unified-memory device)

Public API:
    from terrazzo import Engine, EngineConfig
    engine = Engine(EngineConfig(model="mlx-community/Qwen3-4B-4bit"))
    print(engine.generate("你好", max_tokens=64))
"""

from importlib import import_module

_EXPORTS = {
    "BlockManager": "blocks",
    "BlockManagerConfig": "blocks",
    "Engine": "engine",
    "EngineConfig": "engine",
    "RequestOutput": "engine",
    "PagedKVCache": "paged_kv",
    "PagedPrefixCache": "paged_kv",
    "SharedPrefix": "paged_kv",
    "PagedSession": "paged_session",
    "Scheduler": "scheduler",
    "SchedulerConfig": "scheduler",
    "SchedulerOutputs": "scheduler",
    "Sequence": "sequence",
    "SequenceStatus": "sequence",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Import heavy submodules (and MLX) only when a symbol is actually used.

    ``import terrazzo`` and the pure-Python scheduler/block tests must work on a
    machine without MLX (e.g. Linux CI); ``from terrazzo import Engine`` still works.
    """
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


__version__ = "0.3.0"
