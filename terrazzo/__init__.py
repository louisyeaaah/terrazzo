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

from .blocks import BlockManager, BlockManagerConfig
from .engine import Engine, EngineConfig, RequestOutput
from .paged_kv import PagedKVCache, PagedPrefixCache, SharedPrefix
from .paged_session import PagedSession
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutputs
from .sequence import Sequence, SequenceStatus

__all__ = [
    "BlockManager",
    "BlockManagerConfig",
    "Engine",
    "EngineConfig",
    "PagedKVCache",
    "PagedPrefixCache",
    "PagedSession",
    "RequestOutput",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutputs",
    "Sequence",
    "SequenceStatus",
    "SharedPrefix",
]

__version__ = "0.3.0"
