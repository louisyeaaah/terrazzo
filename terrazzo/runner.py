"""MLX execution layer.

This is the analogue of DeepInfer's ``worker/model_runner.py`` + the CUDA kernel
stack, collapsed to what Apple Silicon offers: one MLX model and mlx-lm's
continuous-batching primitives.

Everything DeepInfer does with FlashMLA / DeepGEMM / DeepEP / CUDA graphs is
handled here by MLX kernels and unified memory. There is no equivalent of
FP8/FP4 KV or EP all-to-all on M-series hardware, so those paths simply do not
exist in this build.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .sequence import SamplingParams, Sequence

try:  # keep the package importable for unit tests without MLX
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    MLX_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on environment
    MLX_AVAILABLE = False
    _MLX_IMPORT_ERROR = exc


def _require_mlx() -> None:
    if not MLX_AVAILABLE:
        raise RuntimeError(
            "MLX / mlx-lm is not available in this environment. "
            "Install it with `pip install mlx-lm` (Apple Silicon only)."
        ) from _MLX_IMPORT_ERROR


class MLXRunner:
    """Owns the model, the tokenizer and the continuous-batching executor."""

    def __init__(
        self,
        model_path: str,
        *,
        completion_batch_size: int = 8,
        prefill_batch_size: int = 8,
        prefill_step_size: int = 512,
        max_kv_size: int | None = None,
        load_model: bool = True,
    ) -> None:
        _require_mlx()
        self.model_path = model_path
        self.completion_batch_size = completion_batch_size
        self.prefill_step_size = prefill_step_size
        self.model = None
        self.tokenizer = None
        if load_model:
            t0 = time.perf_counter()
            self.model, self.tokenizer = mlx_load(model_path)
            self.load_time = time.perf_counter() - t0
            self.eos_token_ids = tuple(sorted(getattr(self.tokenizer, "eos_token_ids", []) or []))
            self.bg = BatchGenerator(
                self.model,
                max_tokens=128,
                stop_tokens=[(t,) for t in self.eos_token_ids],
                completion_batch_size=completion_batch_size,
                prefill_batch_size=prefill_batch_size,
                prefill_step_size=prefill_step_size,
                max_kv_size=max_kv_size,
            )
        else:  # pragma: no cover - test seam
            self.load_time = 0.0
            self.eos_token_ids = ()
            self.bg = None
        self.uid_to_seq: dict[int, Sequence] = {}

    # ------------------------------------------------------------------ tokenize
    def encode(self, text: str) -> list[int]:
        _require_mlx()
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: list[int]) -> str:
        _require_mlx()
        try:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:  # older tokenizer wrappers
            return self.tokenizer.decode(token_ids)

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        _require_mlx()
        tokens = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=False
        )
        return list(tokens)

    # ----------------------------------------------------------------- sampling
    def _sampler(self, params: SamplingParams) -> Callable[[Any], Any]:
        return make_sampler(
            temp=params.temperature,
            top_p=params.top_p if params.top_p else 1.0,
            top_k=params.top_k,
        )

    # ------------------------------------------------------------------- insert
    def insert(
        self,
        seq: Sequence,
        cache: list[Any] | None = None,
        cached_tokens: int = 0,
    ) -> int:
        """Insert a sequence into the executor, optionally with a warm prefix cache."""
        _require_mlx()
        tokens = seq.prompt_token_ids
        if cached_tokens > 0 and cache is not None:
            rest = tokens[cached_tokens:]
            if not rest:
                # everything cached: re-run the final token so sampling can start
                cached_tokens -= 1
                rest = tokens[cached_tokens:]
            segments = [[rest]]
            all_tokens = [tokens[:cached_tokens]]
        else:
            segments = [[tokens]]
            all_tokens = [[]]
        (uid,) = self.bg.insert_segments(
            segments=segments,
            max_tokens=[seq.sampling_params.max_tokens],
            caches=[cache],
            all_tokens=all_tokens,
            samplers=[self._sampler(seq.sampling_params)],
        )
        self.uid_to_seq[uid] = seq
        seq.cached_prompt_tokens = cached_tokens
        return uid

    # --------------------------------------------------------------------- step
    def step(self):
        """One executor step: returns (prompt_responses, generation_responses)."""
        _require_mlx()
        return self.bg.next()

    def remove(self, seqs: list[Sequence], return_caches: bool = False) -> dict:
        _require_mlx()
        uids = [uid for uid, s in self.uid_to_seq.items() if s in seqs]
        for uid in uids:
            self.uid_to_seq.pop(uid, None)
        if not uids:
            return {}
        return self.bg.remove(uids, return_prompt_caches=return_caches)

    def extract_cache(self, seq: Sequence):
        _require_mlx()
        uid = next((u for u, s in self.uid_to_seq.items() if s is seq), None)
        if uid is None:
            return None
        try:
            return self.bg.extract_cache([uid]).get(uid)
        except Exception:
            return None

    def prompt_cache_bytes(self) -> int:
        _require_mlx()
        try:
            return self.bg.prompt_cache_nbytes
        except Exception:
            return 0

    def close(self) -> None:
        if self.bg is not None:
            self.bg.close()

    @property
    def num_active(self) -> int:
        return len(self.uid_to_seq)
