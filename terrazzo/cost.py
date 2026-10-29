"""Predicted per-request cost: prefilling, copying a cached prefix, and KV growth.

This is deliberately a *prediction*, not a measurement of the current request:
prefill/decode time and copy overhead depend on context length, batch shape,
cache layout and memory pressure.  The model keeps a few calibration points per
operation and a confidence band.  Decisions that matter (reuse vs recompute) only
take the aggressive branch when the predicted ranges do not overlap; outside the
calibrated range the model falls back to the conservative branch and marks the
prediction ``confident=False`` so experiments can filter it out.

Phase-0 note: the numbers here are only used for ordering and the cache decision.
The executor is still mlx-lm's BatchGenerator, so a predicted prefill budget is
not a guarantee that the executor will honour it (see Engine.execution_control).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model_profile import load_config

# mlx-lm caches are fp16 today; keep the constant explicit so a future bf16 path
# cannot silently halve the estimate.
KV_DTYPE_BYTES = 2
DEFAULT_BAND = 0.30


def kv_bytes_per_token(model_path: str, *, dtype_bytes: int = KV_DTYPE_BYTES) -> int | None:
    """KV bytes for one token: layers × 2 (K,V) × n_kv_heads × head_dim × dtype."""
    try:
        cfg = load_config(model_path)
    except Exception:
        return None
    layers = cfg.get("num_hidden_layers")
    heads = cfg.get("num_kv_heads") or cfg.get("num_key_value_heads")
    hidden = cfg.get("hidden_size")
    attn_heads = cfg.get("num_attention_heads")
    head_dim = cfg.get("head_dim")
    if head_dim is None and hidden and attn_heads:
        head_dim = hidden // attn_heads
    if not (layers and heads and head_dim):
        return None
    return int(layers * 2 * heads * head_dim * dtype_bytes)


@dataclass
class Prediction:
    ms: float
    low_ms: float
    high_ms: float
    confident: bool
    source: str

    def as_dict(self) -> dict:
        return {"ms": round(self.ms, 3), "low_ms": round(self.low_ms, 3),
                "high_ms": round(self.high_ms, 3), "confident": self.confident,
                "source": self.source}


@dataclass
class PrefixCost:
    prompt_tokens: int
    cached_tokens: int
    uncached_tokens: int
    snapshot_bytes: int
    copy_ms: float
    recompute_ms: float
    reuse: bool
    reason: str
    certificate: str = "n/a"
    copy: Prediction | None = None
    recompute: Prediction | None = None
    uncertain: bool = False

    def as_dict(self) -> dict:
        d = {
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "uncached_tokens": self.uncached_tokens,
            "snapshot_mb": round(self.snapshot_bytes / 1e6, 1),
            "copy_ms": round(self.copy_ms, 3),
            "recompute_ms": round(self.recompute_ms, 3),
            "reuse": self.reuse,
            "reason": self.reason,
            "certificate": self.certificate,
            "uncertain": self.uncertain,
        }
        if self.copy:
            d["copy_range"] = self.copy.as_dict()
        if self.recompute:
            d["recompute_range"] = self.recompute.as_dict()
        return d


def _piecewise_ms(tokens: int, points: list[tuple[int, float]], default_rate: float,
                  band: float = DEFAULT_BAND) -> Prediction:
    """Interpolate a rate from (tokens, ms) points; conservative outside the range."""
    tokens = max(1, int(tokens))
    if not points:
        ms = 1000.0 * tokens / max(default_rate, 1.0)
        return Prediction(ms, ms * (1 - band), ms * (1 + band), True, "default-rate")
    pts = sorted(points)
    if tokens <= pts[0][0]:
        ms = 1000.0 * tokens / max(pts[0][0] / max(pts[0][1], 1e-6), 1.0)
        return Prediction(ms, ms * (1 - band), ms * (1 + band), False, "below-calibration")
    if tokens >= pts[-1][0]:
        rate = pts[-1][0] / max(pts[-1][1], 1e-6)
        ms = 1000.0 * tokens / max(rate, 1.0)
        return Prediction(ms, ms * (1 - band), ms * (1 + band), False, "above-calibration")
    for (a_t, a_ms), (b_t, b_ms) in zip(pts, pts[1:]):
        if a_t <= tokens <= b_t:
            f = (tokens - a_t) / max(1, b_t - a_t)
            ms = a_ms + f * (b_ms - a_ms)
            return Prediction(ms, ms * (1 - band), ms * (1 + band), True, "interpolated")
    ms = 1000.0 * tokens / max(default_rate, 1.0)
    return Prediction(ms, ms * (1 - band), ms * (1 + band), False, "fallback")


@dataclass
class CostModel:
    """Predicted-cost model shared by the cache decision and the scheduler."""

    kv_bytes_per_token: int = 0
    prefill_tok_s: float = 450.0
    copy_tok_s: float = 8000.0
    copy_overhead_ms: float = 0.5
    reuse_margin: float = 0.9
    alpha: float = 0.05
    band: float = DEFAULT_BAND
    prefill_observations: int = 0
    copy_observations: int = 0
    prefill_points: list[tuple[int, float]] = field(default_factory=list)
    copy_points: list[tuple[int, float]] = field(default_factory=list)
    last_cost: PrefixCost | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ predict
    def predict_prefill(self, tokens: int) -> Prediction:
        if self.prefill_points:
            return _piecewise_ms(tokens, self.prefill_points, self.prefill_tok_s, self.band)
        return _piecewise_ms(tokens, [], self.prefill_tok_s, self.band)

    def predict_copy(self, tokens: int) -> Prediction:
        if tokens <= 0:
            return Prediction(0.0, 0.0, 0.0, True, "no-copy")
        if self.copy_points:
            p = _piecewise_ms(tokens, self.copy_points, self.copy_tok_s, self.band)
        else:
            p = _piecewise_ms(tokens, [], self.copy_tok_s, self.band)
        # fixed preparation overhead is outside the per-token rate
        return Prediction(p.ms + self.copy_overhead_ms,
                          p.low_ms + self.copy_overhead_ms,
                          p.high_ms + self.copy_overhead_ms,
                          p.confident, p.source)

    def estimate(self, prompt_tokens: int, cached_tokens: int, snapshot_bytes: int = 0,
                 policy: str = "auto") -> PrefixCost:
        prompt_tokens = max(1, int(prompt_tokens))
        cached_tokens = max(0, min(int(cached_tokens), prompt_tokens))
        uncached = max(0, prompt_tokens - cached_tokens)
        copy = self.predict_copy(cached_tokens)
        recompute = self.predict_prefill(prompt_tokens)
        recompute_cached = self.predict_prefill(cached_tokens)
        reuse = False
        uncertain = False
        certificate = "n/a"
        certified_reuse = (
            copy.confident and recompute_cached.confident
            and copy.high_ms < recompute_cached.low_ms * self.reuse_margin
        )
        certified_recompute = (
            copy.confident and recompute_cached.confident
            and copy.low_ms > recompute_cached.high_ms * self.reuse_margin
        )
        if policy == "never" or cached_tokens <= 0:
            reuse, reason = False, ("policy:never" if policy == "never" else "no-cache-hit")
        elif policy == "always":
            reuse, reason = True, "policy:always"
            certificate = "policy:always"
        elif policy == "certified":
            # DoSA-style gate: only take the copy branch when both interval
            # predictions are calibrated and the ranges do not overlap.
            reuse = certified_reuse
            uncertain = not (copy.confident and recompute_cached.confident)
            reason = "certified:copy-cheaper" if reuse else "certified:recompute"
            certificate = "certified_reuse" if reuse else "certified_recompute"
        else:
            # conservative: only reuse if even the optimistic copy range beats the
            # pessimistic recompute range for the cached span
            reuse = copy.high_ms < recompute_cached.low_ms * self.reuse_margin
            uncertain = not (copy.confident and recompute_cached.confident)
            reason = "copy-cheaper" if reuse else (
                "recompute-cheaper" if copy.high_ms >= recompute_cached.high_ms * self.reuse_margin
                else "uncertain-conservative")
            if reuse:
                certificate = "certified_reuse" if certified_reuse else "uncertain_reuse"
            else:
                certificate = ("certified_recompute" if certified_recompute
                               else "uncertain_recompute")
        c = PrefixCost(prompt_tokens=prompt_tokens, cached_tokens=cached_tokens,
                       uncached_tokens=uncached, snapshot_bytes=int(snapshot_bytes),
                       copy_ms=copy.ms, recompute_ms=recompute.ms,
                       reuse=reuse, reason=reason, certificate=certificate,
                       copy=copy, recompute=recompute,
                       uncertain=uncertain)
        self.last_cost = c
        return c

    # ---------------------------------------------------------------- calibration
    def observe_step(self, prefill_tokens: int, seconds: float) -> None:
        if prefill_tokens <= 0 or seconds <= 0:
            return
        rate = prefill_tokens / seconds
        if rate <= 0:
            return
        if self.prefill_observations == 0:
            self.prefill_tok_s = rate
        else:
            self.prefill_tok_s = (1 - self.alpha) * self.prefill_tok_s + self.alpha * rate
        self.prefill_observations += 1
        self.add_prefill_point(prefill_tokens, seconds)

    def observe_copy(self, tokens: int, seconds: float) -> None:
        if tokens <= 0 or seconds <= 0:
            return
        rate = tokens / seconds
        if rate <= 0:
            return
        if self.copy_observations == 0:
            self.copy_tok_s = rate
        else:
            self.copy_tok_s = (1 - self.alpha) * self.copy_tok_s + self.alpha * rate
        self.copy_observations += 1
        self.add_copy_point(tokens, seconds)

    def add_prefill_point(self, tokens: int, seconds: float) -> None:
        if tokens > 0 and seconds > 0 and len(self.prefill_points) < 32:
            self.prefill_points.append((int(tokens), float(seconds) * 1000.0))

    def add_copy_point(self, tokens: int, seconds: float) -> None:
        if tokens > 0 and seconds > 0 and len(self.copy_points) < 32:
            self.copy_points.append((int(tokens), float(seconds) * 1000.0))

    # --------------------------------------------------------------------- stats
    def stats(self) -> dict:
        return {
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "prefill_tok_s": round(self.prefill_tok_s, 1),
            "copy_tok_s": round(self.copy_tok_s, 1),
            "copy_overhead_ms": self.copy_overhead_ms,
            "band": self.band,
            "prefill_observations": self.prefill_observations,
            "copy_observations": self.copy_observations,
            "prefill_points": len(self.prefill_points),
            "copy_points": len(self.copy_points),
            "last_cost": self.last_cost.as_dict() if self.last_cost else None,
        }

    def priority_tokens(self, prompt_tokens: int, cached_tokens: int,
                        memory_tokens: int = 0, memory_weight: float = 0.0) -> float:
        """Scheduler key: predicted remaining prefill + a small memory term."""
        return float(max(0, int(prompt_tokens) - int(cached_tokens))
                     + memory_weight * max(0, int(memory_tokens)))
