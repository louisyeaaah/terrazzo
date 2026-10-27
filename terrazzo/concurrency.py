"""Small load-aware concurrency rule for the cap4/cap16 experiment.

This is deliberately a rule, not a controller with a model.  It only decides
whether *new* admissions should use the low or high cap; requests already running
are left alone.  Inputs are the current mix of recent requests (cache hit ratio),
queue depth and memory pressure.  A minimum hold time prevents flapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AutoCapConfig:
    low_cap: int = 4
    high_cap: int = 16
    # shared-prefix traffic: most of the prompt is a cache hit -> higher concurrency
    hit_high: float = 0.75
    # mixed traffic: unique prompts pull the ratio down -> lower concurrency
    hit_low: float = 0.50
    min_hold_s: float = 6.0
    # block/KV or MLX active-memory fraction above this always chooses the low cap
    memory_high: float = 0.70
    window: int = 16
    # fresh, insufficient samples should not change the cap
    min_samples: int = 8
    # average uncached prompt tokens: shared-prefix traffic is tiny, mixed traffic
    # carries one full unique prompt per pair of sessions
    uncached_high: float = 512.0
    uncached_low: float = 256.0


@dataclass
class CapDecision:
    cap: int
    reason: str
    hit_ratio: float
    queue: int
    memory_ratio: float
    at: float
    avg_uncached: int = 0


class AutoCap:
    """Hysteretic cap4/cap16 selector.

    ``decide`` returns a new cap or ``None`` when no change is warranted.
    """

    def __init__(self, cfg: AutoCapConfig | None = None, *, current: int | None = None,
                 now: float = 0.0) -> None:
        self.cfg = cfg or AutoCapConfig()
        self.current = int(current or self.cfg.high_cap)
        self.last_change = now
        self.history: list[CapDecision] = []

    def observe_inputs(self, costs: list[dict]) -> tuple[float, int, int]:
        """(hit ratio, avg uncached prompt tokens, sample count) from recent costs."""
        rows = [c for c in costs[-self.cfg.window:] if c.get("prompt_tokens")]
        if not rows:
            return 0.0, 0, 0
        hit = sum(min(c.get("cached_estimate", 0), c["prompt_tokens"])
                  for c in rows) / sum(c["prompt_tokens"] for c in rows)
        uncached = sum(c.get("uncached_estimate", 0) for c in rows) / len(rows)
        return float(hit), int(round(uncached)), len(rows)

    def decide(self, *, now: float, hit_ratio: float, avg_uncached: int,
               queue: int, memory_ratio: float, samples: int | None = None) -> int | None:
        cfg = self.cfg
        if samples is None:
            samples = cfg.window
        if samples < cfg.min_samples:
            return None
        target = self.current
        reason = "hold"
        if memory_ratio >= cfg.memory_high:
            target, reason = cfg.low_cap, "memory-pressure"
        elif avg_uncached >= cfg.uncached_high:
            target, reason = cfg.low_cap, "mixed-prefix"
        elif avg_uncached <= cfg.uncached_low and hit_ratio >= cfg.hit_low:
            # require BOTH a cache-hit-heavy mix and small uncached work; hit ratio
            # alone stays high while the previous phase's requests age out of the
            # window, which caused cap flapping in the reversed-phase validation
            target, reason = cfg.high_cap, "shared-prefix"
        elif hit_ratio <= cfg.hit_low and avg_uncached >= cfg.uncached_low:
            target, reason = cfg.low_cap, "mixed-prefix"
        elif self.current == cfg.high_cap:
            target, reason = cfg.high_cap, "hysteresis-high"
        else:
            target, reason = cfg.low_cap, "hysteresis-low"
        decision = CapDecision(cap=target, reason=reason, hit_ratio=round(hit_ratio, 3),
                               queue=queue, memory_ratio=round(memory_ratio, 3), at=now,
                               avg_uncached=int(avg_uncached))
        if target == self.current:
            return None
        if now - self.last_change < cfg.min_hold_s:
            return None
        self.current = target
        self.last_change = now
        self.history.append(decision)
        if len(self.history) > 64:
            self.history = self.history[-64:]
        return target

    def stats(self) -> dict:
        return {
            "enabled": True,
            "current_cap": self.current,
            "config": self.cfg.__dict__.copy(),
            "decisions": [d.__dict__ for d in self.history],
        }
