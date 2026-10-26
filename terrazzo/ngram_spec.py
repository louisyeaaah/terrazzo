"""N-gram / prompt-lookup speculative decoding (no draft model).

At long context the decode step is dominated by weight + KV reads, and *each* token
pays them. Speculative decoding amortises that: propose k tokens from the prompt's
own n-grams (free), verify them with one forward pass (q_len = k+1), accept the
longest correct prefix, roll the KV cache back. Greedy output is identical to the
baseline by construction — verification only accepts exact argmax matches.

    from terrazzo.ngram_spec import ngram_generate
    text, stats = ngram_generate(model, tok, prompt_ids, max_tokens=128, k=4)

``n`` (n-gram size) and ``k`` (proposals per step) are the knobs; this module is
used by ``Engine.generate(spec_ngram=True)`` and the ``--spec-ngram`` CLI flag.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SpecStats:
    steps: int = 0
    proposed: int = 0
    accepted: int = 0
    tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def decode_tok_s(self) -> float:
        decode = self.total_s - self.ttft_s
        return self.tokens / decode if decode > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "tokens": self.tokens,
            "ttft_s": round(self.ttft_s, 3),
            "total_s": round(self.total_s, 3),
            "decode_tok_s": round(self.decode_tok_s, 2),
            **self.extra,
        }


def _build_ngram_index(prompt: list[int], n: int) -> dict[tuple, list[int]]:
    """Map every (n+1)-gram to the token that follows it."""
    index: dict[tuple, list[int]] = {}
    for i in range(len(prompt) - n):
        key = tuple(prompt[i : i + n])
        index.setdefault(key, []).append(prompt[i + n])
    return index


def index_append(index: dict, ctx: list[int], start: int, n: int) -> None:
    """Add the n-grams introduced by ctx[start:] to the index (prompt + generated)."""
    for pos in range(max(n, start), len(ctx)):
        key = tuple(ctx[pos - n : pos])
        index.setdefault(key, []).append(ctx[pos])


def propose(prompt: list[int], generated: list[int], index: dict, n: int,
            k: int) -> list[int]:
    """Propose up to k tokens following the longest suffix match (prompt lookup)."""
    ctx = prompt + generated
    for size in range(min(n, len(ctx)), 0, -1):
        candidates = index.get(tuple(ctx[-size:]))
        if candidates:
            return candidates[:k]
    return []


def ngram_generate(model, tokenizer, prompt_ids: list[int], *, max_tokens: int = 128,
                   k: int = 4, n: int = 3, temperature: float = 0.0,
                   ignore_eos: bool = True) -> tuple[str, dict]:
    """Greedy speculative generation with prompt-lookup proposals."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    eos = set(getattr(tokenizer, "eos_token_ids", []) or [])
    cache = make_prompt_cache(model)
    stats = SpecStats()
    t0 = time.perf_counter()

    # prefill everything except the last prompt token
    ctx = list(prompt_ids)
    last = ctx[-1]
    prefill = ctx[:-1]
    if prefill:
        chunk = 1024
        for i in range(0, len(prefill), chunk):
            logits = model(mx.array([prefill[i : i + chunk]]), cache=cache)
            mx.eval(logits)

    index = _build_ngram_index(prompt_ids, n)
    generated: list[int] = []
    pending = last
    stats.ttft_s = time.perf_counter() - t0
    while len(generated) < max_tokens:
        drafts = propose(prompt_ids, generated, index, n, k)
        input_ids = [pending] + drafts
        logits = model(mx.array([input_ids]), cache=cache)
        mx.eval(logits)
        stats.steps += 1
        stats.proposed += len(drafts)
        accepted: list[int] = []
        matched = 0
        for i, draft in enumerate(drafts):
            if ignore_eos is False and draft in eos:
                break
            target = int(mx.argmax(logits[0, i]))
            if target == draft:
                accepted.append(draft)
                matched += 1
            else:
                accepted.append(target)  # corrected token, then stop
                break
        if matched == len(drafts):
            # all drafts accepted: take the bonus token from the last position
            accepted.append(int(mx.argmax(logits[0, len(drafts)])))
        if not ignore_eos and accepted and accepted[-1] in eos:
            accepted = accepted[:-1]
            finished = True
        else:
            finished = False
        stats.accepted += matched
        # roll the cache back to (new context length - 1)
        new_ctx_len = len(ctx) + len(accepted)
        fed = len(input_ids)
        keep = new_ctx_len - 1            # last token stays pending for the next step
        excess = (len(ctx) - 1 + fed) - keep
        if excess > 0:
            trim_prompt_cache(cache, excess)
        ctx.extend(accepted)
        index_append(index, ctx, len(ctx) - len(accepted), n)
        generated.extend(accepted)
        stats.tokens = len(generated)
        pending = ctx[-1]
        if finished:
            break
        if len(generated) >= max_tokens:
            break
    stats.total_s = time.perf_counter() - t0
    text = tokenizer.decode(generated[:max_tokens]) if hasattr(tokenizer, "decode") else ""
    try:
        text = tokenizer.decode(generated[:max_tokens], skip_special_tokens=True)
    except TypeError:
        pass
    out = stats.as_dict()
    out["token_ids"] = generated[:max_tokens]
    return text, out


def plain_generate(model, tokenizer, prompt_ids: list[int], *, max_tokens: int = 128,
                   ignore_eos: bool = True) -> tuple[str, dict]:
    """Greedy baseline with the same cache handling (for A/B checks)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    ctx = list(prompt_ids)
    t0 = time.perf_counter()
    chunk = 1024
    prefill = ctx[:-1]
    for i in range(0, len(prefill), chunk):
        logits = model(mx.array([prefill[i : i + chunk]]), cache=cache)
        mx.eval(logits)
    tokens: list[int] = []
    pending = ctx[-1]
    ttft = time.perf_counter() - t0
    steps = 0
    while len(tokens) < max_tokens:
        logits = model(mx.array([[pending]]), cache=cache)
        mx.eval(logits)
        steps += 1
        nxt = int(mx.argmax(logits[0, -1]))
        tokens.append(nxt)
        pending = nxt
    total = time.perf_counter() - t0
    decode = total - ttft
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text, {"steps": steps, "tokens": len(tokens), "ttft_s": round(ttft, 3),
                  "total_s": round(total, 3),
                  "decode_tok_s": round(len(tokens) / decode, 2) if decode > 0 else 0.0,
                  "token_ids": tokens}
