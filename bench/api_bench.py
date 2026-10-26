"""Benchmark any OpenAI-compatible local server (streaming).

Measures TTFT, decode tok/s and concurrency throughput for a fixed prompt set so
that terrazzo can be compared against mlx-lm.server / Ollama / LM Studio on
the *same* machine and model.

    python bench/api_bench.py --base http://127.0.0.1:8103 --label mlx-lm
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SYSTEM = (
    "你是一个严谨的中文技术助理。回答必须准确、简洁，并在必要时给出示例。"
    "以下是背景资料，请在所有回答中遵循：" + "知识库条目：KV cache 用于缓存注意力键值，避免重复计算；" * 20
)
QUESTION = "用户问：KV cache 有什么好处？请用两句话回答。"
QUESTION2 = "用户问：分页缓存和连续批处理有什么关系？请用两句话回答。"
SHORT = "用一句话解释 KV cache。"


def _post(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    """Non-streaming call: authoritative token counts + total wall time."""
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        obj = json.loads(r.read())
    wall = time.perf_counter() - t0
    usage = obj.get("usage") or {}
    return {"wall_s": wall, "tokens": int(usage.get("completion_tokens") or 0)}


def _post_stream(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = (delta.get("content") or delta.get("reasoning")
                     or delta.get("reasoning_content") or "")
            if piece:
                if first is None:
                    first = time.perf_counter()
                n += 1
    total = time.perf_counter() - t0
    if first is None:
        return {"ttft_s": round(total, 3), "pieces": n, "total_s": round(total, 3)}
    return {"ttft_s": round(first - t0, 3), "pieces": n, "total_s": round(total, 3)}


def _case(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    """TTFT from a streaming call, token count from a non-streaming call."""
    st = _post_stream(base, model, messages, max_tokens)
    ns = _post(base, model, messages, max_tokens)
    tokens = ns["tokens"]
    decode_s = max(st["total_s"] - st["ttft_s"], 1e-9)
    return {
        "ttft_s": st["ttft_s"],
        "tokens": tokens,
        "total_s": round(ns["wall_s"], 3),
        "decode_tok_s": round((tokens - 1) / decode_s, 1) if tokens > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default="default_model")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--system-file", default=None,
                    help="text file with the system prompt (default: built-in SYSTEM)")
    args = ap.parse_args()

    global SYSTEM
    if args.system_file:
        SYSTEM = Path(args.system_file).read_text()
    msgs_short = [{"role": "user", "content": SHORT}]
    msgs_sys = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION}]
    msgs_sys2 = [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": QUESTION2}]

    out = {"label": args.label, "base": args.base, "cases": {}}

    # cold: the very first long-system request this server has seen
    out["cases"]["system_cold"] = _case(args.base, args.model, msgs_sys, args.max_tokens)
    # warm identical
    out["cases"]["system_warm_same"] = _case(args.base, args.model, msgs_sys, args.max_tokens)
    # warm, different user question -> partial prefix reuse
    out["cases"]["system_warm_diff"] = _case(args.base, args.model, msgs_sys2, args.max_tokens)

    # no system prompt baseline
    short = [_case(args.base, args.model, msgs_short, args.max_tokens)
             for _ in range(args.repeat)]
    out["cases"]["short"] = {
        "ttft_s": round(statistics.median(x["ttft_s"] for x in short), 3),
        "decode_tok_s": round(statistics.median(x["decode_tok_s"] for x in short), 1),
    }

    # concurrency: long system + different questions
    def ask(i: int):
        m = [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"用户问：第{i}个问题，KV cache 有什么好处？"}]
        return _post(args.base, args.model, m, args.max_tokens)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(ask, range(args.concurrency)))
    wall = time.perf_counter() - t0
    total = sum(r["tokens"] for r in res)
    out["cases"]["concurrent"] = {
        "n": args.concurrency, "wall_s": round(wall, 2), "tokens": total,
        "aggregate_tok_s": round(total / wall, 1),
        "per_req_tok_s": round(total / args.concurrency / wall, 1),
    }
    print(json.dumps(out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
