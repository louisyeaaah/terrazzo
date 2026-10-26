"""Soak test: run concurrent agent sessions for a duration and report stability.

    python bench/soak.py --base http://127.0.0.1:8501 --model terrazzo \
        --sessions 4 --duration 600 --system-file system.txt

Records per-request TTFT/total time, error kinds (HTTP status / exception), and samples
/stats (memory guard, MLX active memory, prefix cache, steps) every 2 seconds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

FILLER = "请根据上面的规范回答：这个操作需要几步确认？" * 2


def chat(base: str, model: str, system: str, max_tokens: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": FILLER}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
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
                if choices:
                    delta = choices[0].get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        tokens += 1
        return {"ok": True, "ttft_s": ttft or (time.perf_counter() - t0),
                "total_s": time.perf_counter() - t0, "tokens": tokens}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}", "status": exc.code}
    except Exception as exc:  # connection reset, timeout, ...
        return {"ok": False, "error": type(exc).__name__}


def fetch_stats(base: str) -> dict:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/stats", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as exc:
        return {"error": type(exc).__name__}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--duration", type=float, default=600, help="seconds")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--system-file", default=None, dest="system_file")
    ap.add_argument("--out", default=None, help="write the raw record JSONL here")
    args = ap.parse_args()

    system = Path(args.system_file).read_text() if args.system_file else "你是一个严谨的技术助理。" * 40
    deadline = time.time() + args.duration
    lock = threading.Lock()
    records: list[dict] = []
    samples: list[dict] = []
    stop = threading.Event()

    def worker() -> None:
        while not stop.is_set() and time.time() < deadline:
            r = chat(args.base, args.model, system, args.max_tokens)
            r["at"] = round(time.time(), 3)
            with lock:
                records.append(r)

    def sampler() -> None:
        while not stop.is_set() and time.time() < deadline:
            st = fetch_stats(args.base)
            mg = st.get("memory_guard") or {}
            samples.append({
                "at": round(time.time(), 3),
                "mlx_active_mb": st.get("mlx_active_mb"),
                "mlx_peak_mb": st.get("mlx_peak_mb"),
                "steps": st.get("steps"),
                "preemptions": st.get("preemptions"),
                "over_memory": mg.get("over_memory"),
                "rejections": mg.get("rejections"),
                "prefix_entries": (st.get("prefix_cache") or {}).get("entries"),
                "error": st.get("error"),
            })
            stop.wait(2.0)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.sessions)]
    threads.append(threading.Thread(target=sampler, daemon=True))
    for t in threads:
        t.start()
    while time.time() < deadline:
        time.sleep(1)
    stop.set()
    for t in threads:
        t.join(timeout=10)

    ok = [r for r in records if r["ok"]]
    errors: dict[str, int] = {}
    for r in records:
        if not r["ok"]:
            errors[r.get("error", "?")] = errors.get(r.get("error", "?"), 0) + 1
    ttfts = [r["ttft_s"] for r in ok]
    total_tokens = sum(r["tokens"] for r in ok)
    summary = {
        "duration_s": args.duration,
        "sessions": args.sessions,
        "requests": len(records),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(records), 3) if records else 0.0,
        "errors": errors,
        "aggregate_tok_s": round(total_tokens / args.duration, 1),
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_p95_s": round(sorted(ttfts)[max(0, int(len(ttfts) * 0.95) - 1)], 3) if ttfts else None,
        "peak_mlx_active_mb": max((s.get("mlx_active_mb") or 0 for s in samples), default=0),
        "any_over_memory": any(s.get("over_memory") for s in samples),
        "stats_errors": sum(1 for s in samples if s.get("error")),
        "samples": len(samples),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0 if len(ok) > 0 and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
