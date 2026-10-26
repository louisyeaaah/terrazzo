"""Agent workload benchmark: one long shared system prompt, N concurrent sessions,
multi-turn. Works against any OpenAI-compatible server.

    python bench/agent_workload.py --base http://127.0.0.1:8501 --model terrazzo \
        --label ours --sessions 4 --turns 2 --system-tokens 2000 --max-tokens 64

Token accounting: each request asks for ``stream_options.include_usage`` and the
aggregate throughput uses the server-reported ``completion_tokens``. If a backend
does not report usage (older servers), the script falls back to counting non-empty
content events and marks the result ``token_source: "chunks(approx)"`` — those
numbers must not be compared across backends as strict token counts.

Multi-turn conversations feed the real assistant answer into the next turn (the
shared system prompt keeps being the common prefix, like real agent traffic).

Every run records environment metadata (host, Python, git commit, backend version,
load average, /stats deltas when available) so results are reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FILLER = (
    "背景资料：本系统是一个运维知识库助手，需要依据下面的规范回答用户问题。"
    "规范一：所有回答必须引用条款编号；规范二：涉及删除操作时必须二次确认；"
    "规范三：遇到不确定的信息要明确说明；规范四：回答保持简洁并给出下一步建议。"
)


def build_system_prompt(target_tokens: int) -> str:
    # ~2.2 chars per token for this Chinese filler; overshoot a little.
    target_chars = int(target_tokens * 2.2)
    text = FILLER * (target_chars // len(FILLER) + 1)
    return text[:target_chars]


class RssSampler:
    def __init__(self, pid: int, interval: float = 0.3) -> None:
        self.pid = pid
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import psutil

            proc = psutil.Process(self.pid)
        except Exception:
            proc = None
        while not self._stop.is_set():
            try:
                if proc is not None:
                    rss = proc.memory_info().rss
                else:
                    out = subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(self.pid)], text=True).strip()
                    rss = int(out) * 1024 if out else 0
                if rss:
                    self.peak = max(self.peak, rss / 1024.0 / 1024.0)  # MB
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=1)
        return self.peak


def find_pid(port: int) -> int | None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def _get_json(url: str, timeout: float = 5.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def probe_environment(base: str, stats_url: str | None) -> dict:
    """Capture what is needed to reproduce/compare a run."""
    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        env["loadavg"] = list(os.getloadavg())
    except (AttributeError, OSError):
        env["loadavg"] = None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=str(Path(__file__).resolve().parents[1]), stderr=subprocess.DEVNULL)
        env["git_commit"] = out.strip()
    except Exception:
        env["git_commit"] = None
    try:
        import terrazzo  # noqa: F401

        env["terrazzo_version"] = getattr(terrazzo, "__version__", None)
    except Exception:
        env["terrazzo_version"] = None
    backend = _get_json(base.rstrip("/") + "/api/version")
    if isinstance(backend, dict) and "version" in backend:
        env["backend_version"] = backend["version"]
    if stats_url:
        env["stats"] = _get_json(stats_url) or {"error": "unavailable"}
    return env


def chat(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    """One streaming request; returns ttft, token usage, content and timings."""
    body = json.dumps({
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    content_events = 0
    content: list[str] = []
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=600) as resp:
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
            usage = obj.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = usage.get("prompt_tokens")
            if usage.get("completion_tokens") is not None:
                completion_tokens = usage.get("completion_tokens")
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                content_events += 1
                content.append(piece)
    total = time.perf_counter() - t0
    if completion_tokens is not None:
        tokens, token_source = int(completion_tokens), "usage"
    else:
        tokens, token_source = content_events, "chunks(approx)"
    return {"ttft_s": ttft if ttft is not None else total,
            "tokens": tokens, "token_source": token_source,
            "content_events": content_events,
            "total_s": total, "prompt_tokens": prompt_tokens,
            "content": "".join(content)}


def run_session(base: str, model: str, system: str, turns: int, max_tokens: int,
                user_text: str) -> list[dict]:
    history = [{"role": "system", "content": system}]
    out = []
    for t in range(turns):
        history.append({"role": "user", "content": f"{user_text}（第 {t + 1} 轮）"})
        r = chat(base, model, history, max_tokens)
        out.append(r)
        # carry the real answer into the next turn (agent-like growing conversation)
        history.append({"role": "assistant", "content": r["content"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--system-file", default=None, dest="system_file",
                    help="use this text as the shared system prompt (else synthesized)")
    ap.add_argument("--stats", default=None, help="optional /stats endpoint for memory metrics")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pid = args.server_pid or find_pid(int(args.base.rsplit(":", 1)[-1].split("/")[0]))
    sampler = RssSampler(pid) if pid else None
    system = (open(args.system_file).read() if args.system_file
              else build_system_prompt(args.system_tokens))

    env_before = probe_environment(args.base, args.stats)
    # warm up the server (first request compiles kernels / fills caches)
    chat(args.base, args.model, [{"role": "user", "content": "热身：你好"}], 8)
    stats_before = _get_json(args.stats) if args.stats else None

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.sessions) as pool:
        futures = [
            pool.submit(run_session, args.base, args.model, system, args.turns,
                        args.max_tokens, f"请总结规范要点 {i}")
            for i in range(args.sessions)
        ]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - t0

    rss_peak = sampler.stop() if sampler else 0.0
    server_stats = {}
    if args.stats:
        try:
            with urllib.request.urlopen(args.stats, timeout=5) as r:
                server_stats = json.loads(r.read().decode())
        except Exception as exc:
            server_stats = {"error": str(exc)}
    flat = [r for session in results for r in session]
    ttfts = [r["ttft_s"] for r in flat]
    output_tokens = sum(r["tokens"] for r in flat)
    content_events = sum(r.get("content_events", 0) for r in flat)
    sources = sorted({r["token_source"] for r in flat})
    token_source = sources[0] if len(sources) == 1 else "mixed"
    prompt_tokens = max((r["prompt_tokens"] or 0) for r in flat)

    server_delta = {}
    if isinstance(stats_before, dict) and isinstance(server_stats, dict):
        for key in ("decode_tokens", "prefill_tokens", "steps", "prefix_hit_tokens"):
            if key in server_stats and key in stats_before:
                server_delta[key] = server_stats[key] - stats_before[key]
        if server_delta.get("decode_tokens"):
            d = server_delta["decode_tokens"]
            server_delta["decode_tok_s"] = round(d / wall, 1)

    summary = {
        "label": args.label, "base": args.base, "model": args.model,
        "sessions": args.sessions, "turns": args.turns,
        "system_tokens": prompt_tokens or args.system_tokens,
        "max_tokens": args.max_tokens,
        "wall_s": round(wall, 2),
        "output_tokens": output_tokens,
        "content_events": content_events,
        "token_source": token_source,
        "aggregate_tok_s": round(output_tokens / wall, 1) if wall else 0.0,
        "per_session_tok_s": round(output_tokens / wall / args.sessions, 1) if wall else 0.0,
        "server_decode_tokens": server_delta.get("decode_tokens"),
        "server_decode_tok_s": server_delta.get("decode_tok_s"),
        "ttft_p50_s": round(statistics.median(ttfts), 3),
        "ttft_p95_s": round(sorted(ttfts)[max(0, int(len(ttfts) * 0.95) - 1)], 3),
        "ttft_min_s": round(min(ttfts), 3),
        "peak_rss_mb": round(rss_peak, 0),
        "environment": env_before,
        "server_memory": {k: v for k, v in server_stats.items()
                          if k in ("mlx_active_mb", "mlx_peak_mb", "mlx_cache_mb", "used",
                                   "num_blocks", "block_size", "prefix_cache", "runtime", "auto_prefix")},
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps({"summary": summary, "runs": flat}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
