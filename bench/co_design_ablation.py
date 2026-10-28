"""Four-arm cache/scheduler ablation for the Phase-1 co-design question.

Arms (all measured on the same machine/model, one server at a time):

  baseline  native mlx-lm server (no pre-registered prefix)
  cache     Terrazzo prefix cache + FIFO admission, reuse always
  sched     Terrazzo cost-aware admission, prefix cache disabled
  combined  Terrazzo prefix cache + cost-aware admission + reuse=auto

Workloads: shared prefix (16 agents), mixed (8 shared + 8 unique prefixes),
long prompt (4 agents, 8k-token system prompt).  TTFT is the first content delta;
tokens come from server usage when available (mlx-lm streaming does not emit usage,
so that arm is marked chunks(approx)).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from agent_workload import build_system_prompt  # noqa: E402

PY = sys.executable


# --------------------------------------------------------------------------- http
def _post(base: str, path: str, body: dict, timeout: float = 1200.0):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _wait(url: str, timeout: float = 180.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def chat_stream(base: str, model: str, messages: list[dict], max_tokens: int) -> dict:
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    content: list[str] = []
    events = 0
    usage_tokens = None
    prompt_tokens = None
    try:
        resp = _post(base, "/v1/chat/completions", body)
    except Exception:
        # mlx-lm may not accept stream_options; retry without it
        body.pop("stream_options", None)
        resp = _post(base, "/v1/chat/completions", body)
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if "error" in obj:
                raise RuntimeError(json.dumps(obj["error"])[:300])
            u = obj.get("usage") or {}
            if u.get("completion_tokens") is not None:
                usage_tokens = int(u["completion_tokens"])
                prompt_tokens = int(u.get("prompt_tokens") or 0)
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if not piece:
                # fall back to reasoning only if no content ever arrives
                piece = delta.get("reasoning") or ""
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                content.append(piece)
                events += 1
    total = time.perf_counter() - t0
    tokens = usage_tokens if usage_tokens is not None else events
    return {"ttft_s": ttft if ttft is not None else total,
            "total_s": total, "tokens": tokens,
            "token_source": "usage" if usage_tokens is not None else "chunks(approx)",
            "content_events": events, "prompt_tokens": prompt_tokens,
            "content": "".join(content)}


# ------------------------------------------------------------------------- server
ARMS = {
    "a": {"label": "A: no cache + FIFO",
          "flags": ["--no-prefix-cache", "--scheduler-policy", "fifo"], "prewarm": False},
    "b": {"label": "B: cache + FIFO",
          "flags": ["--prefix-reuse", "always", "--scheduler-policy", "fifo"], "prewarm": True},
    "c": {"label": "C: no cache + cost-aware",
          "flags": ["--no-prefix-cache", "--scheduler-policy", "cost"], "prewarm": False},
    "d": {"label": "D: cache + cost-aware",
          "flags": ["--prefix-reuse", "auto", "--scheduler-policy", "cost"], "prewarm": True},
    "b_cap2": {"label": "B-cap2: cache+FIFO, 2 slots", "profile": "default",
               "flags": ["--prefix-reuse", "always", "--scheduler-policy", "fifo",
                         "--max-num-seqs", "2"], "prewarm": True},
    "b_cap4": {"label": "B-cap4: cache+FIFO, 4 slots", "profile": "default",
               "flags": ["--prefix-reuse", "always", "--scheduler-policy", "fifo",
                         "--max-num-seqs", "4"], "prewarm": True},
    "b_cap6": {"label": "B-cap6: cache+FIFO, 6 slots", "profile": "default",
               "flags": ["--prefix-reuse", "always", "--scheduler-policy", "fifo",
                         "--max-num-seqs", "6"], "prewarm": True},
    "b_cap8": {"label": "B-cap8: cache+FIFO, 8 slots", "profile": "default",
               "flags": ["--prefix-reuse", "always", "--scheduler-policy", "fifo",
                         "--max-num-seqs", "8"], "prewarm": True},
    "b_cap12": {"label": "B-cap12: cache+FIFO, 12 slots", "profile": "default",
                "flags": ["--prefix-reuse", "always", "--scheduler-policy", "fifo",
                          "--max-num-seqs", "12"], "prewarm": True},
    "d_fixed": {"label": "D-fixed: cost, frozen estimates",
                "flags": ["--prefix-reuse", "auto", "--scheduler-policy", "cost",
                          "--no-cost-calibration"], "prewarm": True},
    "d_always": {"label": "D-always: cost, always reuse",
                 "flags": ["--prefix-reuse", "always", "--scheduler-policy", "cost"],
                 "prewarm": True},
    "d_never": {"label": "D-never: cost, never reuse",
                "flags": ["--prefix-reuse", "never", "--scheduler-policy", "cost"],
                "prewarm": True},
    "baseline": {"label": "native mlx-lm", "flags": [], "prewarm": False, "native": True},
}


def start_server(arm: str, model: str, port: int, system_text: str | None,
                 log_path: Path) -> dict:
    spec = ARMS[arm]
    if spec.get("native"):
        cmd = [PY, "-m", "mlx_lm.server", "--model", model, "--host", "127.0.0.1",
               "--port", str(port), "--max-tokens", "256",
               "--chat-template-args", '{"enable_thinking": false}']
        ready = f"http://127.0.0.1:{port}/v1/models"
        env = os.environ.copy()
        env["HF_HOME"] = str(ROOT / ".hf-home")
        request_model = model
    else:
        cmd = [PY, "-m", "terrazzo", "serve", "--model", model,
               "--host", "127.0.0.1", "--port", str(port),
               "--profile", spec.get("profile", "agent"), "--no-thinking"]
        cmd += list(spec["flags"])
        ready = f"http://127.0.0.1:{port}/health"
        env = os.environ.copy()
        # let all 16 agent requests arrive before admission, otherwise the cost
        # scheduler only sees whatever the HTTP race admitted first
        env.setdefault("TERRAZZO_ADMIT_SETTLE_MS", "50")
        env.setdefault("TERRAZZO_STACK_DUMP", "1")
        request_model = "terrazzo"
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    if not _wait(ready, timeout=180):
        log.close()
        proc.kill()
        raise RuntimeError(f"server failed to start ({arm}); see {log_path}")
    prewarm_s = 0.0
    base = f"http://127.0.0.1:{port}"
    if spec["prewarm"] and system_text:
        t0 = time.perf_counter()
        try:
            with _post(base, "/v1/prewarm", {"system": system_text}, timeout=900) as r:
                r.read()
        except BaseException:
            # do not leave a half-started server behind on prewarm failure
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            log.close()
            raise
        prewarm_s = time.perf_counter() - t0
    # warm up kernels; not measured
    try:
        chat_stream(base, request_model,
                    [{"role": "user", "content": "warm up: say hi"}], 8)
    except Exception:
        pass
    stats0 = {}
    try:
        with urllib.request.urlopen(base + "/stats", timeout=5) as r:
            stats0 = json.loads(r.read())
    except Exception:
        pass
    return {"proc": proc, "log": log, "base": base, "model": request_model,
            "prewarm_s": prewarm_s, "stats0": stats0}


def stop_server(srv: dict) -> None:
    proc = srv["proc"]
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        srv["log"].close()
    except Exception:
        pass


# ------------------------------------------------------------------------ workload
def build_systems(kind: str, sessions: int, system_tokens: int) -> tuple[list[str], str]:
    base = build_system_prompt(system_tokens)
    if kind == "shared":
        return [base] * sessions, base
    if kind == "mixed":
        systems = [base if i % 2 == 0 else f"[session {i}]\n{base}"[: int(system_tokens * 2.2)]
                   for i in range(sessions)]
        return systems, base
    if kind == "long":
        return [base] * sessions, base
    raise ValueError(kind)


def run_workload(srv: dict, systems: list[str], turns: int, max_tokens: int,
                 stagger_ms: float = 0.0) -> dict:
    sessions = len(systems)
    if stagger_ms <= 0:
        barrier = threading.Barrier(sessions)
    else:
        barrier = None
    rows: list[dict] = []
    lock = threading.Lock()
    errors: "queue.Queue[str]" = queue.Queue()

    def one(i: int) -> None:
        messages = [{"role": "system", "content": systems[i]}]
        if barrier is not None:
            barrier.wait()
        else:
            time.sleep(i * stagger_ms / 1000.0)
        for turn in range(turns):
            messages.append({"role": "user",
                             "content": f"Turn {turn + 1}: summarize the key points."})
            try:
                started_at = time.perf_counter()
                r = chat_stream(srv["base"], srv["model"], messages, max_tokens)
            except Exception as exc:  # keep other sessions running
                errors.put(f"session {i} turn {turn}: {exc}")
                return
            r.update({"session": i, "turn": turn, "started_at": started_at,
                      "group": "shared" if systems[i] == systems[0] else "unique"})
            with lock:
                rows.append(r)
            # feed a short real answer back so later turns share only the system prefix
            messages.append({"role": "assistant", "content": r["content"][:200] or "ok"})

    t0 = time.perf_counter()
    threads = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    if not errors.empty():
        raise RuntimeError("; ".join(list(errors.queue)[:3]))
    ttfts = sorted(r["ttft_s"] for r in rows)
    totals = sorted(r["total_s"] for r in rows)
    by_session: dict[int, float] = {}
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_session[r["session"]] = by_session.get(r["session"], 0.0) + r["total_s"]
        by_group.setdefault(r.get("group", "shared"), []).append(r)

    def pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        return xs[min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))]

    tokens = sum(r["tokens"] for r in rows)
    stats1 = {}
    try:
        with urllib.request.urlopen(srv["base"] + "/stats", timeout=5) as r:
            stats1 = json.loads(r.read())
    except Exception:
        pass
    return {
        "sessions": sessions, "turns": turns, "wall_s": round(wall, 3),
        "requests": len(rows),
        "output_tokens": tokens,
        "token_sources": sorted({r["token_source"] for r in rows}),
        "aggregate_tok_s": round(tokens / max(wall, 1e-9), 1),
        "ttft_p50_s": round(pct(ttfts, 0.50), 3),
        "ttft_p95_s": round(pct(ttfts, 0.95), 3),
        "ttft_min_s": round(ttfts[0], 3) if ttfts else None,
        "total_p95_s": round(pct(totals, 0.95), 3),
        "total_max_s": round(max(totals), 3) if totals else None,
        "slowest_session_s": round(max(by_session.values()), 3) if by_session else None,
        "mlx_peak_mb": stats1.get("mlx_peak_mb"),
        "prefix_cache": stats1.get("prefix_cache"),
        "cost_model": stats1.get("cost_model"),
        "scheduler_policy": stats1.get("scheduler_policy"),
        "request_stages": stats1.get("request_stages"),
        "peak_running": stats1.get("peak_running"),
        "peak_waiting": stats1.get("peak_waiting"),
        "peak_kv_tokens": stats1.get("peak_kv_tokens"),
        "peak_used_blocks": stats1.get("peak_used"),
        "groups": {
            g: {
                "requests": len(rs),
                "ttft_p50_s": round(pct(sorted(x["ttft_s"] for x in rs), 0.50), 3),
                "ttft_p95_s": round(pct(sorted(x["ttft_s"] for x in rs), 0.95), 3),
                "total_p95_s": round(pct(sorted(x["total_s"] for x in rs), 0.95), 3),
                "total_max_s": round(max(x["total_s"] for x in rs), 3),
            } for g, rs in by_group.items()
        },
        "stagger_ms": stagger_ms,
        "execution_control": stats1.get("execution_control"),
        "rows": rows,
    }


# ---------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-model", required=True)
    ap.add_argument("--arms", default="all",
                    help="comma list of a,b,c,d,baseline or 'all' (=a,b,c,d)")
    ap.add_argument("--workloads", default="shared,mixed,long")
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--long-system-tokens", type=int, default=8000)
    ap.add_argument("--port-base", type=int, default=8930)
    ap.add_argument("--stagger-ms", type=float, default=0.0,
                    help="spread session starts by i*stagger_ms (0 = burst arrival)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = os.path.abspath(os.path.expanduser(args.mlx_model))
    default_arms = ["a", "b", "c", "d"]
    arms = (default_arms if args.arms == "all"
            else [a.strip() for a in args.arms.split(",")])
    for a in arms:
        if a not in ARMS:
            ap.error(f"unknown arm {a!r}")
    workloads = [w.strip() for w in args.workloads.split(",")]
    out_path = Path(args.out) if args.out else ROOT / "bench" / "co_design_results.json"
    results = []

    for workload in workloads:
        sessions = args.sessions
        turns = args.turns
        system_tokens = args.system_tokens
        if workload == "long":
            sessions = min(sessions, 4)
            turns = 1
            system_tokens = args.long_system_tokens
        systems, shared_system = build_systems(workload, sessions, system_tokens)
        for idx, arm in enumerate(arms):
            port = args.port_base + idx
            log_path = Path(f"/tmp/ablation_{workload}_{arm}.log")
            print(f"[ablation] workload={workload} arm={arm} sessions={sessions} "
                  f"turns={turns} system_tokens~{system_tokens}", flush=True)
            srv = start_server(arm, model, port, shared_system, log_path)
            try:
                summary = run_workload(srv, systems, turns, args.max_tokens,
                                       stagger_ms=args.stagger_ms)
            finally:
                stop_server(srv)
            summary.update({"workload": workload, "arm": arm,
                            "arm_label": ARMS[arm]["label"],
                            "prewarm_s": round(srv["prewarm_s"], 2),
                            "model": model})
            results.append(summary)
            print(f"  ttft p50={summary['ttft_p50_s']} p95={summary['ttft_p95_s']} "
                  f"tok/s={summary['aggregate_tok_s']} mlx_peak={summary['mlx_peak_mb']}",
                  flush=True)

    payload = {
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "platform": sys.platform,
            "python": sys.version,
            "model": model,
            "loadavg": list(os.getloadavg()),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                capture_output=True, text=True).stdout.strip(),
        },
        "args": vars(args),
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\nwrote {out_path}")
    print("\n| workload | arm | sessions | tok/s | TTFT p50 | TTFT p95 | p95 total | slowest | peak MLX MB | peak KV tok | peak run | prewarm s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['workload']} | {r['arm_label']} | {r['sessions']} | {r['aggregate_tok_s']} | "
              f"{r['ttft_p50_s']} | {r['ttft_p95_s']} | {r['total_p95_s']} | "
              f"{r['slowest_session_s']} | {r['mlx_peak_mb']} | {r['peak_kv_tokens']} | "
              f"{r['peak_running']} | {r['prewarm_s']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
