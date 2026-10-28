"""Benchmark a matrix of modern small MLX models on the same agent workload.

Each model is served by Terrazzo with --profile agent and an identical prewarmed
system prompt. We run 1, 4 and 16 concurrent sessions (2 turns, 64 tokens) and
report server-side token counts, TTFT, throughput and MLX peak memory.

    python bench/model_matrix.py --models qwen3.5-0.8b,qwen3.5-2b,qwen3.5-4b,gemma4-e2b
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import co_design_ablation as A  # noqa: E402

DEFAULT_MODELS = {
    "qwen3.5-0.8b": ("mlx-community/Qwen3.5-0.8B-4bit", "Qwen3.5 0.8B · 4bit"),
    "qwen3.5-2b": ("mlx-community/Qwen3.5-2B-4bit", "Qwen3.5 2B · 4bit"),
    "qwen3.5-4b": ("mlx-community/Qwen3.5-4B-4bit", "Qwen3.5 4B · 4bit"),
    "gemma4-e2b": ("mlx-community/gemma-4-e2b-it-4bit", "Gemma 4 E2B · 4bit"),
}
MODEL_ROOT = ROOT / ".hf-models"


def model_path(name: str) -> Path:
    p = MODEL_ROOT / name
    if not p.exists():
        raise SystemExit(f"missing model directory: {p}")
    return p


def dir_size_gb(p: Path) -> float:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1e9, 2)


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--sessions", default="1,4,16")
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--system-tokens", type=int, default=2000)
    ap.add_argument("--port-base", type=int, default=8960)
    ap.add_argument("--cache-mode", choices=["off", "prefix"], default="off",
                    help="off: compare raw model serving; prefix: use prewarmed prefix cache")
    ap.add_argument("--out", default=str(ROOT / "bench" / "model_matrix_results.json"))
    args = ap.parse_args()
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    session_counts = [int(n) for n in args.sessions.split(",")]
    results = []
    for idx, name in enumerate(names):
        repo, label = DEFAULT_MODELS.get(name, (name, name))
        path = model_path(name)
        port = args.port_base + idx
        log_path = Path(f"/tmp/model_matrix_{name}.log")
        print(f"[matrix] {name} ({repo}) port={port}", flush=True)
        record = {"name": name, "label": label, "repo": repo,
                  "model_path": str(path), "size_gb": dir_size_gb(path),
                  "cache_mode": args.cache_mode, "workloads": {}, "errors": []}
        try:
            systems, shared = A.build_systems("shared", max(session_counts),
                                              args.system_tokens)
            t0 = time.perf_counter()
            if args.cache_mode == "prefix":
                srv = A.start_server("b", str(path), port, shared, log_path)
            else:
                # raw model comparison: no prefix cache, so it works for hybrid
                # architectures (Qwen3.5 ArraysCache) that cannot snapshot KV yet
                srv = A.start_server("a", str(path), port, None, log_path)
            record["startup_s"] = round(time.perf_counter() - t0, 2)
            record["prewarm_s"] = round(srv.get("prewarm_s", 0.0), 2)
            try:
                for sessions in session_counts:
                    systems_n, _ = A.build_systems("shared", sessions, args.system_tokens)
                    print(f"  sessions={sessions}", flush=True)
                    summary = A.run_workload(
                        srv, systems_n, args.turns, args.max_tokens,
                        user_template="请总结规范要点 {session}（第 {turn} 轮）")
                    prompt_tokens = [r.get("prompt_tokens") or 0 for r in summary["rows"]]
                    record["workloads"][str(sessions)] = {
                        "sessions": sessions,
                        "wall_s": summary["wall_s"],
                        "output_tokens": summary["output_tokens"],
                        "aggregate_tok_s": summary["aggregate_tok_s"],
                        "ttft_p50_s": summary["ttft_p50_s"],
                        "ttft_p95_s": summary["ttft_p95_s"],
                        "total_p95_s": summary["total_p95_s"],
                        "total_max_s": summary["total_max_s"],
                        "prompt_tokens_median": int(median(prompt_tokens)),
                        "token_sources": summary["token_sources"],
                        "mlx_peak_mb": summary["mlx_peak_mb"],
                        "peak_kv_tokens": summary["peak_kv_tokens"],
                        "peak_running": summary["peak_running"],
                    }
                    print(f"    tok/s={summary['aggregate_tok_s']} "
                          f"ttft p50={summary['ttft_p50_s']} p95={summary['ttft_p95_s']} "
                          f"peak={summary['mlx_peak_mb']}MB", flush=True)
            finally:
                A.stop_server(srv)
        except Exception as exc:
            record["errors"].append(f"{type(exc).__name__}: {exc}")
            print(f"  ERROR: {exc}", flush=True)
        results.append(record)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                     cwd=str(ROOT), capture_output=True,
                                     text=True).stdout.strip(),
        "loadavg": list(os.getloadavg()),
        "args": vars(args),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}\n")
    print("| model | size | sessions | tok/s | TTFT p50 | TTFT p95 | p95 total | peak MLX | prompt tok |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        for key, w in r["workloads"].items():
            print(f"| {r['label']} | {r['size_gb']} GB | {w['sessions']} | "
                  f"{w['aggregate_tok_s']} | {w['ttft_p50_s']} | {w['ttft_p95_s']} | "
                  f"{w['total_p95_s']} | {w['mlx_peak_mb']} MB | {w['prompt_tokens_median']} |")
        for err in r["errors"]:
            print(f"| {r['label']} | ERROR | - | - | - | - | - | - | {err[:80]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
