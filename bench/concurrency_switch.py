"""Load-aware concurrency experiment: shared -> mixed -> shared in one process.

Arms:
  fixed4   FIFO, admission cap 4 for all phases
  fixed16  FIFO, admission cap 16 for all phases
  oracle   cap changes at phase boundaries with foreknowledge (16/4/16)
  auto     simple hit-ratio rule (AutoCap) changes cap 4<->16, min hold 6 s

All arms use the same server implementation/model; each arm is one process that stays
alive across the three phases. Only *new* admissions are affected by the cap.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from agent_workload import build_system_prompt  # noqa: E402
from co_design_ablation import _post, _wait, chat_stream, run_workload  # noqa: E402

PY = sys.executable
MODEL_DEFAULT = os.environ.get("TERRAZZO_MAC_MODEL", "")


def systems_for(phase: str, base: str, sessions: int, phase_idx: int = 0) -> list[str]:
    if phase == "shared":
        return [base] * sessions
    # mixed: half share the prewarmed prefix, half carry a unique system prompt.
    # Include the phase index so a later mixed phase is genuinely cold; otherwise
    # the same unique prompts would hit the prefix cache from the earlier phase.
    return [base if i % 2 == 0 else
            f"[phase {phase_idx} session {i}]\n{base}"[: int(2000 * 2.2)]
            for i in range(sessions)]


def start(arm: str, model: str, port: int, system_text: str, log_path: Path) -> dict:
    cmd = [PY, "-m", "terrazzo", "serve", "--model", model,
           "--host", "127.0.0.1", "--port", str(port), "--no-thinking"]
    if arm == "fixed4":
        # isolate the admission cap: keep the same runner batch as the other arms
        cmd += ["--profile", "default", "--max-num-seqs", "4",
                "--completion-batch-size", "16"]
    else:
        cmd += ["--profile", "agent"]
    if arm in ("oracle", "auto"):
        cmd += []
    env = os.environ.copy()
    # one admission configuration for every arm: otherwise fixed4 (default profile)
    # and agent-profile arms differ in ADMIT_BATCH/ADMIT_WAIT, not just the cap
    env.pop("TERRAZZO_AUTO_CAP", None)
    env.pop("TERRAZZO_ENABLE_CONTROL", None)
    env["TERRAZZO_ADMIT_BATCH"] = "16"
    env["TERRAZZO_ADMIT_WAIT"] = "4"
    env["TERRAZZO_ADMIT_SETTLE_MS"] = "50"
    env["TERRAZZO_STACK_DUMP"] = "1"
    if arm in ("oracle", "auto"):
        env["TERRAZZO_ENABLE_CONTROL"] = "1"
    if arm == "auto":
        env["TERRAZZO_AUTO_CAP"] = "1"
        env["TERRAZZO_AUTO_MIN_HOLD"] = "6"
    log = open(log_path, "wb")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
    if not _wait(f"http://127.0.0.1:{port}/health", timeout=180):
        log.close(); proc.kill()
        raise RuntimeError(f"server failed to start ({arm})")
    with _post(f"http://127.0.0.1:{port}", "/v1/prewarm", {"system": system_text}, timeout=900) as r:
        r.read()
    # warm up kernels; not part of the phases
    chat_stream(f"http://127.0.0.1:{port}", "terrazzo",
                [{"role": "user", "content": "warm up: say hi"}], 8)
    return {"proc": proc, "log": log, "base": f"http://127.0.0.1:{port}",
            "model": "terrazzo", "port": port}


def set_cap(srv: dict, n: int) -> dict:
    with _post(srv["base"], "/v1/concurrency", {"max_num_seqs": n}, timeout=10) as r:
        return json.loads(r.read())


def stop(srv: dict) -> None:
    try:
        srv["proc"].terminate(); srv["proc"].wait(timeout=10)
    except Exception:
        try: srv["proc"].kill()
        except Exception: pass
    try: srv["log"].close()
    except Exception: pass


def summarize_phase(rows: list[dict], wall: float) -> dict:
    def pct(vals, p):
        vals = sorted(vals)
        return vals[min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))] if vals else 0.0
    ttfts = [r["ttft_s"] for r in rows]
    totals = [r["total_s"] for r in rows]
    by_start = sorted(rows, key=lambda r: r.get("started_at", 0))
    head = by_start[:8]
    tail = by_start[-8:]
    return {
        "requests": len(rows),
        "wall_s": round(wall, 3),
        "tok_s": round(sum(r["tokens"] for r in rows) / max(wall, 1e-9), 1),
        "ttft_p50_s": round(pct(ttfts, 0.50), 3),
        "ttft_p95_s": round(pct(ttfts, 0.95), 3),
        "total_p95_s": round(pct(totals, 0.95), 3),
        "switch_wave_ttft_p95_s": round(pct([r["ttft_s"] for r in head], 0.95), 3),
        "steady_ttft_p95_s": round(pct([r["ttft_s"] for r in tail], 0.95), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-model", default=MODEL_DEFAULT, required=not bool(MODEL_DEFAULT))
    ap.add_argument("--arms", default="fixed4,fixed16,oracle,auto")
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--port-base", type=int, default=8900)
    ap.add_argument("--phases", default="shared,mixed,shared")
    ap.add_argument("--stagger-ms", type=float, default=800.0,
                    help="spread session starts within a phase so cap changes affect later arrivals")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model = os.path.abspath(os.path.expanduser(args.mlx_model))
    base = build_system_prompt(2000)
    phases = [p.strip() for p in args.phases.split(",")]
    for ph in phases:
        if ph not in ("shared", "mixed"):
            ap.error(f"unknown phase {ph!r}")
    caps = {"shared": 16, "mixed": 4, "shared2": 16}
    out = Path(args.out) if args.out else ROOT / "bench" / "concurrency_switch_results.json"
    results = []
    for idx, arm in enumerate([a.strip() for a in args.arms.split(",")]):
        port = args.port_base + idx
        log_path = Path(f"/tmp/concurrency_switch_{arm}.log")
        print(f"[switch] arm={arm} port={port}", flush=True)
        srv = start(arm, model, port, base, log_path)
        try:
            phase_results = []
            for phase_idx, phase in enumerate(phases):
                if arm == "oracle":
                    target = caps["mixed"] if phase == "mixed" else 16
                    set_cap(srv, target)
                    time.sleep(0.5)
                systems = systems_for(phase, base, args.sessions, phase_idx)
                t0 = time.perf_counter()
                summary = run_workload(srv, systems, args.turns, args.max_tokens,
                                       stagger_ms=args.stagger_ms)
                wall = time.perf_counter() - t0
                phase_summary = summarize_phase(summary["rows"], wall)
                phase_summary.update({"phase": phase, "phase_index": phase_idx,
                                      "arm": arm})
                # server state after the phase (auto decisions are cumulative)
                try:
                    with urllib.request.urlopen(srv["base"] + "/stats", timeout=5) as r:
                        st = json.loads(r.read())
                    phase_summary["auto_concurrency"] = st.get("auto_concurrency")
                    phase_summary["max_num_seqs"] = st.get("max_num_seqs")
                    phase_summary["mlx_peak_mb"] = st.get("mlx_peak_mb")
                    phase_summary["peak_kv_tokens"] = st.get("peak_kv_tokens")
                    phase_summary["admission"] = st.get("admission")
                    phase_summary["completion_batch_size"] = st.get("completion_batch_size")
                except Exception:
                    pass
                phase_results.append(phase_summary)
                print(f"  {phase}: tok/s={phase_summary['tok_s']} "
                      f"ttft p50={phase_summary['ttft_p50_s']} p95={phase_summary['ttft_p95_s']} "
                      f"cap={phase_summary.get('max_num_seqs')}", flush=True)
            results.append({"arm": arm, "phases": phase_results,
                            "log": str(log_path)})
        finally:
            stop(srv)
    payload = {
        "environment": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": model,
            "loadavg": list(os.getloadavg()),
            "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=str(ROOT), capture_output=True,
                                         text=True).stdout.strip(),
        },
        "args": vars(args),
        "phases": phases,
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")
    print("\n| arm | phase | tok/s | TTFT p50 | TTFT p95 | p95 total | switch-wave p95 | steady p95 | cap | peak MLX |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        for p in r["phases"]:
            print(f"| {r['arm']} | {p['phase']} | {p['tok_s']} | {p['ttft_p50_s']} | "
                  f"{p['ttft_p95_s']} | {p['total_p95_s']} | {p['switch_wave_ttft_p95_s']} | "
                  f"{p['steady_ttft_p95_s']} | {p.get('max_num_seqs')} | {p.get('mlx_peak_mb')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
