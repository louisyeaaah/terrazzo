"""Reliability checks around the agent workflow: prewarm, abort, eviction, recovery."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from agent_review_workflow import build_system_prompt, TOOLS  # noqa: E402
PY = sys.executable


def wait(base, timeout=180):
    t0 = time.perf_counter()
    for _ in range(timeout):
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not become healthy")


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read().decode())


def stream_first(base, body):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=120)


def chat(base, messages):
    return post(base, "/v1/chat/completions", {
        "model": "terrazzo", "messages": messages, "max_tokens": 16,
        "temperature": 0.0, "stream": False,
    }, timeout=120)


def acceptance_report(checks: dict) -> dict:
    ev = checks.get("eviction", {})
    st = checks.get("stats", {})
    pc = st.get("prefix_cache", {})
    return {
        "abort_success_and_recovery": (
            checks.get("abort", {}).get("aborted") is True
            and checks.get("after_abort") in ("stop", "length")),
        "malformed_http_400": checks.get("malformed_status") == 400,
        "malformed_recovery": checks.get("after_malformed") in ("stop", "length"),
        "eviction_observed_and_recovery": (
            (ev.get("evictions") or 0) > 0
            and checks.get("after_eviction") in ("stop", "length")),
        # Missing active is an unobserved condition, not a pass.
        "idle_terminal_state": (
            st.get("running") == 0 and st.get("waiting") == 0
            and pc.get("active") == 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/yezhipeng/project/dsh-other/deepinfer-study/.hf-cache/hub/models--mlx-community--Qwen3-4B-4bit/snapshots/4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25")
    ap.add_argument("--port", type=int, default=9360)
    ap.add_argument("--out", default=str(ROOT / "bench" / "agent_reliability_results.json"))
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith("TERRAZZO_")}
    env.update({"TERRAZZO_AUTO_CAP": "0", "TERRAZZO_ADMIT_SETTLE_MS": "0"})
    cmd = [PY, "-m", "terrazzo", "serve", "--model", args.model,
           "--host", "127.0.0.1", "--port", str(args.port),
           "--profile", "agent", "--no-thinking", "--prefix-reuse", "always",
           "--scheduler-policy", "fifo", "--prefix-cache-mb", "400",
           "--max-num-seqs", "16"]
    log = open("/tmp/agent_reliability.log", "wb")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
    result = {"startup_s": None, "prewarm_ms": [], "checks": {}}
    try:
        result["startup_s"] = round(wait(base), 2)
        system = build_system_prompt(20)
        pw = post(base, "/v1/prewarm", {"system": system}, timeout=900)
        result["prewarm_ms"].append(pw.get("ms"))
        result["prewarm_registered"] = pw.get("registered")

        # Warm one real tool-call request.
        first = chat(base, [{"role": "system", "content": system},
                            {"role": "user", "content": "读取 terrazzo/blocks.py 后给出一句结论。"}])
        result["checks"]["normal"] = first.get("choices", [{}])[0].get("finish_reason")

        # Abort a streaming generation after the first event, then prove recovery.
        req_id = "reliability-abort"
        stream = stream_first(base, {
            "model": "terrazzo", "request_id": req_id,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "写一段很长的代码审查报告。"}],
            "max_tokens": 256, "temperature": 0.0, "stream": True,
        })
        first_line = stream.readline()
        abort = post(base, "/v1/abort", {"request_id": req_id}, timeout=20)
        stream.close()
        result["checks"]["abort"] = abort
        time.sleep(0.3)
        result["checks"]["after_abort"] = chat(
            base, [{"role": "system", "content": system},
                   {"role": "user", "content": "回答一个字：好"}]).get(
                       "choices", [{}])[0].get("finish_reason")

        # Fill a deliberately small prefix-cache budget with distinct prefixes.
        for i in range(3):
            post(base, "/v1/prewarm", {"system": f"[variant {i}]\n{system}"}, timeout=900)
        ev = get(base, "/stats").get("prefix_cache", {})
        result["checks"]["eviction"] = {k: ev.get(k) for k in
                                        ("entries", "evictions", "active", "nbytes")}
        result["checks"]["after_eviction"] = chat(
            base, [{"role": "system", "content": system},
                   {"role": "user", "content": "回答一个字：好"}]).get(
                       "choices", [{}])[0].get("finish_reason")

        # Malformed request must fail without poisoning the executor.
        try:
            req = urllib.request.Request(base + "/v1/chat/completions",
                                         data=b"{not-json",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                code = r.status
        except urllib.error.HTTPError as exc:
            code = exc.code
        result["checks"]["malformed_status"] = code
        result["checks"]["after_malformed"] = chat(
            base, [{"role": "user", "content": "回答一个字：好"}]).get(
                "choices", [{}])[0].get("finish_reason")
        stats = get(base, "/stats")
        result["checks"]["stats"] = {
            "running": stats.get("running"), "waiting": stats.get("waiting"),
            "request_stages": stats.get("request_stages"),
            "prefix_cache": {k: stats.get("prefix_cache", {}).get(k) for k in
                             ("hits", "evictions", "entries", "active")},
        }
        result["acceptance"] = acceptance_report(result["checks"])
        # Input-error handling is not executor failure recovery; leave that
        # acceptance explicitly absent rather than inferring it from HTTP 400.
        result["ok"] = all(result["acceptance"].values())
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        log.close()
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print("wrote", args.out)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
