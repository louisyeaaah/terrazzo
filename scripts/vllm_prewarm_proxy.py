"""OpenAI-compatible reverse proxy that pre-warms an engine's prefix cache at startup.

Why: vLLM (incl. vllm-metal) has block-granular automatic prefix caching, but the cache
is empty until one request has actually run. Agent traffic shares one long system prompt,
so the *first* request of the day pays the full prefill (3.7 s for a 2544-token prompt in
our measurements) while every later request hits the cache. Pre-registering that prompt
on startup removes the cold-start penalty without touching the engine.

    python scripts/vllm_prewarm_proxy.py \
        --backend http://127.0.0.1:8401 --port 8402 --system-tokens 2544

Clients then point at the proxy instead of the backend. The proxy forwards
/v1/chat/completions (streaming and non-streaming), /v1/models and /health verbatim.

The system prompt must match what the clients send: pass `--system-file`, or let the
proxy rebuild the same text that `bench/agent_workload.py` generates (`--system-tokens`).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "bench"))


def build_system_text(system_tokens: int, system_file: str | None) -> str:
    if system_file:
        return Path(system_file).read_text()
    from agent_workload import build_system_prompt  # same text the benchmark sends

    return build_system_prompt(system_tokens)


def backend_model(backend: str) -> str:
    with urllib.request.urlopen(backend.rstrip("/") + "/v1/models", timeout=30) as r:
        return json.loads(r.read())["data"][0]["id"]


def prewarm(backend: str, model: str, system_text: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": "warmup"},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        backend.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read())
    wall = time.perf_counter() - t0
    usage = obj.get("usage") or {}
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "system_tokens_hint": len(system_text),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backend = ""
    prewarm_stats: dict = {}

    def log_message(self, fmt, *args):  # quiet
        pass

    def _forward(self, method: str, body: bytes | None, stream: bool) -> None:
        url = self.backend.rstrip("/") + self.path
        req = urllib.request.Request(url, data=body, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = r.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:  # pragma: no cover - proxy failure path
            payload = json.dumps({"error": {"message": str(exc)}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/stats":
            payload = json.dumps({"prewarm": self.prewarm_stats}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._forward("GET", None, False)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        stream = False
        if body:
            try:
                stream = bool(json.loads(body).get("stream"))
            except Exception:
                pass
        self._forward("POST", body, stream)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8401")
    ap.add_argument("--port", type=int, default=8402)
    ap.add_argument("--system-tokens", type=int, default=2544)
    ap.add_argument("--system-file", default=None)
    ap.add_argument("--skip-prewarm", action="store_true")
    args = ap.parse_args()

    model = backend_model(args.backend)
    stats = {"backend": args.backend, "model": model}
    if not args.skip_prewarm:
        system_text = build_system_text(args.system_tokens, args.system_file)
        stats.update(prewarm(args.backend, model, system_text))
        stats["system_chars"] = len(system_text)
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    Handler.backend = args.backend
    Handler.prewarm_stats = stats
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[prewarm-proxy] listening on http://127.0.0.1:{args.port} -> {args.backend}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
