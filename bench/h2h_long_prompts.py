"""Long system prompt head-to-head: terrazzo vs Ollama (2k / 4k / 8k tokens).

Measures, per system-prompt length:
  * cold TTFT      -- first request with a never-seen system prompt
  * warm-same      -- identical request again (pure prefix reuse)
  * warm-diff      -- same system prompt, different user question (agent pattern)
  * decode tok/s   -- from non-streaming call (Ollama: server counters; ours: wall - ttft)
  * server-side prefill where available (Ollama prompt_eval_duration)

One service at a time; random front marker so no cross-target prefix reuse.

    python bench/h2h_long_prompts.py --mlx-model <dir> --lengths 2000,4000,8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import os
from pathlib import Path

STUDY = Path(os.environ.get("TERRAZZO_STUDY_ROOT",
                            str(Path(__file__).resolve().parents[2] / "deepinfer-study")))
ROOT = Path(__file__).resolve().parents[1]
UNIT = ("知识库条目：KV cache 用于缓存注意力键值，避免重复计算；"
        "规范一：所有回答必须引用条款编号；规范二：涉及删除操作时必须二次确认；"
        "规范三：遇到不确定的信息要明确说明；规范四：回答保持简洁并给出下一步建议。")
QUESTION = "用户问：第1条规范和KV cache有什么关系？请用两句话回答。"
QUESTION2 = "用户问：分页缓存和连续批处理是什么关系？请用两句话回答。"
PY = str(STUDY / ".venv/bin/python")


def _post_json(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _stream_first_token(url: str, body: dict, timeout: float = 600) -> tuple[float, float]:
    """Return (ttft_s, total_s) for an SSE/JSON-lines streaming chat call."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line:
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                piece = ""
                ch = obj.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    piece = d.get("content") or d.get("reasoning") or ""
            else:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                piece = ((obj.get("message") or {}).get("content") or "")
                if obj.get("done"):
                    break
            if piece and first is None:
                first = time.perf_counter()
    total = time.perf_counter() - t0
    return (first - t0 if first is not None else total), total


class Deepinfer:
    def __init__(self, model: str, port: int):
        self.model, self.port = model, port
        self.proc = None
        self.base = f"http://127.0.0.1:{port}"

    def start(self, system_file: str | None = None) -> None:
        args = [PY, "-m", "terrazzo", "serve", "--model", self.model,
                "--port", str(self.port)]
        if system_file:
            args += ["--system-file", system_file]
        self.proc = subprocess.Popen(args, cwd=str(ROOT),
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                urllib.request.urlopen(self.base + "/health", timeout=2)
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("terrazzo server did not become healthy")
        # warm kernels with a short throwaway (does not touch the long prompt prefix)
        self.post([{"role": "user", "content": "热身：说一句你好。"}], 8)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()
            self.proc = None
        time.sleep(1)

    def stream(self, messages, max_tokens: int = 64):
        body = {"model": "terrazzo", "messages": messages, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": True}
        return _stream_first_token(self.base + "/v1/chat/completions", body)

    def post(self, messages, max_tokens: int = 64):
        body = {"model": "terrazzo", "messages": messages, "max_tokens": max_tokens,
                "temperature": 0.0, "stream": False}
        t0 = time.perf_counter()
        obj = _post_json(self.base + "/v1/chat/completions", body)
        wall = time.perf_counter() - t0
        tok = int((obj.get("usage") or {}).get("completion_tokens") or 0)
        return wall, tok


class Ollama:
    def __init__(self, model: str, num_ctx: int, base: str = "http://127.0.0.1:11434"):
        self.model, self.num_ctx, self.base = model, num_ctx, base

    def _body(self, messages, max_tokens, stream):
        return {"model": self.model, "messages": messages, "stream": stream,
                "options": {"num_ctx": self.num_ctx, "temperature": 0.0},
                "think": False, "max_tokens": max_tokens}

    def warm(self):
        subprocess.run(["ollama", "stop", self.model], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.post([{"role": "user", "content": "热身：说一句你好。"}], 8)

    def stream(self, messages, max_tokens: int = 64):
        body = {"model": self.model, "messages": messages, "stream": True,
                "options": {"num_ctx": self.num_ctx, "temperature": 0.0,
                            "num_predict": max_tokens},
                "think": False}
        return _stream_first_token(self.base + "/api/chat", body)

    def post(self, messages, max_tokens: int = 64):
        body = {"model": self.model, "messages": messages, "stream": False,
                "options": {"num_ctx": self.num_ctx, "temperature": 0.0,
                            "num_predict": max_tokens},
                "think": False}
        t0 = time.perf_counter()
        obj = _post_json(self.base + "/api/chat", body)
        wall = time.perf_counter() - t0
        ev = obj.get("eval_duration") or 0
        decode = (obj.get("eval_count") or 0) / (ev / 1e9) if ev else 0.0
        return wall, int(obj.get("eval_count") or 0), decode, (obj.get("prompt_eval_duration") or 0) / 1e9


def build_system(tok, n_tokens: int, marker: str) -> str:
    unit_ids = tok.encode(UNIT)
    k = max(1, n_tokens // len(unit_ids))
    text = f"[run {marker}] " + UNIT * k
    return text


def run_block(name: str, client, system: str, max_tokens: int,
              tok=None) -> dict:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": QUESTION}]
    msgs2 = [{"role": "system", "content": system},
             {"role": "user", "content": QUESTION2}]
    t0 = time.perf_counter()
    cold_ttft, _ = client.stream(msgs, max_tokens)
    cold_total = time.perf_counter() - t0
    warm_same_ttft, _ = client.stream(msgs, max_tokens)
    diff_ttft, diff_total = client.stream(msgs2, max_tokens)
    out = {"target": name,
           "cold_ttft_s": round(cold_ttft, 3),
           "cold_total_s": round(cold_total, 2),
           "warm_same_ttft_s": round(warm_same_ttft, 3),
           "warm_diff_ttft_s": round(diff_ttft, 3)}
    if tok is not None:
        out["prompt_tokens"] = len(tok.encode(system)) + 20
    if isinstance(client, Ollama):
        wall, tokc, decode, prefill = client.post(msgs2, max_tokens)
        out["decode_tok_s"] = round(decode, 1)
        out["server_prefill_s"] = round(prefill, 3)
        out["completion_tokens"] = tokc
    else:
        wall, tokc = client.post(msgs2, max_tokens)
        decode_s = wall - diff_ttft
        if decode_s > 0.01:
            out["decode_tok_s"] = round((max(tokc - 1, 1)) / decode_s, 1)
        else:
            out["decode_tok_s"] = None
        out["completion_tokens"] = tokc
        out["post_wall_s"] = round(wall, 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-model", required=True)
    ap.add_argument("--ollama-model", default="qwen3:4b")
    ap.add_argument("--lengths", default="2000,4000,8000")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--only", default="all", choices=["all", "ours", "ollama"])
    ap.add_argument("--ctx-per-len", default="4096,8192,16384",
                    help="Ollama num_ctx per length (minimal KV allocation per prompt size)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from mlx_lm.utils import load_tokenizer
    tok = load_tokenizer(Path(args.mlx_model))
    lengths = [int(x) for x in args.lengths.split(",")]
    results = []

    if args.only == "all":
        pass
    # 1) our default server (no pre-registration)
    srv = None
    if args.only in ("all", "ours"):
        srv = Deepinfer(args.mlx_model, 8620)
    if srv is not None:
        srv.start()
        try:
            for n in lengths:
                marker = f"def{n}"
                system = build_system(tok, n, marker)
                row = run_block("terrazzo-default", srv, system, args.max_tokens, tok)
                row["target"] += f" n={n}"
                results.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            srv.stop()

    # 2) our pre-registered prefix (start a server per length)
    for n in (lengths if args.only in ("all", "ours") else []):
        marker = f"pre{n}"
        system = build_system(tok, n, marker)
        sysfile = Path(f"/tmp/h2h_long_sys_{n}.txt")
        sysfile.write_text(system)
        srv = Deepinfer(args.mlx_model, 8621)
        srv.start(system_file=str(sysfile))
        try:
            row = run_block("terrazzo-prereg", srv, system, args.max_tokens, tok)
            row["target"] += f" n={n}"
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        finally:
            srv.stop()

    # 3) Ollama (minimal num_ctx per length; a short request absorbs the reload)
    ctxs = [int(x) for x in args.ctx_per_len.split(",")]
    if args.only in ("all", "ollama"):
        for idx, n in enumerate(lengths):
            ctx = ctxs[idx] if idx < len(ctxs) else args.num_ctx
            oll = Ollama(args.ollama_model, ctx)
            oll.post([{"role": "user", "content": "热身：说一句你好。"}], 8)  # loads with this ctx
            time.sleep(3)
            marker = f"oll{n}"
            system = build_system(tok, n, marker)
            row = run_block("ollama", oll, system, args.max_tokens, tok)
            row["target"] += f" n={n} ctx={ctx}"
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    out_path = Path(args.out or "/tmp/h2h_long.jsonl")
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    print(f"WROTE {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
