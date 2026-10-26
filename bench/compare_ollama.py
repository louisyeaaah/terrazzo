"""Fair on-machine comparison: terrazzo engine vs Ollama, decode-only.

Runs sequentially so the two stacks never share memory (Ollama's default 256k
context can hold ~43 GB of KV cache and would otherwise slow MLX down):

  1. Ollama: fixed num_ctx, think=false, temperature 0, warmup, then N tokens.
     decode rate = eval_count / eval_duration.
  2. `ollama stop` (free its memory), then load the MLX model, warmup,
     decode rate = decode tokens / (total - ttft).

    python bench/compare_ollama.py --mlx-model <dir> --ollama-model qwen3:4b
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPT = "用三句话解释 KV cache，每句不超过 20 个字。"


def ollama_generate(base: str, model: str, prompt: str, *, num_predict: int,
                    num_ctx: int, think: bool) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "think": think,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def measure_ollama(base: str, model: str, prompt: str, max_tokens: int, ctx: int) -> dict:
    ollama_generate(base, model, prompt, num_predict=8, num_ctx=ctx, think=False)  # warmup
    runs = []
    for _ in range(3):
        d = ollama_generate(base, model, prompt, num_predict=max_tokens, num_ctx=ctx, think=False)
        ev = d.get("eval_duration", 1) / 1e9
        runs.append({
            "eval_count": d.get("eval_count"),
            "decode_tok_s": d.get("eval_count", 0) / max(1e-9, ev),
            "ttft_s": d.get("prompt_eval_duration", 0) / 1e9,
        })
    return {"runs": runs,
            "decode_tok_s": sorted(r["decode_tok_s"] for r in runs)[len(runs) // 2],
            "ttft_s": runs[0]["ttft_s"]}


def measure_engine(model: str, prompt: str, max_tokens: int) -> dict:
    from terrazzo import Engine, EngineConfig

    eng = Engine(EngineConfig(model=model, num_gpu_blocks=0, max_num_seqs=4,
                              enable_prefix_cache=False))
    tok = eng.runner.tokenizer
    try:
        ids = list(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True,
                                           enable_thinking=False))
    except TypeError:
        ids = list(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True))
    eng.generate(prompt_token_ids=ids, max_tokens=8)  # warmup
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        eng.generate(prompt_token_ids=ids, max_tokens=1)
        ttft = time.perf_counter() - t0
        s0 = eng.stats()["decode_tokens"]
        t0 = time.perf_counter()
        eng.generate(prompt_token_ids=ids, max_tokens=max_tokens)
        total = time.perf_counter() - t0
        dec = eng.stats()["decode_tokens"] - s0
        runs.append({"decode_tokens": dec,
                     "decode_tok_s": dec / max(1e-6, total - ttft),
                     "ttft_s": ttft})
    eng.close()
    return {"runs": runs,
            "decode_tok_s": sorted(r["decode_tok_s"] for r in runs)[len(runs) // 2],
            "ttft_s": runs[0]["ttft_s"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-model", required=True, dest="mlx_model")
    ap.add_argument("--ollama-model", default="qwen3:4b", dest="ollama_model")
    ap.add_argument("--ollama-base", default="http://127.0.0.1:11434", dest="ollama_base")
    ap.add_argument("--num-ctx", type=int, default=8192, dest="num_ctx")
    ap.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    ollama = measure_ollama(args.ollama_base, args.ollama_model, args.prompt,
                            args.max_tokens, args.num_ctx)
    subprocess.run(["ollama", "stop", args.ollama_model], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    engine = measure_engine(args.mlx_model, args.prompt, args.max_tokens)

    rows = [
        {"engine": f"ollama:{args.ollama_model}", "num_ctx": args.num_ctx,
         "decode_tok_s": round(ollama["decode_tok_s"], 1),
         "ttft_s": round(ollama["ttft_s"], 3)},
        {"engine": "terrazzo", "num_ctx": None,
         "decode_tok_s": round(engine["decode_tok_s"], 1),
         "ttft_s": round(engine["ttft_s"], 3)},
    ]
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    print(f"ratio (ours / ollama): {rows[1]['decode_tok_s'] / rows[0]['decode_tok_s']:.2f}x",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
