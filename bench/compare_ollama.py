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
            "decode_s": ev,
            "decode_tok_s": d.get("eval_count", 0) / max(1e-9, ev),
            "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
        })
    return {"runs": runs,
            "decode_tok_s": sorted(r["decode_tok_s"] for r in runs)[len(runs) // 2],
            "decode_tokens": runs[0]["eval_count"],
            "decode_s": runs[0]["decode_s"],
            "prompt_eval_s": runs[0]["prompt_eval_s"]}


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
        from terrazzo.sequence import SamplingParams
        t0 = time.perf_counter()
        seq = eng.add_request(prompt_token_ids=ids,
                              params=SamplingParams(max_tokens=max_tokens, temperature=0.0))
        prev = 0
        first_t = last_t = None
        while not seq.status.is_finished:
            eng.step()
            n = len(seq.output_token_ids)
            if n > prev:
                now = time.perf_counter()
                if first_t is None:
                    first_t = now
                last_t = now
                prev = n
        n = len(seq.output_token_ids)
        decode_s = (last_t - first_t) if (first_t is not None and last_t is not None) else 0.0
        runs.append({"decode_tokens": n,
                     "decode_s": round(decode_s, 4),
                     # first token is not part of the inter-token interval
                     "decode_tok_s": ((n - 1) / decode_s) if n >= 2 and decode_s > 0 else None,
                     "first_token_s": (first_t - t0) if first_t is not None else None})
    eng.close()
    valid = [r["decode_tok_s"] for r in runs if r["decode_tok_s"] is not None]
    return {"runs": runs,
            "decode_tok_s": (sorted(valid)[len(valid) // 2] if valid else None),
            "decode_tokens": runs[0]["decode_tokens"],
            "decode_s": runs[0]["decode_s"],
            "first_token_s": runs[0]["first_token_s"]}


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
         "decode_tokens": ollama["decode_tokens"],
         "decode_s": round(ollama["decode_s"], 4),
         "prompt_eval_s": round(ollama["prompt_eval_s"], 3),
         "timing_source": "server eval_count/eval_duration"},
        {"engine": "terrazzo", "num_ctx": None,
         "decode_tok_s": (round(engine["decode_tok_s"], 1)
                          if engine["decode_tok_s"] is not None else None),
         "decode_tokens": engine["decode_tokens"],
         "decode_s": engine["decode_s"],
         "first_token_s": (round(engine["first_token_s"], 3)
                           if engine["first_token_s"] is not None else None),
         "timing_source": "(n-1)/(last_token-first_token)"},
    ]
    print(json.dumps(rows, ensure_ascii=False), flush=True)
    if rows[0]["decode_tok_s"] and rows[1]["decode_tok_s"]:
        print(f"rough cross-check (different timing boundaries), ours/ollama: "
              f"{rows[1]['decode_tok_s'] / rows[0]['decode_tok_s']:.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
