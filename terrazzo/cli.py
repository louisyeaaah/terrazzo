"""Command line interface: generate / serve / bench / tests."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unittest

from .sequence import SamplingParams


def _resolve_model(model: str) -> str:
    """Use the locally cached snapshot for a repo id when one exists.

    `chat`/`generate` default to a repo id; without HF_HOME pointing at the study
    cache mlx-lm would try to download. Prefer the on-disk snapshot if present.
    """
    from glob import glob
    from os import environ
    from pathlib import Path

    p = Path(model).expanduser()
    if p.exists():
        return str(p)
    try:
        from .models import resolve as _resolve_store

        return str(_resolve_store(model))
    except Exception:
        pass
    roots = []
    if environ.get("HF_HOME"):
        roots.append(Path(environ["HF_HOME"]) / "hub")
    study = Path(__file__).resolve().parents[2]
    roots.extend([study / ".hf-cache" / "hub", Path.home() / ".cache" / "huggingface" / "hub"])
    repo_dir = "models--" + model.replace("/", "--")
    for root in roots:
        hits = sorted(glob(str(root / repo_dir / "snapshots" / "*")))
        if hits:
            return hits[-1]
    return model


def _auto_prefix_budget() -> tuple[int, int]:
    """Prefix snapshot cap/budget: long conversations need more than 4k tokens.

    On >=48 GB machines 8k tokens (~1.2 GB fp16) is affordable; smaller machines
    keep the 4k/1 GB default.
    """
    import os

    total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    if total_gb >= 48:
        # multi-snapshot budget: 2-3 agent system prompts (e.g. 2k + 4k + 8k ~ 2.1 GB)
        # must fit at once; 2 GB only held one 8k snapshot plus small ones.
        return 8192, 4096 * 1000 * 1000
    return 4096, 1 << 30


def _cfg(args):
    from .engine import EngineConfig

    max_tokens, budget = _auto_prefix_budget()
    return EngineConfig(
        model=_resolve_model(args.model),
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        prefill_chunk_size=(getattr(args, "prefill_step_size", None)
                            or args.prefill_chunk_size),
        prefill_batch_size=getattr(args, "prefill_batch_size", 8),
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        max_model_len=args.max_model_len,
        prefix_cache_max_tokens=(args.prefix_max_tokens
                                 if getattr(args, "prefix_max_tokens", None) else max_tokens),
        prefix_cache_bytes=(int(args.prefix_cache_mb * 1e6)
                            if getattr(args, "prefix_cache_mb", None) else budget),
        enable_prefix_cache=not args.no_prefix_cache,
        prefix_reuse=getattr(args, "prefix_reuse", "auto"),
        scheduler_policy=getattr(args, "scheduler_policy", "fifo"),
        scheduler_aging_tokens_per_s=getattr(args, "scheduler_aging", 100.0),
        memory_cost_weight=getattr(args, "memory_cost_weight", 0.1),
        cost_calibration=not getattr(args, "no_cost_calibration", False),
        completion_batch_size=(getattr(args, "completion_batch_size", None)
                               or args.max_num_seqs),
    )


def cmd_generate(args) -> int:
    from .engine import Engine
    engine = Engine(_cfg(args))
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k)
    tok = engine.runner.tokenizer
    if getattr(args, "raw", False):
        # raw text completion: no chat template (Qwen3-Instruct will loop on this)
        prompt_ids = tok.encode(args.prompt)
    else:
        messages = []
        if getattr(args, "system", None):
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        kwargs = {}
        if getattr(args, "thinking", None) is False:
            kwargs["enable_thinking"] = False
        try:
            prompt_ids = list(tok.apply_chat_template(
                messages, add_generation_prompt=True, **kwargs))
        except TypeError:  # tokenizer template does not accept enable_thinking
            prompt_ids = list(tok.apply_chat_template(
                messages, add_generation_prompt=True))
    json_schema = None
    schema_arg = getattr(args, "json_schema", None)
    if schema_arg:
        from .json_schema import SchemaError, compile_schema

        try:
            raw = schema_arg
            if raw.startswith("@"):
                raw = open(raw[1:], encoding="utf-8").read()
            json_schema = compile_schema(json.loads(raw))
        except (OSError, json.JSONDecodeError, SchemaError) as exc:
            print(f"[terrazzo] invalid json schema: {exc}", file=sys.stderr)
            return 2
    t0 = time.perf_counter()
    text = engine.generate(None, prompt_token_ids=prompt_ids, params=params,
                           spec_ngram=getattr(args, "spec_ngram", False),
                           spec_k=getattr(args, "spec_k", 4),
                           spec_n=getattr(args, "spec_n", 3),
                           json_schema=json_schema)
    dt = time.perf_counter() - t0
    print(text)
    st = engine.stats()
    print(f"\n[terrazzo] {dt:.2f}s  decode={st['decode_tokens']} tok  "
          f"prefill={st['prefill_tokens']} tok  steps={st['steps']}")
    spec_stats = getattr(engine, "last_spec_stats", None)
    if spec_stats:
        print(f"[terrazzo] spec: {json.dumps(spec_stats, ensure_ascii=False)}")
    json_stats = getattr(engine, "last_json_stats", None)
    if json_stats:
        print(f"[terrazzo] json: {json.dumps(json_stats, ensure_ascii=False)}")
    if args.metrics:
        print(json.dumps(st, indent=2, ensure_ascii=False))
    engine.close()
    return 0



def cmd_chat(args) -> int:
    """
    from .engine import EngineInteractive terminal chatbot (multi-turn, streamed), ollama-run style."""
    import queue as _queue

    print("[terrazzo] loading model...", flush=True)
    engine = Engine(_cfg(args))
    tok = engine.runner.tokenizer
    thinking = bool(getattr(args, "thinking", False))
    system = getattr(args, "system", None)
    history: list[dict] = []
    if system:
        history.append({"role": "system", "content": system})

    print(f"[terrazzo] ready. model={args.model}", flush=True)
    print("  输入问题回车即可；/help 查看命令，/exit 退出。", flush=True)

    def template(messages: list[dict]) -> list[int]:
        kwargs = {"enable_thinking": thinking}
        try:
            return list(tok.apply_chat_template(messages, add_generation_prompt=True, **kwargs))
        except TypeError:
            return list(tok.apply_chat_template(messages, add_generation_prompt=True))

    def stream_turn(prompt_ids: list[int]) -> tuple[list[int], float, float]:
        params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                                top_p=args.top_p, top_k=args.top_k)
        if getattr(args, "spec_ngram", False):
            # speculative path is blocking (no token streaming) and greedy-only
            t0 = time.perf_counter()
            text = engine.generate(None, prompt_token_ids=prompt_ids, params=params,
                                   spec_ngram=True, spec_k=args.spec_k, spec_n=args.spec_n)
            print(text, end="", flush=True)
            stats = getattr(engine, "last_spec_stats", None) or {}
            ttft = stats.get("ttft_s", 0.0)
            total = time.perf_counter() - t0
            return [], ttft, total
        seq = engine.add_request(prompt_token_ids=prompt_ids, params=params)
        q = engine.stream(seq.request_id)
        out: list[int] = []
        printed = 0
        t0 = time.perf_counter()
        ttft = None
        while True:
            progressed = engine.has_unfinished()
            if progressed:
                engine.step()
            got = False
            while True:
                try:
                    delta = q.get_nowait()
                except _queue.Empty:
                    break
                got = True
                d = delta if isinstance(delta, dict) else {
                    "token_ids": list(getattr(delta, "token_ids", []) or []),
                    "finished": bool(getattr(delta, "finished", False)),
                    "finish_reason": getattr(delta, "finish_reason", None),
                }
                for tid in (d.get("token_ids") or []):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    out.append(tid)
                    # incremental decode: a token can split a multi-byte UTF-8 char,
                    # so decode the whole prefix and print only the safe delta
                    text = engine.runner.decode(out)
                    safe = text.rstrip("\ufffd")
                    if len(safe) > printed:
                        print(safe[printed:], end="", flush=True)
                        printed = len(safe)
                if d.get("finished"):
                    return out, (ttft if ttft is not None else time.perf_counter() - t0), \
                        time.perf_counter() - t0
            if not progressed and not got:
                time.sleep(0.0005)

    try:
        import readline  # noqa: F401  (line editing / history)
        readline.set_history_length(500)
    except Exception:
        pass

    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd = line.split(maxsplit=1)
            name = cmd[0].lower()
            if name in ("/exit", "/quit"):
                break
            if name == "/help":
                print("  /exit          退出")
                print("  /clear         清空对话历史（保留系统提示）")
                print("  /system <文本>  设置系统提示（会清空历史）")
                print("  /think on|off  切换 Qwen3 thinking（默认 off）")
                print("  /stats         打印引擎统计")
                continue
            if name == "/clear":
                history = [m for m in history if m["role"] == "system"]
                if engine.prefix_cache is not None:
                    engine.prefix_cache.clear()
                print("  已清空对话历史。")
                continue
            if name == "/system":
                text = cmd[1] if len(cmd) > 1 else ""
                history = [{"role": "system", "content": text}] if text else []
                print("  系统提示已更新。" if text else "  已移除系统提示。")
                continue
            if name == "/think":
                arg = (cmd[1].lower() if len(cmd) > 1 else "")
                if arg in ("on", "1", "true"):
                    thinking = True
                elif arg in ("off", "0", "false"):
                    thinking = False
                print(f"  thinking={'on' if thinking else 'off'}")
                continue
            if name == "/stats":
                print(json.dumps(engine.stats(), indent=2, ensure_ascii=False))
                continue
            print(f"  未知命令 {name}，输入 /help 查看。")
            continue

        messages = history + [{"role": "user", "content": line}]
        prompt_ids = template(messages)
        out, ttft, total = stream_turn(prompt_ids)
        text = engine.runner.decode(out) if out else getattr(engine, "_last_spec_text", "")
        history = messages + [{"role": "assistant", "content": text}]
        if out:
            n = len(out)
            rate = n / max(1e-6, total - ttft) if ttft else 0
        else:
            spec = getattr(engine, "last_spec_stats", None) or {}
            n = spec.get("tokens", 0)
            rate = spec.get("decode_tok_s", 0.0)
        print(f"\n  [ttft {ttft:.2f}s · {n} tok · {rate:.1f} tok/s]", flush=True)
        if thinking and "<think>" in text and "</think>" not in text:
            print("  [!] thinking 用完了 max_tokens，答案被截断；用 --max-tokens 600+ 或 /think off",
                  flush=True)

    engine.close()
    print("[terrazzo] bye")
    return 0

def cmd_bench(args) -> int:
    from .engine import Engine
    engine = Engine(_cfg(args))
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    prompt = args.prompt if args.prompt else open(args.prompt_file).read()

    def one(_i: int):
        t0 = time.perf_counter()
        text = engine.generate(prompt, params) if args.concurrency == 1 else None
        return text, time.perf_counter() - t0

    # single-request latency first (sequential)
    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        engine.generate(prompt, SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
        times.append(time.perf_counter() - t0)
    per_req = args.max_tokens / statistics.mean(times)
    print(json.dumps({
        "engine": "terrazzo",
        "case": f"single_mt{args.max_tokens}",
        "repeat": args.repeat,
        "wall_s": round(statistics.mean(times), 3),
        "decode_tok_s_per_req": round(per_req, 1),
    }, ensure_ascii=False))

    # concurrency via the streaming API
    for n in [int(x) for x in args.concurrency.split(",") if x]:
        if n <= 1:
            continue
        seqs = [engine.add_request(prompt, SamplingParams(max_tokens=args.max_tokens, temperature=0.0))
                for _ in range(n)]
        t0 = time.perf_counter()
        while engine.has_unfinished():
            engine.step()
        wall = time.perf_counter() - t0
        total = sum(len(s.output_token_ids) for s in seqs)
        print(json.dumps({
            "engine": "terrazzo",
            "case": f"conc_n{n}_mt{args.max_tokens}",
            "n": n,
            "wall_s": round(wall, 2),
            "gen_tokens_total": total,
            "decode_tok_s_per_req": round(total / n / wall, 1),
            "aggregate_decode_tok_s": round(total / wall, 1),
        }, ensure_ascii=False))
    engine.close()
    return 0


def apply_agent_profile(args) -> None:
    """Tuned FIFO presets.

    * ``agent``: many sessions share one long prefix. 16 slots + pre-registration
      measured best for shared-prefix traffic.
    * ``mixed``: shared + unique prefixes. A FIFO cap of 4 beat uncapped FIFO and
      cost-aware ordering on aggregate throughput/latency/memory in the cap sweep;
      treat it as a starting point, not a universal optimum.
    Explicit user flags win; only the generic default (8) is changed.
    """
    profile = getattr(args, "profile", None)
    import os

    if profile == "mixed":
        if getattr(args, "max_num_seqs", 8) == 8:
            args.max_num_seqs = 4
        # Match the measured fixed4 configuration exactly: admission cap 4,
        # 16-wide runner batch, and the same wave settings as the other arms.
        if getattr(args, "completion_batch_size", None) is None:
            args.completion_batch_size = 16
        os.environ.setdefault("TERRAZZO_ADMIT_BATCH", "16")
        os.environ.setdefault("TERRAZZO_ADMIT_WAIT", "4")
        return
    if profile != "agent":
        return
    if getattr(args, "max_num_seqs", 8) == 8:
        args.max_num_seqs = 16
    os.environ.setdefault("TERRAZZO_ADMIT_BATCH", "16")
    os.environ.setdefault("TERRAZZO_ADMIT_WAIT", "4")


def cmd_serve(args) -> int:
    if getattr(args, "memory_cap_mb", None):
        import os

        os.environ["TERRAZZO_MEMORY_CAP_MB"] = str(args.memory_cap_mb)
    apply_agent_profile(args)
    from .server import serve

    # store name -> local path, then apply the Modelfile recipe (if any)
    try:
        from .models import load_recipe
    except Exception:  # pragma: no cover - defensive
        load_recipe = lambda _name: None  # noqa: E731
    requested = args.model
    recipe = load_recipe(requested)
    args.model = _resolve_model(requested)
    if recipe is None:
        recipe = load_recipe(args.model)
    system_text = None
    if args.system_file:
        system_text = open(args.system_file).read()
    elif args.system:
        system_text = args.system
    elif recipe and recipe.get("system"):
        system_text = recipe["system"]
    default_params = dict((recipe or {}).get("parameters") or {})
    default_params.pop("stop", None)   # stop strings are handled per request
    mode = "pool" if args.pool else ("paged" if args.paged else "batch")
    serve(_cfg(args), host=args.host, port=args.port, mode=mode,
          system_text=system_text, use_metal=getattr(args, "metal", False),
          target_tok_s=args.target_tok_s, allow_below_target=args.allow_below_target,
          api_key=args.api_key, rate_limit_rps=args.rate_limit_rps,
          no_thinking=args.no_thinking, default_params=default_params or None,
          tls_cert=getattr(args, "tls_cert", None),
          tls_key=getattr(args, "tls_key", None))
    return 0


def _post_json(url: str, body: dict, timeout: float = 900.0) -> dict:
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _apply_recipe(args, recipe: dict | None) -> dict:
    """Merge Modelfile SYSTEM/PARAMETER values into CLI defaults (CLI flags win)."""
    stop: list[str] = []
    if not recipe:
        return {"system": getattr(args, "system", None), "stop": stop}
    params = recipe.get("parameters") or {}
    system = getattr(args, "system", None) or recipe.get("system")
    for attr, key in (("temperature", "temperature"), ("top_p", "top_p"),
                      ("top_k", "top_k"), ("max_tokens", "max_tokens")):
        if getattr(args, attr, None) in (None, 0, 0.0) or getattr(args, attr, None) == 1.0:
            if key in params:
                setattr(args, key, params[key])
    stop = list(params.get("stop") or [])
    return {"system": system, "stop": stop}


def _stream_run(engine, prompt_ids: list[int], params, stop: list[str],
                verbose: bool) -> tuple[str, int, float]:
    """One-shot streaming completion for `run` (prints only the answer)."""
    import queue as _queue

    seq = engine.add_request(prompt_token_ids=prompt_ids, params=params)
    q = engine.stream(seq.request_id)
    out: list[int] = []
    printed = 0
    text = ""
    t0 = time.perf_counter()
    ttft = None
    while True:
        if engine.has_unfinished():
            engine.step()
        got = False
        while True:
            try:
                delta = q.get_nowait()
            except _queue.Empty:
                break
            got = True
            d = delta if isinstance(delta, dict) else {
                "token_ids": list(getattr(delta, "token_ids", []) or []),
                "finished": bool(getattr(delta, "finished", False)),
                "finish_reason": getattr(delta, "finish_reason", None),
            }
            for tid in d.get("token_ids") or []:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out.append(tid)
                text = engine.runner.decode(out)
                safe = text.rstrip("\ufffd")
                cut = len(safe)
                for marker in stop:
                    idx = safe.find(marker)
                    if idx >= 0:
                        cut = min(cut, idx)
                if cut > printed:
                    print(safe[printed:cut], end="", flush=True)
                    printed = cut
                if any(marker in safe for marker in stop):
                    d = {"finished": True}
                    break
            if d.get("finished"):
                total = time.perf_counter() - t0
                if verbose:
                    n = len(out)
                    rate = n / max(1e-6, total - (ttft or 0)) if ttft else 0
                    print(f"\n[terrazzo] ttft={ttft or 0:.2f}s tokens={n} "
                          f"{rate:.1f} tok/s", flush=True)
                return text, len(out), total
        if not engine.has_unfinished() and not got:
            time.sleep(0.0005)


def cmd_run(args) -> int:
    """
    from .engine import Engine`ollama run MODEL [PROMPT]`: one-shot when a prompt is given, chat otherwise."""
    model = getattr(args, "run_model", None) or args.model
    args.model = model
    try:
        from .models import load_recipe

        recipe = load_recipe(model)
    except Exception:
        recipe = None
    applied = _apply_recipe(args, recipe)
    if not hasattr(args, "spec_ngram"):
        args.spec_ngram = False
    if not applied["system"] and getattr(args, "prompt", None) is None and not recipe:
        pass
    args.system = applied["system"]
    prompt = getattr(args, "prompt", None)
    if prompt is None:
        # interactive: delegate to the existing chat REPL
        return cmd_chat(args)
    engine = Engine(_cfg(args))
    tok = engine.runner.tokenizer
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k)
    messages = []
    if applied["system"]:
        messages.append({"role": "system", "content": applied["system"]})
    messages.append({"role": "user", "content": prompt})
    kwargs = {"enable_thinking": bool(getattr(args, "thinking", False))}
    try:
        prompt_ids = list(tok.apply_chat_template(messages, add_generation_prompt=True, **kwargs))
    except TypeError:
        prompt_ids = list(tok.apply_chat_template(messages, add_generation_prompt=True))
    _stream_run(engine, prompt_ids, params, applied["stop"],
                bool(getattr(args, "verbose", False)))
    engine.close()
    return 0


def cmd_create(args) -> int:
    from .modelfile import parse_modelfile
    from .models import create

    try:
        if args.file == "-":
            text = sys.stdin.read()
        else:
            text = open(args.file, encoding="utf-8").read()
        recipe = parse_modelfile(text)
        if args.base:
            recipe["base"] = args.base
        info = create(args.name, recipe)
    except (OSError, ValueError, FileExistsError) as exc:
        print(f"create failed: {exc}", file=sys.stderr)
        return 1
    print(f"created model {info['name']} from {info['base']}")
    return 0


def cmd_cp(args) -> int:
    from .models import copy

    try:
        info = copy(args.source, args.destination)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"cp failed: {exc}", file=sys.stderr)
        return 1
    print(f"copied {args.source} -> {info['destination']}")
    return 0


def cmd_stop(args) -> int:
    print(f"{args.model}: nothing to unload (single-model server keeps the model "
          f"resident); use Ctrl-C on `terrazzo serve` to stop it.")
    return 0


def cmd_push(args) -> int:
    print("push is not supported: there is no terrazzo registry. Publish an MLX "
          "snapshot on Hugging Face and use `terrazzo pull <repo-id>` instead.",
          file=sys.stderr)
    return 1


def cmd_version(args) -> int:
    from . import __version__

    print(f"terrazzo version {__version__}")
    return 0


def cmd_prewarm(args) -> int:
    """Register a system prompt on a running server so the first request hits."""
    text = None
    if args.system_file:
        text = open(args.system_file).read()
    elif args.system:
        text = args.system
    if not text:
        print("prewarm: provide --system-file or --system", file=sys.stderr)
        return 2
    try:
        result = _post_json(args.base.rstrip("/") + "/v1/prewarm", {"system": text})
    except Exception as exc:
        print(f"prewarm failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


def cmd_list(args) -> int:
    from .models import list_models

    rows = list_models()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no models in the local store (terrazzo pull <hf-repo-id>)")
        return 0
    print(f"{'NAME':32s} {'SIZE':>10s} {'MODIFIED':16s} ARCH")
    for r in rows:
        size = r["size_bytes"] / 1e9
        print(f"{r['name']:32s} {size:9.2f}G {r['modified_at']:16s} {r.get('architecture') or '-'}")
    return 0


def cmd_show(args) -> int:
    from .models import show

    try:
        info = show(args.model)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if getattr(args, "modelfile", False):
        if info.get("modelfile"):
            print(info["modelfile"], end="")
        else:
            print(f"FROM {info['path']}")
        return 0
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    print(f"name:  {info['name']}")
    print(f"path:  {info['path']}")
    print(f"size:  {info['size_bytes'] / 1e9:.2f} GB")
    cfg = info.get("config") or {}
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("files:")
    for f in info.get("files", [])[:12]:
        print(f"  {f['name']:32s} {f['size_bytes'] / 1e6:8.1f} MB")
    return 0


def cmd_rm(args) -> int:
    from .models import remove

    try:
        ok = remove(args.model)
    except Exception as exc:
        print(f"rm failed: {exc}", file=sys.stderr)
        return 1
    print(f"removed {args.model}" if ok else f"model '{args.model}' not found")
    return 0 if ok else 1


def cmd_pull(args) -> int:
    from .models import pull

    try:
        info = pull(args.repo, name=args.name)
    except Exception as exc:
        print(f"pull failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(info, ensure_ascii=False))
    return 0


def cmd_import(args) -> int:
    from .models import import_local

    try:
        info = import_local(args.path, name=args.name)
    except Exception as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(info, ensure_ascii=False))
    return 0


def cmd_ps(args) -> int:
    import urllib.request

    try:
        with urllib.request.urlopen(args.base.rstrip("/") + "/api/ps", timeout=10) as r:
            data = json.loads(r.read())
    except Exception as exc:
        print(f"ps failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    for m in data.get("models", []):
        print(f"{m.get('name')}  {m.get('size', 0) / 1e9:.2f}G  runtime={m.get('runtime')}  "
              f"uptime={m.get('uptime_s')}s  steps={m.get('steps')}")
    return 0


def cmd_slo(args) -> int:
    """Can this (chip, model) reach the target tok/s? MacTurbo admission."""
    from .slo import feasibility, plan

    if getattr(args, "model_path", None):
        from .slo_router import capability, enforce

        cap = capability(args.model_path, target_tok_s=args.target, bits=args.bits,
                         )
        ok, msg = enforce(cap)
        print(json.dumps(cap, ensure_ascii=False, indent=2))
        print(msg)
        return 0 if ok else 1
    if args.plan:
        out = plan(args.chip, args.model_b, args.target, active_params_b=args.active_b)
    else:
        out = feasibility(args.chip, args.model_b, args.target,
                          active_params_b=args.active_b, bits=args.bits)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_test(args) -> int:
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


def cmd_paged(args) -> int:
    """Shared-prefix (paged KV) generation: register a system prefix once, reuse it."""
    from mlx_lm import load as mlx_load

    from .paged_session import PagedSession

    model, tok = mlx_load(args.model)
    session = PagedSession(model, tok, block_size=args.block_size,
                           num_blocks=args.num_gpu_blocks,
                           eos_token_ids=tuple(tok.eos_token_ids))

    system_text = args.system
    if args.system_file:
        system_text = open(args.system_file).read()
    if system_text:
        sys_ids = tok.encode(system_text)
        t0 = time.perf_counter()
        prefix = session.register_prefix(sys_ids)
        if prefix is None:
            print(f"[paged] WARNING: prefix is too short to cache "
                  f"({len(sys_ids)} tokens < {session.block_manager.block_size} block size)",
                  flush=True)
        else:
            print(f"[paged] registered prefix: {prefix.num_tokens} tokens, "
                  f"{len(prefix.block_ids)} blocks, "
                  f"{prefix.nbytes / 1e6:.1f} MB KV "
                  f"in {time.perf_counter() - t0:.2f}s", flush=True)
        prompt_ids = sys_ids + tok.encode(args.prompt)
    else:
        prompt_ids = tok.encode(args.prompt)

    stats = session.generate(prompt_ids, max_tokens=args.max_tokens,
                             temperature=args.temperature)
    print(tok.decode(stats.tokens))
    print(f"\n[paged] {stats.as_dict()}  {session.describe()}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="terrazzo", description="Terrazzo — local LLM inference for Apple Silicon")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
        sp.add_argument("--max-num-seqs", type=int, default=8, dest="max_num_seqs")
        sp.add_argument("--completion-batch-size", type=int, default=None,
                        dest="completion_batch_size",
                        help="BatchGenerator decode batch; defaults to --max-num-seqs")
        sp.add_argument("--max-num-batched-tokens", type=int, default=2048, dest="max_num_batched_tokens")
        sp.add_argument("--prefill-chunk-size", type=int, default=512, dest="prefill_chunk_size")
        sp.add_argument("--prefill-batch-size", type=int, default=8, dest="prefill_batch_size")
        sp.add_argument("--prefill-step-size", type=int, default=None, dest="prefill_step_size",
                        help="BatchGenerator prefill chunk; defaults to --prefill-chunk-size")
        sp.add_argument("--num-gpu-blocks", type=int, default=4096, dest="num_gpu_blocks",
                        help="KV blocks; 0 = derive from RAM (35 percent KV budget)")
        sp.add_argument("--block-size", type=int, default=16, dest="block_size")
        sp.add_argument("--max-model-len", type=int, default=32768, dest="max_model_len",
                        help="max prompt+output tokens per request (default 32768)")
        sp.add_argument("--no-prefix-cache", action="store_true")
        sp.add_argument("--prefix-reuse", choices=["auto", "always", "never", "certified"],
                        default="auto", dest="prefix_reuse",
                        help="reuse a cached prefix only when copy beats recompute "
                             "(auto), always, never, or certified (experimental strict "
                             "gate; can disable reuse when ranges are uncalibrated)")
        sp.add_argument("--scheduler-policy", choices=["fifo", "cost"], default="fifo",
                        dest="scheduler_policy",
                        help="fifo (default) or cost-aware admission order")
        sp.add_argument("--scheduler-aging", type=float, default=100.0,
                        dest="scheduler_aging",
                        help="cost policy aging: prefill tokens credited per waiting second")
        sp.add_argument("--memory-cost-weight", type=float, default=0.1,
                        dest="memory_cost_weight",
                        help="cost policy: weight of predicted new KV tokens")
        sp.add_argument("--no-cost-calibration", action="store_true",
                        dest="no_cost_calibration",
                        help="freeze the predicted-cost rates at their defaults")
        sp.add_argument("--prefix-max-tokens", type=int, default=None,
                        dest="prefix_max_tokens",
                        help="cap a stored prefix snapshot (default 4096); raise for long chats")
        sp.add_argument("--prefix-cache-mb", type=float, default=None, dest="prefix_cache_mb",
                        help="prefix snapshot memory budget in MB (default 1024)")

    g = sub.add_parser("generate", help="generate one completion")
    common(g)
    g.add_argument("--prompt", default="用三句话解释 KV cache")
    g.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    g.add_argument("--top-k", type=int, default=0, dest="top_k")
    g.add_argument("--metrics", action="store_true")
    g.add_argument("--raw", action="store_true",
                   help="raw text completion without the chat template")
    g.add_argument("--spec-ngram", action="store_true", dest="spec_ngram",
                   help="n-gram/prompt-lookup speculative decoding (greedy, experimental)")
    g.add_argument("--spec-k", type=int, default=4, dest="spec_k",
                   help="draft tokens per speculative step")
    g.add_argument("--spec-n", type=int, default=3, dest="spec_n",
                   help="n-gram size for prompt lookup")
    g.add_argument("--json-schema", default=None, dest="json_schema",
                   help="JSON schema (inline JSON or @file) for constrained decoding")
    g.add_argument("--system", default=None, help="system prompt for chat mode")
    g.add_argument("--thinking", dest="thinking", action="store_true", default=False,
                   help="enable Qwen3 thinking (default off for one-shot answers)")
    g.add_argument("--no-thinking", dest="thinking", action="store_false",
                   help="disable Qwen3 thinking (default)")
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("bench", help="latency / throughput benchmark")
    common(b)
    b.add_argument("--prompt", default=None)
    b.add_argument("--prompt-file", default=None, dest="prompt_file")
    b.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    b.add_argument("--repeat", type=int, default=3)
    b.add_argument("--concurrency", default="1,4,8")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("serve", help="OpenAI-compatible server")
    common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--paged", action="store_true",
                   help="sequential paged runtime (v0.2): dense shared prefix, no batching")
    s.add_argument("--pool", action="store_true",
                   help="v0.4 runtime: true block pool, prefix by block reference")
    s.add_argument("--metal", action="store_true",
                   help="with --pool: decode attention via the MLX Metal kernel")
    s.add_argument("--target-tok-s", type=float, default=None, dest="target_tok_s",
                   help="MacTurbo SLO: refuse to serve if predicted tok/s is below this")
    s.add_argument("--allow-below-target", action="store_true", dest="allow_below_target",
                   help="serve even if the predicted tok/s misses the target")
    s.add_argument("--api-key", default=None, dest="api_key",
                   help="require 'Authorization: Bearer <key>' on all endpoints except /health")
    s.add_argument("--tls-cert", default=None, dest="tls_cert",
                   help="TLS certificate chain file (enables HTTPS; also TERRAZZO_TLS_CERT)")
    s.add_argument("--tls-key", default=None, dest="tls_key",
                   help="TLS private key file (also TERRAZZO_TLS_KEY)")
    s.add_argument("--rate-limit", type=float, default=0.0, dest="rate_limit_rps",
                   help="global requests/second limit (token bucket); 0 disables")
    s.add_argument("--memory-cap-mb", type=float, default=None, dest="memory_cap_mb",
                   help="MLX memory cap in MB (default: 70 percent of RAM)")
    s.add_argument("--no-thinking", action="store_true", dest="no_thinking",
                   help="disable Qwen3 thinking in chat-template requests")
    s.add_argument("--profile", default="default",
                   choices=["default", "agent", "mixed"],
                   help="agent: 16 slots for shared-prefix traffic; mixed: 4 slots for "
                        "shared+unique traffic (tuned FIFO starting points)")
    s.add_argument("--system", default=None, help="system prompt to register as a shared prefix")
    s.add_argument("--system-file", default=None, dest="system_file")
    s.set_defaults(func=cmd_serve)

    pl = sub.add_parser("prewarm", help="pre-register a system prompt on a running server")
    pl.add_argument("--base", default="http://127.0.0.1:8080")
    pl.add_argument("--system", default=None)
    pl.add_argument("--system-file", default=None, dest="system_file")
    pl.set_defaults(func=cmd_prewarm)

    lm = sub.add_parser("list", help="list models in the local store")
    lm.add_argument("--json", action="store_true")
    lm.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="show a model's config and files")
    sh.add_argument("model")
    sh.add_argument("--json", action="store_true")
    sh.add_argument("--modelfile", action="store_true",
                    help="print the stored Modelfile (if the model was created from one)")
    sh.set_defaults(func=cmd_show)

    rm = sub.add_parser("rm", help="remove a model from the local store")
    rm.add_argument("model")
    rm.set_defaults(func=cmd_rm)

    pu = sub.add_parser("pull", help="download a Hugging Face snapshot into the store")
    pu.add_argument("repo", help="e.g. mlx-community/Qwen3-4B-4bit")
    pu.add_argument("--name", default=None)
    pu.set_defaults(func=cmd_pull)

    im = sub.add_parser("import", help="link a local model directory into the store")
    im.add_argument("path")
    im.add_argument("--name", default=None)
    im.set_defaults(func=cmd_import)

    ps = sub.add_parser("ps", help="models running on a server (/api/ps)")
    ps.add_argument("--base", default="http://127.0.0.1:8080")
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_ps)

    rn = sub.add_parser("run", help="run a model: one-shot with PROMPT, chat REPL without")
    common(rn)
    rn.add_argument("run_model", nargs="?", help="model name in the store, or a local path")
    rn.add_argument("prompt", nargs="?", default=None, help="prompt for one-shot mode")
    rn.add_argument("--system", default=None, help="system prompt (overrides Modelfile SYSTEM)")
    rn.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    rn.add_argument("--temperature", type=float, default=0.7)
    rn.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    rn.add_argument("--top-k", type=int, default=0, dest="top_k")
    rn.add_argument("--thinking", action="store_true", default=False,
                    help="enable Qwen3 thinking (default off)")
    rn.add_argument("--verbose", action="store_true", help="print ttft/tok-s stats")
    rn.set_defaults(func=cmd_run)

    ct = sub.add_parser("create", help="create a model from a Modelfile")
    ct.add_argument("name", help="new model name")
    ct.add_argument("-f", "--file", default="Modelfile", help="Modelfile path ('-' = stdin)")
    ct.add_argument("--base", default=None, help="override FROM")
    ct.set_defaults(func=cmd_create)

    cp = sub.add_parser("cp", help="copy a model in the local store")
    cp.add_argument("source")
    cp.add_argument("destination")
    cp.set_defaults(func=cmd_cp)

    st = sub.add_parser("stop", help="stop a running model (nothing to unload here)")
    st.add_argument("model", nargs="?", default="terrazzo")
    st.set_defaults(func=cmd_stop)

    pu2 = sub.add_parser("push", help="push a model to a registry (unsupported)")
    pu2.add_argument("model")
    pu2.set_defaults(func=cmd_push)

    ver = sub.add_parser("version", help="print the version")
    ver.set_defaults(func=cmd_version)

    c = sub.add_parser("chat", help="interactive terminal chatbot (multi-turn, streamed)")
    common(c)
    c.add_argument("--system", default=None, help="system prompt")
    c.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    c.add_argument("--top-k", type=int, default=0, dest="top_k")
    c.add_argument("--thinking", dest="thinking", action="store_true", default=False,
                   help="enable Qwen3 thinking (default off)")
    c.add_argument("--spec-ngram", action="store_true", dest="spec_ngram",
                   help="n-gram speculative decoding for chat turns (blocking, experimental)")
    c.add_argument("--spec-k", type=int, default=4, dest="spec_k")
    c.add_argument("--spec-n", type=int, default=3, dest="spec_n")
    c.set_defaults(func=cmd_chat)

    pg = sub.add_parser("paged", help="shared-prefix generation (paged KV, v0.2)")
    common(pg)
    pg.add_argument("--system", default=None, help="system prefix text to register once")
    pg.add_argument("--system-file", default=None, dest="system_file")
    pg.add_argument("--prompt", default="用三句话解释 KV cache")
    pg.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    pg.add_argument("--temperature", type=float, default=0.0)
    pg.set_defaults(func=cmd_paged)

    sl = sub.add_parser("slo", help="MacTurbo: check whether a (chip, model) reaches a tok/s target")
    sl.add_argument("--chip", default=None, help="e.g. m1_max, m4_max, m3_ultra (auto if omitted)")
    sl.add_argument("--model-b", type=float, default=None, dest="model_b",
                    help="total parameters in billions")
    sl.add_argument("--model-path", default=None, dest="model_path",
                    help="local HF model dir: detect active params from config.json")
    sl.add_argument("--active-b", type=float, default=None, dest="active_b",
                    help="MoE active parameters in billions (omit for dense)")
    sl.add_argument("--bits", type=int, default=4)
    sl.add_argument("--target", type=float, default=200.0, help="tok/s SLO")
    sl.add_argument("--plan", action="store_true", help="find the lowest bit width that meets the SLO")
    sl.set_defaults(func=cmd_slo)

    t = sub.add_parser("test", help="run unit tests")
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=cmd_test)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "prompt_file", None):
        pass
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
