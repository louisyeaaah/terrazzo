"""A small, replayable local code-review agent workflow.

The model gets one shared system prompt plus a read_file tool, reviews fixed
tasks in the Terrazzo repo, executes tool calls locally, answers a follow-up, and
records every request/response with token usage and timings.

    python bench/agent_review_workflow.py --base http://127.0.0.1:9300 --label cache-on
"""
from __future__ import annotations
import argparse, json, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "bench"))
from agent_workload import RssSampler, find_pid, probe_environment, _get_json  # noqa: E402

REPO = ROOT
TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 source file inside the repository. Path is relative to repo root.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}]

RULES = (
    "你是本地代码审查 agent。工作目录是 Terrazzo 仓库。"
    "必须先调用 read_file 读取相关文件，再基于真实代码回答；不要编造行号或接口。"
    "审查时优先指出具体风险、触发条件、影响和最小修复方向；不要泛泛而谈。"
    "如果工具返回内容不足，可以再次调用 read_file；最多 3 次工具调用。"
    "最终回答用中文，先给结论，再给一条最值得修复的建议。"
)

TASKS = [
    {"id": "refcount", "set": "tune", "file": "terrazzo/blocks.py",
     "prompt": "审查文件 `terrazzo/blocks.py` 的块分配与引用计数，指出一个可能导致块泄漏或重复释放的具体风险，并给出修复方向。先用 read_file 读取这个路径。",
     "followup": "只保留最严重的一项，说明触发条件和应该在哪个检查点验证。",
     "rubric": [("引用计数", "ref_count", "refcount"), ("释放", "release", "释放")]},
    {"id": "cache", "set": "tune", "file": "terrazzo/engine.py",
     "prompt": "审查文件 `terrazzo/engine.py` 的 prefix cache 完整命中与回退逻辑，指出一个会导致缓存统计或输出不一致的具体风险。先用 read_file 读取这个路径。",
     "followup": "如果 runner 最终没有复用 cache，num_hits 和 prefix_hit_tokens 应该怎么处理？",
     "rubric": [("num_hits", "undo_hit", "prefix_hit_tokens")]},
    {"id": "mask", "set": "tune", "file": "terrazzo/mask_patch.py",
     "prompt": "审查文件 `terrazzo/mask_patch.py` 的 BatchKVCache decode mask 补丁，指出 batch layout 改变后可能失效的具体路径。先用 read_file 读取这个路径。",
     "followup": "extend/filter 之后应该重新计算什么？",
     "rubric": [("left_padding", "_dsh_has_padding", "padding")]},
    {"id": "scheduler", "set": "holdout", "file": "terrazzo/scheduler.py",
     "prompt": "审查文件 `terrazzo/scheduler.py` 的抢占和等待队列，指出一个可能导致请求饥饿或块泄漏的具体风险。先用 read_file 读取这个路径。",
     "followup": "修复后应该用什么可观察指标验证它不再发生？",
     "rubric": [("preempt", "抢占", "block")]},
    {"id": "request", "set": "holdout", "file": "terrazzo/server.py",
     "prompt": "审查文件 `terrazzo/server.py` 的请求超时、取消和失败恢复路径，指出一个会让请求卡住或统计失真的具体风险。先用 read_file 读取这个路径。",
     "followup": "取消后应该释放哪些执行器/调度器状态？",
     "rubric": [("abort", "取消", "timeout", "超时")]},
    {"id": "golden_refcount_leak", "set": "golden",
     "file": "bench/agent_fixtures/block_pool_review.py",
     "prompt": "审查文件 `bench/agent_fixtures/block_pool_review.py`。这个池子的约定是最后一个引用释放时块回到 free_blocks。请找出具体缺陷、触发条件，并给出最小修复。先用 read_file 读取文件。",
     "followup": "请给出一个最小复现步骤，并明确修复后的判断条件。",
     "rubric": [("== 1", "==1", "泄漏", "leak", "never"),
                ("最后一个引用", "last reference", "归零", "zero", "release"),
                ("== 0", "==0", "free_blocks", "free list", "改为 0", "改为0")]},
]


def build_system_prompt(repeat: int) -> str:
    return (RULES + "\n") * repeat


def _read_tool(path: str) -> str:
    p = (REPO / path).resolve()
    if REPO.resolve() not in p.parents and p != REPO.resolve():
        return f"ERROR: path escapes repo: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > 30000:
        data = data[:30000] + "\n... [truncated]"
    return f"FILE {path}\n{data}"


def chat(base: str, model: str, messages: list[dict], max_tokens: int,
         timeout: float = 600.0) -> dict:
    body = json.dumps({
        "model": model, "messages": messages, "tools": TOOLS,
        "tool_choice": "auto", "max_tokens": max_tokens,
        "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode())
    total = time.perf_counter() - t0
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = obj.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "latency_s": total,
    }


def _tool_message(call: dict) -> dict:
    fn = call.get("function") or {}
    if fn.get("name") != "read_file":
        content = f"ERROR: unknown tool {fn.get('name')!r}"
    else:
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except Exception:
            args = {}
        path = args.get("path") if isinstance(args, dict) else None
        content = _read_tool(str(path)) if path else "ERROR: missing path"
    return {"role": "tool", "tool_call_id": call.get("id", ""),
            "name": fn.get("name", "read_file"), "content": content}


def classify_tool_trace(task: dict, tools_used: list[dict]) -> bool:
    """Shared online/replay rule: read the requested file successfully.

    Extra files are allowed (a reviewer may inspect related code); unknown tools,
    failed reads and missing requested-file reads are not.
    """
    expected = task.get("file")
    def same_path(value):
        if not expected or not isinstance(value, str):
            return False
        try:
            return Path(value).as_posix() == Path(expected).as_posix()
        except Exception:
            return False
    if not tools_used:
        return False
    if not all(t.get("name") == "read_file" and t.get("ok") for t in tools_used):
        return False
    def tool_path(t):
        if t.get("path") is not None:
            return t.get("path")
        try:
            parsed = json.loads(t.get("args") or "{}")
            return parsed.get("path") if isinstance(parsed, dict) else None
        except Exception:
            return None
    return any(same_path(tool_path(t)) for t in tools_used)


def run_task(base: str, model: str, system: str, task: dict, max_tokens: int,
             max_tool_rounds: int) -> dict:
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task["prompt"]}]
    requests = []
    final = ""
    final_r = None
    tools_used = []
    pending_tool_calls = False
    tool_rounds_exhausted = False
    t0 = time.perf_counter()
    for round_idx in range(max_tool_rounds + 1):
        r = chat(base, model, messages, max_tokens)
        requests.append({"messages": json.loads(json.dumps(messages)),
                         "response": {k: r[k] for k in
                                      ("finish_reason", "prompt_tokens",
                                       "completion_tokens", "total_tokens",
                                       "latency_s")}})
        if r["tool_calls"]:
            assistant = {"role": "assistant", "content": r["content"] or None,
                         "tool_calls": r["tool_calls"]}
            messages.append(assistant)
            for call in r["tool_calls"]:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                path = args.get("path") if isinstance(args, dict) else None
                tool_msg = _tool_message(call)
                content = tool_msg.get("content") or ""
                tools_used.append({"name": fn.get("name"),
                                   "path": path,
                                   "args": fn.get("arguments"),
                                   "ok": not content.startswith("ERROR:"),
                                   "error": content[:160] if content.startswith("ERROR:") else None})
                messages.append(tool_msg)
            if round_idx == max_tool_rounds:
                pending_tool_calls = True
                tool_rounds_exhausted = True
            continue
        final_r = r
        final = r["content"]
        break
    # Follow-up turn with the actual answer in history.
    messages.append({"role": "assistant", "content": final})
    messages.append({"role": "user", "content": task["followup"]})
    r2 = chat(base, model, messages, max_tokens)
    if r2["tool_calls"]:
        pending_tool_calls = True
    requests.append({"messages": json.loads(json.dumps(messages)),
                     "response": {k: r2[k] for k in
                                  ("finish_reason", "prompt_tokens",
                                   "completion_tokens", "total_tokens",
                                   "latency_s")}})
    text = (final + "\n" + r2["content"]).lower()
    rubric = [[w.lower() for w in group] for group in task["rubric"]]
    keyword_pass = all(any(w in text for w in group) for group in rubric)
    tool_trace_ok = classify_tool_trace(task, tools_used)
    finish_reasons = [final_r.get("finish_reason") if final_r else None,
                      r2.get("finish_reason")]
    truncated = any(x == "length" for x in finish_reasons)
    complete_review = (tool_trace_ok and not truncated and not pending_tool_calls
                       and not tool_rounds_exhausted
                       and finish_reasons == ["stop", "stop"]
                       and bool(final.strip()) and bool(r2["content"].strip()))
    return {
        "task_id": task["id"], "set": task["set"],
        "keyword_pass": keyword_pass,
        "tool_trace_ok": tool_trace_ok,
        "truncated": truncated,
        "complete_review": complete_review,
        "pending_tool_calls": pending_tool_calls,
        "tool_rounds_exhausted": tool_rounds_exhausted,
        "final_finish_reason": finish_reasons[0],
        "followup_finish_reason": finish_reasons[1],
        "wall_s": time.perf_counter() - t0,
        "requests": requests,
        "tools_used": tools_used,
        "final": final, "followup": r2["content"],
        "prompt_tokens": sum((q["response"].get("prompt_tokens") or 0) for q in requests),
        "completion_tokens": sum((q["response"].get("completion_tokens") or 0) for q in requests),
    }


def run_replay(base: str, model: str, recording: str, out: str,
               label: str, stats: str | None, server_pid: int | None) -> int:
    data = json.load(open(recording))
    tasks = data.get("tasks") or []
    pid = server_pid or find_pid(int(base.rsplit(":", 1)[-1].split("/")[0]))
    sampler = RssSampler(pid) if pid else None
    env = probe_environment(base, stats)
    # warmup outside timing
    chat(base, model, [{"role": "user", "content": "热身：你好"}], 8)
    rows = []
    started = time.perf_counter()
    for task in tasks:
        req_rows = []
        for req in task.get("requests", []):
            r = chat(base, model, req["messages"], 128)
            req_rows.append({k: r[k] for k in
                             ("finish_reason", "prompt_tokens",
                              "completion_tokens", "total_tokens", "latency_s")})
        rows.append({"task_id": task.get("task_id"), "requests": req_rows,
                     "latency_s": sum(q["latency_s"] for q in req_rows),
                     "completion_tokens": sum((q["completion_tokens"] or 0)
                                              for q in req_rows)})
    wall = time.perf_counter() - started
    rss_peak = sampler.stop() if sampler else 0.0
    stats_after = _get_json(stats) if stats else None
    summary = {"label": label, "mode": "replay", "base": base, "model": model,
               "recording": recording, "wall_s": round(wall, 2),
               "environment": env, "tasks": rows,
               "peak_rss_mb": round(rss_peak, 0),
               "stats_after": {k: stats_after[k] for k in
                               ("prefix_cache", "prefix_hit_tokens", "mlx_peak_mb")
                               if isinstance(stats_after, dict) and k in stats_after},
               "output_tokens": sum(r["completion_tokens"] for r in rows)}
    Path(out).write_text(json.dumps({"summary": summary, "tasks": tasks},
                                    ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))
    print("wrote", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="terrazzo")
    ap.add_argument("--label", required=True)
    ap.add_argument("--task-set", choices=["tune", "holdout", "golden", "all"], default="all")
    ap.add_argument("--mode", choices=["agent", "replay"], default="agent")
    ap.add_argument("--recording", default=None,
                    help="replay mode: JSON produced by an agent run")
    ap.add_argument("--system-repeat", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-tool-rounds", type=int, default=2)
    ap.add_argument("--server-pid", type=int, default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--prewarm", action="store_true",
                    help="POST the shared system prompt to /v1/prewarm first")
    ap.add_argument("--out", default=str(ROOT / "bench" / "agent_workflow_results.json"))
    args = ap.parse_args()
    if args.mode == "replay":
        if not args.recording:
            return print("replay mode requires --recording") or 2
        return run_replay(args.base, args.model, args.recording, args.out,
                          args.label, args.stats, args.server_pid)
    pid = args.server_pid or find_pid(int(args.base.rsplit(":", 1)[-1].split("/")[0]))
    sampler = RssSampler(pid) if pid else None
    system = build_system_prompt(args.system_repeat)
    env = probe_environment(args.base, args.stats)
    warm = None
    try:
        if args.prewarm:
            body = json.dumps({"system": system}).encode()
            req = urllib.request.Request(args.base.rstrip("/") + "/v1/prewarm",
                                         data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                warm = json.loads(resp.read().decode())
        # one warmup request, outside task timing
        chat(args.base, args.model, [{"role": "user", "content": "热身：你好"}], 8)
    except Exception as exc:
        return print(json.dumps({"error": f"warmup/prewarm: {exc}"})) or 2
    tasks = [t for t in TASKS if args.task_set == "all" or t["set"] == args.task_set]
    started = time.perf_counter()
    results = []
    for task in tasks:
        print(f"[agent] {task['id']}", flush=True)
        results.append(run_task(args.base, args.model, system, task,
                                args.max_tokens, args.max_tool_rounds))
    wall = time.perf_counter() - started
    rss_peak = sampler.stop() if sampler else 0.0
    stats_after = _get_json(args.stats) if args.stats else None
    prefix_before = (env.get("stats") or {}).get("prefix_cache") if isinstance(env.get("stats"), dict) else None
    prefix_after = (stats_after or {}).get("prefix_cache") if isinstance(stats_after, dict) else None
    summary = {
        "label": args.label, "base": args.base, "model": args.model,
        "task_set": args.task_set, "system_chars": len(system),
        "system_tokens_est": len(system) // 2,
        "prewarm": warm, "environment": env,
        "wall_s": round(wall, 2),
        "tasks": [{k: r[k] for k in
                   ("task_id", "set", "keyword_pass", "tool_trace_ok",
                    "truncated", "complete_review", "wall_s", "prompt_tokens",
                    "completion_tokens", "final_finish_reason",
                    "followup_finish_reason", "pending_tool_calls",
                    "tool_rounds_exhausted", "tools_used")
                   if k in r} for r in results],
        "keyword_pass_rate": sum(1 for r in results if r["keyword_pass"]) / max(1, len(results)),
        "complete_review_rate": sum(1 for r in results if r["complete_review"]) / max(1, len(results)),
        "truncated_tasks": [r["task_id"] for r in results if r["truncated"]],
        "tool_trace_failures": [r["task_id"] for r in results if not r["tool_trace_ok"]],
        "peak_rss_mb": round(rss_peak, 0),
        "stats_after": {k: stats_after[k] for k in
                        ("prefix_cache", "prefix_hit_tokens", "mlx_peak_mb")
                        if isinstance(stats_after, dict) and k in stats_after},
        "prefix_hits_before": (prefix_before or {}).get("hits"),
        "prefix_hits_after": (prefix_after or {}).get("hits"),
        "output_tokens": sum(r["completion_tokens"] for r in results),
        "aggregate_tok_s": round(sum(r["completion_tokens"] for r in results) / wall, 1)
        if wall else 0.0,
    }
    payload = {"summary": summary, "tasks": results}
    with open(args.out, "w") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
