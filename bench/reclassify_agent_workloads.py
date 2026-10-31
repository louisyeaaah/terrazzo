"""Post-process existing agent recordings without rerunning any model."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from agent_review_workflow import TASKS  # noqa: E402
TASK_MAP = {t["id"]: t for t in TASKS}


def classify(task_rec, task_def):
    reqs = task_rec.get("requests", [])
    final_req = reqs[-2]["response"] if len(reqs) >= 2 else {}
    follow_req = reqs[-1]["response"] if reqs else {}
    text = ((task_rec.get("final") or "") + "\n" + (task_rec.get("followup") or "")).lower()
    rubric = [[w.lower() for w in g] for g in task_def["rubric"]]
    keyword_pass = bool(text) and all(any(w in text for w in g) for g in rubric)
    tool_ok = True
    expected = task_def.get("file")
    expected_read = False
    for r in reqs:
        for msg in r.get("messages", []):
            if msg.get("role") == "tool":
                content = str(msg.get("content") or "")
                if content.startswith("ERROR:"):
                    tool_ok = False
        for t in task_rec.get("tools_used", []):
            args = str(t.get("args") or "")
            try:
                parsed = json.loads(args) if args else {}
            except Exception:
                parsed = {}
            path = parsed.get("path") if isinstance(parsed, dict) else None
            if (t.get("name") == "read_file" and expected
                    and isinstance(path, str)
                    and Path(path).as_posix() == Path(expected).as_posix()):
                expected_read = True
    tool_trace_ok = bool(expected_read and tool_ok
                         and all(t.get("name") == "read_file" for t in
                                 task_rec.get("tools_used", [])))
    finish_reasons = [final_req.get("finish_reason"), follow_req.get("finish_reason")]
    pending = any(x == "tool_calls" for x in finish_reasons)
    truncated = any(x == "length" for x in finish_reasons)
    complete_review = (tool_trace_ok and not truncated and not pending
                       and finish_reasons == ["stop", "stop"]
                       and bool((task_rec.get("final") or "").strip())
                       and bool((task_rec.get("followup") or "").strip()))
    return {"keyword_pass": keyword_pass, "tool_trace_ok": tool_trace_ok,
            "truncated": truncated, "complete_review": complete_review,
            "pending_tool_calls": pending,
            "final_finish_reason": finish_reasons[0],
            "followup_finish_reason": finish_reasons[1]}


def main():
    changed = []
    for path in sorted((ROOT / "bench" / "agent_workloads").glob("*.json")):
        data = json.loads(path.read_text())
        summary = data.get("summary") or {}
        if summary.get("mode") == "replay":
            continue
        tasks = data.get("tasks") or []
        for t in tasks:
            task_def = TASK_MAP.get(t.get("task_id"))
            if task_def is None:
                continue
            t.pop("success", None)
            t.update(classify(t, task_def))
        summary.pop("success_rate", None)
        summary["tasks"] = [{k: t.get(k) for k in
                             ("task_id", "set", "keyword_pass", "tool_trace_ok",
                              "truncated", "complete_review", "pending_tool_calls", "wall_s",
                              "prompt_tokens", "completion_tokens",
                              "final_finish_reason", "followup_finish_reason",
                              "tools_used") if k in t} for t in tasks]
        summary["keyword_pass_rate"] = (sum(1 for t in tasks if t.get("keyword_pass"))
                                        / max(1, len(tasks)))
        summary["complete_review_rate"] = (sum(1 for t in tasks if t.get("complete_review"))
                                           / max(1, len(tasks)))
        summary["truncated_tasks"] = [t["task_id"] for t in tasks if t.get("truncated")]
        summary["tool_trace_failures"] = [t["task_id"] for t in tasks
                                          if not t.get("tool_trace_ok")]
        data["summary"] = summary
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        changed.append((path.name, summary["keyword_pass_rate"],
                        summary["complete_review_rate"],
                        summary["truncated_tasks"]))
    print("reclassified:")
    for row in changed:
        print(row)


if __name__ == "__main__":
    main()
