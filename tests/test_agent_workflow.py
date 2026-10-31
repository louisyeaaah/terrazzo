"""Classifier checks for the agent workflow: keyword hits are not task success."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from agent_review_workflow import TASKS, run_task  # noqa: E402
from agent_reliability import acceptance_report  # noqa: E402


def _resp(content="", tool_calls=None, finish="stop"):
    return {"content": content, "tool_calls": tool_calls or [],
            "finish_reason": finish, "prompt_tokens": 10,
            "completion_tokens": 2, "total_tokens": 12, "latency_s": 0.01}


class AgentClassifierTest(unittest.TestCase):
    def test_padding_without_tool_is_not_complete(self):
        task = TASKS[0]
        replies = [_resp("padding " + " ".join(g[0] for g in task["rubric"]), finish="length"),
                   _resp("padding " + " ".join(g[0] for g in task["rubric"]), finish="length")]
        with patch("agent_review_workflow.chat", side_effect=replies):
            r = run_task("http://unused", "m", "system", task, 8, 1)
        self.assertTrue(r["keyword_pass"])
        self.assertFalse(r["tool_trace_ok"])
        self.assertTrue(r["truncated"])
        self.assertFalse(r["complete_review"])

    def test_successful_tool_and_natural_stop_is_complete(self):
        task = TASKS[0]
        call = {"id": "c1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": "{\"path\": \"terrazzo/blocks.py\"}"}}
        text = "引用计数 ref_count 释放 release 建议"
        replies = [_resp(tool_calls=[call], finish="tool_calls"),
                   _resp(text, finish="stop"),
                   _resp(text, finish="stop")]
        with patch("agent_review_workflow.chat", side_effect=replies):
            r = run_task("http://unused", "m", "system", task, 96, 1)
        self.assertTrue(r["tool_trace_ok"])
        self.assertFalse(r["truncated"])
        self.assertTrue(r["complete_review"])


    def test_followup_pending_tool_call_is_not_complete(self):
        task = TASKS[0]
        call = {"id": "c1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": "{\"path\": \"terrazzo/blocks.py\"}"}}
        text = "引用计数 ref_count 释放 release 建议"
        replies = [_resp(tool_calls=[call], finish="tool_calls"),
                   _resp(text, finish="stop"),
                   _resp("还需要再调用一次工具", tool_calls=[call],
                         finish="tool_calls")]
        with patch("agent_review_workflow.chat", side_effect=replies):
            r = run_task("http://unused", "m", "system", task, 96, 1)
        self.assertTrue(r["pending_tool_calls"])
        self.assertFalse(r["complete_review"])

    def test_wrong_tool_path_is_not_complete(self):
        task = TASKS[0]
        bad = {"id": "c1", "type": "function",
               "function": {"name": "read_file",
                            "arguments": "{\"path\": \"terrazzo/engine.py\"}"}}
        text = "引用计数 ref_count 释放 release 建议"
        replies = [_resp(tool_calls=[bad], finish="tool_calls"),
                   _resp(text, finish="stop"),
                   _resp(text, finish="stop")]
        with patch("agent_review_workflow.chat", side_effect=replies):
            r = run_task("http://unused", "m", "system", task, 96, 1)
        self.assertFalse(r["tool_trace_ok"])
        self.assertFalse(r["complete_review"])

    def test_tool_rounds_exhausted_is_not_complete(self):
        task = TASKS[0]
        call = {"id": "c1", "type": "function",
                "function": {"name": "read_file",
                             "arguments": "{\"path\": \"terrazzo/blocks.py\"}"}}
        replies = [_resp(tool_calls=[call], finish="tool_calls"),
                   _resp(tool_calls=[call], finish="tool_calls"),
                   _resp(tool_calls=[call], finish="tool_calls")]
        with patch("agent_review_workflow.chat", side_effect=replies):
            r = run_task("http://unused", "m", "system", task, 96, 1)
        self.assertTrue(r["tool_rounds_exhausted"])
        self.assertFalse(r["complete_review"])

    def test_extra_file_read_does_not_fail_tool_trace(self):
        task = TASKS[0]
        calls = [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file",
                          "arguments": "{\"path\": \"terrazzo/blocks.py\"}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "read_file",
                          "arguments": "{\"path\": \"README.md\"}"}},
        ]
        text = "引用计数 ref_count 释放 release 建议"
        replies = [_resp(tool_calls=calls, finish="tool_calls"),
                   _resp(text, finish="stop"),
                   _resp(text, finish="stop")]
        with patch("agent_review_workflow.chat", side_effect=replies):
            r = run_task("http://unused", "m", "system", task, 96, 1)
        self.assertTrue(r["tool_trace_ok"])
        self.assertTrue(r["complete_review"])

    def test_missing_final_active_is_not_idle(self):
        checks = {
            "abort": {"aborted": True}, "after_abort": "stop",
            "malformed_status": 400, "after_malformed": "stop",
            "eviction": {"evictions": 1}, "after_eviction": "stop",
            "stats": {"running": 0, "waiting": 0, "prefix_cache": {}},
        }
        acc = acceptance_report(checks)
        self.assertFalse(acc["idle_terminal_state"])
        self.assertFalse(all(acc.values()))

if __name__ == "__main__":
    unittest.main()
