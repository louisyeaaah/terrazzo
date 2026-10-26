"""Tool/function-calling helpers: validation and output parsing."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.tools import (apply_tool_choice, normalize_messages,  # noqa: E402
                                 parse_tool_calls, validate_tools)


class ValidateToolsTest(unittest.TestCase):
    def test_normalizes(self):
        tools = validate_tools([{"type": "function", "function": {
            "name": "get_weather", "description": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}])
        self.assertEqual(tools[0]["function"]["name"], "get_weather")
        self.assertIn("city", tools[0]["function"]["parameters"]["properties"])

    def test_missing_parameters_gets_empty_object(self):
        tools = validate_tools([{"function": {"name": "ping"}}])
        self.assertEqual(tools[0]["function"]["parameters"], {"type": "object", "properties": {}})

    def test_errors(self):
        for bad in ([], None, [{"function": {}}], [{"function": {"name": ""}}]):
            with self.assertRaises(ValueError):
                validate_tools(bad)


class NormalizeMessagesTest(unittest.TestCase):
    def test_assistant_null_content_becomes_empty(self):
        msgs = normalize_messages([{"role": "assistant", "content": None,
                                    "tool_calls": [{"id": "c1", "type": "function",
                                                    "function": {"name": "f",
                                                                 "arguments": '{"a": 1}'}}]}])
        self.assertEqual(msgs[0]["content"], "")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})

    def test_tool_message_and_list_content(self):
        msgs = normalize_messages([{"role": "tool", "content": [{"type": "text"}]},
                                   {"role": "user", "content": "hi"}])
        self.assertIsInstance(msgs[0]["content"], str)
        self.assertEqual(msgs[1]["content"], "hi")


class ToolChoiceTest(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "a"}},
             {"type": "function", "function": {"name": "b"}}]

    def test_none(self):
        tools, extra = apply_tool_choice(self.TOOLS, "none")
        self.assertEqual(tools, [])
        self.assertEqual(extra, [])

    def test_specific_function_filters_and_instructs(self):
        tools, extra = apply_tool_choice(
            self.TOOLS, {"type": "function", "function": {"name": "b"}})
        self.assertEqual([t["function"]["name"] for t in tools], ["b"])
        self.assertIn("`b`", extra[0]["content"])

    def test_required_instructs(self):
        tools, extra = apply_tool_choice(self.TOOLS, "required")
        self.assertEqual(len(tools), 2)
        self.assertIn("must call", extra[0]["content"])

    def test_unknown_function_raises(self):
        with self.assertRaises(ValueError):
            apply_tool_choice(self.TOOLS, {"type": "function", "function": {"name": "x"}})


class ParseToolCallsTest(unittest.TestCase):
    def test_single_call_block(self):
        text = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n'
                '</tool_call>')
        cleaned, calls = parse_tool_calls(text)
        self.assertEqual(cleaned, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["city"], "Beijing")

    def test_call_with_surrounding_text(self):
        text = ('好的，我来查一下。\n<tool_call>{"name":"get_weather",'
                '"arguments":{"city":"上海"}}</tool_call>')
        cleaned, calls = parse_tool_calls(text)
        self.assertIn("我来查一下", cleaned)
        self.assertNotIn("tool_call", cleaned)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")

    def test_two_calls(self):
        text = ('<tool_call>{"name":"a","arguments":{}}</tool_call>'
                '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>')
        _, calls = parse_tool_calls(text)
        self.assertEqual([c["function"]["name"] for c in calls], ["a", "b"])

    def test_bare_json_fallback(self):
        _, calls = parse_tool_calls('{"name": "ping", "arguments": {}}')
        self.assertEqual(calls[0]["function"]["name"], "ping")

    def test_plain_answer_untouched(self):
        text = "KV cache 缓存注意力键值，避免重复计算。"
        cleaned, calls = parse_tool_calls(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(calls, [])

    def test_thinking_is_stripped(self):
        text = ('<think>用户要查天气，应该调用 get_weather。</think>'
                '<tool_call>{"name":"get_weather","arguments":{"city":"北京"}}'
                '</tool_call>')
        cleaned, calls = parse_tool_calls(text)
        self.assertEqual(cleaned, "")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")

    def test_malformed_block_is_ignored(self):
        cleaned, calls = parse_tool_calls('<tool_call>{not json}</tool_call>')
        self.assertEqual(calls, [])
        self.assertIn("tool_call", cleaned)
