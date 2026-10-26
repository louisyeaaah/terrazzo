"""Tests for server hardening: auth, rate limiting, abort, memory 503."""

import json
import os
import queue
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.server import BaseRuntime, make_handler  # noqa: E402


class FakeRuntime(BaseRuntime):
    name = "fake"

    def __init__(self, *, fail_memory: bool = False):
        self.fail_memory = fail_memory
        self.aborted: list[str] = []
        self.submitted: list[str] = []
        self.text = "ok"
        self.tools_seen = None

    def submit(self, prompt_ids, params, request_id=None):
        if self.fail_memory:
            raise MemoryError("over cap")
        rid = request_id or "req-fake"
        self.submitted.append(rid)
        q: "queue.Queue" = queue.Queue()
        q.put({"request_id": rid, "token_ids": [1], "finished": False, "finish_reason": None})
        q.put({"request_id": rid, "token_ids": [], "finished": True, "finish_reason": "stop"})
        return rid, q

    def stats(self) -> dict:
        return {"runtime": self.name}

    def abort(self, request_id: str) -> bool:
        self.aborted.append(request_id)
        return True

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, tokens) -> str:
        return self.text

    def apply_chat_template(self, messages, tools=None, enable_thinking=None):
        self.tools_seen = tools
        self.messages_seen = messages
        self.thinking_seen = enable_thinking
        return [1, 2, 3, 4]

    def generate_json(self, prompt_ids, schema, *, max_tokens=512, timeout=600.0):
        self.json_calls = getattr(self, "json_calls", 0) + 1
        self.last_json_schema = schema
        return '{"ok": true}', {"steps": 8, "complete": True}


def start(runtime, **kwargs):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime, **kwargs))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


class JsonSchemaEndpointTest(unittest.TestCase):
    def _post(self, base, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_json_schema_response(self):
        rt = FakeRuntime()
        httpd, base = start(rt)
        try:
            status, body = self._post(base, {
                "messages": [{"role": "user", "content": "give me json"}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "x",
                                                    "schema": {"type": "object",
                                                               "properties": {"ok": {"type": "boolean"}},
                                                               "required": ["ok"]}}},
            })
            self.assertEqual(status, 200)
            content = body["choices"][0]["message"]["content"]
            self.assertEqual(json.loads(content), {"ok": True})
            self.assertEqual(rt.json_calls, 1)
        finally:
            httpd.shutdown()

    def test_invalid_schema_is_400(self):
        rt = FakeRuntime()
        httpd, base = start(rt)
        try:
            status, body = self._post(base, {
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "x",
                                                    "schema": {"$ref": "#/a"}}},
            })
            self.assertEqual(status, 400)
            self.assertIn("invalid json schema", body["error"]["message"])
            self.assertEqual(getattr(rt, "json_calls", 0), 0)
        finally:
            httpd.shutdown()

    def test_unsupported_response_format_is_400(self):
        rt = FakeRuntime()
        httpd, base = start(rt)
        try:
            status, _ = self._post(base, {"messages": [{"role": "user", "content": "x"}],
                                          "response_format": {"type": "xml"}})
            self.assertEqual(status, 400)
        finally:
            httpd.shutdown()


class HardeningTest(unittest.TestCase):
    def test_auth_required(self):
        httpd, base = start(FakeRuntime(), api_key="k")
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(base + "/v1/models", timeout=5)
            self.assertEqual(ctx.exception.code, 401)
            req = urllib.request.Request(base + "/v1/models",
                                         headers={"Authorization": "Bearer k"})
            with urllib.request.urlopen(req, timeout=5) as r:
                self.assertEqual(r.status, 200)
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                self.assertEqual(r.status, 200)
        finally:
            httpd.shutdown()

    def test_rate_limit(self):
        httpd, base = start(FakeRuntime(), rate_limit_rps=1)
        try:
            codes = []
            for _ in range(5):
                body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": 2}).encode()
                req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        codes.append(r.status)
                except urllib.error.HTTPError as exc:
                    codes.append(exc.code)
            self.assertIn(429, codes)
            self.assertIn(200, codes)
        finally:
            httpd.shutdown()

    def test_abort_endpoint(self):
        rt = FakeRuntime()
        httpd, base = start(rt)
        try:
            body = json.dumps({"request_id": "abc"}).encode()
            req = urllib.request.Request(base + "/v1/abort", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                payload = json.loads(r.read())
            self.assertTrue(payload["aborted"])
            self.assertEqual(rt.aborted, ["abc"])
        finally:
            httpd.shutdown()

    def test_memory_rejection_is_503(self):
        httpd, base = start(FakeRuntime(fail_memory=True))
        try:
            body = json.dumps({"messages": [{"role": "user", "content": "hi"}],
                               "max_tokens": 2}).encode()
            req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req, timeout=10)
            self.assertEqual(ctx.exception.code, 503)
            self.assertEqual(ctx.exception.headers.get("Retry-After"), "5")
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    unittest.main()


class AutoPrefixCapTest(unittest.TestCase):
    """auto-registration cap must follow the configured snapshot cap (8k agents)."""

    def test_cap_follows_cfg(self):
        from terrazzo.engine import EngineConfig
        from terrazzo.server import auto_prefix_cap

        self.assertEqual(auto_prefix_cap(EngineConfig(prefix_cache_max_tokens=8192)), 8192)
        self.assertEqual(auto_prefix_cap(EngineConfig(prefix_cache_max_tokens=4096)), 4096)
        # never below the historical 4096 floor
        self.assertEqual(auto_prefix_cap(EngineConfig(prefix_cache_max_tokens=1024)), 4096)


class _PrewarmRuntime(FakeRuntime):
    def prewarm(self, system_text: str) -> dict:
        if not system_text.strip():
            raise ValueError("system prompt is empty")
        return {"registered": True, "tokens": 42, "ms": 1.0}


class ControlEndpointsTest(unittest.TestCase):
    def _get(self, base, path):
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, json.loads(r.read())

    def _post(self, base, path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_api_tags_and_ps(self):
        httpd, base = start(FakeRuntime())
        try:
            status, tags = self._get(base, "/api/tags")
            self.assertEqual(status, 200)
            self.assertIn("models", tags)
            status, ps = self._get(base, "/api/ps")
            self.assertEqual(status, 200)
            self.assertEqual(ps["models"][0]["runtime"], "fake")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_prewarm_unsupported_is_501(self):
        httpd, base = start(FakeRuntime())
        try:
            status, body = self._post(base, "/v1/prewarm", {"system": "hello"})
            self.assertEqual(status, 501)
            self.assertEqual(body["error"]["type"], "unsupported")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_prewarm_supported(self):
        httpd, base = start(_PrewarmRuntime())
        try:
            status, body = self._post(base, "/v1/prewarm", {"system": "long system prompt"})
            self.assertEqual(status, 200)
            self.assertEqual(body["tokens"], 42)
            status, body = self._post(base, "/v1/prewarm", {"system": "   "})
            self.assertEqual(status, 400)
        finally:
            httpd.shutdown()
            httpd.server_close()

class ToolsEndpointTest(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]

    def _post(self, base, body):
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_non_stream_tool_call(self):
        rt = FakeRuntime()
        rt.text = ('<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}'
                   '</tool_call>')
        httpd, base = start(rt)
        try:
            status, body = self._post(base, {
                "messages": [{"role": "user", "content": "北京天气"}],
                "tools": self.TOOLS, "max_tokens": 32})
            self.assertEqual(status, 200)
            choice = body["choices"][0]
            self.assertEqual(choice["finish_reason"], "tool_calls")
            msg = choice["message"]
            self.assertIsNone(msg["content"])
            call = msg["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "get_weather")
            self.assertEqual(json.loads(call["function"]["arguments"])["city"], "Beijing")
            self.assertEqual(rt.tools_seen[0]["function"]["name"], "get_weather")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_plain_answer_with_tools(self):
        rt = FakeRuntime()
        rt.text = "不需要调用工具，直接回答。"
        httpd, base = start(rt)
        try:
            status, body = self._post(base, {
                "messages": [{"role": "user", "content": "你好"}],
                "tools": self.TOOLS, "max_tokens": 32})
            self.assertEqual(status, 200)
            choice = body["choices"][0]
            self.assertEqual(choice["finish_reason"], "stop")
            self.assertEqual(choice["message"]["content"], "不需要调用工具，直接回答。")
            self.assertNotIn("tool_calls", choice["message"])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_invalid_tools_is_400(self):
        httpd, base = start(FakeRuntime())
        try:
            status, body = self._post(base, {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {}}]})
            self.assertEqual(status, 400)
            self.assertEqual(body["error"]["type"], "invalid_request")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_streaming_tool_calls(self):
        rt = FakeRuntime()
        rt.text = '<tool_call>{"name":"get_weather","arguments":{"city":"上海"}}</tool_call>'
        httpd, base = start(rt)
        try:
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "上海天气"}],
                                 "tools": self.TOOLS, "stream": True,
                                 "max_tokens": 32}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                raw = r.read().decode()
            self.assertIn("tool_calls", raw)
            self.assertIn("get_weather", raw)
            self.assertIn('"finish_reason": "tool_calls"', raw)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_default_system_prompt_injected(self):
        rt = FakeRuntime()
        rt.default_system = "你是运维助手。"
        httpd, base = start(rt)
        try:
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({"messages": [{"role": "user", "content": "hi"}],
                                 "max_tokens": 8}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            self.assertEqual(rt.messages_seen[0]["role"], "system")
            self.assertEqual(rt.messages_seen[0]["content"], "你是运维助手。")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_tool_choice_none_disables_tools(self):
        rt = FakeRuntime()
        rt.text = '<think>先想想</think>不需要调用工具，直接回答。'
        httpd, base = start(rt)
        try:
            status, body = self._post(base, {
                "messages": [{"role": "user", "content": "北京天气"}],
                "tools": self.TOOLS, "tool_choice": "none", "max_tokens": 32})
            self.assertEqual(status, 200)
            self.assertEqual(rt.tools_seen, [])
            # no tools injected -> no structured tool_calls, thinking is stripped
            msg = body["choices"][0]["message"]
            self.assertNotIn("tool_calls", msg)
            self.assertEqual(msg["content"], "不需要调用工具，直接回答。")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_tools_disable_thinking_by_default(self):
        rt = FakeRuntime()
        httpd, base = start(rt)
        try:
            self._post(base, {"messages": [{"role": "user", "content": "hi"}],
                              "tools": self.TOOLS, "max_tokens": 8})
            self.assertIs(rt.thinking_seen, False)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_multi_turn_assistant_null_content_normalized(self):
        rt = FakeRuntime()
        rt.text = "北京晴，26 度。"
        httpd, base = start(rt)
        try:
            status, body = self._post(base, {
                "messages": [
                    {"role": "user", "content": "北京天气"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": '{"city": "北京"}'}}]},
                    {"role": "tool", "tool_call_id": "call_1",
                     "content": '{"temp_c": 26, "sky": "晴"}'},
                ],
                "tools": self.TOOLS, "max_tokens": 32})
            self.assertEqual(status, 200)
            self.assertEqual(body["choices"][0]["message"]["content"], "北京晴，26 度。")
            sent = rt.messages_seen
            self.assertEqual(sent[1]["content"], "")
            self.assertEqual(sent[1]["tool_calls"][0]["function"]["arguments"], {"city": "北京"})
        finally:
            httpd.shutdown()
            httpd.server_close()


class OllamaApiTest(unittest.TestCase):
    """Ollama-compatible endpoints: /api/version, /api/chat, /api/generate, ..."""

    def _post(self, base, path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _post_json(self, base, path, body):
        status, raw = self._post(base, path, body)
        return status, json.loads(raw or b"{}")

    def test_version(self):
        httpd, base = start(FakeRuntime())
        try:
            with urllib.request.urlopen(base + "/api/version", timeout=10) as r:
                data = json.loads(r.read())
            self.assertIn("version", data)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_chat_stream_and_nonstream(self):
        rt = FakeRuntime()
        rt.text = "ok"
        httpd, base = start(rt)
        try:
            status, raw = self._post(base, "/api/chat", {
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "stream": True})
            self.assertEqual(status, 200)
            lines = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
            self.assertTrue(any(l.get("message", {}).get("content") == "ok" for l in lines))
            self.assertTrue(lines[-1]["done"])
            status, data = self._post_json(base, "/api/chat", {
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "stream": False})
            self.assertEqual(status, 200)
            self.assertEqual(data["message"]["content"], "ok")
            self.assertTrue(data["done"])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_generate(self):
        rt = FakeRuntime()
        rt.text = "hello"
        httpd, base = start(rt)
        try:
            status, data = self._post_json(base, "/api/generate", {
                "model": "m", "prompt": "hi", "stream": False})
            self.assertEqual(status, 200)
            self.assertEqual(data["response"], "hello")
            self.assertTrue(data["done"])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_chat_with_tools(self):
        rt = FakeRuntime()
        rt.text = '<tool_call>{"name":"get_weather","arguments":{"city":"北京"}}</tool_call>'
        httpd, base = start(rt)
        try:
            status, data = self._post_json(base, "/api/chat", {
                "model": "m", "messages": [{"role": "user", "content": "天气"}],
                "tools": [{"type": "function", "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
                "stream": False})
            self.assertEqual(status, 200)
            calls = data["message"]["tool_calls"]
            self.assertEqual(calls[0]["function"]["name"], "get_weather")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_unsupported_endpoints_are_explicit(self):
        httpd, base = start(FakeRuntime())
        try:
            for path in ("/api/embed", "/api/embeddings", "/api/pull", "/api/push"):
                status, data = self._post_json(base, path, {"model": "x"})
                self.assertEqual(status, 501, path)
                self.assertIn("error", data)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_show_create_copy_delete(self):
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("TERRAZZO_MODELS_DIR")
            os.environ["TERRAZZO_MODELS_DIR"] = str(_Path(tmp) / "store")
            try:
                base_dir = _Path(tmp) / "basemodel"
                base_dir.mkdir()
                (base_dir / "config.json").write_text(json.dumps(
                    {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}))
                (base_dir / "model.safetensors").write_bytes(b"x")
                from terrazzo import models as store

                store.import_local(str(base_dir), name="base")
                httpd, base = start(FakeRuntime())
                try:
                    status, data = self._post_json(base, "/api/create", {
                        "model": "custom",
                        "modelfile": "FROM base\nSYSTEM \"Be terse.\"\nPARAMETER temperature 0.2\n",
                    })
                    self.assertEqual(status, 200, data)
                    self.assertEqual(store.load_recipe("custom")["system"], "Be terse.")
                    status, data = self._post_json(base, "/api/show", {"model": "custom"})
                    self.assertEqual(status, 200)
                    self.assertIn("FROM base", data["modelfile"])
                    status, _ = self._post_json(base, "/api/copy",
                                                {"source": "custom", "destination": "copy1"})
                    self.assertEqual(status, 200)
                    with urllib.request.urlopen(
                            base + "/api/show?name=copy1", timeout=10) as r:
                        self.assertEqual(r.status, 200)
                    req = urllib.request.Request(
                        base + "/api/delete", data=json.dumps({"model": "copy1"}).encode(),
                        headers={"Content-Type": "application/json"}, method="DELETE")
                    with urllib.request.urlopen(req, timeout=10) as r:
                        self.assertEqual(r.status, 200)
                    self.assertFalse((_Path(tmp) / "store" / "copy1").exists())
                finally:
                    httpd.shutdown()
                    httpd.server_close()
            finally:
                if old is None:
                    os.environ.pop("TERRAZZO_MODELS_DIR", None)
                else:
                    os.environ["TERRAZZO_MODELS_DIR"] = old

    def test_thinking_is_split_from_content(self):
        rt = FakeRuntime()
        rt.text = "<think>secret reasoning</think>final answer"
        httpd, base = start(rt)
        try:
            status, data = self._post_json(base, "/api/chat", {
                "model": "m", "stream": False,
                "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(status, 200)
            self.assertEqual(data["message"]["content"], "final answer")
            self.assertNotIn("thinking", data["message"])
            status, data = self._post_json(base, "/api/chat", {
                "model": "m", "stream": False, "think": True,
                "messages": [{"role": "user", "content": "hi"}]})
            self.assertEqual(data["message"]["content"], "final answer")
            self.assertEqual(data["message"]["thinking"], "secret reasoning")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_streaming_hides_thinking(self):
        rt = FakeRuntime()
        rt.text = "<think>secret reasoning</think>final answer"
        httpd, base = start(rt)
        try:
            status, raw = self._post(base, "/api/chat", {
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "stream": True})
            text = "".join(json.loads(l).get("message", {}).get("content", "")
                           for l in raw.decode().splitlines() if l.strip())
            self.assertEqual(text, "final answer")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_stop_option_truncates(self):
        rt = FakeRuntime()
        rt.text = "hello STOP world"
        httpd, base = start(rt)
        try:
            status, data = self._post_json(base, "/api/generate", {
                "model": "m", "prompt": "hi", "stream": False,
                "options": {"stop": ["STOP"]}})
            self.assertEqual(status, 200)
            self.assertEqual(data["response"], "hello ")
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_format_json_uses_structured_decode(self):
        rt = FakeRuntime()
        httpd, base = start(rt)
        try:
            status, data = self._post_json(base, "/api/chat", {
                "model": "m", "stream": False, "format": "json",
                "messages": [{"role": "user", "content": "give json"}]})
            self.assertEqual(status, 200)
            self.assertEqual(data["message"]["content"], '{"ok": true}')
        finally:
            httpd.shutdown()
            httpd.server_close()
