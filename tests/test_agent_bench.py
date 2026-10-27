"""Agent-workload benchmark: native Ollama token accounting."""

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

import agent_workload  # noqa: E402


class _OllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    eval_count = 5

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for piece in ("a", "b", "c"):
            self.wfile.write((json.dumps({"message": {"role": "assistant",
                                                      "content": piece},
                                          "done": False}) + "\n").encode())
        done = {"message": {"role": "assistant", "content": ""}, "done": True,
                "prompt_eval_count": 100}
        if self.eval_count is not None:
            done["eval_count"] = self.eval_count
        self.wfile.write((json.dumps(done) + "\n").encode())
        self.wfile.flush()

    def log_message(self, *args):
        pass


def start_server(handler_cls):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


class OllamaAccountingTest(unittest.TestCase):
    def test_eval_count_is_authoritative(self):
        httpd, base = start_server(_OllamaHandler)
        try:
            result = agent_workload.chat_ollama(
                base, "m", [{"role": "user", "content": "hi"}], 16)
            self.assertEqual(result["tokens"], 5)
            self.assertEqual(result["token_source"], "ollama-eval")
            self.assertEqual(result["prompt_tokens"], 100)
            self.assertEqual(result["content"], "abc")
            self.assertEqual(result["content_events"], 3)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_missing_eval_count_falls_back_and_is_marked_approximate(self):
        class NoCountHandler(_OllamaHandler):
            eval_count = None

        httpd, base = start_server(NoCountHandler)
        try:
            result = agent_workload.chat_ollama(
                base, "m", [{"role": "user", "content": "hi"}], 16)
            self.assertEqual(result["tokens"], 3)
            self.assertEqual(result["token_source"], "chunks(approx)")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
