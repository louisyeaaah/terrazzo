"""Prewarm proxy helpers: the proxy's warmup prompt must match what clients send."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "bench"))


class SystemTextTest(unittest.TestCase):
    def test_matches_agent_workload_generator(self):
        from agent_workload import build_system_prompt
        from vllm_prewarm_proxy import build_system_text

        for tokens in (512, 2544, 4000):
            self.assertEqual(build_system_text(tokens, None), build_system_prompt(tokens))

    def test_system_file_wins(self):
        from vllm_prewarm_proxy import build_system_text

        p = Path("/tmp/prewarm_proxy_test.txt")
        p.write_text("hello system")
        try:
            self.assertEqual(build_system_text(2544, str(p)), "hello system")
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
