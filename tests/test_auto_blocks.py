"""Unit test for RAM-derived KV block sizing (no model weights needed)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.engine import auto_num_gpu_blocks  # noqa: E402


class AutoBlocksTest(unittest.TestCase):
    def test_derives_blocks_from_config_and_ram(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "hidden_size": 2560,
                "num_attention_heads": 20,
            }
            Path(tmp, "config.json").write_text(json.dumps(cfg))
            blocks = auto_num_gpu_blocks(tmp, block_size=16)
            # 68.7 GB RAM * 35% / (36*2*8*128*2 bytes * 16 tokens/block) ≈ 10.2k
            self.assertGreater(blocks, 1000)
            self.assertLessEqual(blocks, 32768)
            # a big-KV model gets fewer blocks than a small-KV one would
            cfg["num_key_value_heads"] = 64
            Path(tmp, "config.json").write_text(json.dumps(cfg))
            self.assertLess(auto_num_gpu_blocks(tmp, block_size=16), blocks)

    def test_falls_back_when_config_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"hidden_size": 128}))
            self.assertEqual(auto_num_gpu_blocks(tmp, block_size=16), 4096)


if __name__ == "__main__":
    unittest.main()
