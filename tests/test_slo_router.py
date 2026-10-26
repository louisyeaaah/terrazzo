"""Tests for chip detection, model profiling and the SLO router."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.chip import _normalize, detect_chip  # noqa: E402
from terrazzo.model_profile import active_params_b, load_config, profile  # noqa: E402
from terrazzo.slo_router import capability, enforce  # noqa: E402

QWEN3_4B = os.environ.get("TERRAZZO_MAC_MODEL", "")
OLMOE = os.environ.get("TERRAZZO_OLMOE_MODEL", "")


class ChipTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(_normalize("Apple M1 Max"), "m1_max")
        self.assertEqual(_normalize("Apple M4 Pro"), "m4_pro")
        self.assertEqual(_normalize("Apple M3 Ultra"), "m3_ultra")
        self.assertEqual(_normalize("Apple M5"), "m5")

    def test_detect(self):
        chip = detect_chip()
        self.assertGreater(chip["peak_bw_gbs"], 0)
        if chip["key"]:
            self.assertTrue(chip["name"].startswith("M"))


@unittest.skipUnless(QWEN3_4B and Path(QWEN3_4B).is_dir(), "set TERRAZZO_MAC_MODEL")
class DenseProfileTest(unittest.TestCase):
    def test_qwen3_4b_profile(self):
        p = profile(QWEN3_4B)
        self.assertFalse(p["is_moe"])
        self.assertGreater(p["active_params_b"], 3.6)
        self.assertLess(p["active_params_b"], 4.5)
        # 4-bit: ~2.2-2.4 GB per token, matching the measured 2.28 GB weight size
        self.assertGreater(p["bytes_per_token_gb"], 2.1)
        self.assertLess(p["bytes_per_token_gb"], 2.5)

    def test_capability_matches_measurement(self):
        cap = capability(QWEN3_4B, target_tok_s=200)
        # calibrated law predicts 81.2 tok/s; measured 81 single / 96 batched
        self.assertGreater(cap["predicted_tok_s"], 70)
        self.assertLess(cap["predicted_tok_s"], 95)
        self.assertFalse(cap["meets_target"])
        ok, msg = enforce(cap)
        self.assertFalse(ok)
        self.assertIn("NOT reachable", msg)
        ok2, _ = enforce(cap, allow_below_target=True)
        self.assertTrue(ok2)


@unittest.skipUnless(OLMOE and Path(OLMOE).is_dir(), "set TERRAZZO_OLMOE_MODEL")
class MoeProfileTest(unittest.TestCase):
    def test_olmoe_profile(self):
        p = profile(OLMOE)
        self.assertTrue(p["is_moe"])
        self.assertLess(p["active_params_b"], 2.0)   # ~1.3B active
        self.assertGreater(p["total_params_b"], 6.0)  # ~7B total
        # small-active MoE: predicted ~159; measured 165 (within the +/-20% band)
        cap = capability(OLMOE)
        self.assertGreater(cap["predicted_tok_s"], 130)
        self.assertLess(cap["predicted_tok_s"], 200)


if __name__ == "__main__":
    unittest.main()
