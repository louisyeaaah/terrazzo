"""Unit tests for the MacTurbo SLO admission model."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo.slo import (  # noqa: E402
    CHIPS,
    feasibility,
    model_bytes,
    plan,
    predict_tok_s,
)


class CalibrationTest(unittest.TestCase):
    def test_m1_max_4b_matches_measurement(self):
        # measured: 12.32 ms/token -> 81 tok/s (single), 96 tok/s (batched engine)
        pred = predict_tok_s("m1_max", 4.02)
        self.assertGreater(pred, 75)
        self.assertLess(pred, 95)

    def test_three_calibration_points(self):
        # the fitted two-term law must reproduce the three measured models
        self.assertAlmostEqual(predict_tok_s("m1_max", 4.02), 81.2, delta=6)
        self.assertAlmostEqual(predict_tok_s("m1_max", 0.6), 197.2, delta=15)
        self.assertAlmostEqual(predict_tok_s("m1_max", 6.92, active_params_b=1.18), 158.8, delta=20)

    def test_model_bytes_4bit(self):
        self.assertAlmostEqual(model_bytes(4), 2.25, places=2)
        self.assertAlmostEqual(model_bytes(7), 3.94, places=2)
        self.assertAlmostEqual(model_bytes(30, active_params_b=3), 1.69, places=2)

    def test_unknown_chip(self):
        with self.assertRaises(KeyError):
            predict_tok_s("m9_ultra", 4)


class Target200Test(unittest.TestCase):
    def test_dense_7b_never_meets_200_on_any_current_chip(self):
        for chip in CHIPS:
            f = feasibility(chip, 7, 200)
            self.assertFalse(f["meets_slo"], f"{chip} claims 200 tok/s for dense 7B")

    def test_dense_4b_never_reaches_200_single_stream(self):
        # calibrated: even an M3 Ultra tops out ~136 tok/s for dense 4B
        for chip in CHIPS:
            self.assertFalse(feasibility(chip, 4.02, 200)["meets_slo"], chip)

    def test_only_tiny_models_reach_200_single_stream(self):
        # 0.6B dense on an Ultra is the smallest real example that crosses 200
        self.assertTrue(plan("m3_ultra", 0.6, 200)["meets_slo"])
        self.assertFalse(feasibility("m1_max", 0.6, 200)["meets_slo"])  # 197 < 200

    def test_moe_30b_a3b_single_stream_below_200(self):
        # measured law: 3B active on an Ultra predicts ~143 tok/s, not 200
        f = feasibility("m3_ultra", 30, 200, active_params_b=3)
        self.assertFalse(f["meets_slo"])
        self.assertGreater(f["predicted_tok_s"], 120)
        # it reports the active-param budget needed for the target
        self.assertLess(f["max_active_b_for_target"], 1.5)

    def test_dense_27b_impossible(self):
        for chip in CHIPS:
            self.assertLess(predict_tok_s(chip, 27), 40)


class PlanTest(unittest.TestCase):
    def test_plan_prefers_4bit_when_possible(self):
        # a 0.6B dense model on an Ultra clears 200 tok/s at 4-bit
        p = plan("m3_ultra", 0.6, 200)
        self.assertTrue(p["meets_slo"])
        self.assertEqual(p["bits"], 4)

    def test_plan_reports_failure_for_moe_a3b(self):
        p = plan("m4_max", 30, 200, active_params_b=3)
        self.assertFalse(p["meets_slo"])
        self.assertIn("advice", p)

    def test_plan_falls_back_to_lower_bits_for_small_models(self):
        # 1.2B-active MoE: 4-bit misses 200 on an Ultra, 3-bit fits
        p = plan("m3_ultra", 6.92, 200, active_params_b=1.18)
        self.assertTrue(p["meets_slo"], p)
        self.assertLess(p["bits"], 4)

    def test_plan_reports_failure_for_dense_7b(self):
        p = plan("m4_max", 7, 200)
        self.assertFalse(p["meets_slo"])
        self.assertIn("advice", p)


if __name__ == "__main__":
    unittest.main()
