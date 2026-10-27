"""Pure tests for the load-aware cap4/cap16 rule."""

import unittest

from terrazzo.concurrency import AutoCap, AutoCapConfig


def inputs(hit, uncached, samples=16):
    return {"hit_ratio": hit, "avg_uncached": uncached, "samples": samples}


class AutoCapTest(unittest.TestCase):
    def test_switches_low_for_mixed_prefix(self):
        ac = AutoCap(current=16, now=0.0)
        target = ac.decide(now=7.0, hit_ratio=0.6, avg_uncached=1200,
                           queue=8, memory_ratio=0.1)
        self.assertEqual(target, 4)
        self.assertEqual(ac.current, 4)

    def test_switches_high_for_shared_prefix(self):
        ac = AutoCap(current=4, now=0.0)
        target = ac.decide(now=7.0, hit_ratio=0.99, avg_uncached=24,
                           queue=8, memory_ratio=0.1)
        self.assertEqual(target, 16)

    def test_min_samples_and_min_hold(self):
        ac = AutoCap(current=16, now=0.0)
        self.assertIsNone(ac.decide(now=1.0, hit_ratio=0.1, avg_uncached=2000,
                                    queue=8, memory_ratio=0.1, samples=4))
        self.assertIsNone(ac.decide(now=1.0, hit_ratio=0.1, avg_uncached=2000,
                                    queue=8, memory_ratio=0.1, samples=16))
        self.assertEqual(ac.decide(now=7.0, hit_ratio=0.1, avg_uncached=2000,
                                   queue=8, memory_ratio=0.1, samples=16), 4)

    def test_memory_pressure_forces_low(self):
        ac = AutoCap(current=16, now=0.0)
        self.assertEqual(ac.decide(now=10.0, hit_ratio=0.99, avg_uncached=20,
                                   queue=8, memory_ratio=0.9, samples=16), 4)

    def test_observe_inputs(self):
        ac = AutoCap()
        costs = [
            {"prompt_tokens": 2000, "cached_estimate": 2000, "uncached_estimate": 0},
            {"prompt_tokens": 2000, "cached_estimate": 0, "uncached_estimate": 2000},
        ]
        hit, avg_uncached, samples = ac.observe_inputs(costs)
        self.assertAlmostEqual(hit, 0.5)
        self.assertEqual(avg_uncached, 1000)
        self.assertEqual(samples, 2)


if __name__ == "__main__":
    unittest.main()
