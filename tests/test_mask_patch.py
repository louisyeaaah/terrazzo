"""Tests for the decode mask patch (no padding -> no mask array)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx  # noqa: E402

from terrazzo.mask_patch import apply_mask_patch, disable_mask_patch  # noqa: E402


class MaskPatchTest(unittest.TestCase):
    def setUp(self):
        from mlx_lm.models.cache import BatchKVCache

        self.cache_cls = BatchKVCache
        self.enabled = apply_mask_patch()

    def tearDown(self):
        disable_mask_patch()

    def test_patch_installs(self):
        self.assertTrue(self.enabled)

    def test_no_padding_decode_skips_mask(self):
        if not self.enabled:
            self.skipTest("patch disabled")
        cache = self.cache_cls([0, 0])
        cache.prepare()
        self.assertIsNone(cache.make_mask(1))

    def test_left_padding_keeps_mask(self):
        if not self.enabled:
            self.skipTest("patch disabled")
        cache = self.cache_cls([0, 2])
        cache.prepare()
        mask = cache.make_mask(1)
        self.assertIsNotNone(mask)
        self.assertEqual(tuple(mask.shape), (2, 1, 1, 1))

    def test_prompt_step_keeps_mask(self):
        if not self.enabled:
            self.skipTest("patch disabled")
        cache = self.cache_cls([0, 0])
        cache.prepare()
        mask = cache.make_mask(4)
        self.assertIsNotNone(mask)
        mx.eval(mask)

    def test_disable_restores_original(self):
        disable_mask_patch()
        cache = self.cache_cls([0, 0])
        cache.prepare()
        self.assertIsNotNone(cache.make_mask(1))


if __name__ == "__main__":
    unittest.main()
