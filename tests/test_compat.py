"""Dependency-compat contract: version range and patched-attribute checks."""

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrazzo import compat  # noqa: E402


class CompatTest(unittest.TestCase):
    def test_tested_version_passes(self):
        self.assertEqual(compat.check_mlx_lm(compat.TESTED_MLX_LM), compat.TESTED_MLX_LM)

    def test_newer_version_raises_with_clear_message(self):
        with self.assertRaises(compat.CompatibilityError) as ctx:
            compat.check_mlx_lm("0.32.0")
        self.assertIn("outside the tested range", str(ctx.exception))

    def test_older_version_raises(self):
        with self.assertRaises(compat.CompatibilityError):
            compat.check_mlx_lm("0.30.9")

    def test_missing_attribute_raises(self):
        with mock.patch("mlx_lm.models.cache.BatchKVCache.prepare", new=None, create=True):
            # patch the class object itself: use a shim class without prepare
            from mlx_lm.models import cache as cache_mod

            class ShimBatch:
                finalize = staticmethod(lambda *a, **k: None)
                update_and_fetch = staticmethod(lambda *a, **k: None)
                filter = staticmethod(lambda *a, **k: None)

            class ShimKVCache:
                update_and_fetch = staticmethod(lambda *a, **k: None)

            with mock.patch.object(cache_mod, "BatchKVCache", ShimBatch):
                with mock.patch.object(cache_mod, "KVCache", ShimKVCache):
                    with self.assertRaises(compat.CompatibilityError) as ctx:
                        compat.check_mlx_lm("0.31.3")
            self.assertIn("BatchKVCache.prepare", str(ctx.exception))

    def test_skip_env_bypasses_version(self):
        with mock.patch.dict("os.environ", {"TERRAZZO_SKIP_MLX_LM_CHECK": "1"}):
            self.assertIsInstance(compat.require_tested_mlx_lm(), str)


if __name__ == "__main__":
    unittest.main()
