"""Run the pure-Python test subset with MLX blocked.

This is the Linux-CI entry point: it proves that `import terrazzo`, the scheduler /
block manager / prefix-cache logic, the HTTP layer (fake runtime) and the CLI helpers
do not secretly require MLX. Tests that need MLX are skipped or live in the macOS job.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


class _BlockMLX:
    """Meta path finder that makes `import mlx` / `import mlx_lm` fail loudly."""

    def find_spec(self, name, path=None, target=None):  # noqa: D401
        if name.split(".")[0] in ("mlx", "mlx_lm"):
            raise ImportError(f"{name} is blocked by scripts/ci_core_tests.py")
        return None


PURE_MODULES = [
    "tests.test_core",
    "tests.test_models",
    "tests.test_modelfile",
    "tests.test_tools",
    "tests.test_json_schema",
    "tests.test_ngram_spec",
    "tests.test_slo",
    "tests.test_slo_router",
    "tests.test_server_hardening",
    "tests.test_long_chat_prefix",
]


def main() -> int:
    import os

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if os.environ.get("TERRAZZO_CI_ALLOW_MLX") != "1":
        sys.meta_path.insert(0, _BlockMLX())
    suite = unittest.defaultTestLoader.loadTestsFromNames(PURE_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    blocked = "MLX blocked" if os.environ.get("TERRAZZO_CI_ALLOW_MLX") != "1" else "MLX available"
    print(f"\ncore suite ({blocked}): {result.testsRun} run, {len(result.failures)} failed, "
          f"{len(result.errors)} errors, {len(result.skipped)} skipped")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
