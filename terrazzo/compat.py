"""Dependency compatibility contract for the mlx-lm internals we patch at runtime.

Terrazzo patches a few mlx-lm cache methods (see ``batch_attn_patch.py`` and
friends). Those are *not* public API, so a new mlx-lm release can silently change
behaviour. Everything here is deliberately narrow:

* ``TESTED_MLX`` / ``TESTED_MLX_LM`` are the exact versions the benchmark numbers and
  test suite were produced with.
* ``require_tested_mlx_lm()`` raises one clear error if the installed version is
  outside the tested range or if the patched attributes are missing.

The package metadata pins ``mlx`` / ``mlx-lm`` to that range (see
``pyproject.toml`` and ``constraints.txt``); this module is the runtime backstop for
people who bypass the pins.
"""

from __future__ import annotations

import re

TESTED_MLX = "0.32.4"
TESTED_MLX_LM = "0.31.3"
MIN_MLX_LM = (0, 31, 3)
MAX_MLX_LM_EXCLUSIVE = (0, 32, 0)


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"(\d+)\.(\d+)\.?(\d*)", version)
    if not match:
        raise ValueError(f"cannot parse version {version!r}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


class CompatibilityError(RuntimeError):
    pass


def check_mlx_lm(version: str | None = None) -> str:
    """Validate the mlx-lm version and the cache attributes the patches rely on."""
    if version is None:
        try:
            from importlib.metadata import version as _dist_version

            version = _dist_version("mlx-lm")
        except Exception as exc:  # pragma: no cover - packaging edge
            raise CompatibilityError(f"mlx-lm is not installed: {exc}") from exc
    parsed = _version_tuple(version)
    if not (MIN_MLX_LM <= parsed < MAX_MLX_LM_EXCLUSIVE):
        raise CompatibilityError(
            f"mlx-lm {version} is outside the tested range "
            f">={'.'.join(map(str, MIN_MLX_LM))},<{'.'.join(map(str, MAX_MLX_LM_EXCLUSIVE))} "
            f"(tested with {TESTED_MLX_LM}). Runtime patches touch non-public mlx-lm "
            f"internals; install the tested version or set "
            f"TERRAZZO_SKIP_MLX_LM_CHECK=1 at your own risk.")
    try:
        from mlx_lm.models.cache import BatchKVCache, KVCache  # noqa: F401
    except ImportError as exc:
        raise CompatibilityError(
            f"mlx-lm {version} is missing mlx_lm.models.cache: {exc}") from exc
    required = {
        "BatchKVCache.prepare": hasattr(BatchKVCache, "prepare"),
        "BatchKVCache.finalize": hasattr(BatchKVCache, "finalize"),
        "BatchKVCache.update_and_fetch": hasattr(BatchKVCache, "update_and_fetch"),
        "BatchKVCache.filter": hasattr(BatchKVCache, "filter"),
        "KVCache.update_and_fetch": hasattr(KVCache, "update_and_fetch"),
    }
    missing = [name for name, ok in required.items() if not ok]
    if missing:
        raise CompatibilityError(
            f"mlx-lm {version} no longer exposes {', '.join(missing)}; "
            f"the runtime patches must be updated (tested with {TESTED_MLX_LM})")
    return version


def require_tested_mlx_lm() -> str:
    """Called by the engine before applying runtime patches."""
    import os

    if os.environ.get("TERRAZZO_SKIP_MLX_LM_CHECK") == "1":
        try:
            from importlib.metadata import version as _dist_version

            return _dist_version("mlx-lm")
        except Exception:
            return "unknown"
    return check_mlx_lm()
