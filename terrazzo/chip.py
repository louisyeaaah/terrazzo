"""Chip detection for the MacTurbo SLO router."""

from __future__ import annotations

import platform
import subprocess

from .slo import CHIPS

_FALLBACK_PEAK = 400.0  # conservative: assume a Max-class chip if unknown


def detect_chip() -> dict:
    """Best-effort chip identity + peak bandwidth on this machine."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return {"key": None, "name": platform.processor() or "unknown",
                "peak_bw_gbs": _FALLBACK_PEAK, "verified": False,
                "note": "not Apple Silicon; using fallback bandwidth"}
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        brand = "Apple Silicon"
    key = _normalize(brand)
    if key in CHIPS:
        return {"key": key, "name": CHIPS[key][1], "peak_bw_gbs": CHIPS[key][0],
                "verified": key in ("m1", "m1_pro", "m1_max", "m1_ultra"),
                "note": "Apple official bandwidth" if key in ("m1_pro", "m1_max", "m1_ultra")
                        else "bandwidth from public specs"}
    return {"key": None, "name": brand, "peak_bw_gbs": _FALLBACK_PEAK, "verified": False,
            "note": "unrecognized Apple chip; using fallback bandwidth"}


def _normalize(brand: str) -> str | None:
    b = brand.replace("Apple", "").strip().lower()  # "m1 max"
    parts = b.split()
    if not parts:
        return None
    chip = parts[0]  # m1..m5
    suffix = "_".join(parts[1:]) if len(parts) > 1 else ""
    return f"{chip}_{suffix}" if suffix else chip
