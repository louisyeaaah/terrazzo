"""SLO admission for Mac inference: can this (chip, model) reach the target tok/s?

Calibrated decode model (measured on this M1 Max, see MAC-TURBO.md):

    t_step_ms ≈ bytes_per_token × 1000 / BW_eff  +  FIXED_MS

with ``BW_eff = 0.664 × peak_bandwidth`` and ``FIXED_MS = 3.8`` (per-step framework
overhead). Fit against three measured points:

| model | bytes/token | measured ms | predicted ms |
|---|---:|---:|---:|
| Qwen3-4B (dense, 4.02B active) | 2.263 | 12.32 | 12.32 |
| Qwen3-0.6B (dense) | 0.340 | 5.08 | 5.08 |
| OLMoE-1B-7B (MoE, 1.18B active) | 0.663 | 6.09 | 6.30 |

The MoE result shows that **active parameters do control speed** on MLX — the
earlier bandwidth-only model underestimated MoE by ~5x only on a cold run; once
warmed up it follows the same two-term law.
"""

from __future__ import annotations

# peak unified-memory bandwidth, GB/s. Apple official for M1/M1 Pro/M1 Max/M1 Ultra;
# the rest are public specs, not verified on this machine.
CHIPS: dict[str, tuple[float, str]] = {
    "m1": (68.25, "M1"),
    "m1_pro": (200.0, "M1 Pro"),
    "m1_max": (400.0, "M1 Max"),
    "m1_ultra": (800.0, "M1 Ultra"),
    "m2": (100.0, "M2"),
    "m2_pro": (200.0, "M2 Pro"),
    "m2_max": (400.0, "M2 Max"),
    "m2_ultra": (800.0, "M2 Ultra"),
    "m3": (100.0, "M3"),
    "m3_pro": (150.0, "M3 Pro"),
    "m3_max": (400.0, "M3 Max"),
    "m3_ultra": (800.0, "M3 Ultra"),
    "m4": (120.0, "M4"),
    "m4_pro": (273.0, "M4 Pro"),
    "m4_max": (546.0, "M4 Max"),
}

# weights per parameter at a given bit width (scales/zero-points included)
BYTES_PER_PARAM = {4: 0.5625, 8: 1.0625, 3: 0.4375, 2: 0.3125}

BW_FRACTION = 0.664   # effective / peak bandwidth, calibrated on M1 Max
FIXED_MS = 3.8        # per-step overhead (framework/launch), ms


def model_bytes(params_b: float, bits: int = 4, active_params_b: float | None = None) -> float:
    """GB read per generated token. MoE: pass active params, not total."""
    bpp = BYTES_PER_PARAM.get(bits)
    if bpp is None:
        bpp = bits / 8 + 0.0625
    p = active_params_b if active_params_b is not None else params_b
    return p * bpp


def step_ms(chip: str, params_b: float, *, active_params_b: float | None = None,
            bits: int = 4, bw_fraction: float = BW_FRACTION,
            fixed_ms: float = FIXED_MS) -> float:
    peak = CHIPS.get(chip, (None, None))[0]
    if peak is None:
        raise KeyError(f"unknown chip {chip!r}; known: {sorted(CHIPS)}")
    bw_eff = bw_fraction * peak
    return model_bytes(params_b, bits, active_params_b) * 1000.0 / bw_eff + fixed_ms


def predict_tok_s(chip: str, params_b: float, *, active_params_b: float | None = None,
                  bits: int = 4, bw_fraction: float = BW_FRACTION,
                  fixed_ms: float = FIXED_MS) -> float:
    return 1000.0 / step_ms(chip, params_b, active_params_b=active_params_b, bits=bits,
                            bw_fraction=bw_fraction, fixed_ms=fixed_ms)


def feasibility(chip: str, params_b: float, target_tok_s: float = 200.0, *,
                active_params_b: float | None = None, bits: int = 4,
                bw_fraction: float = BW_FRACTION,
                fixed_ms: float = FIXED_MS) -> dict:
    """Whether (chip, model, bits) can meet the SLO, and what it would take."""
    peak = CHIPS[chip][0]
    bw_eff = bw_fraction * peak
    bpt = model_bytes(params_b, bits, active_params_b)
    tps = 1000.0 / (bpt * 1000.0 / bw_eff + fixed_ms)
    out = {
        "chip": CHIPS[chip][1],
        "model_b": params_b,
        "active_b": active_params_b,
        "bits": bits,
        "bytes_per_token_gb": round(bpt, 3),
        "bw_eff_gbs": round(bw_eff, 1),
        "fixed_ms": fixed_ms,
        "predicted_tok_s": round(tps, 1),
        "target_tok_s": target_tok_s,
        "meets_slo": tps >= target_tok_s,
        "peak_bw_needed_gbs": round(
            (bpt * 1000.0 / (1000.0 / target_tok_s - fixed_ms)), 1)
        if (1000.0 / target_tok_s - fixed_ms) > 0 else None,
    }
    if not out["meets_slo"]:
        out["max_active_b_for_target"] = round(
            bw_eff * (1000.0 / target_tok_s - fixed_ms) / 1000.0 / BYTES_PER_PARAM.get(bits, bits / 8), 2)
        out["advice"] = _advice(params_b, active_params_b, bits, peak, target_tok_s)
    return out


def _advice(params_b: float, active: float | None, bits: int, peak: float, target: float) -> str:
    per_token_budget = (1000.0 / target - FIXED_MS) * BW_FRACTION * peak / 1000.0
    if per_token_budget <= 0:
        return "target below the per-step overhead floor; unreachable at any model size"
    active_budget = per_token_budget / BYTES_PER_PARAM.get(bits, bits / 8)
    if active is None and params_b > active_budget:
        return (f"dense model too large for this chip: the SLO budget is "
                f"{per_token_budget:.2f} GB/token (~{active_budget:.1f}B params at {bits}-bit); "
                f"use a MoE with active params below that, or a faster Mac")
    return (f"needs active params <= {active_budget:.1f}B at {bits}-bit on this chip; "
            f"otherwise throughput is only reachable in aggregate with batching")


def plan(chip: str, params_b: float, target_tok_s: float = 200.0, *,
         active_params_b: float | None = None) -> dict:
    """Find the lowest bit width that meets the SLO on this chip, if any."""
    for bits in (8, 4, 3, 2):
        f = feasibility(chip, params_b, target_tok_s, active_params_b=active_params_b, bits=bits)
        if f["meets_slo"]:
            return {"chip": CHIPS[chip][1], "bits": bits, **f}
    return {"chip": CHIPS[chip][1], "meets_slo": False,
            "advice": "no bit width reaches the SLO on this chip; use a smaller-active model "
                      "or a faster Mac"}
