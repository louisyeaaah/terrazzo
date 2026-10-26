"""SLO router: should this Mac serve this model at the requested tok/s?

Combines chip detection, the per-token weight estimate and the placement formula
from MAC-TURBO.md, then gives the server a capability record and an admission
verdict.
"""

from __future__ import annotations

from .chip import detect_chip
from .model_profile import profile
from .slo import BW_FRACTION, FIXED_MS, step_ms


def capability(model_path: str, *, target_tok_s: float | None = None,
               bits: int = 4, chip: dict | None = None) -> dict:
    chip = chip or detect_chip()
    prof = profile(model_path, bits)
    bw_eff = BW_FRACTION * chip["peak_bw_gbs"]
    t_ms = step_ms("m1_max", 1.0, bits=bits)  # placeholder never used; kept for clarity
    t_ms = prof["bytes_per_token_gb"] * 1000.0 / bw_eff + FIXED_MS
    predicted = 1000.0 / t_ms if t_ms > 0 else 0.0
    meets = target_tok_s is None or predicted >= target_tok_s
    cap = {
        "chip": chip["name"],
        "chip_key": chip["key"],
        "chip_verified": chip["verified"],
        "peak_bw_gbs": chip["peak_bw_gbs"],
        "bw_fraction": BW_FRACTION,
        "bw_eff_gbs": round(bw_eff, 1),
        "step_ms": round(t_ms, 2),
        "model": str(model_path),
        "is_moe": prof["is_moe"],
        "active_params_b": prof["active_params_b"],
        "total_params_b": prof["total_params_b"],
        "bits": bits,
        "bytes_per_token_gb": prof["bytes_per_token_gb"],
        "predicted_tok_s": round(predicted, 1),
        "target_tok_s": target_tok_s,
        "meets_target": meets,
        "estimate_band_pct": 20,
        "detail": prof["detail"],
    }
    if target_tok_s is not None and not meets:
        cap["admission"] = "reject"
        cap["advice"] = advice_for(cap, target_tok_s)
    else:
        cap["admission"] = "accept"
    return cap


def advice_for(cap: dict, target: float) -> str:
    """What would have to change to meet the target."""
    budget_gb = max(0.0, (1000.0 / target - FIXED_MS)) * cap["bw_eff_gbs"] / 1000.0
    active_budget = budget_gb / 0.5625  # 4-bit
    return (f"single-stream budget on this chip is ~{budget_gb:.2f} GB/token "
            f"(~{active_budget:.1f}B active params at 4-bit). "
            f"This model needs {cap['bytes_per_token_gb']:.2f} GB/token; "
            f"{'switch to a smaller-active MoE' if cap['is_moe'] else 'use a small-active MoE or a faster Mac'} "
            f"or accept aggregate throughput with batching")


def enforce(cap: dict, *, allow_below_target: bool = False) -> tuple[bool, str]:
    """(ok, message) for server admission."""
    if cap.get("meets_target", True) or allow_below_target:
        return True, (f"[MacTurbo] {cap['chip']}: predicted {cap['predicted_tok_s']} tok/s "
                      f"({cap['bytes_per_token_gb']} GB/token at {cap['bits']}-bit)")
    return False, (f"[MacTurbo] target {cap['target_tok_s']} tok/s NOT reachable: predicted "
                   f"{cap['predicted_tok_s']} tok/s. {cap['advice']}. "
                   f"Use --allow-below-target to serve anyway.")
