"""Estimate how many weight bytes a model reads per generated token.

The SLO router needs this *before* loading the model. We derive it from the HF
``config.json``:

  * dense: every weight is read per token, except the token-embedding table
    (only one row is gathered). The LM head IS read (unless tied).
  * MoE: only the routed experts + shared experts for this token are read.

Estimates are marked ``estimated`` and carry a +/-20% band; the matrix in
MAC-TURBO.md gets recalibrated from measured runs (see §MoE calibration).
"""

from __future__ import annotations

import json
from pathlib import Path

# architectures we can model from config fields
_MOE_KEYS = (
    "num_experts", "num_local_experts", "n_routed_experts",
    "num_experts_per_tok", "n_shared_experts", "moe_intermediate_size",
    "intermediate_size", "shared_expert_intermediate_size",
)


def load_config(model_path: str) -> dict:
    p = Path(model_path)
    cfg_file = p / "config.json" if p.is_dir() else None
    if cfg_file is None or not cfg_file.is_file():
        raise FileNotFoundError(f"no config.json under {model_path}")
    cfg = json.loads(cfg_file.read_text())
    # multimodal configs keep the text model under text_config
    return cfg.get("text_config", cfg)


def _g(cfg: dict, *names, default=None):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


def active_params_b(cfg: dict) -> tuple[float, dict]:
    """(active params in billions, detail dict). Includes lm_head, excludes embedding."""
    h = _g(cfg, "hidden_size", "n_embd", "d_model")
    layers = _g(cfg, "num_hidden_layers", "n_layer", "num_layers")
    nh = _g(cfg, "num_attention_heads", "n_head", "num_heads")
    nkv = _g(cfg, "num_key_value_heads", "n_kv_heads", default=nh)
    hd = _g(cfg, "head_dim", default=(h // nh) if h and nh else None)
    vocab = _g(cfg, "vocab_size", default=0)
    tied = bool(_g(cfg, "tie_word_embeddings", default=False))
    if not (h and layers and nh and hd):
        raise ValueError("config lacks hidden_size/layers/heads")

    # attention per layer: q + k + v + o
    attn = layers * (h * nh * hd + 2 * h * nkv * hd + nh * hd * h)

    n_experts = _g(cfg, "num_experts", "num_local_experts", "n_routed_experts")
    detail: dict = {"attn_b": round(attn / 1e9, 3)}
    if n_experts:
        topk = _g(cfg, "num_experts_per_tok", "num_experts_per_token", default=2)
        moe_inter = _g(cfg, "moe_intermediate_size", "intermediate_size", default=4 * h)
        n_shared = _g(cfg, "n_shared_experts", "num_shared_experts", default=0) or 0
        # routed experts active per token + shared experts + router gate
        routed = layers * topk * 3 * h * moe_inter
        shared = layers * n_shared * 3 * h * moe_inter if n_shared else 0
        gate = layers * h * n_experts
        mlp = routed + shared + gate
        detail.update({"routed_b": round(routed / 1e9, 3), "shared_b": round(shared / 1e9, 3),
                       "gate_b": round(gate / 1e9, 3), "n_experts": n_experts, "topk": topk})
    else:
        inter = _g(cfg, "intermediate_size", default=4 * h)
        mlp = layers * 3 * h * inter
        detail["mlp_b"] = round(mlp / 1e9, 3)

    # the LM head matrix is read in full on every token (tied or not: when tied the
    # embedding table is reused as the head, so the bytes are still spent)
    lm_head = vocab * h
    detail["lm_head_b"] = round(lm_head / 1e9, 3)
    detail["lm_head_tied"] = tied
    total_active = attn + mlp + lm_head
    return total_active / 1e9, detail


def total_params_b(cfg: dict) -> float:
    """Total parameters (for MoE this is much larger than the active count)."""
    h = _g(cfg, "hidden_size")
    layers = _g(cfg, "num_hidden_layers")
    vocab = _g(cfg, "vocab_size", default=0)
    n_experts = _g(cfg, "num_experts", "num_local_experts", "n_routed_experts")
    inter = _g(cfg, "intermediate_size", "moe_intermediate_size", default=4 * h)
    nh = _g(cfg, "num_attention_heads")
    nkv = _g(cfg, "num_key_value_heads", default=nh)
    hd = _g(cfg, "head_dim", default=h // nh)
    attn = layers * (h * nh * hd + 2 * h * nkv * hd + nh * hd * h)
    if n_experts:
        moe_inter = _g(cfg, "moe_intermediate_size", "intermediate_size", default=4 * h)
        mlp = layers * n_experts * 3 * h * moe_inter
    else:
        mlp = layers * 3 * h * inter
    embed = vocab * h
    return (attn + mlp + embed + vocab * h) / 1e9


def profile(model_path: str, bits: int = 4) -> dict:
    from .slo import model_bytes

    cfg = load_config(model_path)
    active, detail = active_params_b(cfg)
    total = total_params_b(cfg)
    return {
        "active_params_b": round(active, 2),
        "total_params_b": round(total, 2),
        "is_moe": bool(_g(cfg, "num_experts", "num_local_experts", "n_routed_experts")),
        "bytes_per_token_gb": round(model_bytes(total, bits, active_params_b=active), 3),
        "bits": bits,
        "estimated": True,
        "detail": detail,
    }
