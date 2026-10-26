"""Fused decode forward for Qwen3: fewer kernels per layer, identical math.

The decode step is kernel-latency bound (36 serial layers). Concatenating weights
lets a layer issue fewer, larger kernels while keeping the same number of FLOPs and
bytes:

* q/k/v projections -> one quantized matmul [hidden -> 6144]
* gate/up projections -> one quantized matmul [hidden -> 19456]

Quantized rows are independent, so concatenating packed weight / scales / biases
along the output dimension is exact. ``fused_decode_step`` mirrors mlx-lm's
Qwen3Model (rmsnorm order, rope offset, GQA layout, tied lm head) for one token.
"""

from __future__ import annotations

import mlx.core as mx


def _cat_weights(modules):
    wq = mx.concatenate([m.weight for m in modules], axis=0)
    sc = mx.concatenate([m.scales for m in modules], axis=0)
    bi = mx.concatenate([m.biases for m in modules], axis=0)
    return (wq, sc, bi)


def _as_w(module):
    return (module.weight, module.scales, module.biases)


def build_fused(model):
    """Per-layer fused qkv / gate-up quantized weights (model is not mutated)."""
    fused = []
    for layer in model.model.layers:
        attn = layer.self_attn
        mlp = layer.mlp
        fused.append({
            "qkv": _cat_weights([attn.q_proj, attn.k_proj, attn.v_proj]),
            "gate_up": _cat_weights([mlp.gate_proj, mlp.up_proj]),
            "qkv_sizes": (attn.q_proj.weight.shape[0], attn.k_proj.weight.shape[0],
                          attn.v_proj.weight.shape[0]),
            "gate_size": mlp.gate_proj.weight.shape[0],
            "group_size": attn.q_proj.group_size,
            "bits": attn.q_proj.bits,
        })
    return fused


def _qlinear(x, w, group_size: int, bits: int):
    return mx.quantized_matmul(x, *w, transpose=True, group_size=group_size, bits=bits)


def fused_decode_step(model, fused, token: int, caches, *, fuse_qkv: bool = True,
                      fuse_gate_up: bool = True) -> mx.array:
    """One decode token -> logits [1, 1, vocab] (same numerics as the model)."""
    from mlx_lm.models.activations import swiglu

    embed = model.model.embed_tokens
    h = embed(mx.array([[token]]))
    for i, (layer, fw) in enumerate(zip(model.model.layers, fused)):
        attn = layer.self_attn
        mlp = layer.mlp
        h_norm = layer.input_layernorm(h)
        B, L, _ = h_norm.shape

        if fuse_qkv:
            qkv = _qlinear(h_norm, fw["qkv"], fw["group_size"], fw["bits"])
            hq, hk, hv = fw["qkv_sizes"]
            q = qkv[..., :hq]
            k = qkv[..., hq:hq + hk]
            v = qkv[..., hq + hk:hq + hk + hv]
        else:
            q = _qlinear(h_norm, _as_w(attn.q_proj), fw["group_size"], fw["bits"])
            k = _qlinear(h_norm, _as_w(attn.k_proj), fw["group_size"], fw["bits"])
            v = _qlinear(h_norm, _as_w(attn.v_proj), fw["group_size"], fw["bits"])

        q = attn.q_norm(q.reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
        k = attn.k_norm(k.reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)

        layer_cache = caches[i]
        q = attn.rope(q, offset=layer_cache.offset)
        k = attn.rope(k, offset=layer_cache.offset)
        k, v = layer_cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        h = h + _qlinear(out, _as_w(attn.o_proj), fw["group_size"], fw["bits"])

        h2 = layer.post_attention_layernorm(h)
        if fuse_gate_up:
            gu = _qlinear(h2, fw["gate_up"], fw["group_size"], fw["bits"])
            hg = fw["gate_size"]
            gate, up = gu[..., :hg], gu[..., hg:]
        else:
            gate = _qlinear(h2, _as_w(mlp.gate_proj), fw["group_size"], fw["bits"])
            up = _qlinear(h2, _as_w(mlp.up_proj), fw["group_size"], fw["bits"])
        h = h + _qlinear(swiglu(gate, up), _as_w(mlp.down_proj), fw["group_size"], fw["bits"])

    return embed.as_linear(model.model.norm(h))

