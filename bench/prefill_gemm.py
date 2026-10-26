"""Prefill ceiling: MLX 4-bit quantized matmul throughput at Qwen3-4B shapes.

Prefill is dominated by weight matmuls. For each layer we need:
  qkv  : [M,2560] x [6144,2560]^T   (q 32*128 + k/v 8*128 each)
  o    : [M,4096] x [2560,4096]^T
  mlp  : [M,2560] x [9728,2560]^T  (gate + up) and [M,9728] x [2560,9728]^T
This script measures each (M in 256..4000), reports effective TFLOP/s and the
projected 4k-token prefill time for 36 layers, plus whether mx.compile helps.

    python bench/prefill_gemm.py [--batches 256,512,1024,2048,4000]
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

HIDDEN = 2560
INTER = 9728
QKV_OUT = 32 * 128 + 2 * 8 * 128   # q + k + v
O_OUT = 32 * 128
LAYERS = 36
GROUP, BITS = 64, 4


def quant_weight(out_dim: int, in_dim: int):
    w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.float16)
    return mx.quantize(w, group_size=GROUP, bits=BITS)


def bench(fn, n: int = 5) -> float:
    fn()
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1000


def mm(x, wq):
    return mx.quantized_matmul(x, *wq, transpose=True, group_size=GROUP, bits=BITS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="256,512,1024,2048,4000")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    w_qkv = quant_weight(QKV_OUT, HIDDEN)
    w_o = quant_weight(O_OUT, 32 * 128)
    w_gate = quant_weight(INTER, HIDDEN)
    w_up = quant_weight(INTER, HIDDEN)
    w_down = quant_weight(HIDDEN, INTER)

    for M in [int(b) for b in args.batches.split(",")]:
        x = mx.random.normal((M, HIDDEN)).astype(mx.float16)
        h = mx.random.normal((M, 32 * 128)).astype(mx.float16)
        mid = mx.random.normal((M, INTER)).astype(mx.float16)

        times = {}
        for name, fn in {
            "qkv": lambda: mm(x, w_qkv),
            "o": lambda: mm(h, w_o),
            "gate": lambda: mm(x, w_gate),
            "up": lambda: mm(x, w_up),
            "down": lambda: mm(mid, w_down),
        }.items():
            times[name] = min(bench(fn) for _ in range(args.repeats))
        layer_ms = times["qkv"] + times["o"] + times["gate"] + times["up"] + times["down"]
        flops = 2 * M * (QKV_OUT * HIDDEN + O_OUT * 32 * 128 + 2 * INTER * HIDDEN
                         + HIDDEN * INTER)
        tflops = flops / (layer_ms / 1000) / 1e12
        prefill_s = layer_ms * LAYERS / 1000
        print(f"M={M:>5}: layer={layer_ms:6.1f}ms ({tflops:4.1f} TFLOP/s)  "
              f"36-layer={prefill_s:5.2f}s  -> {M / prefill_s:6.1f} tok/s  "
              f"[qkv {times['qkv']:.1f} o {times['o']:.1f} gate {times['gate']:.1f} "
              f"up {times['up']:.1f} down {times['down']:.1f}]", flush=True)

    # does mx.compile help for a fixed-shape matmul chain?
    M = 512
    x = mx.random.normal((M, HIDDEN)).astype(mx.float16)

    def chain():
        a = mm(x, w_gate)
        b = mm(x, w_up)
        return mm(mx.maximum(a, 0) * b, w_down)

    compiled = mx.compile(chain)
    plain_ms = min(bench(chain) for _ in range(3))
    try:
        comp_ms = min(bench(compiled) for _ in range(3))
        print(f"mx.compile: plain={plain_ms:.1f}ms compiled={comp_ms:.1f}ms "
              f"({plain_ms / comp_ms:.2f}x)", flush=True)
    except Exception as exc:
        print(f"mx.compile failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
