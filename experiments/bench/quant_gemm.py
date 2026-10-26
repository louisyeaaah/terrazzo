"""Decompose MLX 4-bit quantized matmul: dequant cost vs GEMM cost vs bandwidth.

For Qwen3-4B shapes and M in {1, 16, 256, 1024, 4000}, compares:
  quant      : mx.quantized_matmul (what the model uses)
  dequant+mm : mx.dequantize(weights) then fp16 matmul, both inside the timing
  fp16 mm    : pre-dequantized weights, fp16 matmul only (no dequant at all)

Reports ms, achieved weight bandwidth (GB/s) and effective TFLOP/s, plus the
projected decode tok/s (M=1, sum over 36 layers + lm_head) and prefill seconds
(M=4000) for each variant.

    python experiments/bench/quant_gemm.py [--m 1,16,256,1024,4000]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

GROUP, BITS = 64, 4
LAYERS = 36
HIDDEN = 2560
INTER = 9728
QKV = 32 * 128 + 2 * 8 * 128
O = 32 * 128
VOCAB = 151936

# (name, out_dim, in_dim, count per layer)
SHAPES = [
    ("qkv", QKV, HIDDEN, 1),
    ("o", HIDDEN, O, 1),
    ("gate", INTER, HIDDEN, 1),
    ("up", INTER, HIDDEN, 1),
    ("down", HIDDEN, INTER, 1),
]
LM_HEAD = ("lm_head", VOCAB, HIDDEN, 1)


def bench(fn, n: int = 6) -> float:
    fn()
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1000


def weight_bytes(out_dim: int, in_dim: int) -> int:
    packed = out_dim * in_dim * BITS // 8
    meta = (out_dim * in_dim // GROUP) * 2 * 2  # scales+biases fp16
    return packed + meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,16,256,1024,4000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chain", action="store_true",
                    help="also time the whole layer stack as one graph (no per-matmul eval)")
    args = ap.parse_args()

    for M in [int(x) for x in args.m.split(",")]:
        totals = {"quant_ms": 0.0, "dequant_ms": 0.0, "fp16_ms": 0.0,
                  "bytes": 0, "flops": 0}
        per_shape = {}
        for name, out_dim, in_dim, count in SHAPES + [LM_HEAD]:
            w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.float16)
            wq = mx.quantize(w, group_size=GROUP, bits=BITS)
            w16 = mx.dequantize(*wq, group_size=GROUP, bits=BITS)
            x = mx.random.normal((M, in_dim)).astype(mx.float16)
            mx.eval(w16, x)

            t_quant = min(bench(lambda: mx.quantized_matmul(
                x, *wq, transpose=True, group_size=GROUP, bits=BITS))
                for _ in range(args.repeats))
            def dq():
                wd = mx.dequantize(*wq, group_size=GROUP, bits=BITS)
                return mx.matmul(x, wd.T)
            t_dequant = min(bench(dq) for _ in range(args.repeats))
            t_fp16 = min(bench(lambda: mx.matmul(x, w16.T)) for _ in range(args.repeats))

            n = LAYERS if name != "lm_head" else 1
            per_shape[name] = (t_quant, t_dequant, t_fp16)
            totals["quant_ms"] += t_quant * n
            totals["dequant_ms"] += t_dequant * n
            totals["fp16_ms"] += t_fp16 * n
            totals["bytes"] += weight_bytes(out_dim, in_dim) * n
            totals["flops"] += 2 * M * out_dim * in_dim * n
            del w, wq, w16, x

        gb = totals["bytes"] / 1e9
        tf = totals["flops"] / 1e12
        def line(label, ms):
            bw = gb / (ms / 1000) if ms else 0
            return f"{label}={ms:7.2f}ms ({bw:5.0f} GB/s, {tf / (ms / 1000) :4.1f} TF/s)"
        print(f"M={M:>5} weights={gb:4.2f} GB  " + "  ".join([
            line("quant", totals["quant_ms"]),
            line("dequant+mm", totals["dequant_ms"]),
            line("fp16", totals["fp16_ms"]),
        ]), flush=True)
        if M == 1:
            print(f"  decode projection: quant {1000/totals['quant_ms']:5.1f} tok/s | "
                  f"dequant+mm {1000/totals['dequant_ms']:5.1f} | fp16 {1000/totals['fp16_ms']:5.1f}",
                  flush=True)
        else:
            scale = 4000 / M
            print(f"  4k-prefill projection: quant {totals['quant_ms']*scale/1000:5.2f}s | "
                  f"dequant+mm {totals['dequant_ms']*scale/1000:5.2f}s | "
                  f"fp16 {totals['fp16_ms']*scale/1000:5.2f}s", flush=True)
    if args.chain:
        chain_mode(args.repeats)
    return 0


def chain_mode(repeats: int) -> None:
    """Time all 36 layers as one queued graph: real streaming behaviour."""
    weights = []
    for layer in range(LAYERS):
        layer_w = []
        for _name, out_dim, in_dim, _count in SHAPES:
            w = (mx.random.normal((out_dim, in_dim)) * 0.02).astype(mx.float16)
            layer_w.append(mx.quantize(w, group_size=GROUP, bits=BITS))
        weights.append(layer_w)
    total_bytes = sum(weight_bytes(o, i) for _n, o, i, _c in SHAPES) * LAYERS

    inputs = {in_dim: mx.random.normal((1, in_dim)).astype(mx.float16)
              for _n, _o, in_dim, _c in SHAPES}

    def run_chain(dequant_first: bool) -> float:
        t0 = time.perf_counter()
        outs = []
        for layer_w in weights:
            for (_name, _o, in_dim, _c), wq in zip(SHAPES, layer_w):
                x = inputs[in_dim]
                if dequant_first:
                    wd = mx.dequantize(*wq, group_size=GROUP, bits=BITS)
                    outs.append(mx.matmul(x, wd.T))
                else:
                    outs.append(mx.quantized_matmul(x, *wq, transpose=True,
                                                    group_size=GROUP, bits=BITS))
        mx.eval(*outs)
        return time.perf_counter() - t0

    gb = total_bytes / 1e9
    run_chain(False)
    q = min(run_chain(False) for _ in range(repeats))
    d = min(run_chain(True) for _ in range(repeats))
    print(f"chain M=1 ({LAYERS} layers, {gb:.2f} GB weights, one graph): "
          f"quant={q * 1000:7.2f}ms ({gb / q:5.0f} GB/s)  "
          f"dequant+matmul={d * 1000:7.2f}ms ({gb / d:5.0f} GB/s)", flush=True)
    print(f"  chain decode projection: quant {1 / q:5.1f} tok/s  "
          f"dequant+mm {1 / d:5.1f} tok/s", flush=True)

    # custom 4-bit GEMV kernel vs MLX for the same 36-layer chain
    from experiments.quant_gemv_metal import gemv_quantized

    def run_gemv() -> float:
        outs = []
        for layer_w in weights:
            for (_name, _o, in_dim, _c), wq in zip(SHAPES, layer_w):
                x = inputs[in_dim]
                outs.append(gemv_quantized(x, *wq))
        mx.eval(*outs)
        return time.perf_counter() - t0

    t0 = time.perf_counter()
    run_gemv()
    g = min(run_gemv() for _ in range(repeats))
    print(f"  custom GEMV chain: {g * 1000:7.2f}ms ({gb / g:5.0f} GB/s)  "
          f"speedup vs MLX = {q / g:.2f}x  decode projection {1 / g:5.1f} tok/s",
          flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
