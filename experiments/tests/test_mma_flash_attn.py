"""MMA flash-attention kernel: numerics vs SDPA, fp16/bf16, padding, lse."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx  # noqa: E402

from experiments.mma_flash_attn import mma_flash_attention  # noqa: E402

D = 128


def sdpa_ref(q, k, v, causal):
    hq, hk = q.shape[0], k.shape[0]
    rep = hq // hk
    krep = mx.repeat(k, rep, axis=0)
    vrep = mx.repeat(v, rep, axis=0)
    return mx.fast.scaled_dot_product_attention(
        q[None], krep[None], vrep[None], scale=D ** -0.5,
        mask="causal" if causal else None)[0]


def fp32_ref(q, kt, v, causal):
    """[Hq,L,D] x [Hk,D,KV] x [Hk,KV,D] -> out + lse, fp32."""
    hq, length, _ = q.shape
    hk = kt.shape[0]
    rep = hq // hk
    outs, lses = [], []
    for h in range(hq):
        kh = h // rep
        scores = mx.matmul(q[h].astype(mx.float32), kt[kh].astype(mx.float32)) * (D ** -0.5)
        kv = scores.shape[1]
        if causal:
            mask = mx.arange(length)[:, None] <= (mx.arange(kv)[None, :] + (kv - length))
            scores = mx.where(mask, scores, -mx.inf)
        lses.append(mx.logsumexp(scores, axis=-1))
        probs = mx.softmax(scores, axis=-1)
        outs.append(mx.matmul(probs, v[kh].astype(mx.float32)))
    return mx.stack(outs), mx.stack(lses)


class MmaFlashAttnTest(unittest.TestCase):
    def _check(self, hq, hk, length, kv, causal, dtype=mx.float16, atol=2e-2,
               check_lse=False):
        mx.random.seed(0)
        q = mx.random.normal((hq, length, D)).astype(dtype)
        k = mx.random.normal((hk, kv, D)).astype(dtype)
        v = mx.random.normal((hk, kv, D)).astype(dtype)
        kt = mx.contiguous(mx.swapaxes(k, 1, 2))
        if check_lse:
            out, lse = mma_flash_attention(q, kt, v, causal=causal, return_lse=True)
            ref, ref_lse = fp32_ref(q, kt, v, causal)
            mx.eval(out, lse, ref, ref_lse)
            self.assertLess(float(mx.max(mx.abs(lse - ref_lse))), 5e-3)
        else:
            out = mma_flash_attention(q, kt, v, causal=causal)
            ref = sdpa_ref(q, k, v, causal)
            mx.eval(out, ref)
        diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        self.assertLess(diff, atol, f"hq={hq} l={length} kv={kv} {dtype} diff={diff}")

    def test_causal_and_noncausal(self):
        for length in (64, 128):
            for causal in (False, True):
                self._check(hq=2, hk=1, length=length, kv=length, causal=causal)

    def test_gqa_mapping(self):
        self._check(hq=8, hk=2, length=64, kv=64, causal=True)

    def test_bf16(self):
        self._check(hq=2, hk=1, length=64, kv=64, causal=True, dtype=mx.bfloat16)

    def test_padded_query_length(self):
        # L smaller than the 8-row MMA tile must still be exact
        self._check(hq=4, hk=1, length=4, kv=256, causal=False, dtype=mx.bfloat16)

    def test_lse(self):
        self._check(hq=2, hk=1, length=8, kv=256, causal=False,
                    dtype=mx.bfloat16, check_lse=True)


if __name__ == "__main__":
    unittest.main()
