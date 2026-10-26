# experiments

One-off kernels and probes kept for reproducibility of the negative results
documented in `../BENCHMARKS.md`. **Nothing here is part of the `terrazzo`
package** and nothing is imported by the engine or server.

Moved here during the 0.3 cleanup because it is not product code:

| module | what it explores | verdict |
|---|---|---|
| `mma_flash_attn.py` | simdgroup MMA flash attention | correct, ~10x slower than SDPA for decode shapes |
| `paged_batch_metal.py`, `paged_gqa_metal.py`, `paged_attn_patch.py` | paged/batched Metal attention | paged decode path loses to dense SDPA |
| `paged_attention.py` | block-gather reference implementation | used as the correctness oracle in tests |
| `kv_quant.py` | KV cache quantization probe | memory wins, throughput loses |
| `quant_gemv_metal.py` | quantized GEMV Metal kernel | no win over MLX matmul |
| `decode_fused.py` | fused decode step | no win |
| `dense_gqa_multi_metal.py` | multi-token GQA Metal kernel | no win |

Tests live in `tests/` and benchmarks in `bench/`:

```bash
python -m unittest discover -s experiments/tests -v
python experiments/experiments/bench/kv_quant.py --model mlx-community/Qwen3-4B-4bit
```
