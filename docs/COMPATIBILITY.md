# Compatibility

Terrazzo targets **macOS on Apple Silicon** and depends on MLX / `mlx-lm`. A few
runtime performance patches touch `mlx-lm` cache classes (not public API), so the
dependency range is intentionally narrow.

## Tested matrix

| Component | Version | Notes |
|---|---|---|
| macOS | 14 / 15 (Apple Silicon) | M1 Max 64 GB used for benchmarks |
| Python | 3.10 | `requires-python = ">=3.10"` |
| `mlx` | **0.32.4** | pinned in `pyproject.toml` / `constraints.txt` |
| `mlx-lm` | **0.31.3** | pinned; runtime patches verified against this |
| Model | `mlx-community/Qwen3-4B-4bit` | Qwen3 chat template (tools/thinking) |

Install exactly:

```bash
python -m pip install -e . -c constraints.txt
```

## Runtime patches and the safety net

`batch_attn_patch.py`, `mask_patch.py`, `multi_query_patch.py` and
`contig_kv_patch.py` modify `mlx_lm.models.cache` behaviour. On engine start,
`terrazzo.compat.require_tested_mlx_lm()` checks:

1. the installed `mlx-lm` version is in `>=0.31.3,<0.32`;
2. the patched attributes still exist (`BatchKVCache.prepare/finalize/update_and_fetch/filter`,
   `KVCache.update_and_fetch`).

If either check fails, the engine refuses to start with a message naming the
mismatch and the tested version. The escape hatch
`TERRAZZO_SKIP_MLX_LM_CHECK=1` bypasses the check — only for testing a new
mlx-lm release, never for production.

## Benchmarks

Every benchmark run records its environment in the output JSON: host, Python,
git commit, `terrazzo` version, backend version, load average, and `/stats`
snapshots (`bench/agent_workload.py`). Absolute numbers are only comparable
within the matrix above; external machine load is also recorded because it
materially changes TTFT on a laptop.

## Not supported

- Linux / Windows hosts (MLX requires Apple Silicon).
- CUDA / multi-GPU / distributed inference.
- Ollama registry or GGUF `pull` (use Hugging Face MLX snapshots).
