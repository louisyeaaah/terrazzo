# terrazzo operations runbook

## Start

```bash
# deterministic prefix + auto KV blocks (35% of RAM for KV)
terrazzo serve --model <mlx-model-dir> --port 8600 --num-gpu-blocks 0 \
    --system-file system.txt --max-num-seqs 8

# with auth and rate limiting
terrazzo serve ... --api-key "$KEY" --rate-limit 5

# supervised (restart on crash)
scripts/serve_watchdog.sh --max-restarts 10 -- terrazzo serve ...
```

`num_gpu_blocks=0` prints the computed value at startup (`auto num_gpu_blocks=...`).
`--system-file` registers the shared prefix at startup (best cold TTFT); without it
the server auto-registers the prefix shared by the first batch (costs one prefill,
~9 s for 2.5k tokens) and then behaves like the pre-registered mode.

## Health and metrics

| Endpoint | Use |
|---|---|
| `GET /health` | liveness (no auth) |
| `GET /stats` | scheduler + `memory_guard` + `prefix_cache` + `auto_prefix` |
| `GET /v1/models` | model + predicted tok/s + capability |

Watch these `/stats` fields:

- `memory_guard.over_memory` — server is shedding load; expect 503s until it clears
- `memory_guard.rejections` / `evictions` — pressure happened
- `prefix_cache.hit_rate`, `entries`, `nbytes` — prefix caching working / growing
- `preemptions` — KV blocks too small for the offered concurrency (raise blocks)
- `mean_step_ms` — decode step cost; rising means throughput is degrading

## Load shedding / memory

- Default cap: 70% of physical RAM (`--memory-cap-mb` or `TERRAZZO_MEMORY_CAP_MB`).
- Over cap: MLX cache + prefix snapshots are dropped; new requests get
  `503` + `Retry-After: 5`; recovery is automatic.
- If `preemptions` grows quickly, either lower `--max-num-seqs` or raise
  `--num-gpu-blocks` (or use `0` to derive from RAM).

## Cancellation

- `POST /v1/abort {"request_id": "..."}` — clients may pass their own `request_id`
  in the completion body.
- A client that disconnects mid-stream aborts its request automatically (KV freed).

## Soak / acceptance

```bash
python bench/soak.py --base http://127.0.0.1:8600 --model terrazzo \
    --sessions 4 --duration 3600 --max-tokens 48 --system-file system.txt
```

Pass criteria used here: `ok == requests`, `error_rate == 0`, `any_over_memory == false`,
peak MLX memory stable across the run, `stats_errors == 0`.
