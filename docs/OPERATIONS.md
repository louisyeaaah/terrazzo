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

## Limits and timeouts

Defaults are deliberately conservative; override with environment variables:

| Variable | Default | Behaviour |
|---|---|---|
| `TERRAZZO_REQUEST_TIMEOUT_S` | 600 | per-request deadline (queue + generation). Non-stream: `504`; stream: `finish_reason="error"` then close |
| `TERRAZZO_MAX_BODY_MB` | 1 | request body cap; larger requests get `413` before JSON parsing |
| `TERRAZZO_MAX_CONCURRENT` | 64 | HTTP worker cap; excess connections get an immediate `503` + `Retry-After: 1` |
| `TERRAZZO_MAX_PENDING` | 256 | inference queue cap; excess submissions get `503` |
| `TERRAZZO_MEMORY_CAP_MB` | 70% RAM | memory guard (existing) |

Auth is checked before the request body is read. A request that hits the deadline is
aborted in the engine so it does not keep occupying KV. After three consecutive
`step()` failures the runtime refuses new work (`503`) until the server is restarted;
watch the server log for `[engine] step failed` lines.

## Cancellation

- `POST /v1/abort {"request_id": "..."}` — clients may pass their own `request_id`
  in the completion body.
- A client that disconnects mid-stream aborts its request automatically (KV freed).

## Known gaps / residual risks

These are deliberate limits of the current hardening, not guesses:

- **No GPU-level preemption.** A hung Metal kernel can keep the MLX worker thread
  blocked; HTTP requests time out (`504`) and the pending queue eventually fills
  (`503`), but the process may need a restart. `handle_failure()` only runs when
  `step()` raises, not when it silently hangs.
- **Paged/Pool runtimes are less hardened.** They terminate per-request exceptions
  and handlers enforce request deadlines, but they do not have the
  `Engine.handle_failure()` + unhealthy-after-3-failures path.
- **Concurrency slots are HTTP threads.** A slow client holding a streaming response
  occupies a slot until completion; there is no write timeout (only a 30 s read
  timeout).
- **No TLS and no constant-time API-key comparison.** Put it behind a reverse proxy
  if it leaves localhost; the API key is a bearer token, not an auth system.
- **CI does not run model E2E or the Metal kernel experiments.** The macOS job covers
  import + core suite + selected MLX unit tests; the `experiments/` kernels and
  `TERRAZZO_MAC_MODEL` E2E tests need real hardware/model access.
- **Benchmark cross-backend token counts are not strictly comparable.** Terrazzo now
  reports true `usage` tokens; backends that do not support
  `stream_options.include_usage` fall back to SSE content-event counting and are
  marked `token_source="chunks(approx)"`.
- **mlx-lm patch contract is version+attribute based**, not behavioural: a patch-level
  release inside the pinned range could still change cache semantics. `constraints.txt`
  pins the exact tested versions.

## Soak / acceptance

```bash
python bench/soak.py --base http://127.0.0.1:8600 --model terrazzo \
    --sessions 4 --duration 3600 --max-tokens 48 --system-file system.txt
```

Pass criteria used here: `ok == requests`, `error_rate == 0`, `any_over_memory == false`,
peak MLX memory stable across the run, `stats_errors == 0`.
