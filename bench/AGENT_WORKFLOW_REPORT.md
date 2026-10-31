# Local code-review agent: first serial, token-capped evaluation

See `AGENT_ACCEPTANCE_PROTOCOL.md` for the pre-registered scenario and thresholds.
Raw recordings and summaries are in `bench/agent_workloads/` and
`bench/agent_workflow_summary.json`.

## What this run actually measures

- The workload is **serial**: one task and one HTTP request at a time; normally
  one outstanding request. It does not validate concurrent admission or batching.
- Every model response is capped at **96 tokens** for the agent run. The stored
  recordings show `finish_reason=length` for the final/follow-up answers, so these
  are **limited-length workflow times**, not full review completion times.
- `keyword_pass` only checks that rubric words appear in the text. It does not
  prove the tool was used correctly, the review is correct, or the answer is
  complete.
- `complete_review` requires: expected file read successfully, no tool error,
  both final and follow-up stop naturally, and non-empty text. In the stored
  runs **complete_review is 0/3 and 0/2** because responses hit the length cap.
- `peak_rss_mb` is host RSS sampling. MLX peak memory was identical for p4 and
  cap1 (4514.2 MB), so lower RSS does not establish lower unified-memory pressure.

## Results (one run/config, loadavg 3.4–6.2, 96-token cap)

Tuning tasks (3):

| Config | wall (limited) | keyword pass | complete review | truncated | RSS |
|---|---:|---:|---:|---:|---:|
| Terrazzo cache off | 228.5 s | 3/3 | 0/3 | 3/3 | 2470 MB |
| Terrazzo cache on | 149.3 s | 3/3 | 0/3 | 3/3 | 2472 MB |
| Terrazzo cache on p4 | **112.9 s** | 3/3 | 0/3 | 3/3 | 2470 MB |
| Terrazzo cap1 | 169.8 s | 3/3 | 0/3 | 3/3 | **1489 MB** |
| native mlx-lm | 116.2 s | 3/3 | 0/3 | 3/3 | 2526 MB |

Holdout tasks (2):

| Config | wall (limited) | keyword pass | complete review | truncated |
|---|---:|---:|---:|---:|
| Terrazzo cache off | 239.9 s | 2/2 | 0/2 | 2/2 |
| Terrazzo cache on | 167.3 s | 2/2 | 0/2 | 2/2 |
| Terrazzo cache on p4 | 127.5 s | 2/2 | 0/2 | 2/2 |
| native mlx-lm | 105.2 s | 2/2 | 0/2 | 2/2 |

Fixed replay:

| Config | wall | cache hits |
|---|---:|---:|
| Terrazzo cache off | 272.6 s | 0 |
| Terrazzo cache on | 197.2 s | 18 (18432 hit tokens) |
| native mlx-lm | 52.5 s | — |

Cache-on versus cache-off is directionally consistent across the three
non-independent workloads (-34.7% tuning, -30.3% holdout, -27.7% replay), but the
magnitude is single-run and the workload is truncated.

## Decision

**No sufficient switch reason versus native mlx-lm in this workflow.**

- Terrazzo cache-on is faster than cache-off in this run.
- p4 (cache on, cap16, prefill batch 4 / step 512) is 112.9 s vs native 116.2 s
  on tuning (-2.8%) but 127.5 s vs 105.2 s on holdout (+21.2%). A 2.8% single-run
  difference is not evidence of a stable win; the holdout direction is opposite.
- This does not meet the pre-registered 15% task-time or 20% memory threshold.
- Cap1's RSS is lower (1489 MB), but MLX peak is 4514.2 MB, same as p4. It is not
  evidence that cap1 reduces actual unified-memory pressure or enables smaller
  machines. Time is also 46% worse than native in this run.
- Ollama's OpenAI-compatible endpoint did not emit tool calls; its native tool
  path was not completed. It is excluded from the quality claim.
- Therefore: pause performance feature expansion. Do not report a Terrazzo win.

## Recommendation

For a latency-first local code-review agent, native mlx-lm remains the safer
choice. If Terrazzo is already deployed, enable prefix cache + prewarm and test
the p4 parameters below; treat cap1 only as a to-be-validated memory experiment.

```bash
python -m terrazzo serve --model <qwen3-4b-4bit> --profile agent --no-thinking \
  --prefix-reuse always --scheduler-policy fifo \
  --max-num-seqs 16 --completion-batch-size 16 \
  --prefill-batch-size 4 --prefill-step-size 512
curl -X POST http://127.0.0.1:PORT/v1/prewarm \
  -H 'Content-Type: application/json' -d '{"system":"<agent system prompt>"}'

python bench/agent_review_workflow.py --base http://127.0.0.1:PORT \
  --label cache-on --task-set tune --stats http://127.0.0.1:PORT/stats

python bench/agent_review_workflow.py --base http://127.0.0.1:PORT \
  --mode replay --recording bench/agent_workloads/tune_on.json --label replay
```

## Reliability / cold start (observed, not fully accepted)

`bench/agent_reliability.py` observed:

| Check | Result |
|---|---|
| server startup to `/health` | 3.04 s |
| prewarm | 7.85 s (2064 tokens) |
| abort + recovery | aborted=true; next request stop |
| malformed JSON | 400; next request stop |
| prefix eviction under 400 MB | evictions observed; active=0 |
| final counters | running=0, waiting=0; added=5, finished=5, aborted=1, failed=0 |

This covers **input-error handling**, not `runner.step()` exception/watchdog
executor recovery. The stored result did not record final `prefix_cache.active`,
so the old automatic `ok=true` is not sufficient; the reclassified result is
`ok=false` pending a proper reliability run.

Cold start before steady state is about 10.9 s (3.04 s server + 7.85 s prewarm).
Model-load time is not separated from this.

## Open / required next

1. Raise the answer budget or use a stopping criterion that produces complete
   reviews; measure **correct/complete review rate**, not keyword hits.
2. Add concurrent agent sessions if admission/batching claims are desired; the
   current serial workload cannot attribute p4 differences to batch capacity.
3. Add a real executor failure (`runner.step` exception / watchdog) recovery test.
4. Repeat configs on an idle machine; do not claim statistical significance from a
   single-run 2.8% difference.
5. Complete Ollama native tool calling or record it as not comparable.
