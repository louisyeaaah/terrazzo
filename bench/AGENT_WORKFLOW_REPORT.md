# Local code-review agent: first acceptance result

See `AGENT_ACCEPTANCE_PROTOCOL.md` for the pre-registered scenario and thresholds.
Raw recordings and summaries are in `bench/agent_workloads/` and
`bench/agent_workflow_summary.json`.

## Workflow

- Shared system prompt: ~3.5 KB (~1.7k tokens) of repository review rules.
- Tool: `read_file(path)`; the model must call it before reviewing.
- Each task: tool call(s), final review, then a follow-up answer. The follow-up
  history contains the real first answer.
- Recorded requests include full messages, tool calls/results, token usage,
  finish reason and latency. Fixed replay re-sends the recorded requests without
  executing tools.

## Results (one run per config, high background load)

Full agent, tuning tasks (3 tasks):

| Config | wall | success | peak RSS | cache hits | vs native |
|---|---:|---:|---:|---:|---:|
| Terrazzo cache off | 228.5 s | 3/3 | 2470 MB | 0 | +96.6% time |
| Terrazzo cache on | 149.3 s | 3/3 | 2472 MB | 18 | +28.5% time |
| Terrazzo cap1 | 169.8 s | 3/3 | **1489 MB** | 18 | +46.1% time |
| native mlx-lm | **116.2 s** | 3/3 | 2526 MB | — | baseline |

Full agent, holdout tasks (2 tasks):

| Config | wall | success | peak RSS |
|---|---:|---:|---:|
| Terrazzo cache off | 239.9 s | 2/2 | 2466 MB |
| Terrazzo cache on | 167.3 s | 2/2 | 2466 MB |
| native mlx-lm | **105.2 s** | 2/2 | 2510 MB |

Fixed replay:

| Config | wall | cache hits |
|---|---:|---:|
| Terrazzo cache off | 272.6 s | 0 |
| Terrazzo cache on | 197.2 s | 18 (18432 hit tokens) |
| native mlx-lm | **52.5 s** | — |

Cache on vs cache off is a real, repeatable-direction benefit: -34.7% tuning,
-30.3% holdout, -27.7% replay, with the same 3/3 and 2/2 rubric success.

## Decision

**No sufficient switch reason versus native mlx-lm in this agent workflow.**

- Terrazzo cache-on is clearly better than Terrazzo cache-off, so the cache is
  useful inside Terrazzo.
- Native mlx-lm is materially faster (tune 116s vs 149s; holdout 105s vs 167s;
  replay 52s vs 197s) with the same rubric success.
- Cap1 lowers peak RSS by ~41% vs native, but task time is ~46% worse; that is a
  memory-constrained tradeoff, not a default recommendation.
- Ollama's OpenAI-compatible endpoint did not call tools and scored 0/3; its
  native tool path was not completed in this round, so it is not used for a
  quality claim.

## Recommendation

For this scenario, use **native mlx-lm** for latency-first local code review.
If Terrazzo is already deployed, enable prefix cache + prewarm (`cache on`);
choose cap1 only when memory is the binding constraint and the latency tradeoff is
acceptable. Do not expand performance features until a different workflow shows
a clear gap.

## Startup / replay

```bash
# Terrazzo cache-on (steady-state; prewarm separately)
python -m terrazzo serve --model <qwen3-4b-4bit> --profile agent --no-thinking \
  --prefix-reuse always --scheduler-policy fifo
curl -X POST http://127.0.0.1:PORT/v1/prewarm \
  -H 'Content-Type: application/json' -d '{"system":"<agent system prompt>"}'

# Full agent (tune or holdout)
python bench/agent_review_workflow.py --base http://127.0.0.1:PORT \
  --label cache-on --task-set tune --stats http://127.0.0.1:PORT/stats

# Fixed replay from a recording
python bench/agent_review_workflow.py --base http://127.0.0.1:PORT \
  --mode replay --recording bench/agent_workloads/tune_on.json --label replay
```

## Open work

- Repeat runs on an idle machine; current loadavg 3.4–6.2.
- Add cancellation, cache-eviction and failure-recovery coverage.
- Measure cold start with model load separated from prewarm.
- Optionally complete Ollama native tool calling for a second cross-engine point.
