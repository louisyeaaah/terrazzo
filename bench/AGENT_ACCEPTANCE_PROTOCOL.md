# Agent workflow acceptance protocol (pre-registered)

Status: first run completed; reliability/cold-start follow-ups still open.

## User and scenario

An Apple Silicon developer running a local multi-turn code-review agent. Multiple
reviews share the same repository rules, tool definitions and project background;
tasks contain tool returns, a follow-up, and different input lengths.

## Fixed environment

- Machine: M1 Max, 10 CPU cores, macOS 26.6.2 arm64.
- Model: Qwen3-4B-4bit (MLX), same local snapshot for Terrazzo and mlx-lm.
- Task set: 3 tuning tasks (`refcount`, `cache`, `mask`) and 2 holdout tasks
  (`scheduler`, `request`). Holdout is not used for configuration selection.
- Workloads: full agent execution and fixed replay from the same recording.
- Background load: loadavg 3.4–6.2; single run per config. Numbers are
  directional until repeated on an idle machine.

## Configurations

1. Terrazzo cache off: `--no-prefix-cache --scheduler-policy fifo --profile agent`.
2. Terrazzo cache on: same + `/v1/prewarm` of the exact shared system prompt.
3. Terrazzo cap1: cache on, `--max-num-seqs 1 --completion-batch-size 1
   --prefill-batch-size 1 --prefill-step-size 512`.
4. Native mlx-lm server, same model.
5. Ollama `qwen3:4b` (different runtime/quant; OpenAI-compatible endpoint did not
   emit tool calls, so it is excluded from quality comparison).

Only existing Terrazzo parameters are changed. Prewarm/model-load time is
reported separately from task timing.

## Metrics and thresholds

Primary: task completion wall time, task success rate, peak memory (RSS and
`/stats.mlx_peak_mb` where available). Secondary: per-request latency, tokens.

A qualifying switch reason requires, on holdout tasks:

- a stable direction across repeats, and
- about 15% lower task completion time **or** about 20% lower peak memory versus
  the strongest baseline, with no quality regression and no unacceptable
  degradation in the other resource.

If the tuned baseline still wins, record "no sufficient switch reason for this
scenario" and stop expanding performance features.

## Required coverage

- multi-turn agent execution: yes (tasks above).
- fixed replay: yes.
- cancellation: not yet covered.
- cache eviction: not yet covered.
- failure recovery: not yet covered.
- cold start vs steady-state: prewarm 8.1s observed; full cold-start matrix not yet.
