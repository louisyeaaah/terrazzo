# Agent workflow acceptance protocol (pre-registered)

Status: first serial, 96-token-capped evaluation completed. Quality/complete-review,
concurrent workload and executor-failure recovery are **not accepted**. Repeated
idle runs and Ollama native tool calling remain open.

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
3. Terrazzo p4: cache on, `--max-num-seqs 16 --completion-batch-size 16
   --prefill-batch-size 4 --prefill-step-size 512`.
4. Terrazzo cap1: cache on, `--max-num-seqs 1 --completion-batch-size 1
   --prefill-batch-size 1 --prefill-step-size 512`.
5. Native mlx-lm server, same model.
6. Ollama `qwen3:4b` (different runtime/quant; OpenAI-compatible endpoint did not
   emit tool calls, so it is excluded from quality comparison).

Only existing Terrazzo parameters are changed. Prewarm/model-load time is
reported separately from task timing.

## Metrics and thresholds

Primary: limited-length task wall time, `complete_review` rate, peak memory (RSS
and `/stats.mlx_peak_mb` where available). Secondary: per-request latency, tokens.
`keyword_pass` is only a rubric word check and must not be called task success.

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
- cancellation: observed (abort+recovery); automatic acceptance still needs final
  active/running/waiting assertions.
- cache eviction: observed (4 distinct prewarms under 400 MB; evictions observed).
- failure recovery: **not covered**: malformed JSON is input validation, not a
  `runner.step()`/watchdog executor failure.
- cold start vs steady-state: Terrazzo startup 3.04 s + prewarm 7.85 s; task timings
  are steady-state. Native baseline cold start not measured.
- concurrency: **not covered**; the agent loop is serial, so no admission/batch
  capacity claim is possible.
