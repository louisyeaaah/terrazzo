# Known-defect code review: single verified task

Goal: one task with a known defect, enough output budget, then manual confirmation
of defect / trigger / fix. This is not a performance comparison.

## Fixture and known defect

`bench/agent_fixtures/refcount_leak.py`:

```python
self.refcounts[block] -= 1
if self.refcounts[block] == 1:   # BUG: should be == 0
    self.free_blocks.append(block)
```

Expected:
- defect: last reference never returns the block to `free_blocks`;
- trigger: `acquire()` then `release()` when refcount becomes zero;
- fix: change `== 1` to `== 0` (and append to `free_blocks`).

## Run

- Config: Terrazzo cache on + prewarm, default agent profile; this is not a tuned
  performance run.
- Output budget: `--max-tokens 512`; no truncation.
- Tool: `read_file("bench/agent_fixtures/refcount_leak.py")`, success.
- Finish reasons: final `stop`, follow-up `stop`.
- `complete_review=true`, `keyword_pass=true`, 362 completion tokens, 10.54 s.
- Full request/response record: `bench/agent_golden_review.json`.

## Leaked-prompt run (flow only)

The original fixture contained comments naming the defect and fix; those comments
were returned to the model. That run proved flow/tool reading and a correct patch
suggestion, but **not independent discovery**.

## Neutral rerun (answer leak removed)

- `bench/agent_fixtures/block_pool_review.py`: neutral comments/name; expected
  defect/trigger/fix moved to `bench/agent_fixtures/block_pool_eval.json`, not sent
  to the model.
- Qwen3-4B run: tool read succeeded, both turns stopped, no truncation, but the
  answer was **incorrect**: it claimed duplicate addition to `free_blocks` instead
  of the `==1` vs `==0` leak. `keyword_pass=false`, `complete_review=true`.
- Qwen3.5 4B attempt: no tool call and follow-up hit the length cap; rejected.
- These raw answers are preserved; they are not rewritten.

## Manual review checklist for the leaked-prompt run (sign-off pending)

| Item | Result | Evidence |
|---|---|---|
| Defect | correct | Identifies `refcount == 1` condition and the leak when count reaches zero |
| Trigger | correct | Acquire then release the last reference; block not returned |
| Fix | correct | Change to `refcount == 0` and append to `free_blocks` |
| Completeness | pass | First stop, follow-up stop, no pending tool calls, no truncation |
| Wording | minor issue | Says block “stays in free_blocks”; actually it is removed on acquire and never returned |

**User review (not a sign-off):** process acceptance passes; core diagnosis/fix
are correct for the leaked-prompt run, but the explanation has the state wording
error. Independent discovery is not evaluated because the answer leaked. The user
explicitly did not provide human sign-off; `accepted=false` remains.

## Preserved failure/truncation records

The earlier 96-token runs remain in `bench/agent_workloads/` and are reclassified:
`keyword_pass` 3/3 or 2/2, `complete_review=0`, `truncated=all` for tune/holdout.
The leaked-prompt run completed without truncation; the neutral rerun also
`complete_review=true` but its diagnosis is wrong. Neither is accepted.

## Next

- Do not expand the task set or rerun performance configs.
- Either try a general review prompt (no answer hints) on another local model, or
  record this as a failed independent-discovery case.
- Human sign-off remains the user's decision.
