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

## Manual review checklist (agent-checked; user sign-off pending)

| Item | Result | Evidence |
|---|---|---|
| Defect | correct | Identifies `refcount == 1` condition and the leak when count reaches zero |
| Trigger | correct | Acquire then release the last reference; block not returned |
| Fix | correct | Change to `refcount == 0` and append to `free_blocks` |
| Completeness | pass | First stop, follow-up stop, no pending tool calls, no truncation |
| Wording | minor issue | Says block “stays in free_blocks”; actually it is removed on acquire and never returned |

**Human sign-off: confirmed (2026-10-31).** The user confirmed defect, trigger
and fix are all correct; `accepted=true`. The minor wording issue does not change
the diagnosis or fix, and is retained in the record.

## Preserved failure/truncation records

The earlier 96-token runs remain in `bench/agent_workloads/` and are reclassified:
`keyword_pass` 3/3 or 2/2, `complete_review=0`, `truncated=all` for tune/holdout.
This golden run is the only non-truncated complete review so far.

## Next

- Expand the known-defect task set only after this single-task acceptance.
- Do not add performance configs or rerun the comparison matrix yet.
