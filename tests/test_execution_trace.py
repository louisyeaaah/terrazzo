import time
from collections import deque
import unittest

try:
    from terrazzo.engine import Engine  # imports MLX-backed paged_kv
except Exception as _exc:  # pragma: no cover - pure-Python CI
    Engine = None
    _IMPORT_ERROR = _exc

from terrazzo.scheduler import ScheduledChunk, SchedulerOutputs
from terrazzo.sequence import Sequence, SamplingParams


@unittest.skipIf(Engine is None, "engine requires MLX")
class TraceTest(unittest.TestCase):
    def test_record_exec_trace_tracks_mismatch(self):
        e = Engine.__new__(Engine)
        e.prefix_cache = None
        e.prefix_hit_tokens = 0
        e.prefix_recompute_skips = 0
        e._exec_trace = deque(maxlen=8)
        e._first_prefill_steps = []
        e._control_totals = {
            "steps": 0, "scheduled_prefill_tokens": 0, "actual_prompt_tokens": 0,
            "scheduled_decode_tokens": 0, "actual_decode_tokens": 0,
            "scheduled_prefill_requests": 0, "inserted_prefill_requests": 0,
            "steps_with_prefill": 0, "steps_with_mismatch": 0,
        }
        e.num_steps = 1
        e._step_prompt_tokens = 64
        e._step_decode_tokens = 0
        seq = Sequence(request_id="r1", prompt_token_ids=list(range(32)),
                       sampling_params=SamplingParams(max_tokens=4))
        outs = SchedulerOutputs(chunks=[ScheduledChunk(seq, 128, True)],
                                num_prefill_tokens=128)
        e._record_exec_trace(outs, ["r1"], {"hits": 0, "stores": 0, "skips": 0,
                                            "hit_tokens": 0}, 0.01)
        self.assertEqual(e._control_totals["steps"], 1)
        self.assertEqual(e._control_totals["steps_with_mismatch"], 1)
        self.assertEqual(e._first_prefill_steps[0]["scheduled_ids"], ["r1"])
        self.assertEqual(e._first_prefill_steps[0]["actual_prompt_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
