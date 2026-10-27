import threading
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


@unittest.skipIf(Engine is None, "engine requires MLX")
class LifecycleTest(unittest.TestCase):
    def _engine(self):
        e = Engine.__new__(Engine)
        e._lock = threading.RLock()
        e._stream_lock = threading.RLock()
        e._streams = {}
        e._outputs = {}
        e._inserted = {}
        e.request_stages = {"added": 0, "first_token": 0, "finished": 0,
                            "aborted": 0, "failed": 0}
        return e

    def test_fail_streams_does_not_block_on_execution_lock(self):
        import queue as _queue
        import threading
        e = self._engine()
        q = _queue.Queue()
        e._streams["r1"] = q
        e._outputs["r1"] = object()
        result = {}
        started = threading.Event()
        def run():
            started.set()
            result["n"] = e.fail_streams(RuntimeError("metal hung"))
        # simulate the worker thread holding _lock inside runner.step()
        with e._lock:
            t = threading.Thread(target=run)
            t.start()
            self.assertTrue(started.wait(2))
            t.join(timeout=2)
            self.assertFalse(t.is_alive(), "fail_streams blocked on the execution lock")
        self.assertEqual(result.get("n"), 1)
        event = q.get(timeout=1)
        self.assertEqual(event["finish_reason"], "error")
        self.assertEqual(e._streams, {})
        self.assertEqual(e._outputs, {})

    def test_collect_finished_releases_request_maps(self):
        import queue as _queue
        e = self._engine()
        seq = Sequence(request_id="done", prompt_token_ids=list(range(8)),
                       sampling_params=SamplingParams(max_tokens=4))
        seq.finish("stop")
        q = _queue.Queue()
        e._streams["done"] = q
        e._outputs["done"] = type("Out", (), {})()
        class FakeScheduler:
            def drain_finished(self):
                return [seq]
        e.scheduler = FakeScheduler()
        e._collect_finished()
        self.assertEqual(e._streams, {})
        self.assertEqual(e._outputs, {})
        final = q.get(timeout=1)
        self.assertTrue(final.finished)
        self.assertEqual(final.finish_reason, "stop")


    def test_abort_removes_executor_request_and_releases_streams(self):
        import queue as _queue
        e = self._engine()
        uid = 7
        e._inserted = {"r-abort": uid}
        removed = []
        class FakeRunner:
            uid_to_seq = {uid: object()}
            def remove(self, uids):
                removed.extend(uids)
        e.runner = FakeRunner()
        class FakeScheduler:
            def abort(self, rid):
                return True
        e.scheduler = FakeScheduler()
        q = _queue.Queue()
        e._streams["r-abort"] = q
        e._outputs["r-abort"] = object()
        self.assertTrue(e.abort("r-abort"))
        self.assertEqual(removed, [uid])
        self.assertNotIn(uid, e.runner.uid_to_seq)
        self.assertEqual(e._streams, {})
        self.assertEqual(e._outputs, {})
        self.assertEqual(q.get(timeout=1)["finish_reason"], "abort")

    def test_worker_continues_after_watchdog_released_outputs(self):
        from types import SimpleNamespace
        e = self._engine()
        e._inserted = {"r1": 7}
        seq = Sequence(request_id="r1", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        class FakeRunner:
            uid_to_seq = {7: seq}
            def remove(self, uids):
                pass
        e.runner = FakeRunner()
        class FakeScheduler:
            finished = []
            def finish(self, s):
                self.finished.append(s)
        e.scheduler = FakeScheduler()
        e.num_decode_tokens = 0
        e._step_decode_tokens = 0
        # fail_streams removed the output/stream, but the executor can still emit
        # one more response before it is drained; this must not raise.
        e.fail_streams(RuntimeError("hung"))
        out = e._process_generation_responses(
            [SimpleNamespace(uid=7, token=3, finish_reason=None, prompt_cache=None)])
        self.assertEqual(seq.output_token_ids, [3])
        self.assertEqual(out, [])
