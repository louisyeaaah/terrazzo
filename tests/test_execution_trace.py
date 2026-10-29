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
        seq = Sequence(request_id="r-abort", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        e._inserted = {"r-abort": uid}
        removed = []
        class FakeRunner:
            uid_to_seq = {uid: seq}
            def remove(self, seqs):
                removed.extend(seqs)
                for u, s in list(self.uid_to_seq.items()):
                    if s in seqs:
                        self.uid_to_seq.pop(u, None)
        e.runner = FakeRunner()
        class FakeScheduler:
            waiting = []
            running = [seq]
            def abort(self, rid):
                return True
        e.scheduler = FakeScheduler()
        q = _queue.Queue()
        e._streams["r-abort"] = q
        e._outputs["r-abort"] = object()
        self.assertTrue(e.abort("r-abort"))
        self.assertEqual(removed, [seq])
        self.assertNotIn(uid, e.runner.uid_to_seq)
        self.assertEqual(e._streams, {})
        self.assertEqual(e._outputs, {})
        self.assertEqual(q.get(timeout=1)["finish_reason"], "abort")

    def test_runner_remove_resolves_sequence_to_uid(self):
        from terrazzo.runner import MLXRunner
        seq = Sequence(request_id="x", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        runner = MLXRunner.__new__(MLXRunner)
        runner.uid_to_seq = {7: seq}
        class FakeBG:
            calls = []
            def remove(self, uids, return_prompt_caches=False):
                self.calls.append(list(uids))
                return {}
        runner.bg = FakeBG()
        runner.remove([seq])
        self.assertEqual(runner.bg.calls, [[7]])
        self.assertEqual(runner.uid_to_seq, {})

    def test_handle_failure_resets_executor_state(self):
        import queue as _queue
        e = self._engine()
        e._executor_broken = False
        seq = Sequence(request_id="r1", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        e._inserted = {"r1": 7}
        reset = {"called": False}
        class FakeRunner:
            uid_to_seq = {7: seq}
            def reset(self):
                reset["called"] = True
                self.uid_to_seq.clear()
        e.runner = FakeRunner()
        aborted = []
        class FakeScheduler:
            waiting = [seq]
            running = []
            def abort(self, rid):
                aborted.append(rid)
                return True
        e.scheduler = FakeScheduler()
        q = _queue.Queue()
        e._streams["r1"] = q
        e._outputs["r1"] = object()
        n = e.handle_failure(RuntimeError("metal boom"))
        self.assertEqual(n, 1)
        self.assertTrue(reset["called"])
        self.assertIn("r1", aborted)
        self.assertEqual(e._inserted, {})
        self.assertEqual(e.runner.uid_to_seq, {})
        self.assertEqual(q.get(timeout=1)["finish_reason"], "error")

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


    def test_runner_remove_keeps_mapping_when_batcher_raises(self):
        from terrazzo.runner import MLXRunner
        seq = Sequence(request_id="x", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        runner = MLXRunner.__new__(MLXRunner)
        runner.uid_to_seq = {7: seq}
        class BadBG:
            def remove(self, uids, return_prompt_caches=False):
                raise RuntimeError("batcher removal failed")
        runner.bg = BadBG()
        with self.assertRaises(RuntimeError):
            runner.remove([seq])
        self.assertEqual(runner.uid_to_seq, {7: seq})

    def test_abort_failure_recovers_and_returns_false(self):
        import queue as _queue
        e = self._engine()
        e._executor_broken = False
        seq = Sequence(request_id="r-bad", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        e._inserted = {"r-bad": 7}
        reset = {"called": False}
        class FakeRunner:
            uid_to_seq = {7: seq}
            def remove(self, seqs):
                raise RuntimeError("batcher removal failed")
            def reset(self):
                reset["called"] = True
                self.uid_to_seq.clear()
        e.runner = FakeRunner()
        class FakeScheduler:
            waiting = [seq]
            running = []
            def abort(self, rid):
                return True
        e.scheduler = FakeScheduler()
        q = _queue.Queue()
        e._streams["r-bad"] = q
        e._outputs["r-bad"] = object()
        self.assertFalse(e.abort("r-bad"))
        self.assertTrue(reset["called"])
        self.assertFalse(e._executor_broken)
        self.assertEqual(e._inserted, {})
        self.assertEqual(e.runner.uid_to_seq, {})
        event = q.get(timeout=1)
        self.assertEqual(event["finish_reason"], "error")

    def test_abort_failure_reset_failure_marks_broken(self):
        import queue as _queue
        e = self._engine()
        e._executor_broken = False
        seq = Sequence(request_id="r-broken", prompt_token_ids=[1, 2],
                       sampling_params=SamplingParams(max_tokens=4))
        e._inserted = {"r-broken": 7}
        class FakeRunner:
            uid_to_seq = {7: seq}
            def remove(self, seqs):
                raise RuntimeError("batcher removal failed")
            def reset(self):
                raise RuntimeError("rebuild failed")
        e.runner = FakeRunner()
        class FakeScheduler:
            waiting = [seq]
            running = []
            def abort(self, rid):
                return True
        e.scheduler = FakeScheduler()
        q = _queue.Queue()
        e._streams["r-broken"] = q
        e._outputs["r-broken"] = object()
        self.assertFalse(e.abort("r-broken"))
        self.assertTrue(e._executor_broken)
        self.assertEqual(q.get(timeout=1)["finish_reason"], "error")
        with self.assertRaises(RuntimeError):
            e.add_request(prompt_token_ids=[1], params=SamplingParams(max_tokens=1))


    def test_prepare_full_hit_kv_rolls_back_one_token(self):
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from terrazzo.runner import MLXRunner
        runner = MLXRunner.__new__(MLXRunner)
        c = KVCache()
        c.keys = mx.ones((1, 1, 8, 4))
        c.values = mx.ones((1, 1, 8, 4))
        c.offset = 8
        cache, keep = runner.prepare_full_hit([c], 8, 8)
        self.assertEqual(keep, 7)
        self.assertIs(cache[0], c)
        self.assertEqual(c.offset, 7)
        self.assertEqual(c.keys.shape[2], 7)
        # with a suffix (partial prompt hit), keep the cache unchanged
        cache2, keep2 = runner.prepare_full_hit([c], 7, 8)
        self.assertEqual(keep2, 7)
        self.assertIs(cache2[0], c)
        # native cache cannot be safely sliced -> recompute
        class Native:
            pass
        cache3, keep3 = runner.prepare_full_hit([Native()], 8, 8)
        self.assertIsNone(cache3)
        self.assertEqual(keep3, 0)


    def test_register_prefix_trims_before_prefill(self):
        import types
        from unittest.mock import patch
        import mlx.core as mx
        from terrazzo.blocks import BlockManager, BlockManagerConfig
        from terrazzo.paged_kv import PagedPrefixCache

        e = Engine.__new__(Engine)
        e.cfg = types.SimpleNamespace(prefill_chunk_size=512,
                                      max_num_batched_tokens=2048)
        e.prefix_cache = PagedPrefixCache(
            BlockManager(BlockManagerConfig(num_blocks=512, block_size=16)),
            max_entries=2, max_bytes=1 << 30)
        seen = []
        class FakeModel:
            def __call__(self, arr, cache=None):
                seen.append(arr.shape[1])
                return mx.zeros((1, 1))
        e.runner = types.SimpleNamespace(model=FakeModel())
        class FakeCache:
            state = []
            meta_state = ""
        with patch("mlx_lm.models.cache.make_prompt_cache", return_value=[FakeCache()]):
            prefix = e.register_prefix(list(range(2003)))
        self.assertEqual(seen, [2000])  # prefill never saw the partial block
        self.assertIsNotNone(prefix)
        self.assertEqual(prefix.num_tokens, 2000)
        self.assertEqual(len(prefix.tokens), 2000)
        self.assertIsNotNone(prefix.native)

    def test_insert_full_hit_replays_last_token(self):
        import types
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache
        from terrazzo.runner import MLXRunner
        r = MLXRunner.__new__(MLXRunner)
        r.uid_to_seq = {}
        calls = []
        class FakeBG:
            def insert_segments(self, **kw):
                calls.append(kw)
                return (7,)
        r.bg = FakeBG()
        seq = Sequence(request_id="x", prompt_token_ids=list(range(8)),
                       sampling_params=SamplingParams(max_tokens=4))
        c = KVCache()
        c.keys = mx.ones((1, 1, 8, 4)); c.values = mx.ones((1, 1, 8, 4)); c.offset = 8
        r.insert(seq, cache=[c], cached_tokens=8)
        self.assertEqual(calls[0]["segments"], [[[7]]])
        self.assertEqual(calls[0]["all_tokens"], [[0,1,2,3,4,5,6]])
        self.assertEqual(c.offset, 7)
        # native full hit must recompute the whole prompt
        calls.clear()
        class Native: pass
        r.insert(seq, cache=[Native()], cached_tokens=8)
        self.assertEqual(calls[0]["segments"], [[list(range(8))]])
        self.assertEqual(calls[0]["all_tokens"], [[]])
        self.assertEqual(calls[0]["caches"], [None])


    def test_insert_new_does_not_count_recompute_as_hit(self):
        import types

        class FakePrefixCache:
            def __init__(self):
                self.num_hits = 0
                self.num_stores = 0
                self.entries = {}
            def peek(self, ids):
                p = types.SimpleNamespace(native=None, nbytes=1, num_tokens=16)
                return p, 16
            def lookup(self, ids):
                self.num_hits += 1
                p = types.SimpleNamespace(native=None, nbytes=1, num_tokens=16)
                return p, 16
            def undo_hit(self, prefix=None):
                self.num_hits = max(0, self.num_hits - 1)
            def stats(self):
                return {"hits": self.num_hits}
        class FakeCost:
            class P:
                reuse = True
            def estimate(self, *a, **k):
                return self.P()
        class FakeRunner:
            def insert(self, seq, cache=None, cached_tokens=0):
                seq.cached_prompt_tokens = 0  # executor fell back to recompute
                return 7
        e = Engine.__new__(Engine)
        e.cfg = types.SimpleNamespace(prefix_reuse="always")
        e.prefix_cache = FakePrefixCache()
        e.cost_model = FakeCost()
        e.runner = FakeRunner()
        e._inserted = {}
        e.prefix_hit_tokens = 0
        e.prefix_reuse_autos = 0
        e.prefix_recompute_skips = 0
        e._seed_caches = lambda prefix, n: []
        seq = Sequence(request_id="r", prompt_token_ids=list(range(32)),
                       sampling_params=SamplingParams(max_tokens=4))
        outs = types.SimpleNamespace(chunks=[types.SimpleNamespace(seq=seq, is_prefill=True)])
        e._insert_new(outs)
        self.assertEqual(e.prefix_cache.num_hits, 0)
        self.assertEqual(e.prefix_hit_tokens, 0)
        self.assertEqual(e.prefix_reuse_autos, 0)
