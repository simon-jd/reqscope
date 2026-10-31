import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adapter_under_test", ROOT / "adapter/vllm/reqscope_adapter/__init__.py")
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


class FakeCollector:
    def __init__(self):
        self.events = []
        self.stack = []
        self.records = []
        self.fail_pop = False
        self.fail_push = False
        self.fail_emit = False

    def push(self, identity):
        self.events.append(("push", identity))
        if self.fail_push:
            raise ADAPTER.CollectorUnavailable("writer failed")
        self.stack.append(identity)

    def pop(self):
        self.events.append(("pop", self.stack.pop()))
        if self.fail_pop:
            raise RuntimeError("collector pop failed")

    def emit(self, record):
        self.events.append(("emit", record["type"]))
        if self.fail_emit:
            raise ADAPTER.CollectorUnavailable("writer failed")
        self.records.append(dict(record))
        return 0


def output(tokens=None):
    return SimpleNamespace(num_scheduled_tokens={"new": 3, "old": 1} if tokens is None else tokens,
                           scheduled_new_reqs=[SimpleNamespace(req_id="new")],
                           scheduled_cached_reqs=SimpleNamespace(is_context_phase=lambda req: False),
                           preempted_req_ids={"paused"}, finished_req_ids={"done"})


def runner_class():
    class Runner:
        def __init__(self):
            self.execute_calls = []
            self.sample_calls = []
            self.result = None
            self.execute_error = None
            self.sample_error = None

        def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
            self.execute_calls.append((scheduler_output, intermediate_tensors, dummy_run))
            if self.execute_error:
                raise self.execute_error
            return self.result

        def sample_tokens(self, grammar=None):
            self.sample_calls.append(grammar)
            if self.sample_error:
                raise self.sample_error
            return "tokens"
    return Runner


class AdapterContractTests(unittest.TestCase):
    def setUp(self):
        self.collector = FakeCollector()
        self.runner_type = runner_class()
        ADAPTER.install(self.runner_type, self.collector, version="0.31.0")
        self.runner = self.runner_type()

    def test_async_schedule_keeps_execute_and_sample_in_same_step(self):
        scheduled = output()
        self.assertIsNone(self.runner.execute_model(scheduled, "intermediate"))
        self.assertEqual(self.collector.records, [])
        self.assertEqual(self.runner.sample_tokens("grammar"), "tokens")
        self.assertEqual(len(self.collector.records), 1)
        row = self.collector.records[0]
        self.assertEqual(row["members"], [{"req_id": "new", "tokens": 3, "phase": "prefill"},
                                          {"req_id": "old", "tokens": 1, "phase": "decode"}])
        self.assertEqual(row["preempted"], ["paused"])
        self.assertEqual(row["finished"], ["done"])
        self.assertLessEqual(row["t_begin"], row["t_end"])
        self.assertEqual(self.collector.events,
                         [("push", row["step_id"]), ("pop", row["step_id"])] * 2 + [("emit", "Step")])
        self.assertEqual(self.collector.stack, [])
        self.assertEqual(self.runner.execute_calls, [(scheduled, "intermediate", False)])
        self.assertEqual(scheduled._reqscope_step_id, row["step_id"])   # read by the lifecycle hook

    def test_dummy_run_never_records_or_correlates(self):
        self.runner.result = "dummy-result"
        self.assertEqual(self.runner.execute_model(output(), dummy_run=True), "dummy-result")
        self.runner.sample_tokens()
        self.assertEqual(self.collector.records, [])
        self.assertEqual(self.collector.events, [])

    def test_empty_step_and_immediate_result_finalize(self):
        self.runner.execute_model(output({}))
        self.assertEqual(self.collector.records[-1]["members"], [])
        self.runner.result = "result"
        self.assertEqual(self.runner.execute_model(output()), "result")
        self.assertEqual(len(self.collector.records), 2)
        self.runner.sample_tokens()
        self.assertEqual(len(self.collector.records), 2)

    def test_engine_exception_balances_correlation_and_marks_loss(self):
        original = ValueError("engine failure")
        self.runner.execute_error = original
        with self.assertRaises(ValueError) as captured:
            self.runner.execute_model(output())
        self.assertIs(captured.exception, original)
        self.assertEqual(self.collector.stack, [])
        self.assertEqual([r["type"] for r in self.collector.records], ["Loss", "Step"])
        self.assertEqual(self.collector.records[-1]["status"], "Partial")

    def test_sample_exception_balances_correlation_and_marks_loss(self):
        self.runner.execute_model(output())
        original = ValueError("sample failure")
        self.runner.sample_error = original
        with self.assertRaises(ValueError) as captured:
            self.runner.sample_tokens()
        self.assertIs(captured.exception, original)
        self.assertEqual(self.collector.stack, [])
        self.assertEqual(self.collector.records[-1]["status"], "Partial")

    def test_collector_cleanup_does_not_replace_engine_exception(self):
        original = ValueError("engine failure must survive")
        self.runner.execute_error = original
        self.collector.fail_pop = True
        with self.assertWarns(RuntimeWarning):
            with self.assertRaises(ValueError) as captured:
                self.runner.execute_model(output())
        self.assertIs(captured.exception, original)

    def test_new_execute_closes_missing_sample_as_partial(self):
        self.runner.execute_model(output())
        self.runner.execute_model(output())
        self.assertEqual([r["type"] for r in self.collector.records], ["Loss", "Step"])
        first = self.collector.records[-1]
        self.assertEqual(first["status"], "Partial")
        self.runner.sample_tokens()
        self.assertNotEqual(first["step_id"], self.collector.records[-1]["step_id"])

    def test_install_is_idempotent(self):
        method = self.runner_type.execute_model
        ADAPTER.install(self.runner_type, self.collector, version="0.31.0")
        self.assertIs(self.runner_type.execute_model, method)
        self.runner.result = "done"
        self.runner.execute_model(output())
        self.assertEqual(len(self.collector.records), 1)

    def test_version_and_signature_fail_before_mutating_class(self):
        fresh = runner_class()
        original = fresh.execute_model
        with self.assertRaises(RuntimeError):
            ADAPTER.install(fresh, self.collector, version="0.30.0")
        self.assertIs(fresh.execute_model, original)
        class Unsupported:
            def execute_model(self, scheduler_output): pass
            def sample_tokens(self): pass
        original = Unsupported.execute_model
        with self.assertRaises(RuntimeError):
            ADAPTER.install(Unsupported, self.collector)
        self.assertIs(Unsupported.execute_model, original)

    def config(self, tp=1, pp=1, executor="uni"):
        return SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                                                                distributed_executor_backend=executor),
                               scheduler_config=SimpleNamespace(async_scheduling=True),
                               compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE"))

    def test_unsupported_parallel_configuration_rejected_before_engine(self):
        for config in (self.config(tp=2), self.config(pp=2), self.config(executor="mp")):
            with self.subTest(config=config):
                fresh = self.runner_type()
                fresh.vllm_config = config
                with self.assertRaises(RuntimeError):
                    fresh.execute_model(output())
                self.assertEqual(fresh.execute_calls, [])
                self.assertEqual(self.collector.events, [])

    def test_effective_configuration_recorded_once(self):
        self.runner.vllm_config = self.config()
        self.runner.result = "done"
        self.runner.execute_model(output())
        self.runner.execute_model(output())
        configs = [r for r in self.collector.records if r["type"] == "AdapterConfig"]
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["executor"], "uni")
        self.assertTrue(configs[0]["async_scheduling"])
        self.assertEqual(configs[0]["cudagraph_mode"], "FULL_AND_PIECEWISE")

    def test_nvtx_push_failure_balances_correlation_and_keeps_serving(self):
        class RangeCollector(FakeCollector):
            def range_push(self, record, phase):
                raise ValueError("NVTX unavailable")
            def range_pop(self):
                raise AssertionError("failed NVTX push must not be popped")
        collector = RangeCollector()
        fresh = runner_class()
        ADAPTER.install(fresh, collector)
        runner = fresh()
        runner.result = "served"
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(runner.execute_model(output()), "served")
        self.assertEqual(collector.stack, [])
        self.assertEqual(len(runner.execute_calls), 1)
        self.assertEqual([r["type"] for r in collector.records], ["Loss", "Step", "Diagnostic"])
        self.assertEqual(collector.records[1]["status"], "Partial")
        self.assertIn("NVTX", collector.records[-1]["reason"])

    def test_nvtx_pop_failure_preserves_engine_exception_and_pops_correlation(self):
        class RangeCollector(FakeCollector):
            def range_push(self, record, phase): pass
            def range_pop(self): raise RuntimeError("NVTX pop failed")
        collector = RangeCollector()
        fresh = runner_class()
        ADAPTER.install(fresh, collector)
        runner = fresh()
        original = ValueError("engine original")
        runner.execute_error = original
        with self.assertWarns(RuntimeWarning):
            with self.assertRaises(ValueError) as captured:
                runner.execute_model(output())
        self.assertIs(captured.exception, original)
        self.assertEqual(collector.stack, [])
        [step] = [r for r in collector.records if r["type"] == "Step"]
        self.assertEqual(step["status"], "Partial")

    def assert_stopped_and_serving(self, runner):
        """Once collection stopped, steps run with no collector calls at all."""
        self.assertTrue(self.collector.reqscope_stopped)
        before = len(self.collector.events)
        runner.result = None
        self.assertIsNone(runner.execute_model(output()))
        self.assertEqual(runner.sample_tokens(), "tokens")
        runner.result = "again"
        self.assertEqual(runner.execute_model(output()), "again")
        self.assertEqual(self.collector.events[before:], [])

    def test_writer_failure_on_push_keeps_serving_and_stops_collecting(self):
        self.collector.fail_push = True
        self.runner.result = "served"
        with self.assertWarnsRegex(RuntimeWarning, "stopped collecting"):
            self.assertEqual(self.runner.execute_model(output()), "served")
        self.assertEqual(len(self.runner.execute_calls), 1)
        [diagnostic] = self.collector.records
        self.assertEqual((diagnostic["type"], diagnostic["source"], diagnostic["stopped"]),
                         ("Diagnostic", "vllm_plugin", True))
        self.assert_stopped_and_serving(self.runner)

    def test_writer_failure_on_emit_keeps_serving_and_stops_collecting(self):
        self.runner.execute_model(output())
        self.collector.fail_emit = True
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(self.runner.sample_tokens(), "tokens")
        self.assertEqual(self.collector.stack, [])
        self.assertEqual(self.collector.records, [])
        self.assert_stopped_and_serving(self.runner)

    def test_pop_failure_without_engine_error_does_not_raise(self):
        self.collector.fail_pop = True
        self.runner.result = "served"
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(self.runner.execute_model(output()), "served")
        self.assertEqual([r["type"] for r in self.collector.records], ["Loss", "Step", "Diagnostic"])
        self.assert_stopped_and_serving(self.runner)

    def test_collector_that_cannot_start_does_not_stop_the_engine(self):
        class Broken:
            def emit(self, record): raise RuntimeError("reqscope collector initialization failed")
            def push(self, step_id): raise RuntimeError("reqscope collector initialization failed")
            def pop(self): raise AssertionError("never pushed")
        fresh = runner_class()
        ADAPTER.install(fresh, Broken())
        runner = fresh()
        runner.vllm_config = self.config()
        runner.result = "served"
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(runner.execute_model(output()), "served")
        self.assertEqual(runner.execute_model(output()), "served")
        self.assertEqual(len(runner.execute_calls), 2)


class LifecycleTests(unittest.TestCase):
    def test_request_record_has_arrival_queueing_tokens_and_finish_in_order(self):
        class Scheduler:
            def add_request(self, request): pass
            def schedule(self): return SimpleNamespace(num_scheduled_tokens={"r1": 4})
            def update_from_output(self, scheduler_output, model_runner_output): return {}
            def _free_request(self, request): return None
        collector = FakeCollector()
        ADAPTER.install_lifecycle(Scheduler, collector)
        ADAPTER.install_lifecycle(Scheduler, collector)          # idempotent
        scheduler = Scheduler()
        request = SimpleNamespace(request_id="r1", num_prompt_tokens=4,
                                  status=SimpleNamespace(name="FINISHED_LENGTH_CAPPED"))
        scheduler.add_request(request)
        out = scheduler.schedule()
        sampled = SimpleNamespace(req_id_to_index={"r1": 0}, sampled_token_ids=[[7]])
        scheduler.update_from_output(out, sampled)
        scheduler.update_from_output(out, SimpleNamespace(req_id_to_index={"r1": 0}, sampled_token_ids=[[]]))
        scheduler.update_from_output(out, sampled)
        scheduler._free_request(request)
        [record] = [r for r in collector.records if r["type"] == "Request"]
        self.assertEqual(record["req_id"], "r1")
        self.assertEqual(record["prompt_tokens"], 4)
        self.assertEqual(len(record["token_times"]), 2)              # the empty sample is not a token
        times = [record["start"], record["queued_until"], *record["token_times"], record["end"]]
        self.assertEqual(times, sorted(times))
        self.assertEqual(record["status"], "FINISHED_LENGTH_CAPPED")
        self.assertNotIn("token_steps", record)                     # outputs were never executed

    def test_final_token_is_kept_when_the_request_is_freed_inside_update(self):
        request = SimpleNamespace(request_id="r1", num_prompt_tokens=1, status=SimpleNamespace(name="FINISHED"))
        class Scheduler:
            def add_request(self, request): pass
            def schedule(self): return SimpleNamespace(num_scheduled_tokens={"r1": 1})
            def update_from_output(self, scheduler_output, model_runner_output):
                self._free_request(request)          # vLLM frees finished requests in here
                return {}
            def _free_request(self, request): return None
        collector = FakeCollector()
        ADAPTER.install_lifecycle(Scheduler, collector)
        scheduler = Scheduler()
        scheduler.add_request(request)
        scheduler.update_from_output(scheduler.schedule(),
                                     SimpleNamespace(req_id_to_index={"r1": 0}, sampled_token_ids=[[7]]))
        [record] = [r for r in collector.records if r["type"] == "Request"]
        self.assertEqual(len(record["token_times"]), 1)

    def test_long_requests_flush_tokens_in_chunks(self):
        class Scheduler:
            def add_request(self, request): pass
            def schedule(self): return SimpleNamespace(num_scheduled_tokens={"r1": 1})
            def update_from_output(self, scheduler_output, model_runner_output): return {}
            def _free_request(self, request): return None
        collector = FakeCollector()
        ADAPTER.install_lifecycle(Scheduler, collector)
        scheduler = Scheduler()
        request = SimpleNamespace(request_id="r1", num_prompt_tokens=1, status=SimpleNamespace(name="FINISHED"))
        scheduler.add_request(request)
        sampled = SimpleNamespace(req_id_to_index={"r1": 0}, sampled_token_ids=[[7]])
        with patch.object(ADAPTER, "TOKEN_CHUNK", 2):
            for step in range(5):
                output = scheduler.schedule()
                output._reqscope_step_id = 10 + step
                scheduler.update_from_output(output, sampled)
            scheduler._free_request(request)
        chunks = [r for r in collector.records if r["type"] == "RequestTokens"]
        [final] = [r for r in collector.records if r["type"] == "Request"]
        self.assertEqual([(c["first"], c["token_steps"]) for c in chunks], [(0, [10, 11]), (2, [12, 13])])
        self.assertEqual((final["token_offset"], final["token_steps"]), (4, [14]))
        times = [t for c in chunks for t in c["token_times"]] + final["token_times"]
        self.assertEqual(times, sorted(times))

    def test_emit_failure_in_the_scheduler_stops_lifecycle_records_without_raising(self):
        class Scheduler:
            def add_request(self, request): pass
            def schedule(self): return SimpleNamespace(num_scheduled_tokens={"r1": 1})
            def update_from_output(self, scheduler_output, model_runner_output): return "updated"
            def _free_request(self, request): return "freed"
        collector = FakeCollector()
        ADAPTER.install_lifecycle(Scheduler, collector)
        scheduler = Scheduler()
        request = SimpleNamespace(request_id="r1", num_prompt_tokens=1, status=SimpleNamespace(name="FINISHED"))
        scheduler.add_request(request)
        sampled = SimpleNamespace(req_id_to_index={"r1": 0}, sampled_token_ids=[[7]])
        self.assertEqual(scheduler.update_from_output(scheduler.schedule(), sampled), "updated")
        collector.fail_emit = True
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(scheduler._free_request(request), "freed")
        before = len(collector.events)
        scheduler.add_request(request)
        self.assertEqual(scheduler.update_from_output(scheduler.schedule(), sampled), "updated")
        self.assertEqual(scheduler._free_request(request), "freed")
        self.assertEqual(collector.events[before:], [])

    def test_tokens_name_the_step_that_produced_them(self):
        class Scheduler:
            def add_request(self, request): pass
            def schedule(self): return SimpleNamespace(num_scheduled_tokens={"r1": 1})
            def update_from_output(self, scheduler_output, model_runner_output): return {}
            def _free_request(self, request): return None
        collector = FakeCollector()
        ADAPTER.install_lifecycle(Scheduler, collector)
        scheduler = Scheduler()
        request = SimpleNamespace(request_id="r1", num_prompt_tokens=1, status=SimpleNamespace(name="FINISHED"))
        scheduler.add_request(request)
        first, second = scheduler.schedule(), scheduler.schedule()
        first._reqscope_step_id, second._reqscope_step_id = 11, 12   # set by the runner hook
        sampled = SimpleNamespace(req_id_to_index={"r1": 0}, sampled_token_ids=[[7]])
        scheduler.update_from_output(first, sampled)
        scheduler.update_from_output(second, sampled)
        scheduler._free_request(request)
        [record] = [r for r in collector.records if r["type"] == "Request"]
        self.assertEqual(record["token_steps"], [11, 12])


if __name__ == "__main__":
    unittest.main()
