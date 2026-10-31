"""vLLM 0.31.0 hooks for reqscope; no vLLM import until plugin registration.

Two sets of hooks, both in the EngineCore process (uni executor, TP=1):

* the V2 GPUModelRunner's execute_model/sample_tokens emit one Step record per
  engine step and push its id as CUPTI external correlation, so every CUDA
  API call and GPU activity of that step links back to it;
* the scheduler's add_request/schedule/update_from_output/_free_request emit
  one Request record per request: arrival, first scheduled, per-token and
  finish times on the same CLOCK_MONOTONIC as the GPU timestamps.

The profiler must never take serving down. If any collector call fails (the
writer hit a full disk, the collector is shutting down, an NVTX range cannot
be opened), the plugin closes what it opened, stops collecting for the rest
of the process, records a Diagnostic if the collector still accepts records,
and warns. Engine exceptions always propagate unchanged.
"""
import ctypes
import functools
import importlib.metadata
import inspect
import itertools
import json
import os
import threading
import time
import warnings

SUPPORTED_VERSION = "0.31.0"
_ids = itertools.count(1)


class CollectorUnavailable(RuntimeError):
    """The collector no longer accepts records: its writer failed or it stopped."""


class Collector:
    def __init__(self, path=None):
        self.lib = ctypes.CDLL(path or os.environ["REQSCOPE_LIBRARY"])
        self.lib.InitializeInjection.argtypes = []
        self.lib.InitializeInjection.restype = ctypes.c_int
        self.lib.reqscope_emit_json.argtypes = [ctypes.c_char_p]
        self.lib.reqscope_emit_json.restype = ctypes.c_int
        self.lib.reqscope_push_step.argtypes = [ctypes.c_uint64]
        self.lib.reqscope_push_step.restype = ctypes.c_int
        self.lib.reqscope_pop_step.argtypes = []
        self.lib.reqscope_pop_step.restype = ctypes.c_int
        if self.lib.InitializeInjection() != 1:
            raise RuntimeError("reqscope collector initialization failed")

    def emit(self, record):
        # -1 (malformed/oversize) and -2 (queue full) are counted as Loss
        # records by the collector itself; -3 means it accepts nothing more.
        result = self.lib.reqscope_emit_json(json.dumps(record, separators=(",", ":")).encode())
        if result == -3:
            raise CollectorUnavailable("reqscope collector is not accepting records")
        return result

    def push(self, step_id):
        result = self.lib.reqscope_push_step(step_id)
        if result == -3:
            raise CollectorUnavailable("reqscope collector is not accepting records")
        if result:
            raise RuntimeError("reqscope external correlation push failed")

    def pop(self):
        result = self.lib.reqscope_pop_step()
        if result == -3:
            raise CollectorUnavailable("reqscope collector is not accepting records")
        if result:
            raise RuntimeError("reqscope external correlation pop failed")

    def range_push(self, record, phase):
        if os.environ.get("REQSCOPE_NVTX") == "1":
            import nvtx
            nvtx.push_range(message=json.dumps({"step_id": record["step_id"],
                                                "members": record["members"], "phase": phase}),
                            domain="reqscope", payload=record["step_id"])

    def range_pop(self):
        if os.environ.get("REQSCOPE_NVTX") == "1":
            import nvtx
            nvtx.pop_range(domain="reqscope")


def collecting(collector):
    return not getattr(collector, "reqscope_stopped", False)


def stop_collecting(collector, action, error):
    """Stop all collection after a collector failure; serving continues."""
    if not collecting(collector):
        return
    collector.reqscope_stopped = True
    reason = f"{action} failed ({error}); reqscope stopped collecting and the engine keeps serving"
    warnings.warn(f"reqscope: {reason}", RuntimeWarning, stacklevel=2)
    try:
        collector.emit({"type": "Diagnostic", "source": "vllm_plugin", "reason": reason, "stopped": True})
    except Exception:
        pass   # the collector is gone; the warning and its own error file remain


def emit_or_stop(collector, record, action):
    if not collecting(collector):
        return
    try:
        collector.emit(record)
    except Exception as error:
        stop_collecting(collector, action, error)


def _members(output):
    new = {r.req_id for r in output.scheduled_new_reqs}
    cached = getattr(output, "scheduled_cached_reqs", None)
    members = []
    for req_id, tokens in output.num_scheduled_tokens.items():
        if not isinstance(req_id, str) or not isinstance(tokens, int) or tokens < 0:
            raise RuntimeError("unsupported SchedulerOutput member")
        phase = "prefill" if req_id in new else "unknown"
        if cached is not None and hasattr(cached, "is_context_phase"):
            phase = "prefill" if req_id in new or cached.is_context_phase(req_id) else "decode"
        members.append({"req_id": req_id, "tokens": tokens, "phase": phase})
    return members


def install(runner_class, collector, version=SUPPORTED_VERSION):
    """Wrap exactly two methods. Exposed separately for dependency-free contract tests."""
    if version != SUPPORTED_VERSION:
        raise RuntimeError(f"reqscope requires vLLM {SUPPORTED_VERSION}, found {version}")
    if getattr(runner_class, "_reqscope_installed", False):
        return
    execute = runner_class.execute_model
    sample = runner_class.sample_tokens
    signature = inspect.signature(execute)
    if not {"scheduler_output", "dummy_run"}.issubset(signature.parameters):
        raise RuntimeError("unsupported V2 execute_model signature")

    def finish(runner, partial=False):
        state = getattr(runner, "_reqscope_pending", None)
        if state is None:
            return
        runner._reqscope_pending = None
        state["t_end"] = time.monotonic_ns()
        if partial:
            state["status"] = "Partial"
            emit_or_stop(collector, {"type": "Loss", "reason": "adapter_incomplete_step", "count": 1,
                                     "time_range": [state["t_begin"], state["t_end"]]}, "recording a step")
        emit_or_stop(collector, state, "recording a step")

    def cleanup(runner, failed, complete):
        """Close the step's NVTX range and correlation. Never raises."""
        problem = None
        try:
            if getattr(runner, "_reqscope_nvtx_open", False):
                runner._reqscope_nvtx_open = False
                collector.range_pop()
        except Exception as error:
            problem = ("closing the step's NVTX range", error)
        try:
            collector.pop()
        except Exception as error:
            problem = problem or ("popping the step's external correlation", error)
        if complete or problem:
            finish(runner, partial=failed or problem is not None)
        if problem:
            stop_collecting(collector, *problem)

    def push(runner, record, phase):
        """Open the step's correlation and NVTX range; False if collection stopped."""
        try:
            collector.push(record["step_id"])
        except Exception as error:
            runner._reqscope_pending = None
            stop_collecting(collector, "pushing the step's external correlation", error)
            return False
        try:
            if hasattr(collector, "range_push"):
                collector.range_push(record, phase)
                runner._reqscope_nvtx_open = True
        except BaseException as error:
            try:
                collector.pop()
            except Exception:
                pass
            finish(runner, partial=True)
            if not isinstance(error, Exception):
                raise
            stop_collecting(collector, "opening the step's NVTX range", error)
            return False
        return True

    def validate_configuration(runner):
        if getattr(runner, "_reqscope_config_checked", False):
            return
        config = getattr(runner, "vllm_config", None)
        if config is not None:
            parallel = config.parallel_config
            if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
                raise RuntimeError("reqscope currently requires TP=1 and PP=1")
            executor = getattr(parallel, "distributed_executor_backend", None)
            if executor != "uni":
                raise RuntimeError(f"reqscope requires uni executor, found {executor}")
            emit_or_stop(collector, {"type": "AdapterConfig", "vllm_version": version,
                                     "runner": "V2 GPUModelRunner", "executor": executor,
                                     "async_scheduling": bool(config.scheduler_config.async_scheduling),
                                     "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
                                     "tensor_parallel_size": 1, "pipeline_parallel_size": 1},
                         "recording the adapter configuration")
        runner._reqscope_config_checked = True

    @functools.wraps(execute)
    def execute_wrapped(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["dummy_run"] or not collecting(collector):
            return execute(self, *args, **kwargs)
        validate_configuration(self)
        finish(self, partial=True)
        if not collecting(collector):
            return execute(self, *args, **kwargs)
        output = bound.arguments["scheduler_output"]
        step_id = next(_ids)
        # The scheduler hands this same object to update_from_output, so the
        # lifecycle hook can say which step produced each token.
        try:
            output._reqscope_step_id = step_id
        except AttributeError:
            pass
        record = {"type": "Step", "step_id": step_id, "t_begin": time.monotonic_ns(),
                  "tid": threading.get_native_id(), "members": _members(output),
                  "preempted": sorted(getattr(output, "preempted_req_ids", None) or []),
                  "finished": sorted(getattr(output, "finished_req_ids", None) or [])}
        self._reqscope_pending = record
        opened = push(self, record, "execute_model")
        failed = False
        try:
            if opened and os.environ.get("REQSCOPE_TEST_SYNCHRONIZE") == "1":
                import torch
                torch.cuda.synchronize()
            result = execute(self, *args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            if opened:
                cleanup(self, failed, failed)
        if opened and (result is not None or not output.num_scheduled_tokens):
            finish(self)
        return result

    @functools.wraps(sample)
    def sample_wrapped(self, *args, **kwargs):
        pending = getattr(self, "_reqscope_pending", None)
        if pending is None or not collecting(collector):
            return sample(self, *args, **kwargs)
        opened = push(self, pending, "sample_tokens")
        failed = False
        try:
            return sample(self, *args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            if opened:
                cleanup(self, failed, True)

    runner_class.execute_model = execute_wrapped
    runner_class.sample_tokens = sample_wrapped
    runner_class._reqscope_installed = True


TOKEN_CHUNK = 1024


def install_lifecycle(scheduler_class, collector):
    """Emit one Request record per request from inside the scheduler.

    Arrival is taken when the scheduler admits the request to its queue, so
    ``queued_until`` (first step that schedules it) measures engine queueing,
    and ``token_times`` are when the engine processed each sampled token.
    ``token_steps`` names the step that produced each token. With async
    scheduling the next step is already launched when a token is processed,
    so the step cannot be recovered from timestamps alone.

    Tokens are recorded before vLLM processes the step's output, because a
    request that finishes in that step is freed inside ``update_from_output``.
    Every ``TOKEN_CHUNK`` tokens they are flushed as a ``RequestTokens``
    record, so no record outgrows a collector queue slot however long the
    request runs.
    """
    if getattr(scheduler_class, "_reqscope_lifecycle", False):
        return
    add_request = scheduler_class.add_request
    schedule = scheduler_class.schedule
    update = scheduler_class.update_from_output
    free = scheduler_class._free_request
    live = {}

    @functools.wraps(add_request)
    def add_request_wrapped(self, request):
        if not collecting(collector):
            live.clear()
            return add_request(self, request)
        live[request.request_id] = {"start": time.monotonic_ns(), "prompt_tokens": request.num_prompt_tokens,
                                    "queued_until": None, "token_times": [], "token_steps": [], "flushed": 0}
        return add_request(self, request)

    @functools.wraps(schedule)
    def schedule_wrapped(self, *args, **kwargs):
        output = schedule(self, *args, **kwargs)
        now = time.monotonic_ns()
        for req_id in output.num_scheduled_tokens:
            state = live.get(req_id)
            if state is not None and state["queued_until"] is None:
                state["queued_until"] = now
        return output

    def flush(req_id, state):
        record = {"type": "RequestTokens", "req_id": req_id, "first": state["flushed"],
                  "token_times": state["token_times"]}
        if None not in state["token_steps"]:
            record["token_steps"] = state["token_steps"]
        emit_or_stop(collector, record, "recording request tokens")
        state["flushed"] += len(state["token_times"])
        state["token_times"], state["token_steps"] = [], []

    @functools.wraps(update)
    def update_wrapped(self, scheduler_output, model_runner_output):
        if not collecting(collector):
            return update(self, scheduler_output, model_runner_output)
        now = time.monotonic_ns()
        sampled = getattr(model_runner_output, "sampled_token_ids", None) or []
        step_id = getattr(scheduler_output, "_reqscope_step_id", None)
        for req_id, index in (getattr(model_runner_output, "req_id_to_index", None) or {}).items():
            state = live.get(req_id)
            if state is not None and index < len(sampled) and len(sampled[index]):
                state["token_times"].append(now)
                state["token_steps"].append(step_id)
                if len(state["token_times"]) >= TOKEN_CHUNK:
                    flush(req_id, state)
        return update(self, scheduler_output, model_runner_output)

    @functools.wraps(free)
    def free_wrapped(self, request, *args, **kwargs):
        state = live.pop(request.request_id, None)
        if state is not None and collecting(collector):
            record = {"type": "Request", "req_id": request.request_id, "start": state["start"],
                      "end": time.monotonic_ns(), "prompt_tokens": state["prompt_tokens"],
                      "token_times": state["token_times"], "token_offset": state["flushed"],
                      "status": request.status.name, "source": "engine-scheduler"}
            if None not in state["token_steps"]:
                record["token_steps"] = state["token_steps"]
            if state["queued_until"] is not None:
                record["queued_until"] = state["queued_until"]
            emit_or_stop(collector, record, "recording a request")
        return free(self, request, *args, **kwargs)

    scheduler_class.add_request = add_request_wrapped
    scheduler_class.schedule = schedule_wrapped
    scheduler_class.update_from_output = update_wrapped
    scheduler_class._free_request = free_wrapped
    scheduler_class._reqscope_lifecycle = True


def register():
    if os.environ.get("REQSCOPE_ENABLED") != "1":
        return
    version = importlib.metadata.version("vllm")
    if version != SUPPORTED_VERSION:
        raise RuntimeError(f"reqscope requires vLLM {SUPPORTED_VERSION}, found {version}")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    # Delay opening the library until the first real runner call in EngineCore.
    # The general plugin also runs in the API server before worker processes fork.
    class LazyCollector:
        inner = None
        def _get(self):
            if self.inner is None:
                self.inner = Collector()
            return self.inner
        def emit(self, record): return self._get().emit(record)
        def push(self, step_id): return self._get().push(step_id)
        def pop(self): return self._get().pop()
        def range_push(self, record, phase): return self._get().range_push(record, phase)
        def range_pop(self): return self._get().range_pop()
    collector = LazyCollector()
    install(GPUModelRunner, collector, version)
    from vllm.v1.core.sched.scheduler import Scheduler
    install_lifecycle(Scheduler, collector)
