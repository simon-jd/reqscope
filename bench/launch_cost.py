"""Host cost of launching decode-like steps under each collector mode, without vLLM.

    python bench/launch_cost.py --mode off|plugin|light|trace|sampled --out results/raw/launch-trace.json

Each step launches one CUDA graph of --nodes small kernels and --eager
kernels outside it, between reqscope_push_step and reqscope_pop_step (as
the vLLM plugin does), then waits for the GPU with an event, as a decode
step waits for its sampled tokens. The child process measures per step:
- launch: host time from the first launch call to the last one returning;
- step: wall time of the whole step, waiting included.
--size sets the elements each kernel touches, and so the GPU time per
step: with the default, a step takes about 5 ms on an RTX 4090, close to a
vLLM decode step, so the collector sees a similar record rate. The launch
time is what CUPTI's per-launch work and any stall of the launching thread
add to. In sampled mode (with --env REQSCOPE_TRACE_EVERY=N) step i is traced
when i % N == 0, counting warm-up steps; the report splits the times into
traced steps, the step right after each (the first back in light mode) and
the others.

The collector runs as in bench/interference.py: CUDA_INJECTION64_PATH (light,
trace), REQSCOPE_MODE, and libcupti preloaded in every mode. Extra
REQSCOPE_* settings can be passed with --env KEY=VALUE. With --lock, the run
holds that file's flock.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from interference import cupti_library  # noqa: E402

CHILD = r"""
import ctypes, json, os, sys, time
import torch
nodes, eager, steps, warmup, size = map(int, sys.argv[1:6])
lib = ctypes.CDLL(os.environ["REQSCOPE_LIBRARY"]) if os.environ.get("REQSCOPE_LIBRARY") else None
if lib:
    lib.reqscope_push_step.argtypes = [ctypes.c_uint64]
    if os.environ.get("REQSCOPE_MODE") == "plugin":
        assert lib.InitializeInjection() == 1
x = torch.zeros(size, device="cuda")
stream = torch.cuda.Stream()
graph = torch.cuda.CUDAGraph()
with torch.cuda.stream(stream):
    for _ in range(3):
        for _ in range(nodes): x.add_(1.0)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(nodes): x.add_(1.0)
done = torch.cuda.Event()
launch, step = [], []
with torch.cuda.stream(stream):
    for i in range(warmup + steps):
        t0 = time.perf_counter_ns()
        if lib: lib.reqscope_push_step(i)
        graph.replay()
        for _ in range(eager): x.mul_(1.0)
        if lib: lib.reqscope_pop_step()
        t1 = time.perf_counter_ns()
        done.record(stream)
        done.synchronize()
        t2 = time.perf_counter_ns()
        if i >= warmup:
            launch.append(t1 - t0); step.append(t2 - t0)
if lib and hasattr(lib, "reqscope_flush"):
    lib.reqscope_flush.restype = ctypes.c_int
    lib.reqscope_flush()
print(json.dumps({"launch_ns": launch, "step_ns": step}))
"""


def quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def summary(values: list[int]) -> dict:
    median = quantile(values, 0.5)
    return {"p50_us": round(median / 1e3, 1), "p90_us": round(quantile(values, 0.9) / 1e3, 1),
            "p99_us": round(quantile(values, 0.99) / 1e3, 1), "max_us": round(max(values) / 1e3, 1),
            "mean_us": round(sum(values) / len(values) / 1e3, 1),
            "over_2x_median": sum(v > 2 * median for v in values)}


def capture_summary(capture: Path) -> dict:
    """Loss counts by reason, GPU records by kind, TracedStep records, and the
    collector's last CollectorStats totals."""
    from capture_stats import frames
    losses, kinds, stats, traced = {}, {}, None, 0
    for segment in sorted(capture.glob("segment-*.rqs")):
        for record, _ in frames(segment):
            if record and record.get("type") == "Loss":
                losses[record["reason"]] = losses.get(record["reason"], 0) + record.get("count", 0)
            elif record and record.get("type") == "GpuActivity":
                kinds[record.get("kind")] = kinds.get(record.get("kind"), 0) + 1
            elif record and record.get("type") == "TracedStep":
                traced += 1
            elif record and record.get("type") == "CollectorStats":
                stats = record
    out = {"losses": losses, "gpu_records_by_kind": kinds, "traced_step_records": traced}
    if stats:
        out.update(writer_cpu_s=round(stats["writer_cpu_ns"] / 1e9, 3), cupti_records=stats["cupti_records"])
        switch = stats["sites"].get("switch_granularity")
        if switch and switch["calls"]:
            out["switch_granularity"] = {"calls": switch["calls"], "mean_us": round(switch["ns"] / switch["calls"] / 1e3, 1),
                                         "max_us": round(switch["max_ns"] / 1e3, 1)}
    return out


def split(values: list[int], warmup: int, every: int) -> dict:
    """Sampled mode: times of traced steps, of the step after each, and of the rest."""
    groups = {"traced": [], "after_traced": [], "other": []}
    for k, value in enumerate(values):
        i = warmup + k
        group = "traced" if i % every == 0 else "after_traced" if i % every == 1 else "other"
        groups[group].append(value)
    return {name: summary(group) for name, group in groups.items() if group}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["off", "plugin", "light", "trace", "sampled"], required=True)
    parser.add_argument("--out", type=Path, required=True, help="result JSON file")
    parser.add_argument("--nodes", type=int, default=400, help="kernels in the graph")
    parser.add_argument("--eager", type=int, default=30, help="kernels launched outside the graph per step")
    parser.add_argument("--size", type=int, default=1 << 20, help="float32 elements per kernel")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--env", action="append", default=[], help="extra KEY=VALUE for the child")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    parser.add_argument("--build", type=Path, default=ROOT.parent / "build")
    parser.add_argument("--lock", type=Path, default=os.environ.get("REQSCOPE_GPU_LOCK"))
    args = parser.parse_args()
    extra = dict(item.split("=", 1) for item in args.env)
    library = args.build / "libreqscope.so"
    with tempfile.TemporaryDirectory(prefix="rqs-launch-") as capture:
        env = {k: v for k, v in os.environ.items() if not k.startswith("REQSCOPE_")}
        env["LD_PRELOAD"] = str(cupti_library(args.cuda))
        if args.mode != "off":
            env.update({"REQSCOPE_LIBRARY": str(library), "REQSCOPE_MODE": args.mode,
                        "REQSCOPE_OUTPUT": capture, "REQSCOPE_SOCKET": capture + "/control.sock",
                        "REQSCOPE_RING_SECONDS": "5", "REQSCOPE_RING_BYTES": str(256 << 20)})
            if args.mode in ("light", "trace", "sampled"):
                env["CUDA_INJECTION64_PATH"] = str(library)
        env.update(extra)
        command = [str(args.python), "-c", CHILD, str(args.nodes), str(args.eager), str(args.steps),
                   str(args.warmup), str(args.size)]
        with open(args.lock, "a+") if args.lock else contextlib.nullcontext() as lock:
            if lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
            load = Path("/proc/loadavg").read_text().split()[0]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise SystemExit(result.stderr[-2000:])
        times = json.loads(result.stdout.strip().splitlines()[-1])
        captured = capture_summary(Path(capture))
    report = {"mode": args.mode, "env": extra, "nodes": args.nodes, "eager": args.eager, "steps": args.steps,
              "size": args.size, "load1": float(load), "launch": summary(times["launch_ns"]),
              "step": summary(times["step_ns"]), "capture": captured}
    if args.mode == "sampled":
        every = int(extra.get("REQSCOPE_TRACE_EVERY", "16"))
        if every > 1:
            report["launch_by_step"] = split(times["launch_ns"], args.warmup, every)
            report["step_by_step"] = split(times["step_ns"], args.warmup, every)
    args.out.write_text(json.dumps(report))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
