"""Can reqscope explain why decode tokens got slow? And what does it cost?

    python bench/interference.py interference --mode trace --out results/raw/interference
    python bench/interference.py overhead --mode off|plugin|light|trace --out results/raw/overhead-off
    python bench/interference.py openloop --mode off|plugin|light|trace --rates 8,16,24 --out results/raw/openloop-off-r1
    python bench/interference.py multicause --mode trace|plugin --seed 1 --out results/raw/multicause-trace-s1

interference
    16 decode streams generate 4,000 tokens each. Starting 2 s in, a
    6,000-token prompt arrives every 3 s, six in all. With chunked prefill
    each prompt's chunks share engine steps with the decodes and stretch
    their inter-token gaps. The client records when each long prompt was sent
    and when its first token came back; the analyzer never sees this and has
    to find the culprits from step membership and GPU time alone.

overhead
    32 closed-loop clients (512-token prompts, 128 output tokens) for 60 s.

openloop
    Poisson arrivals of the same requests at each rate in --rates (requests
    per second), --seconds of arrivals per rate, on one server.

multicause
    16 decode streams generate --decode-tokens tokens each while the client
    injects causes of slow steps at random times drawn from --seed, three of
    each kind by default:
      interference  a 6,000-token prompt whose prefill shares steps with the decodes
      gpu_slowdown  a separate CUDA process (--python) runs fp16 matmuls for --burst-ms
      host_delay    the EngineCore process is stopped (SIGSTOP) for --stall-ms
    Each event's cause and actual [start, end] go into client.json, which
    the analyzer never reads; bench/score_causes.py compares them with
    blame's per-step causes.

Collector modes:
    off     no plugin, no collector
    plugin  the vLLM plugin with the collector's manual backend: steps,
            membership and request lifecycles, no CUPTI
    light   plugin + CUPTI injection, one record per CUDA graph launch
    trace   plugin + CUPTI injection, one record per kernel

The server is vLLM 0.31.0 from the Python environment given by --python
(default: this interpreter), with the reqscope plugin installed in it (see
README). Every path has a flag; nothing is machine-specific. With --lock,
the run holds that file's flock for the whole run, from server start to
shutdown.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CUPTI_NAMES = ("lib64/libcupti.so.13", "lib64/libcupti.so", "extras/CUPTI/lib64/libcupti.so")


def cupti_library(cuda: Path) -> Path:
    for name in CUPTI_NAMES:
        if (cuda / name).exists():
            return cuda / name
    raise SystemExit(f"no libcupti under {cuda}; pass --cuda with the CUDA 13.2 toolkit root")


def server(args, out: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REQSCOPE_", "OTEL_", "VLLM_"))}
    env.update({"VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_NO_USAGE_STATS": "1", "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": args.device,
                # libcupti is preloaded in every mode, so all modes load the same libraries.
                "LD_PRELOAD": str(cupti_library(args.cuda))})
    if args.hf_home:
        env["HF_HOME"] = str(args.hf_home)
    if args.offline:
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if args.cache_root:
        env.update({"XDG_CACHE_HOME": str(args.cache_root), "VLLM_CACHE_ROOT": str(args.cache_root / "vllm")})
    library = args.build / "libreqscope.so"
    if args.mode != "off":
        if not library.is_file():
            raise SystemExit(f"{library} not found; build reqscope first (README) or pass --build")
        env.update({"REQSCOPE_LIBRARY": str(library), "REQSCOPE_ENABLED": "1", "VLLM_PLUGINS": "reqscope",
                    "REQSCOPE_OUTPUT": str(out / "capture"), "REQSCOPE_RING_SECONDS": "3600",
                    "REQSCOPE_RING_BYTES": str(args.ring_bytes), "REQSCOPE_QUEUE_SLOTS": "1024"})
        if args.mode == "plugin":
            env["REQSCOPE_MODE"] = "plugin"           # the plugin's records only; CUPTI is never started
        else:
            env.update({"CUDA_INJECTION64_PATH": str(library), "REQSCOPE_MODE": args.mode,
                        "REQSCOPE_CUPTI_BUFFERS": "256"})
    # A local model directory is used as is; a Hub id is pinned to --revision.
    pinned = [] if Path(args.model).is_dir() else ["--revision", args.revision, "--tokenizer-revision", args.revision]
    command = [str(args.python), "-B", "-m", "vllm.entrypoints.cli.main", "serve", args.model, *pinned,
               "--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", "m",
               "--enable-request-id-headers", "--distributed-executor-backend", "uni", "--async-scheduling",
               "--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "32",
               "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--no-enable-prefix-caching", "--seed", "42"]
    log = open(out / "server.log", "w")
    return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(process: subprocess.Popen, port: int, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited during startup; see server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError("vLLM did not become ready")


def flush_collectors(pid: int) -> int:
    """Make every collector in the server's process tree drain and sync its trace.

    vLLM's shutdown SIGKILLs EngineCore almost at once, so records still in
    the collector's queue (the last requests' lifecycle records among them)
    would otherwise be lost. Returns the number of collectors flushed.
    """
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/stat") as stat:
                    children.setdefault(int(stat.read().rsplit(")", 1)[1].split()[1]), []).append(int(entry))
            except OSError:
                continue
    flushed, stack = 0, [pid]
    while stack:
        current = stack.pop()
        stack += children.get(current, [])
        path = f"/tmp/reqscope-{current}.sock"
        if not os.path.exists(path):
            continue
        with socket.socket(socket.AF_UNIX) as control:
            control.settimeout(120)
            control.connect(path)
            control.sendall(b'{"command":"flush"}\n')
            if b'"ok":true' not in control.makefile("rb").readline():
                raise RuntimeError(f"collector {current} did not flush")
        flushed += 1
    return flushed


async def stream(session, port: int, request_id: str, prompt: list[int], max_tokens: int) -> dict:
    body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0.0, "stream": True}
    sent, chunks = time.monotonic_ns(), []
    async with session.post(f"http://127.0.0.1:{port}/v1/completions", json=body,
                            headers={"X-Request-Id": request_id}) as response:
        async for line in response.content:
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                chunks.append(time.monotonic_ns())
    return {"id": request_id, "sent_ns": sent, "chunk_ns": chunks}


def tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1000, 100_000) for _ in range(n)]


async def interference(args) -> list[dict]:
    import aiohttp
    rng = random.Random(args.seed)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        decodes = [asyncio.create_task(stream(session, args.port, f"decode-{i}", tokens(rng, 128), 4000))
                   for i in range(16)]
        longs = []
        await asyncio.sleep(2.0)
        for j in range(6):
            longs.append(asyncio.create_task(stream(session, args.port, f"long-{j}", tokens(rng, 6000), 1)))
            await asyncio.sleep(3.0)
        return list(await asyncio.gather(*decodes, *longs))


async def overhead(args) -> list[dict]:
    import aiohttp
    rng = random.Random(args.seed)
    stop = time.monotonic() + args.seconds
    results: list[dict] = []

    async def client(k: int):
        i = 0
        while time.monotonic() < stop:
            results.append(await stream(session, args.port, f"load-{k}-{i}", tokens(rng, 512), 128))
            i += 1
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        await asyncio.gather(*(client(k) for k in range(32)))
    return results


async def openloop(args) -> list[dict]:
    """One block per rate: Poisson arrivals for --seconds, then wait for all of them."""
    import aiohttp
    blocks = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        for rate in args.rates:
            rng = random.Random(f"{args.seed}-{rate}")
            pending, started = [], time.monotonic_ns()
            arrival = time.monotonic()
            stop = arrival + args.seconds
            i = 0
            while True:
                arrival += rng.expovariate(rate)
                if arrival >= stop:
                    break
                await asyncio.sleep(max(0.0, arrival - time.monotonic()))
                pending.append(asyncio.create_task(
                    stream(session, args.port, f"open-{rate:g}-{i}", tokens(rng, 512), 128)))
                i += 1
            results = list(await asyncio.gather(*pending))
            blocks.append({"rate": rate, "seconds": args.seconds, "started_ns": started,
                           "ended_ns": time.monotonic_ns(), "results": results})
            await asyncio.sleep(2.0)
    return blocks


BURNER = r"""
import sys, time, torch
x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
print("ready", flush=True)
for line in sys.stdin:
    start = time.monotonic_ns()
    stop = start + int(float(line) * 1e6)
    while time.monotonic_ns() < stop:
        y = x @ x
        torch.cuda.synchronize()
    print(start, time.monotonic_ns(), flush=True)
"""


def engine_core(server_pid: int) -> int:
    """The pid of the server's EngineCore process."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            with contextlib.suppress(OSError):
                with open(f"/proc/{entry}/stat") as stat:
                    children.setdefault(int(stat.read().rsplit(")", 1)[1].split()[1]), []).append(int(entry))
    stack = [server_pid]
    while stack:
        pid = stack.pop()
        with contextlib.suppress(OSError):
            if Path(f"/proc/{pid}/comm").read_text().startswith("VLLM::EngineCor"):
                return pid
        stack += children.get(pid, [])
    raise RuntimeError("no EngineCore process under the server")


def schedule(args) -> list[tuple[float, str]]:
    """Event start times (seconds after the decodes start) and causes, from --seed."""
    rng = random.Random(f"multicause-{args.seed}")
    causes = [c for c in args.causes for _ in range(args.events_per_cause)]
    rng.shuffle(causes)
    events, at = [], 2.0
    for cause in causes:
        events.append((at, cause))
        at += rng.uniform(1.5, 3.0)
    return events


async def multicause(args) -> dict:
    import aiohttp
    rng = random.Random(args.seed)
    burner = subprocess.Popen([str(args.python), "-c", BURNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.device})
    engine = engine_core(args.server_pid)
    events, tasks = [], []
    try:
        if burner.stdout.readline().strip() != "ready":
            raise RuntimeError("GPU burner did not start")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
            decodes = [asyncio.create_task(stream(session, args.port, f"decode-{i}", tokens(rng, 128),
                                                  args.decode_tokens)) for i in range(16)]
            origin = time.monotonic()
            for j, (at, cause) in enumerate(schedule(args)):
                await asyncio.sleep(max(0.0, origin + at - time.monotonic()))
                if cause == "interference":
                    tasks.append(asyncio.create_task(
                        stream(session, args.port, f"long-{j}", tokens(rng, 6000), 1)))
                    events.append({"cause": cause, "request": f"long-{j}"})
                elif cause == "gpu_slowdown":
                    burner.stdin.write(f"{args.burst_ms}\n")
                    burner.stdin.flush()
                    events.append({"cause": cause})
                else:
                    start = time.monotonic_ns()
                    os.kill(engine, signal.SIGSTOP)
                    try:
                        await asyncio.sleep(args.stall_ms / 1000)
                    finally:
                        os.kill(engine, signal.SIGCONT)
                    events.append({"cause": cause, "start_ns": start, "end_ns": time.monotonic_ns()})
            longs = {r["id"]: r for r in await asyncio.gather(*tasks)}
            results = list(await asyncio.gather(*decodes)) + list(longs.values())
    finally:
        burner.stdin.close()
        bursts = [list(map(int, line.split())) for line in burner.stdout]
        burner.wait(timeout=60)
    for event in events:
        if event["cause"] == "interference":
            result = longs[event["request"]]
            event.update(start_ns=result["sent_ns"], end_ns=result["chunk_ns"][0])
        elif event["cause"] == "gpu_slowdown":
            event["start_ns"], event["end_ns"] = bursts.pop(0)
    return {"seed": args.seed, "events": events, "streams": results}


async def stream_warmup(port: int) -> None:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        await stream(session, port, "warmup", list(range(1000, 1064)), 8)


@contextlib.contextmanager
def load_samples(samples: list, every: float = 5.0):
    """Record [monotonic ns, 1-min load, 5-min load] while the block runs.

    The host may be shared, and its load changes host-side timings."""
    done = threading.Event()

    def sample():
        while True:
            load = Path("/proc/loadavg").read_text().split()
            samples.append([time.monotonic_ns(), float(load[0]), float(load[1])])
            if done.wait(every):
                return
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    try:
        yield
    finally:
        done.set()
        worker.join()


@contextlib.contextmanager
def gpu_lock(path: Path | None):
    if path is None:
        yield
        return
    with open(path, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["interference", "overhead", "openloop", "multicause"])
    parser.add_argument("--mode", choices=["off", "plugin", "light", "trace"], default="trace")
    parser.add_argument("--out", type=Path, required=True, help="new run directory")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=None,
                        help="load duration: 60 for overhead, 45 per rate for openloop")
    parser.add_argument("--rates", default="8,16,24", help="openloop request rates per second")
    parser.add_argument("--round", type=int, default=1, help="label recorded in client.json")
    parser.add_argument("--causes", default="interference,gpu_slowdown,host_delay", help="multicause causes")
    parser.add_argument("--events-per-cause", type=int, default=3)
    parser.add_argument("--burst-ms", type=float, default=300.0, help="multicause GPU burst length")
    parser.add_argument("--stall-ms", type=float, default=200.0, help="multicause EngineCore stop length")
    parser.add_argument("--decode-tokens", type=int, default=6000, help="multicause tokens per decode")
    parser.add_argument("--model", default=os.environ.get("REQSCOPE_BENCH_MODEL", MODEL),
                        help="Hugging Face model id, or a local model directory (e.g. a cached snapshot)")
    parser.add_argument("--revision", default=os.environ.get("REQSCOPE_BENCH_REVISION", REVISION),
                        help="Hub revision for a model id")
    parser.add_argument("--hf-home", type=Path, default=None, help="HF_HOME for the server (default: inherited)")
    parser.add_argument("--offline", action="store_true", help="use only the local Hugging Face cache")
    parser.add_argument("--cache-root", type=Path, default=None, help="vLLM/torch compile cache directory")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="interpreter of the environment with vLLM and the plugin")
    parser.add_argument("--cuda", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
                        help="CUDA toolkit root that holds libcupti")
    parser.add_argument("--build", type=Path, default=ROOT / "build", help="reqscope build directory")
    parser.add_argument("--lock", type=Path, default=os.environ.get("REQSCOPE_GPU_LOCK"),
                        help="flock this file for the whole run (default: no lock)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES for the server")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--ring-bytes", type=int, default=8 << 30, help="collector ring size on disk")
    args = parser.parse_args(argv)
    args.rates = [float(r) for r in args.rates.split(",")]
    args.causes = args.causes.split(",")
    if set(args.causes) - {"interference", "gpu_slowdown", "host_delay"}:
        parser.error("--causes takes interference, gpu_slowdown and host_delay")
    if args.seconds is None:
        args.seconds = 45.0 if args.experiment == "openloop" else 60.0
    if not 0 < args.gpu_memory_utilization <= 0.95:
        parser.error("--gpu-memory-utilization must be in (0, 0.95]")
    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run = {"interference": interference, "overhead": overhead, "openloop": openloop,
           "multicause": multicause}[args.experiment]
    load: list = []
    with gpu_lock(args.lock), load_samples(load):
        process = server(args, args.out)
        try:
            wait_ready(process, args.port)
            args.server_pid = process.pid
            asyncio.run(stream_warmup(args.port))
            started = time.monotonic_ns()
            results = asyncio.run(run(args))
            ended = time.monotonic_ns()
            if args.mode != "off" and not flush_collectors(process.pid):
                raise RuntimeError("no collector control socket found under the server")
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    (args.out / "client.json").write_text(json.dumps({
        "experiment": args.experiment, "mode": args.mode, "seed": args.seed, "round": args.round,
        "model": args.model, "revision": args.revision, "gpu_memory_utilization": args.gpu_memory_utilization,
        "started_ns": started, "ended_ns": ended, "loadavg": load, "results": results}))
    print(args.out)


if __name__ == "__main__":
    main()
