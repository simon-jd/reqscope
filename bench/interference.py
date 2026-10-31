"""Can reqscope explain why decode tokens got slow? And what does it cost?

    python bench/interference.py interference --mode trace --out results/raw/interference
    python bench/interference.py overhead --mode off|light|trace --out results/raw/overhead-off

interference
    16 decode streams generate 4,000 tokens each. Starting 2 s in, a
    6,000-token prompt arrives every 3 s, six in all. With chunked prefill
    each prompt's chunks share engine steps with the decodes and stretch
    their inter-token gaps. The client records when each long prompt was sent
    and when its first token came back; the analyzer never sees this and has
    to find the culprits from step membership and GPU time alone.

overhead
    32 closed-loop clients (512-token prompts, 128 output tokens) for 60 s,
    with no collector, the light collector or the trace collector.

Runs vLLM 0.31.0 from this project's environment (see README), holding the
machine-wide GPU lock for the whole run.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".runtime/reqscope-cpython-3.12.13-021044895e95/env/bin/python"
MODEL = ROOT / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LIBRARY = ROOT / "build/libreqscope.so"
CUPTI = Path("/usr/local/cuda-13.2/lib64/libcupti.so.13")
PORT = 8123


def server(mode: str, out: Path) -> subprocess.Popen:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REQSCOPE_", "OTEL_", "VLLM_"))}
    cache = ROOT / ".cache/runtime"
    env.update({"HF_HOME": str(ROOT / ".cache/huggingface"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "XDG_CACHE_HOME": str(cache), "VLLM_CACHE_ROOT": str(cache / "vllm"),
                "VLLM_USE_V2_MODEL_RUNNER": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_NO_USAGE_STATS": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0",
                "LD_PRELOAD": str(CUPTI)})
    if mode != "off":
        env.update({"CUDA_INJECTION64_PATH": str(LIBRARY), "REQSCOPE_LIBRARY": str(LIBRARY),
                    "REQSCOPE_ENABLED": "1", "VLLM_PLUGINS": "reqscope", "REQSCOPE_MODE": mode,
                    "REQSCOPE_OUTPUT": str(out / "capture"), "REQSCOPE_RING_SECONDS": "3600",
                    "REQSCOPE_RING_BYTES": str(8 << 30), "REQSCOPE_QUEUE_SLOTS": "1024",
                    "REQSCOPE_CUPTI_BUFFERS": "256"})
    command = [str(PYTHON), "-B", str(ROOT / ".venv/bin/vllm"), "serve", str(MODEL),
               "--host", "127.0.0.1", "--port", str(PORT), "--served-model-name", "m",
               "--enable-request-id-headers", "--distributed-executor-backend", "uni", "--async-scheduling",
               "--dtype", "bfloat16", "--max-model-len", "8192", "--max-num-seqs", "32",
               "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.6",
               "--no-enable-prefix-caching", "--seed", "42"]
    log = open(out / "server.log", "w")
    return subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def wait_ready(process: subprocess.Popen, timeout: float = 600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited during startup; see server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
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


async def stream(session, request_id: str, prompt: list[int], max_tokens: int) -> dict:
    body = {"model": "m", "prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0.0, "stream": True}
    sent, chunks = time.monotonic_ns(), []
    async with session.post(f"http://127.0.0.1:{PORT}/v1/completions", json=body,
                            headers={"X-Request-Id": request_id}) as response:
        async for line in response.content:
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                chunks.append(time.monotonic_ns())
    return {"id": request_id, "sent_ns": sent, "chunk_ns": chunks}


def tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1000, 100_000) for _ in range(n)]


async def interference(seed: int) -> list[dict]:
    import aiohttp
    rng = random.Random(seed)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        decodes = [asyncio.create_task(stream(session, f"decode-{i}", tokens(rng, 128), 4000)) for i in range(16)]
        longs = []
        await asyncio.sleep(2.0)
        for j in range(6):
            longs.append(asyncio.create_task(stream(session, f"long-{j}", tokens(rng, 6000), 1)))
            await asyncio.sleep(3.0)
        return list(await asyncio.gather(*decodes, *longs))


async def overhead(seed: int, seconds: float = 60.0) -> list[dict]:
    import aiohttp
    rng = random.Random(seed)
    stop = time.monotonic() + seconds
    results: list[dict] = []

    async def client(k: int):
        i = 0
        while time.monotonic() < stop:
            results.append(await stream(session, f"load-{k}-{i}", tokens(rng, 512), 128))
            i += 1
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        await asyncio.gather(*(client(k) for k in range(32)))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["interference", "overhead"])
    parser.add_argument("--mode", choices=["off", "light", "trace"], default="trace")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with open("/var/tmp/rtx4090.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = server(args.mode, args.out)
        try:
            wait_ready(process)
            asyncio.run(stream_warmup())
            started = time.monotonic_ns()
            results = asyncio.run(interference(args.seed) if args.experiment == "interference" else overhead(args.seed))
            ended = time.monotonic_ns()
            if args.mode != "off" and not flush_collectors(process.pid):
                raise RuntimeError("no collector control socket found under the server")
        finally:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    (args.out / "client.json").write_text(json.dumps({"experiment": args.experiment, "mode": args.mode,
        "seed": args.seed, "started_ns": started, "ended_ns": ended, "results": results}))
    print(args.out)


async def stream_warmup() -> None:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        await stream(session, "warmup", list(range(1000, 1064)), 8)


if __name__ == "__main__":
    main()
