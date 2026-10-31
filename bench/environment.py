"""Print the software and hardware versions a benchmark run depends on, as JSON.

    python bench/environment.py [--cuda /usr/local/cuda] > results/environment.json

Run it with the interpreter that serves vLLM. Uses only the standard library
plus nvidia-smi and the CUDA toolkit's version.json.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi() -> dict:
    try:
        line = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                               "--format=csv,noheader"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    name, driver, memory = [field.strip() for field in line.splitlines()[0].split(",")]
    return {"gpu": name, "driver": driver, "gpu_memory": memory}


def toolkit(cuda: Path) -> dict:
    info = {"cuda_root": str(cuda)}
    try:
        data = json.loads((cuda / "version.json").read_text())
        info["cuda_toolkit"] = data["cuda"]["version"]
        info["cupti"] = data.get("cuda_cupti", {}).get("version")
    except (OSError, KeyError, ValueError):
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=Path, default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")))
    args = parser.parse_args()
    torch_cuda = None
    try:
        import torch
        torch_cuda = torch.version.cuda
    except ImportError:
        pass
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    print(json.dumps({
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("vllm", "torch", "transformers", "aiohttp", "reqscope-vllm")},
        "torch_cuda": torch_cuda,
        **toolkit(args.cuda.resolve()),
        **nvidia_smi(),
        "kernel": platform.release(),
        "cpu": cpu,
        "cpus": os.cpu_count(),
    }, indent=2))


if __name__ == "__main__":
    main()
