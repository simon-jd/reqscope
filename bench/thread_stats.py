"""Summarize the EngineCore thread samples of bench/interference.py runs.

    python bench/thread_stats.py results/raw/openloop-trace-r1 ...

For each run, over the client's load window, per thread (the EngineCore
main thread is "engine"; other threads by name, numbered when repeated):
- cpu: CPU time divided by the window, in cores;
- wait: time spent runnable but waiting for a CPU (schedstat run delay),
  divided by the window;
- ecore: share of samples whose last CPU is an efficiency core, on hybrid
  Intel hosts that list them in /sys/devices/cpu_atom/cpus;
- involuntary: involuntary context switches per second.

Threads under 0.5% of a core are summed into "other". Uses only the
standard library.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


def cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            low, _, high = part.partition("-")
            cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def efficiency_cores() -> set[int]:
    try:
        return cpu_list(Path("/sys/devices/cpu_atom/cpus").read_text())
    except OSError:
        return set()


def summarize(run: Path, ecores: set[int]) -> dict:
    client = json.loads((run / "client.json").read_text())
    threads = client.get("threads") or {}
    samples = threads.get("samples") or []
    if not samples:
        return {"run": run.name, "mode": client["mode"], "threads": {}}
    begin, end = client["started_ns"], client["ended_ns"]
    first, last, names, cpus = {}, {}, {}, {}
    for t, rows in samples:
        for tid, (exec_ns, delay_ns, cpu, switches, name) in rows.items():
            names[tid] = name
            if t <= begin or tid not in first:
                first[tid] = (exec_ns, delay_ns, switches)
            if t <= end:
                last[tid] = (exec_ns, delay_ns, switches)
                if t >= begin:
                    cpus.setdefault(tid, Counter())[cpu in ecores] += 1
    seconds = (end - begin) / 1e9
    pid, seen, out, other = str(threads["pid"]), Counter(), {}, [0.0, 0.0, 0.0]
    for tid in sorted(last, key=int):
        cpu = (last[tid][0] - first[tid][0]) / 1e9 / seconds
        wait = (last[tid][1] - first[tid][1]) / 1e9 / seconds
        involuntary = (last[tid][2] - first[tid][2]) / seconds
        if tid != pid and cpu < 0.005:
            other = [other[0] + cpu, other[1] + wait, other[2] + involuntary]
            continue
        name = "engine" if tid == pid else names[tid]
        seen[name] += 1
        key = name if seen[name] == 1 else f"{name}#{seen[name]}"
        on = cpus.get(tid, Counter())
        out[key] = {"cpu": round(cpu, 3), "wait": round(wait, 4), "involuntary": round(involuntary, 1),
                    "ecore": round(on[True] / max(1, sum(on.values())), 3) if ecores else None}
    out["other"] = {"cpu": round(other[0], 3), "wait": round(other[1], 4), "involuntary": round(other[2], 1),
                    "ecore": None}
    return {"run": run.name, "mode": client["mode"], "seconds": round(seconds, 1), "threads": out}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ecores = efficiency_cores()
    print(json.dumps([summarize(Path(run), ecores) for run in sys.argv[1:]], indent=1))


if __name__ == "__main__":
    main()
