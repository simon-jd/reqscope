"""Where trace mode's host time goes, from runs of bench/interference.py.

    python bench/trace_cost.py results/raw/tracecost-light results/raw/tracecost-trace ...

For each run (an openloop run at one rate), over the client's load window:
- latency: TTFT and TPOT percentiles (bench/score.py openloop);
- collector: the collector's own sites from its CollectorStats records
  (calls per second, microseconds per call, longest call, CPU cores, share
  of calls on an engine thread), the writer's and probe's CPU, and CUPTI
  records per second;
- threads: CPU, run-queue wait and efficiency-core share per EngineCore
  thread (bench/thread_stats.py);
- apis: per recorded CUDA API, calls and duration percentiles: CUPTI does
  its per-launch work inside these calls, so light against trace shows what
  per-kernel tracing adds to each;
- slow_launches: kernel and copy launches (not graph launches) over 100 us,
  how many came in runs of consecutive slow launches, and the share of slow
  and of other launches that overlapped a writer busy interval
  (CollectorActivity);
- graph_launch_stalls: graph launches over 1 ms, and the time between them.

Reads only ApiCall, CollectorStats and CollectorActivity frames, so a
multi-GB trace capture takes about a minute. Uses only the standard library.
"""
from __future__ import annotations

import bisect
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import mean_load, openloop  # noqa: E402
from thread_stats import efficiency_cores, summarize as thread_summary  # noqa: E402

KEEP = (b'"ApiCall"', b'"CollectorStats"', b'"CollectorActivity"')


def records(capture: Path):
    for path in sorted(capture.glob("segment-*.rqs")):
        data = path.read_bytes()
        offset = 8
        while offset + 8 <= len(data):
            length, _ = struct.unpack_from("<II", data, offset)
            payload = data[offset + 8:offset + 8 + length]
            offset += 8 + length
            if any(key in payload for key in KEEP):
                record = json.loads(payload)
                if record.get("type") in ("ApiCall", "CollectorStats", "CollectorActivity"):
                    yield record


def quantile(values: list[int], q: float) -> int:
    return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]


def collector_sites(stats: list[dict], begin: int, end: int) -> dict | None:
    window = [r for r in stats if begin <= r["timestamp"] <= end]
    if len(window) < 2:
        return None
    a, z = window[0], window[-1]
    seconds = (z["timestamp"] - a["timestamp"]) / 1e9
    sites = {}
    for name, last in z["sites"].items():
        first = a["sites"].get(name, {"calls": 0, "ns": 0, "engine_calls": 0})
        calls = last["calls"] - first["calls"]
        if not calls:
            continue
        ns = last["ns"] - first["ns"]
        sites[name] = {"per_s": round(calls / seconds, 1), "us_per_call": round(ns / calls / 1e3, 2),
                       "max_us": round(last["max_ns"] / 1e3, 1), "cores": round(ns / 1e9 / seconds, 4),
                       "engine_share": round((last["engine_calls"] - first["engine_calls"]) / calls, 2)}
    per_s = lambda key: round((z[key] - a[key]) / seconds)
    return {"seconds": round(seconds, 1), "sites": sites,
            "writer_cpu_cores": round((z["writer_cpu_ns"] - a["writer_cpu_ns"]) / 1e9 / seconds, 3),
            "probe_cpu_cores": round((z["probe_cpu_ns"] - a["probe_cpu_ns"]) / 1e9 / seconds, 4),
            "cupti_records_per_s": per_s("cupti_records"), "cupti_bytes_per_s": per_s("cupti_bytes"),
            "timestamp_callbacks_per_s": per_s("timestamp_callbacks")}


def summarize(run: Path, ecores: set[int]) -> dict:
    client = json.loads((run / "client.json").read_text())
    begin, end = client["started_ns"], client["ended_ns"]
    calls, stats, busy = [], [], []
    for record in records(run / "capture"):
        if record["type"] == "ApiCall":
            if begin <= record["start"] <= end:
                calls.append((record["start"], record["end"], record["name"].split("_v")[0]))
        elif record["type"] == "CollectorStats":
            stats.append(record)
        else:
            busy.append((record["start"], record["end"]))
    calls.sort()
    apis = {}
    for name in sorted({c[2] for c in calls}):
        durations = sorted(e - s for s, e, n in calls if n == name)
        apis[name] = {"calls": len(durations), "p50_us": round(quantile(durations, 0.5) / 1e3, 1),
                      "p90_us": round(quantile(durations, 0.9) / 1e3, 1),
                      "p99_us": round(quantile(durations, 0.99) / 1e3, 1),
                      "max_us": round(durations[-1] / 1e3, 1), "total_s": round(sum(durations) / 1e9, 3)}
    launches = [c for c in calls if ("Launch" in c[2] or "Memcpy" in c[2]) and "Graph" not in c[2]]
    slow = [c[1] - c[0] > 100_000 for c in launches]
    in_runs, run_length, longest = 0, 0, 0
    for is_slow in slow + [False]:
        if is_slow:
            run_length += 1
        else:
            if run_length >= 3:
                in_runs += run_length
            longest, run_length = max(longest, run_length), 0
    busy.sort()
    starts = [b[0] for b in busy]

    def during_writer(call) -> bool:
        i = bisect.bisect_right(starts, call[1]) - 1
        return i >= 0 and busy[i][1] > call[0]
    slow_calls = [c for c, s in zip(launches, slow) if s]
    other_calls = [c for c, s in zip(launches, slow) if not s]
    share = lambda group: round(sum(map(during_writer, group)) / len(group), 2) if group and busy else None
    stalls = [s for s, e, n in calls if "GraphLaunch" in n and e - s > 1_000_000]
    stall_us = sorted(e - s for s, e, n in calls if "GraphLaunch" in n and e - s > 1_000_000)
    between = sorted(b - a for a, b in zip(stalls, stalls[1:]))
    seconds = (end - begin) / 1e9
    return {"run": run.name, "mode": client["mode"], "collector_env": client.get("collector_env", {}),
            "load1_mean": mean_load(client.get("loadavg", []), begin, end), "window_s": round(seconds, 1),
            "latency": {k: v for k, v in openloop([run], [])["runs"][0].items()
                        if k.startswith(("ttft", "tpot", "output"))},
            "collector": collector_sites(stats, begin, end),
            "threads": thread_summary(run, ecores)["threads"],
            "apis": apis,
            "slow_launches": {"launches": len(launches), "over_100us": len(slow_calls),
                              "in_runs_of_3_or_more": in_runs, "longest_run": longest,
                              "extra_s": round(sum(c[1] - c[0] for c in slow_calls) / 1e9, 3),
                              "slow_share_during_writer_busy": share(slow_calls),
                              "other_share_during_writer_busy": share(other_calls)},
            "graph_launch_stalls": {"over_1ms": len(stalls), "per_s": round(len(stalls) / seconds, 2),
                                    "median_us": round(quantile(stall_us, 0.5) / 1e3) if stall_us else None,
                                    "median_between_s": round(quantile(between, 0.5) / 1e9, 2) if between else None}}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ecores = efficiency_cores()
    print(json.dumps([summarize(Path(run), ecores) for run in sys.argv[1:]], indent=1))


if __name__ == "__main__":
    main()
