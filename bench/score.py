"""Score runs from bench/interference.py.

    python bench/score.py interference results/raw/interference
    python bench/score.py overhead results/raw/overhead-{off,light,trace}
    python bench/score.py openloop [--load load.csv] results/raw/openloop-*

interference
    Runs `reqscope blame` on the capture, then checks
    blame against the client's ground truth: a long prompt interferes from
    when it was sent until its first token came back. Every slow decode gap
    inside such a window should be blamed on that prompt, and nothing outside
    the windows should be blamed on a long prompt.

overhead
    Throughput, TTFT and TPOT per collector mode, from the client's timings.

openloop
    Per run and rate: throughput, TTFT and TPOT (p50 and p99). Then, per mode
    and rate, the median over rounds with the range (min, max), and the change
    of each median against mode "off". Each run also gets the host's mean
    1-minute load average over its load window, from client.json or, for
    runs that did not record it, from a bench/sample_load.py CSV (--load).

`reqscope` is $REQSCOPE_BIN, else build/reqscope. Reading a large capture
needs more than the reader's default 2 GiB budget, so the trace budget
defaults to $REQSCOPE_TRACE_BUDGET_MIB or 8192 MiB.
"""
from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("REQSCOPE_BIN", ROOT / "build/reqscope"))
TRACE_BUDGET_MIB = os.environ.get("REQSCOPE_TRACE_BUDGET_MIB", "8192")


def reqscope(*args: str) -> dict:
    env = {**os.environ, "REQSCOPE_TRACE_BUDGET_MIB": TRACE_BUDGET_MIB}
    return json.loads(subprocess.run([str(BIN), *args], check=True, capture_output=True, text=True,
                                     env=env).stdout)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def ms(ns: float) -> float:
    return round(ns / 1e6, 2)


def interference(run: Path) -> dict:
    client = json.loads((run / "client.json").read_text())
    started = time.monotonic()
    blame = reqscope("blame", str(run / "capture"))
    analysis_seconds = time.monotonic() - started
    (run / "blame.json").write_text(json.dumps(blame))
    windows = {r["id"]: (r["sent_ns"], r["chunk_ns"][0]) for r in client["results"] if r["id"].startswith("long-")}

    def long_of(req_id: str) -> str | None:
        return next((w for w in windows if w in req_id), None)

    gaps = inside = blamed_right = false_blame = 0
    excess = {"long": 0, "other": 0, "unexplained": 0}
    producer_busy, typical_gap = [], []
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        typical_gap.append(row["median_gap_ns"])
        for gap in row["slow_gaps"]:
            gaps += 1
            overlapping = {w for w, (a, b) in windows.items() if gap["begin_ns"] < b and gap["end_ns"] > a}
            culprits = {long_of(x["req_id"]) or x["req_id"] for x in gap["blamed"]}
            inside += bool(overlapping)
            blamed_right += bool(overlapping & culprits)
            false_blame += bool(not overlapping and any(c in windows for c in culprits))
            for x in gap["blamed"]:
                excess["long" if long_of(x["req_id"]) else "other"] += x["excess_ns"]
            if not gap["blamed"]:
                excess["unexplained"] += gap["excess_ns"]
            producer_busy += [s["gpu_busy_ns"] for s in gap["steps"]]
    total = sum(excess.values()) or 1
    ranked = [long_of(c["req_id"]) or c["req_id"] for c in blame["culprits"]]
    summary = blame["summary"]
    capture_bytes = sum(f.stat().st_size for f in (run / "capture").iterdir())
    return {
        "decode_slow_gaps": gaps,
        "slow_gaps_inside_a_long_prompt_window": inside,
        "of_those_blamed_on_that_prompt": blamed_right,
        "blamed_on_a_long_prompt_outside_every_window": false_blame,
        "excess_share": {k: round(v / total, 4) for k, v in excess.items()},
        "top_culprits": ranked[:len(windows)],
        "top_culprits_are_the_long_prompts": sorted(ranked[:len(windows)]) == sorted(windows),
        "typical_decode_gap_ms": ms(statistics.median(typical_gap)),
        "slow_gap_producer_gpu_busy_ms": {"p50": ms(percentile(producer_busy, 0.5)),
                                          "p90": ms(percentile(producer_busy, 0.9))} if producer_busy else None,
        "clocks": blame["clocks"],
        **summary,
        "capture_bytes": capture_bytes,
        "capture_seconds": (client["ended_ns"] - client["started_ns"]) / 1e9,
        "blame_seconds": round(analysis_seconds, 1),
    }


def latency(done: list[dict], seconds: float) -> dict:
    ttft = [r["chunk_ns"][0] - r["sent_ns"] for r in done]
    tpot = [(r["chunk_ns"][-1] - r["chunk_ns"][0]) / (len(r["chunk_ns"]) - 1) for r in done]
    return {"requests": len(done),
            "output_tokens_per_s": round(sum(len(r["chunk_ns"]) for r in done) / seconds, 1),
            "ttft_p50_ms": ms(percentile(ttft, 0.5)), "ttft_p99_ms": ms(percentile(ttft, 0.99)),
            "tpot_p50_ms": ms(percentile(tpot, 0.5)), "tpot_p99_ms": ms(percentile(tpot, 0.99))}


def overhead(runs: list[Path]) -> list[dict]:
    rows = []
    for run in runs:
        client = json.loads((run / "client.json").read_text())
        done = [r for r in client["results"] if len(r["chunk_ns"]) >= 2]
        seconds = (client["ended_ns"] - client["started_ns"]) / 1e9
        ttft = [r["chunk_ns"][0] - r["sent_ns"] for r in done]
        tpot = [(r["chunk_ns"][-1] - r["chunk_ns"][0]) / 127 for r in done]
        rows.append({"mode": client["mode"], "requests": len(done),
                     "output_tokens_per_s": round(128 * len(done) / seconds, 1),
                     "ttft_p50_ms": ms(percentile(ttft, 0.5)), "ttft_p99_ms": ms(percentile(ttft, 0.99)),
                     "tpot_p50_ms": ms(percentile(tpot, 0.5)), "tpot_p99_ms": ms(percentile(tpot, 0.99))})
    return rows


METRICS = ("output_tokens_per_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms")


def load_samples(path: str | None) -> list:
    if not path:
        return []
    return [[int(row[0]), float(row[2]), float(row[3])]
            for row in (line.split(",") for line in Path(path).read_text().splitlines()) if row[0]]


def mean_load(samples: list, begin: int, end: int) -> float | None:
    window = [s[1] for s in samples if begin <= s[0] <= end]
    return round(statistics.mean(window), 2) if window else None


def openloop(runs: list[Path], load: list) -> dict:
    rows = []
    for run in runs:
        client = json.loads((run / "client.json").read_text())
        samples = client.get("loadavg") or load
        for block in client["results"]:
            done = [r for r in block["results"] if len(r["chunk_ns"]) >= 2]
            rows.append({"run": run.name, "mode": client["mode"], "round": client["round"], "rate": block["rate"],
                         "sent": len(block["results"]),
                         **latency(done, (block["ended_ns"] - block["started_ns"]) / 1e9),
                         "load1_mean": mean_load(samples, block["started_ns"], block["ended_ns"])})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["rate"], row["mode"])].append(row)
    summary = []
    for (rate, mode), group in sorted(groups.items(), key=lambda item: (item[0][0], MODES.index(item[0][1]))):
        entry = {"rate": rate, "mode": mode, "rounds": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            entry[metric] = {"median": round(statistics.median(values), 2), "min": min(values), "max": max(values)}
        summary.append(entry)
    for entry in summary:
        base = next((e for e in summary if e["rate"] == entry["rate"] and e["mode"] == "off"), None)
        if base and entry is not base:
            entry["change_vs_off"] = {m: round(entry[m]["median"] / base[m]["median"] - 1, 4) for m in METRICS}
    return {"runs": rows, "summary": summary}


MODES = ["off", "plugin", "light", "trace"]


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("interference", "overhead", "openloop"):
        sys.exit(__doc__)
    args = sys.argv[2:]
    load = []
    if args[:1] == ["--load"]:
        load, args = load_samples(args[1]), args[2:]
    runs = [Path(p) for p in args]
    result = {"interference": lambda: interference(runs[0]), "overhead": lambda: overhead(runs),
              "openloop": lambda: openloop(runs, load)}[sys.argv[1]]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
