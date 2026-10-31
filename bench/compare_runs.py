"""Compare interference runs scored by score.py, e.g. with and without CUPTI.

    python bench/compare_runs.py [--load load.csv] results/raw/ablation-trace results/raw/ablation-plugin

For each run directory (client.json and the blame.json written by
`bench/score.py interference`): the collector mode, the per-step scores of
score_steps.py, blame's split of the excess latency, and the median gap,
GPU busy time and host span of the slow steps blame attributed to a long
prompt and of those it left unexplained. With CUPTI off ("plugin" mode)
there is no GPU record, so GPU busy time is 0 everywhere. Each run also
gets the host's mean 1-minute load average over the run, from client.json
or from a bench/sample_load.py CSV (--load); null when neither covers it.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import load_samples, mean_load  # noqa: E402
from score_steps import score  # noqa: E402


def medians(rows: list[dict]) -> dict:
    if not rows:
        return {"steps": 0}
    fields = {"median_gap_ms": "gap_ms_median", "median_gpu_busy_ms": "gpu_busy_ms", "median_host_ms": "host_ms"}
    return {"steps": len(rows), **{name: round(statistics.median(r[key] for r in rows), 2)
                                   for name, key in fields.items()}}


def compare(run: Path, load: list) -> dict:
    client = json.loads((run / "client.json").read_text())
    blame = json.loads((run / "blame.json").read_text())
    steps = score(run)
    excess = {"long": 0, "other": 0, "unexplained": 0}
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        for gap in row["slow_gaps"]:
            for culprit in gap["blamed"]:
                excess["long" if "long-" in culprit["req_id"] else "other"] += culprit["excess_ns"]
            if not gap["blamed"]:
                excess["unexplained"] += gap["excess_ns"]
    total = sum(excess.values()) or 1
    blamed = [r for r in steps["steps"] if r["blamed_long_prompts"]]
    unexplained = [r for r in steps["steps"] if not r["blamed_long_prompts"] and not r["blamed_other_requests"]]
    return {
        "run": run.name, "mode": client["mode"],
        "load1_mean": mean_load(client.get("loadavg") or load, client["started_ns"], client["ended_ns"]),
        "gpu_activities": blame["summary"]["gpu_activities"],
        **{k: v for k, v in steps.items() if k not in ("steps", "long_prompt_windows_ns")},
        "excess_share": {k: round(v / total, 4) for k, v in excess.items()},
        "top_culprits": [next((w for w in steps["long_prompt_windows_ns"] if w in c["req_id"]), c["req_id"])
                         for c in blame["culprits"][:6]],
        "blamed_steps": medians(blamed),
        "unexplained_steps": medians(unexplained),
        "trace_status": blame.get("trace", {}).get("status"),
        "warnings": blame.get("warnings"),
    }


def main() -> None:
    args, load = sys.argv[1:], []
    if args[:1] == ["--load"]:
        load, args = load_samples(args[1]), args[2:]
    if not args:
        sys.exit(__doc__)
    print(json.dumps([compare(Path(p), load) for p in args], indent=2))


if __name__ == "__main__":
    main()
