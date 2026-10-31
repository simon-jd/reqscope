"""Score an interference run per engine step instead of per token gap.

    python bench/score_steps.py results/raw/interference

Reads RUN/blame.json (written by `bench/score.py interference`) and the
client's ground truth in RUN/client.json. The 16 decodes run in lockstep,
so one slow engine step shows up as up to 16 slow gaps; this groups the
decode slow gaps by the step that produced the token and reports, per step:
the long-prompt windows its gaps overlap, whom blame named, its GPU busy
time and its gap. The long prompts' windows (sent until first token) are
included so the result can be checked against blame.json alone.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys


def ms(ns: float) -> float:
    return round(ns / 1e6, 2)


def score(run: Path) -> dict:
    client = json.loads((run / "client.json").read_text())
    blame = json.loads((run / "blame.json").read_text())
    windows = {r["id"]: (r["sent_ns"], r["chunk_ns"][0]) for r in client["results"] if r["id"].startswith("long-")}

    def long_of(req_id: str) -> str | None:
        return next((w for w in windows if w in req_id), None)

    steps: dict[int, dict] = {}
    gaps = 0
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        for gap in row["slow_gaps"]:
            gaps += 1
            for step in gap["steps"]:
                entry = steps.setdefault(step["step_id"], {"gaps": [], "busy": set(), "windows": set(),
                                                           "long": set(), "other": set()})
                entry["gaps"].append(gap["gap_ns"])
                entry["busy"].add(step["gpu_busy_ns"])
                entry["windows"] |= {w for w, (a, b) in windows.items() if gap["begin_ns"] < b and gap["end_ns"] > a}
                for culprit in gap["blamed"]:
                    name = long_of(culprit["req_id"])
                    (entry["long"] if name else entry["other"]).add(name or culprit["req_id"])
    rows = [{"step_id": step_id,
             "slow_gaps": len(e["gaps"]),
             "gap_ms_median": ms(statistics.median(e["gaps"])),
             "gpu_busy_ms": ms(max(e["busy"])),
             "long_prompt_windows": sorted(e["windows"]),
             "blamed_long_prompts": sorted(e["long"]),
             "blamed_other_requests": len(e["other"])}
            for step_id, e in sorted(steps.items())]
    inside = [r for r in rows if r["long_prompt_windows"]]
    unexplained = [r for r in rows if not r["blamed_long_prompts"] and not r["blamed_other_requests"]]
    return {
        "long_prompt_windows_ns": {k: list(v) for k, v in sorted(windows.items())},
        "decode_slow_gaps": gaps,
        "slow_steps": len(rows),
        "slow_steps_inside_a_long_prompt_window": len(inside),
        "of_those_blamed_on_that_prompt": sum(bool(set(r["long_prompt_windows"]) & set(r["blamed_long_prompts"]))
                                              for r in inside),
        "slow_steps_blamed_on_a_long_prompt_outside_every_window":
            sum(bool(r["blamed_long_prompts"]) and not r["long_prompt_windows"] for r in rows),
        "slow_steps_blamed_only_on_other_requests":
            sum(bool(r["blamed_other_requests"]) and not r["blamed_long_prompts"] for r in rows),
        "slow_steps_unexplained": len(unexplained),
        "of_those_inside_a_long_prompt_window": sum(bool(r["long_prompt_windows"]) for r in unexplained),
        "steps": rows,
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(json.dumps(score(Path(sys.argv[1])), indent=2))


if __name__ == "__main__":
    main()
