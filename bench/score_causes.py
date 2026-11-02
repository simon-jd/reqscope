"""Score blame's slow-step causes against the injected causes of multicause runs.

    python bench/score_causes.py results/raw/multicause-trace-s1 results/raw/multicause-plugin-s1 ...

For each run directory: runs `reqscope blame` on the capture unless
blame.json exists, then reads the injected events from client.json, which
blame never sees. The 16 decodes run in lockstep, so slow gaps are grouped
by the engine step that produced them; a step's predicted cause is the most
common cause blame gave its gaps. Its true cause is the injected event whose
window [start, end + 20 ms] overlaps the step's gap the most, or "none" when
no event overlaps it (the run's own slow steps).

Per run and summed per collector mode:
- the confusion matrix, true cause by predicted cause;
- per cause: recall over the steps with that true cause; precision over
  all steps given that cause, and over the steps inside some event window
  (excluding the run's own slow steps);
- per cause: the share of injected events with at least one step given the
  right cause;
- per run, the host's mean 1-minute load average during the run;
- host-delay verdicts that blame flagged "collector_active" (the
  collector was busy for at least half of the host extra), by true cause,
  and host-delay precision over the verdicts that were not flagged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import mean_load, reqscope  # noqa: E402

CAUSES = ["interference", "gpu_slowdown", "host_delay"]
PREDICTED = CAUSES + ["unexplained", "undetermined"]
SPILL_NS = 20_000_000


def steps_of(blame: dict) -> dict[int, dict]:
    steps: dict[int, dict] = {}
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        for gap in row["slow_gaps"]:
            if not gap["steps"]:
                continue
            entry = steps.setdefault(gap["steps"][0]["step_id"],
                                     {"begin": gap["begin_ns"], "end": gap["end_ns"], "causes": Counter(),
                                      "flagged": False})
            entry["causes"][gap["cause"]] += 1
            entry["flagged"] |= "collector_active" in gap.get("flags", [])
    return steps


def truth(step: dict, events: list[dict]) -> tuple[str, int | None]:
    best, index, most = "none", None, 0
    for i, event in enumerate(events):
        overlap = min(step["end"], event["end_ns"] + SPILL_NS) - max(step["begin"], event["start_ns"])
        if overlap > most:
            best, index, most = event["cause"], i, overlap
    return best, index


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def metrics(confusion: Counter, detected: Counter, injected: Counter, flagged: Counter) -> dict:
    per_cause = {}
    for cause in CAUSES:
        true = sum(n for (t, _), n in confusion.items() if t == cause)
        given = sum(n for (_, p), n in confusion.items() if p == cause)
        given_inside = sum(n for (t, p), n in confusion.items() if p == cause and t != "none")
        right = confusion[(cause, cause)]
        per_cause[cause] = {"true_steps": true, "recall": ratio(right, true),
                            "precision": ratio(right, given), "precision_inside_windows": ratio(right, given_inside),
                            "events": injected[cause], "events_detected": detected[cause]}
    matrix = {t: {p: confusion[(t, p)] for p in PREDICTED} for t in CAUSES + ["none"]}
    given = sum(n for (_, p), n in confusion.items() if p == "host_delay")
    unflagged_right = confusion[("host_delay", "host_delay")] - flagged[("host_delay", "host_delay")]
    unflagged = given - sum(n for (_, p), n in flagged.items() if p == "host_delay")
    flags = {"host_delay_flagged_by_true_cause": {t: flagged[(t, "host_delay")] for t in CAUSES + ["none"]},
             "host_delay_precision_unflagged": ratio(unflagged_right, unflagged),
             "host_delay_recall_unflagged": ratio(unflagged_right, sum(n for (t, _), n in confusion.items()
                                                                        if t == "host_delay"))}
    return {"per_cause": per_cause, "confusion": matrix, "collector_active_flags": flags}


def score(run: Path) -> tuple[dict, Counter, Counter, Counter, Counter]:
    client = json.loads((run / "client.json").read_text())
    blame_path = run / "blame.json"
    if not blame_path.exists():
        blame_path.write_text(json.dumps(reqscope("blame", str(run / "capture"))))
    blame = json.loads(blame_path.read_text())
    events = client["results"]["events"]
    confusion, detected, injected = Counter(), Counter(), Counter(e["cause"] for e in events)
    flagged = Counter()
    hits = defaultdict(set)
    for step in steps_of(blame).values():
        cause, index = truth(step, events)
        predicted = step["causes"].most_common(1)[0][0]
        confusion[(cause, predicted)] += 1
        flagged[(cause, predicted)] += step["flagged"]
        if index is not None and predicted == cause:
            hits[cause].add(index)
    for cause, found in hits.items():
        detected[cause] = len(found)
    result = {"run": run.name, "mode": client["mode"], "seed": client["results"]["seed"],
              "load1_mean": mean_load(client.get("loadavg", []), client["started_ns"], client["ended_ns"]),
              "slow_steps": sum(confusion.values()), "trace_status": blame.get("trace", {}).get("status"),
              **metrics(confusion, detected, injected, flagged)}
    return result, confusion, detected, injected, flagged


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs, totals = [], defaultdict(lambda: [Counter(), Counter(), Counter(), Counter(), []])
    for path in sys.argv[1:]:
        result, confusion, detected, injected, flagged = score(Path(path))
        runs.append(result)
        total = totals[result["mode"]]
        total[0].update(confusion); total[1].update(detected); total[2].update(injected); total[3].update(flagged)
        total[4].append(result["seed"])
    by_mode = {mode: {"seeds": seeds, "slow_steps": sum(c.values()), **metrics(c, d, i, f)}
               for mode, (c, d, i, f, seeds) in totals.items()}
    print(json.dumps({"by_mode": by_mode, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
