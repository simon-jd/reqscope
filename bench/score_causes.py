"""Score blame's slow-step causes against the injected causes of multicause runs.

    python bench/score_causes.py [--rule RULE] results/raw/multicause-trace-s1 results/raw/multicause-plugin-s1 ...

For each run directory: runs `reqscope blame` on the capture unless
blame.json exists, then reads the injected events from client.json, which
blame never sees. The 16 decodes run in lockstep, so slow gaps are grouped
by the engine step that produced them; a step's predicted cause is the most
common cause blame gave its gaps. Its true cause is the injected event whose
window [start, end + 20 ms] overlaps the step's gap the most, or "none" when
no event overlaps it (the run's own slow steps).

--rule picks how a slow step's cause is predicted; the slow steps and the
scoring are the same for every rule:
- blame (default): the most common cause blame gave the step's gaps;
- trivial: interference if the step had any co-scheduled prefill, else host
  delay;
- prefill_only: interference if the step had any co-scheduled prefill, else
  undetermined: what per-step token counts alone can say.

Per run and summed per collector mode:
- the confusion matrix, true cause by predicted cause;
- per cause: recall over the steps with that true cause; precision over
  all steps given that cause, and over the steps inside some event window
  (excluding the run's own slow steps);
- per cause: the share of injected events with at least one step given the
  right cause;
- per run, the host's mean 1-minute load average during the run;
- for steps blame gave their true cause, GPU slowdown or host delay, the
  medians of the evidence behind the verdict (the first gap's
  cause_evidence): GPU busy time and its baseline, launch wait and its
  baseline, and host extra;
- host-delay verdicts that blame flagged "collector_active" (the
  collector was busy for at least half of the host extra), by true cause,
  and host-delay precision over the verdicts that were not flagged.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import mean_load, reqscope  # noqa: E402

CAUSES = ["interference", "gpu_slowdown", "host_delay"]
PREDICTED = CAUSES + ["unexplained", "undetermined"]
SPILL_NS = 20_000_000
EVIDENCE = ["gpu_busy_ns", "gpu_busy_baseline_ns", "launch_wait_ns", "launch_wait_baseline_ns", "host_extra_ns"]
RULES = ["blame", "trivial", "prefill_only"]


def steps_of(blame: dict) -> dict[int, dict]:
    steps: dict[int, dict] = {}
    for row in blame["requests"]:
        if "decode-" not in row["req_id"]:
            continue
        for gap in row["slow_gaps"]:
            if not gap["steps"]:
                continue
            step = gap["steps"][0]
            entry = steps.setdefault(step["step_id"],
                                     {"begin": gap["begin_ns"], "end": gap["end_ns"], "causes": Counter(),
                                      "flagged": False, "evidence": gap.get("cause_evidence", {}),
                                      "prefill": any(m["phase"] == "prefill" for m in step["co_scheduled"])})
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


def predict(rule: str, step: dict) -> str:
    if rule == "blame":
        return step["causes"].most_common(1)[0][0]
    if step["prefill"]:
        return "interference"
    return "host_delay" if rule == "trivial" else "undetermined"


def ratio(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def medians(evidence: dict) -> dict:
    return {cause: {"steps": len(values[EVIDENCE[0]]),
                    **{f: statistics.median(values[f]) if values[f] else None for f in EVIDENCE}}
            for cause, values in sorted(evidence.items())}


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


def score(run: Path, rule: str = "blame") -> tuple[dict, Counter, Counter, Counter, Counter, dict]:
    client = json.loads((run / "client.json").read_text())
    blame_path = run / "blame.json"
    if not blame_path.exists():
        blame_path.write_text(json.dumps(reqscope("blame", str(run / "capture"))))
    blame = json.loads(blame_path.read_text())
    events = client["results"]["events"]
    confusion, detected, injected = Counter(), Counter(), Counter(e["cause"] for e in events)
    flagged = Counter()
    hits = defaultdict(set)
    evidence = defaultdict(lambda: defaultdict(list))
    for step in steps_of(blame).values():
        cause, index = truth(step, events)
        predicted = predict(rule, step)
        confusion[(cause, predicted)] += 1
        flagged[(cause, predicted)] += step["flagged"]
        if index is not None and predicted == cause:
            hits[cause].add(index)
        if rule == "blame" and predicted == cause and cause in ("gpu_slowdown", "host_delay"):
            for field in EVIDENCE:
                if step["evidence"].get(field) is not None:
                    evidence[cause][field].append(step["evidence"][field])
    for cause, found in hits.items():
        detected[cause] = len(found)
    result = {"run": run.name, "rule": rule, "mode": client["mode"], "seed": client["results"]["seed"],
              "load1_mean": mean_load(client.get("loadavg", []), client["started_ns"], client["ended_ns"]),
              "slow_steps": sum(confusion.values()), "trace_status": blame.get("trace", {}).get("status"),
              **metrics(confusion, detected, injected, flagged), "verdict_evidence_medians": medians(evidence)}
    return result, confusion, detected, injected, flagged, evidence


def main() -> None:
    args = sys.argv[1:]
    rule = "blame"
    if args[:1] == ["--rule"] and len(args) > 1:
        rule, args = args[1], args[2:]
    if not args or rule not in RULES:
        sys.exit(__doc__)
    runs, totals = [], defaultdict(lambda: [Counter(), Counter(), Counter(), Counter(), [],
                                            defaultdict(lambda: defaultdict(list))])
    for path in args:
        result, confusion, detected, injected, flagged, evidence = score(Path(path), rule)
        runs.append(result)
        total = totals[result["mode"]]
        total[0].update(confusion); total[1].update(detected); total[2].update(injected); total[3].update(flagged)
        total[4].append(result["seed"])
        for cause, fields in evidence.items():
            for field, values in fields.items():
                total[5][cause][field] += values
    by_mode = {mode: {"seeds": seeds, "slow_steps": sum(c.values()), **metrics(c, d, i, f),
                      "verdict_evidence_medians": medians(e)}
               for mode, (c, d, i, f, seeds, e) in totals.items()}
    print(json.dumps({"rule": rule, "by_mode": by_mode, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
